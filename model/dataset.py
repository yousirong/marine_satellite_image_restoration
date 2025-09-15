import os
import cv2
import numpy as np
import torch
import random
import re

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, land_sea_mask_path,
                 mask_mode, target_size,
                 augment=False, training=True, mask_reverse=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.target_size = target_size if isinstance(target_size, int) else target_size[0]
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse

        # 실제 데이터 분포에 맞춘 전처리 파라미터
        self.missing_value = -999
        self.valid_min = -1000      # 유효한 데이터 최소값
        self.valid_max = 1000       # 유효한 데이터 최대값
        self.min_valid_ratio = 0.05 # 최소 5% 유효 픽셀 (더 관대하게)

        # Load data
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)
        self.land_sea_mask = self.load_land_sea_mask(land_sea_mask_path)

        print(f"Total training samples: {len(self.data)}")
        print(f"Total masks: {len(self.mask_data)}")

        if len(self.mask_data) == 0:
            print(f"Warning: No mask files found in the directory: {mask_path}")

        # 데이터 품질 검사
        if self.training:
            self.validate_dataset_quality()

    def validate_dataset_quality(self):
        """
        데이터셋 품질 검사
        """
        print("=== Dataset Quality Check ===")
        valid_samples = 0
        total_checked = min(100, len(self.data))  # 처음 100개 샘플 검사

        for i in range(total_checked):
            img = cv2.imread(self.data[i], cv2.IMREAD_UNCHANGED)
            if img is not None:
                img = img.astype(np.float32)

                # 유효한 픽셀 비율 계산
                valid_pixels = np.sum((img != self.missing_value) &
                                    (img >= self.valid_min) &
                                    (img <= self.valid_max) &
                                    (~np.isnan(img)))
                total_pixels = img.size
                valid_ratio = valid_pixels / total_pixels

                if valid_ratio >= self.min_valid_ratio:
                    valid_samples += 1

        print(f"Valid samples: {valid_samples}/{total_checked} ({valid_samples/total_checked*100:.1f}%)")
        if valid_samples < total_checked * 0.5:
            print("⚠️  Warning: Low quality dataset detected!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, mask = self.load_item(index, test_mode=not self.training)

        if img is None or mask is None:
            return self.__getitem__((index + 1) % len(self.data))

        # 최종 데이터 품질 검사 (더 관대한 기준)
        if (np.isnan(img).any() or np.isinf(img).any() or
            np.isnan(mask).any() or np.isinf(mask).any()):
            if self.training:
                print(f"  NaN/Inf detected, trying next sample...")
                return self.__getitem__((index + 1) % len(self.data))
            else:
                # 테스트 시에는 더 강력하게 정리
                img = np.nan_to_num(img, nan=0.0, posinf=100.0, neginf=-900.0)
                mask = np.nan_to_num(mask, nan=1.0).astype(np.uint8)
                print(f"  Cleaned NaN/Inf for test sample")

        filename = os.path.basename(self.data[index])
        img_tensor = self.to_tensor(img)
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return img_tensor, mask_tensor, filename

    def load_item(self, index, test_mode=False):
        """
        실제 데이터 분포에 맞춘 아이템 로딩
        """
        max_attempts = 15 if self.training else 3

        for attempt in range(max_attempts):
            try:
                current_index = (index + attempt) % len(self.data)
                img = cv2.imread(self.data[current_index], cv2.IMREAD_UNCHANGED)

                if img is None:
                    continue

                # 채널 정규화
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=2)
                elif img.shape[2] == 1:
                    img = np.concatenate([img]*3, axis=2)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]
                elif img.shape[2] != 3:
                    continue

                img = img.astype(np.float32)

                # ===== 실제 데이터 분포 분석 및 전처리 =====

                # 1. 극값 제거 (매우 비정상적인 값들)
                extreme_mask = (img < -2000) | (img > 2000)
                img[extreme_mask] = self.missing_value  # 극값을 결측값으로 변환

                # 2. 유효한 픽셀 비율 확인 (-999가 아닌 픽셀들)
                valid_pixels = np.sum(img != self.missing_value)
                total_pixels = img.size
                valid_ratio = valid_pixels / total_pixels

                # if self.training and attempt == 0:  # 첫 번째 시도에서만 출력
                #     print(f"Sample {current_index}: valid ratio = {valid_ratio:.3f}")

                if valid_ratio < self.min_valid_ratio:
                    if self.training:
                        if attempt == 0:
                            print(f"  Skipping due to low valid ratio: {valid_ratio:.3f}")
                        continue
                    else:
                        if attempt == 0:
                            print(f"  Low valid ratio but processing for test: {valid_ratio:.3f}")

                # 3. 결측값 처리 (훈련용과 테스트용 다르게)
                if self.training:
                    # 훈련 시: 채널별 적응적 채우기
                    img = self.adaptive_fill_missing(img)
                else:
                    # 테스트 시: 간단한 보간
                    img = self.simple_fill_missing(img)

                # 4. 크기 조정
                if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                    img = cv2.resize(img, (self.target_size, self.target_size))

                # 5. 채널 순서 변경
                img = np.transpose(img, (2, 0, 1))

                # 6. 마스크 생성
                mask = self.load_mask(img, current_index)
                land_sea_mask_patch = self.get_land_sea_mask_patch(img, current_index, self.land_sea_mask)

                if land_sea_mask_patch.shape[1:] != (self.target_size, self.target_size):
                    land_sea_mask_patch = cv2.resize(
                        land_sea_mask_patch.transpose(1, 2, 0),
                        (self.target_size, self.target_size)
                    )
                    land_sea_mask_patch = np.transpose(land_sea_mask_patch, (2, 0, 1))

                land_removed_mask = self.remove_land_from_mask(mask, land_sea_mask_patch)

                # 7. 해양 영역 검증
                sea_mask = (land_sea_mask_patch == 1).astype(np.uint8)  # 1=바다, 0=육지
                total_ocean = sea_mask.sum()

                if total_ocean == 0:
                    continue

                ocean_holes = (land_removed_mask == 0) & (sea_mask == 1)  # 0=hole, 1=valid
                ocean_hole_ratio = ocean_holes.sum() / total_ocean

                if ocean_hole_ratio >= 0.005:  # 0.5% 이상 (더 관대하게)
                    # if self.training and attempt == 0:
                    #     print(f"  Success: ocean hole ratio = {ocean_hole_ratio:.3f}")
                    return img, land_removed_mask
                else:
                    if self.training and attempt == 0:
                        print(f"  Low ocean hole ratio: {ocean_hole_ratio:.3f}")

            except Exception as e:
                if self.training and attempt == 0:
                    print(f"Error loading sample {current_index}: {e}")
                continue

        if self.training:
            print(f"Failed to load valid sample after {max_attempts} attempts")
            raise RuntimeError(f"Failed to load valid sample after {max_attempts} attempts")
        else:
            print(f"Failed to load sample at index {index}, returning None to skip")
            return None, None

    def adaptive_fill_missing(self, img):
        """
        적응적 결측값 채우기 (훈련용)
        """
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            missing_mask = (channel == self.missing_value)

            if missing_mask.sum() > 0:
                valid_values = channel[~missing_mask]

                if len(valid_values) > 0:
                    # 유효한 값들의 분포 분석
                    unique_values = np.unique(valid_values)

                    if len(unique_values) >= 5:
                        # 분포가 다양하면 percentile 기반 채우기
                        p25 = np.percentile(valid_values, 25)
                        p75 = np.percentile(valid_values, 75)

                        # 25-75 percentile 범위에서 랜덤 채우기
                        fill_values = np.random.uniform(p25, p75, missing_mask.sum())
                        channel[missing_mask] = fill_values
                    else:
                        # 값의 종류가 적으면 가장 빈번한 값으로 채우기
                        unique_vals, counts = np.unique(valid_values, return_counts=True)
                        most_frequent = unique_vals[np.argmax(counts)]
                        channel[missing_mask] = most_frequent
                else:
                    # 유효한 값이 없으면 0으로
                    channel[missing_mask] = 0.0

        return img

    def simple_fill_missing(self, img):
        """
        간단한 결측값 채우기 (테스트용)
        """
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            missing_mask = (channel == self.missing_value)

            if missing_mask.sum() > 0:
                valid_values = channel[~missing_mask]

                if len(valid_values) > 0:
                    # median으로 채우기
                    median_val = np.median(valid_values)
                    channel[missing_mask] = median_val
                else:
                    channel[missing_mask] = 0.0

        return img


    def load_mask(self, img, index):
        imgh, imgw = img.shape[1:]
        path = random.choice(self.mask_data)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((imgh, imgw), dtype=np.uint8)
        m = cv2.resize(m, (self.target_size, self.target_size))
        m = (m == 0).astype(np.uint8)
        m = np.repeat(m[np.newaxis, :, :], 3, axis=0)
        if self.mask_reverse:
            m = 1 - m
        return m

    def get_land_sea_mask_patch(self, img, index, land_sea_mask):
        imgh, imgw = img.shape[1:]
        fname = os.path.basename(self.data[index])
        row, col = self.extract_row_col(fname)
        if row is None or col is None:
            raise ValueError(f"Can't parse row/col: {fname}")
        if row+imgh > land_sea_mask.shape[0] or col+imgw > land_sea_mask.shape[1]:
            raise ValueError("Land-sea patch out of bounds")
        patch = land_sea_mask[row:row+imgh, col:col+imgw]
        patch = np.repeat(patch[np.newaxis, :, :], 3, axis=0)
        return patch

    def remove_land_from_mask(self, mask_image, land_sea_mask_patch):
        """
        해양 영역에서만 결측치 마스크를 생성하는 함수
        mask_image: np.array [3,H,W], 1=hole,0=valid
        land_sea_mask_patch: np.array [3,H,W], 1=sea,0=land

        반환: np.array [3,H,W], 해양 영역의 결측치만 0(검은색)으로 표시, 나머지는 1(흰색)
        """
        # 1) 1채널로 축소
        sea = land_sea_mask_patch[0]    # 1=sea, 0=land
        hole = mask_image[0]            # 1=hole, 0=valid

        # 2) 기본값을 1(흰색)으로 세팅
        final = np.ones_like(hole, dtype=np.uint8)

        # 3) 해양 영역(sea==1)의 결측치(hole==1)만 0(검은색)으로 변경
        # 육지 영역의 결측치는 무시하고 해양 영역의 결측치만 복원 대상으로 설정
        ocean_holes = (sea == 1) & (hole == 1)
        final[ocean_holes] = 0

        # 4) 3채널로 확장
        return np.repeat(final[np.newaxis, :, :], 3, axis=0)

    def extract_row_col(self, filename):
        m = re.search(r'r(\d+)_c(\d+)', filename)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    def load_land_sea_mask(self, path):
        """
        육지-해양 마스크 로드 및 변환
        원본: 1=육지(흰색), 999=바다(검은색)
        변환: 0=육지, 1=바다
        """
        lm = np.load(path)
        print("Before conversion:", np.unique(lm))
        lm = np.where(lm == 999, 0, 1).astype(np.uint8)  # 0=육지(999), 1=바다(others)
        print("After conversion:", np.unique(lm))
        return lm

    def to_tensor(self, img):
        """
        Convert the numpy array to a PyTorch tensor.
        Ensures the image is in a valid format and data type.
        """
        # If the input is a NumPy array, convert it to a PyTorch tensor
        if isinstance(img, np.ndarray):
            # Check if dtype conversion is necessary (e.g., from object type to float32)
            if img.dtype == object:
                print(f"Converting image dtype from object to float32.")
                img = img.astype(np.float32)

            # If the image is in 2D, expand it to 3D with a single channel
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=0)

            # Convert NumPy array to PyTorch tensor
            return torch.Tensor(img)

        # If the input is already a PyTorch tensor, just return it as-is
        elif isinstance(img, torch.Tensor):
            return img

        else:
            raise TypeError(f"Unsupported data type: {type(img)}")

    def load_list(self, path):
        if isinstance(path, str):
            if os.path.isdir(path):
                files = []
                for root, _, fnames in os.walk(path):
                    for f in fnames:
                        if f.lower().endswith(('.tiff', '.png')):
                            files.append(os.path.join(root, f))
                files.sort()
                return files
            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=str, encoding='utf-8').tolist()
                except:
                    return [path]
        return []