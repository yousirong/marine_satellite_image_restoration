import sys
import os
import re
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

# ─── 프로젝트 루트를 sys.path에 추가 ───
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model.model import RFRNetModel


def check_ocean_missing_data(image_dir, land_sea_mask_path, patch_size=256):
    """
    해당 이미지 디렉토리에서 해양 영역에 결측치가 있는지 미리 확인
    Returns: (has_ocean_missing, sample_count)
    """
    try:
        # 육지-해양 마스크 로드
        lm = np.load(land_sea_mask_path)
        print(f"    DEBUG: Land mask unique values: {np.unique(lm)}")
        land_sea_mask = np.where(lm == 999, 0, 1).astype(np.uint8)  # 0=육지(999), 1=바다(others)
        print(f"    DEBUG: Converted mask unique values: {np.unique(land_sea_mask)}")

        # TIFF 파일 목록 가져오기
        files = []
        for root, _, fnames in os.walk(image_dir):
            for f in fnames:
                if f.lower().endswith('.tiff'):
                    files.append(os.path.join(root, f))

        if len(files) == 0:
            return False, 0

        # 샘플링하여 빠른 검사 (최대 10개 파일만 확인, 다양성을 위해 균등하게 선택)
        step = max(1, len(files) // 10)
        sample_files = files[::step][:10]

        ocean_missing_patches = 0
        total_ocean_missing_pixels = 0

        for path in sample_files:
            try:
                # 이미지 로드
                img = tiff.imread(path).astype(np.float32)

                # 채널 정규화 (첫 번째 채널만 사용하여 중복 계산 방지)
                if img.ndim == 2:
                    img_channel = img
                elif len(img.shape) == 3:
                    img_channel = img[:, :, 0]  # 첫 번째 채널만 사용
                else:
                    continue

                # 파일명에서 좌표 추출
                m = re.search(r'y(\d+)_x(\d+)', os.path.basename(path))
                if not m:
                    continue
                y0, x0 = map(int, m.groups())

                # 해당 패치의 육지-해양 마스크 추출
                try:
                    land_sea_mask_patch = land_sea_mask[y0:y0 + patch_size, x0:x0 + patch_size]
                except IndexError:
                    continue

                if land_sea_mask_patch.shape != (patch_size, patch_size):
                    land_sea_mask_patch = cv2.resize(land_sea_mask_patch.astype(np.uint8), (patch_size, patch_size))

                # 이미지 크기 조정
                if img_channel.shape[0] != patch_size or img_channel.shape[1] != patch_size:
                    img_channel = cv2.resize(img_channel, (patch_size, patch_size))

                # 해양 영역에서 결측치 확인
                ocean_mask = (land_sea_mask_patch == 1)  # 해양 영역
                land_mask = (land_sea_mask_patch == 0)   # 육지 영역
                ocean_pixels = np.sum(ocean_mask)
                land_pixels = np.sum(land_mask)

                # 디버깅: 패치별 육지/해양 비율 확인
                print(f"    DEBUG: {os.path.basename(path)} - Ocean: {ocean_pixels}, Land: {land_pixels}, Total: {ocean_pixels + land_pixels}")

                if ocean_pixels == 0:  # 해양 영역이 없으면 스킵
                    print(f"    DEBUG: {os.path.basename(path)} - SKIP: No ocean pixels")
                    continue

                # 결측치 정의: -999, NaN, 0, 1000
                missing_mask = ((img_channel == -999) | np.isnan(img_channel) | (img_channel == 0) | (img_channel == 1000))

                # 해양 영역의 결측치 확인
                ocean_missing = ocean_mask & missing_mask
                ocean_missing_in_patch = np.sum(ocean_missing)

                if ocean_missing_in_patch > 0:
                    ocean_missing_patches += 1
                    total_ocean_missing_pixels += ocean_missing_in_patch

                    print(f"    DEBUG: {os.path.basename(path)} - Ocean missing: {ocean_missing_in_patch}/{ocean_pixels} pixels")
                else:
                    print(f"    DEBUG: {os.path.basename(path)} - No missing pixels in ocean area")

            except Exception as e:
                print(f"    DEBUG: Error processing {os.path.basename(path)}: {e}")
                continue

        # 실제로 해양 결측치가 있는 패치가 있는지 확인
        has_ocean_missing = ocean_missing_patches > 0

        if has_ocean_missing:
            print(f"    DEBUG: Found {ocean_missing_patches} patches with ocean missing data, total {total_ocean_missing_pixels} pixels")

        return has_ocean_missing, total_ocean_missing_pixels

    except Exception:
        return False, 0


def find_dates_with_ocean_missing(base_data_dir, land_sea_mask_path):
    """
    모든 년도/날짜/시간에서 해양 결측치가 있는 것들만 찾아서 반환
    Returns: [(year, date, time), ...]
    """
    valid_dates = []

    if not os.path.exists(base_data_dir):
        print(f"[ERROR] Base data directory not found: {base_data_dir}")
        return valid_dates

    print(f"[INFO] Scanning all years in: {base_data_dir}")

    # 모든 년도 폴더 검색
    for year_item in sorted(os.listdir(base_data_dir)):
        year_path = os.path.join(base_data_dir, year_item)
        if not os.path.isdir(year_path):
            continue

        print(f"\n[INFO] Checking year: {year_item}")
        year_count = 0

        # 각 년도의 날짜 폴더 검색
        for date_item in sorted(os.listdir(year_path)):
            date_path = os.path.join(year_path, date_item)
            if not os.path.isdir(date_path):
                continue

            # 각 날짜의 시간 폴더 검색
            for time_item in sorted(os.listdir(date_path)):
                time_path = os.path.join(date_path, time_item)
                if not os.path.isdir(time_path):
                    continue

                # 시간 필터링 (00-07시만)
                if time_item[:2] not in {'00', '01', '02', '03', '04', '05', '06', '07'}:
                    continue

                # 해양 결측치 확인
                has_missing, missing_count = check_ocean_missing_data(time_path, land_sea_mask_path)

                if has_missing:
                    valid_dates.append((year_item, date_item, time_item))
                    year_count += 1
                    print(f"  ✓ {date_item} {time_item}: {missing_count} ocean missing pixels - ADDED")

        print(f"[SUMMARY] Year {year_item}: Found {year_count} valid time periods")

    return valid_dates


class GOCIEvalDataset(Dataset):
    """
    GOCI 평가용 데이터셋.
    - dataset.py와 model.py의 test 함수 로직을 적용
    - 해양 영역에서만 결측치를 복원 대상으로 처리
    - 육지는 마스킹에서 제외
    """
    def __init__(self, image_dir, land_sea_mask_path, patch_size=256):
        self.patch_size = patch_size

        # 실제 데이터 분포에 맞춘 전처리 파라미터
        # -999, NaN, 0값을 모두 결측값으로 사용하여 복원
        self.missing_value = -999
        self.valid_min = -1000
        self.valid_max = 1000
        self.min_valid_ratio = 0.05

        # 복원 조건 임계값 설정 (해양 영역에서만 처리하도록 더 엄격하게 조정)
        self.min_hole_ratio = 0.01   # 1% - 복원할 부분이 이것보다 적으면 복원하지 않음
        self.max_hole_ratio = 0.95   # 95% - 복원할 부분이 이것보다 많으면 데이터 부족으로 복원하지 않음
        self.min_ocean_ratio = 0.3   # 30% - 패치에서 해양 영역이 이것보다 적으면 건너뜀

        # 육지-해양 마스크 로드 및 변환 (dataset.py의 load_land_sea_mask와 동일)
        lm = np.load(land_sea_mask_path)
        print("Before conversion:", np.unique(lm))
        self.land_sea_mask = np.where(lm == 999, 0, 1).astype(np.uint8)  # 0=육지(999), 1=바다
        print("After conversion:", np.unique(self.land_sea_mask))

        # 이미지 파일 목록 생성
        self.files = []
        for root, _, fnames in os.walk(image_dir):
            for f in fnames:
                if f.lower().endswith('.tiff'):
                    self.files.append(os.path.join(root, f))
        self.files.sort()
        if len(self.files) == 0:
            print(f"[Dataset] WARNING: No tiles found in '{image_dir}'")
        else:
            print(f"[Dataset] Found {len(self.files)} tiles")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 인덱스가 범위를 벗어나면 예외 발생
        if index >= len(self.files):
            raise IndexError(f"Index {index} is out of range for dataset with {len(self.files)} files")

        # 지정된 인덱스의 파일만 처리 (순환 참조 방지)
        path = self.files[index]
        attempts = 1

        try:
                # 1. 원본 이미지 로드
                img = tiff.imread(path).astype(np.float32)

                # 채널 정규화 (dataset.py와 동일)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=2)
                elif len(img.shape) == 3 and img.shape[2] == 1:
                    img = np.concatenate([img]*3, axis=2)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                elif len(img.shape) == 3 and img.shape[2] != 3:
                    img = np.stack([img[:,:,0]]*3, axis=2)
                elif img.ndim != 3:
                    # 2D 이미지를 3D로 변환
                    img = np.stack([img]*3, axis=2)

                # 극값을 -999로 처리
                extreme_mask = (img < -1000) | (img > 1000)
                img[extreme_mask] = self.missing_value

                # 2. 테스트용 결측값 채우기 먼저 수행
                img_filled = self.simple_fill_missing(img.copy())

                # 유효한 픽셀 비율 확인 (채워진 이미지에서)
                missing_mask = (img == self.missing_value) | np.isnan(img) | (img == 0)
                valid_pixels = np.sum(~missing_mask)
                total_pixels = img.size
                valid_ratio = valid_pixels / total_pixels

                # 임계값을 더 관대하게 조정 (0.05 -> 0.01)
                if valid_ratio < 0.01:  # 1%
                    # 조용히 GT passthrough 처리 (출력 없음)
                    return self.create_passthrough_sample(img_filled, path, verbose=False)

                # 3. 크기 조정
                if img_filled.shape[0] != self.patch_size or img_filled.shape[1] != self.patch_size:
                    img_filled = cv2.resize(img_filled, (self.patch_size, self.patch_size))

                # 4. 채널 순서 변경 (H,W,C) -> (C,H,W)
                img_filled = np.transpose(img_filled, (2, 0, 1))

                # 5. 파일명에서 좌표 추출
                m = re.search(r'y(\d+)_x(\d+)', os.path.basename(path))
                if not m:
                    print(f"Cannot extract coords from filename: {path}, returning GT as-is")
                    return self.create_passthrough_sample(img_filled, path)
                y0, x0 = map(int, m.groups())

                # 6. 육지-해양 마스크 패치 가져오기 (dataset.py의 get_land_sea_mask_patch 로직)
                try:
                    land_sea_mask_patch = self.land_sea_mask[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
                except IndexError as e:
                    print(f"Land-sea mask patch out of bounds for {path}, returning GT as-is")
                    return self.create_passthrough_sample(img_filled, path)

                if land_sea_mask_patch.shape != (self.patch_size, self.patch_size):
                    land_sea_mask_patch = cv2.resize(land_sea_mask_patch.astype(np.uint8), (self.patch_size, self.patch_size))

                # --- 육지 영역을 흰색으로 강제 설정 ---
                # land_sea_mask_patch에서 육지(0)인 영역을 찾아
                # 입력 이미지(img_filled)의 해당 픽셀 값을 흰색(valid_max)으로 변경
                white_value = self.valid_max
                img_filled[:, land_sea_mask_patch == 0] = white_value
                # ------------------------------------

                land_sea_mask_patch = np.repeat(land_sea_mask_patch[np.newaxis, :, :], 3, axis=0)

                # 7. 마스크 생성 (dataset.py의 remove_land_from_mask 로직)
                # ⚠️ 중요: 원본 이미지를 사용해서 결측값 위치를 정확히 찾고,
                # 육지 영역만 따로 제외하는 방식으로 수정
                img_for_mask = img.copy()  # 원본 이미지 사용 (결측값 보존)
                if img_for_mask.shape[0] != self.patch_size or img_for_mask.shape[1] != self.patch_size:
                    img_for_mask = cv2.resize(img_for_mask, (self.patch_size, self.patch_size))
                img_for_mask = np.transpose(img_for_mask, (2, 0, 1))  # (H,W,C) -> (C,H,W)
                
                # 해양 영역에서만 결측치를 처리하도록 수정
                land_mask_2d = land_sea_mask_patch[0]  # 0=land, 1=sea
                
                # 원본 결측값 마스크 먼저 생성 (1=hole, 0=valid)
                # -999, NaN, 0, 1000값을 모두 결측값으로 처리
                hole_mask = ((img_for_mask == self.missing_value) | np.isnan(img_for_mask) | (img_for_mask == 0) | (img_for_mask == 1000)).astype(np.uint8)
                
                # 해양 영역에서만 결측치를 유지하고, 육지 영역의 결측치는 제거
                for c in range(3):
                    # 육지 영역(land_mask_2d == 0)의 결측치를 유효값으로 변경
                    land_holes = hole_mask[c] & (land_mask_2d == 0)
                    hole_mask[c][land_holes] = 0  # 육지의 결측치를 유효값으로 설정

                # 디버깅: 실제 데이터 값 분포 확인
                print(f"  - Image unique values: {np.unique(img_for_mask[0])[:10]}...")  # 첫 10개만 출력
                print(f"  - Missing value count (-999): {np.sum(img_for_mask == self.missing_value)}")
                print(f"  - NaN count: {np.sum(np.isnan(img_for_mask))}")
                print(f"  - Zero count: {np.sum(img_for_mask == 0)}")
                print(f"  - 1000 count (GT invalid): {np.sum(img_for_mask == 1000)}")
                print(f"  - Hole mask sum: {hole_mask.sum()}")  # 실제 hole 개수
                
                # 육지/해양 구분 디버깅
                land_mask_2d = land_sea_mask_patch[0]  # 0=land, 1=sea
                land_holes = np.sum(hole_mask[0] & (land_mask_2d == 0))  # 육지 위 hole
                sea_holes = np.sum(hole_mask[0] & (land_mask_2d == 1))   # 해양 위 hole
                print(f"  - Land holes (should be 0 after fix): {land_holes}")
                print(f"  - Sea holes (reconstruction target): {sea_holes}")

                # dataset.py의 remove_land_from_mask 로직 적용
                land_removed_mask = self.remove_land_from_mask(hole_mask, land_sea_mask_patch)

                # 8. 해양 영역 검증 및 복원 필요성 판단
                sea_mask = (land_sea_mask_patch == 1).astype(np.uint8)  # 1=바다, 0=육지
                total_pixels = land_sea_mask_patch.size
                total_ocean = sea_mask.sum()
                ocean_ratio = total_ocean / total_pixels

                # 디버깅 정보
                print(f"  - Ocean pixels: {total_ocean}/{total_pixels} ({ocean_ratio:.1%})")

                # 해양 영역이 없거나 너무 적으면 건너뜀
                if total_ocean == 0:
                    print(f"  - SKIP: No ocean pixels in {os.path.basename(path)}")
                    return self.create_passthrough_sample(img_filled, path, verbose=False)

                if ocean_ratio < self.min_ocean_ratio:
                    print(f"  - SKIP: Ocean ratio {ocean_ratio:.1%} < {self.min_ocean_ratio:.1%} in {os.path.basename(path)}")
                    return self.create_passthrough_sample(img_filled, path, verbose=False)

                # 해양 영역에서 실제 결측 부분 계산  
                ocean_holes = (land_removed_mask[0] == 0) & (sea_mask[0] == 1)  # 0=hole, 1=valid
                ocean_hole_count = ocean_holes.sum()
                ocean_hole_ratio = ocean_hole_count / total_ocean

                # 복원할 부분이 충분하지 않으면 GT 그대로 사용
                if ocean_hole_ratio < self.min_hole_ratio:
                    print(f"  - SKIP: Ocean hole ratio {ocean_hole_ratio:.1%} < {self.min_hole_ratio:.1%} in {os.path.basename(path)}")
                    return self.create_passthrough_sample(img_filled, path, verbose=False)
                elif ocean_hole_ratio > self.max_hole_ratio:
                    print(f"  - SKIP: Ocean hole ratio {ocean_hole_ratio:.1%} > {self.max_hole_ratio:.1%} in {os.path.basename(path)}")
                    return self.create_passthrough_sample(img_filled, path, verbose=False)
                else:
                    # 실제 복원되는 경우만 출력
                    print(f"🔄 PROCESSING: {os.path.basename(path)} - Ocean: {total_ocean}, Holes: {ocean_hole_count} ({ocean_hole_ratio:.1%})")
                    print(f"   → This patch has sufficient ocean missing data for reconstruction")

                # 9. model.py와 동일한 전처리 적용
                # img_filled를 torch tensor로 변환 (채널 순서 변경 전에)
                img_filled_torch = torch.from_numpy(img_filled.astype(np.float32))
                
                # RRS 데이터 전처리 (model.py와 동일)
                gt_processed, gt_valid_mask = self.preprocess_rrs_data_like_model(img_filled_torch)
                
                # 스마트 정규화 (model.py와 동일)  
                gt_normalized, data_min, data_max = self.smart_normalize_like_model(gt_processed, gt_valid_mask)
                
                # 마스크를 0-1 범위로 정규화
                mask_normalized = torch.from_numpy(land_removed_mask.astype(np.float32))
                mask_normalized = (mask_normalized > 0).float()
                
                print(f"   Preprocessing applied: range [{gt_normalized.min():.3f}, {gt_normalized.max():.3f}]")
                
                return gt_normalized, mask_normalized, os.path.basename(path)

        except Exception as e:
            # 오류는 조용히 처리 (출력 최소화)
            # 기본적인 이미지 로딩이라도 시도
            try:
                img = tiff.imread(path).astype(np.float32)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=2)
                elif len(img.shape) == 3 and img.shape[2] != 3:
                    img = np.stack([img[:,:,0]]*3, axis=2)

                if img.shape[0] != self.patch_size or img.shape[1] != self.patch_size:
                    img = cv2.resize(img, (self.patch_size, self.patch_size))
                img = np.transpose(img, (2, 0, 1))
                return self.create_passthrough_sample(img, path, verbose=False)
            except:
                # 완전히 실패한 경우 0으로 채운 이미지 반환
                img = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
                return self.create_passthrough_sample(img, path, verbose=False)

    def create_passthrough_sample(self, img_filled, path, verbose=True):
        """
        복원 조건을 만족하지 않는 샘플에 대해 GT 그대로 반환
        마스크는 모두 유효(1)로 설정하여 모델이 복원하지 않도록 함
        """
        # 이미지 텐서 변환
        if img_filled.ndim == 3 and img_filled.shape[2] == 3:
            # (H,W,C) -> (C,H,W) 변환이 필요한 경우
            img_filled = np.transpose(img_filled, (2, 0, 1))
        elif img_filled.ndim == 2:
            # 2D인 경우 3채널로 확장
            img_filled = np.stack([img_filled]*3, axis=0)

        # model.py와 동일한 전처리 적용
        img_tensor = torch.from_numpy(img_filled.astype(np.float32))
        
        # RRS 데이터 전처리 (model.py와 동일)
        gt_processed, gt_valid_mask = self.preprocess_rrs_data_like_model(img_tensor)
        
        # 스마트 정규화 (model.py와 동일)  
        gt_normalized, data_min, data_max = self.smart_normalize_like_model(gt_processed, gt_valid_mask)

        # 마스크는 모두 유효(1)로 설정 - 복원할 부분이 없음을 의미
        # 이렇게 하면 모델이 실제로는 아무것도 복원하지 않고 GT를 그대로 사용
        mask_tensor = torch.ones((3, self.patch_size, self.patch_size), dtype=torch.float32)

        if verbose:
            print(f"  -> GT passthrough for {os.path.basename(path)}: no reconstruction needed")
            print(f"     Applied model preprocessing: range [{gt_normalized.min():.3f}, {gt_normalized.max():.3f}]")
        
        return gt_normalized, mask_tensor, os.path.basename(path)

    def simple_fill_missing(self, img):
        """
        간단한 결측값 채우기 (테스트용) - -999, NaN, 0값을 모두 결측값으로 처리하여 복원
        다양성을 위해 약간의 노이즈 추가
        """
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            # -999, NaN, 0값을 모두 결측값으로 처리
            missing_mask = (channel == self.missing_value) | np.isnan(channel) | (channel == 0)

            if missing_mask.sum() > 0:
                valid_values = channel[~missing_mask]

                if len(valid_values) > 0:
                    # 유효한 값의 분포를 고려한 채우기
                    if len(valid_values) >= 5:
                        # 충분한 데이터가 있으면 25-75 percentile 범위에서 랜덤 채우기
                        q25 = np.percentile(valid_values, 25)
                        q75 = np.percentile(valid_values, 75)
                        if q75 > q25:
                            fill_values = np.random.uniform(q25, q75, missing_mask.sum())
                        else:
                            fill_values = np.full(missing_mask.sum(), np.median(valid_values))
                    else:
                        # 데이터가 적으면 median으로 채우되 약간의 노이즈 추가
                        median_val = np.median(valid_values)
                        noise = np.random.normal(0, abs(median_val) * 0.1, missing_mask.sum())
                        fill_values = median_val + noise
                    
                    channel[missing_mask] = fill_values
                else:
                    # 유효한 값이 없으면 작은 양수값으로 설정
                    channel[missing_mask] = 0.001

        return img

    def preprocess_rrs_data_like_model(self, data, mode='adaptive_fill'):
        """
        model.py의 preprocess_rrs_data와 동일한 전처리
        """
        processed_data = data.clone()
        missing_value = -999

        # 1. 결측값(-999) 마스크 생성
        missing_mask = (processed_data == missing_value)

        # 2. 매우 이상한 값들 제거
        extreme_mask = (processed_data < -1000) | (processed_data > 1000)

        # 3. 전체 무효값 마스크
        invalid_mask = missing_mask | extreme_mask

        if mode == 'adaptive_fill':
            # 배치별, 채널별 적응적 채우기
            for c in range(data.size(0)):  # eval에서는 채널이 첫 번째 차원
                channel_data = processed_data[c]
                channel_invalid = invalid_mask[c]

                # 유효한 값들만 추출
                valid_values = channel_data[~channel_invalid]

                if len(valid_values) > 0:
                    # 유효한 값들의 분포 분석
                    valid_sorted = torch.sort(valid_values)[0]
                    n_valid = len(valid_sorted)

                    if n_valid >= 10:
                        # 충분한 데이터가 있으면 10~90 percentile 사용
                        p10_idx = max(0, n_valid // 10)
                        p90_idx = min(n_valid - 1, (n_valid * 9) // 10)
                        fill_min = valid_sorted[p10_idx]
                        fill_max = valid_sorted[p90_idx]

                        # 범위 내에서 랜덤하게 채우기
                        fill_range = fill_max - fill_min
                        if fill_range > 0:
                            random_fill = fill_min + torch.rand_like(channel_data[channel_invalid]) * fill_range
                        else:
                            random_fill = torch.full_like(channel_data[channel_invalid], fill_min)
                        processed_data[c][channel_invalid] = random_fill
                    else:
                        # 데이터가 적으면 median으로 채우기
                        median_value = torch.median(valid_values)
                        processed_data[c][channel_invalid] = median_value
                else:
                    # 모든 값이 무효한 경우 0으로 설정
                    processed_data[c][channel_invalid] = 0.0

        return processed_data, ~invalid_mask  # 유효한 픽셀 마스크도 반환

    def smart_normalize_like_model(self, data, valid_mask=None):
        """
        model.py의 smart_normalize와 동일한 정규화
        """
        if valid_mask is not None:
            valid_data = data[valid_mask]
            if len(valid_data) > 0:
                # 실제 데이터 범위 확인
                data_min = torch.min(valid_data)
                data_max = torch.max(valid_data)

                # 데이터 분포에 맞는 정규화 범위 설정
                if data_min < -500:  # 큰 음수 값들이 있는 경우
                    norm_min = max(data_min.item(), -900)  # -900 이하는 클리핑
                    norm_max = min(data_max.item(), 100)   # 100 이상은 클리핑
                else:
                    norm_min = data_min.item()
                    norm_max = data_max.item()

                # 범위가 너무 작으면 기본값 사용
                if (norm_max - norm_min) < 1e-3:
                    norm_min, norm_max = -1.0, 1.0

                # Min-Max 정규화 (0-1 범위로)
                normalized = torch.clamp((data - norm_min) / (norm_max - norm_min), 0, 1)
                return normalized, norm_min, norm_max

        # fallback: 기본 정규화
        return torch.clamp((data + 900) / 950, 0, 1), -900.0, 50.0

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

        # 2) 기본값을 1(흰색)으로 세팅 - 모든 영역을 유효값으로 초기화
        final = np.ones_like(hole, dtype=np.uint8)

        # 3) 해양 영역(sea==1)의 결측치(hole==1)만 0(검은색)으로 변경
        # 육지 영역의 결측치는 무시하고 해양 영역의 결측치만 복원 대상으로 설정
        ocean_holes = (sea == 1) & (hole == 1)
        final[ocean_holes] = 0

        # 4) 3채널로 확장
        return np.repeat(final[np.newaxis, :, :], 3, axis=0)


if __name__ == '__main__':
    # 1. 모델 초기화 (model.py 사용)
    model = RFRNetModel()
    model.initialize_model(
        path='/home/juneyonglee/Desktop/AY_ust/mydata/5th_years/GOCI_RRS_band3_1day/g_1000000.pth',
        train=False,
        gpu_ids=[0, 1]
    )
    model.cuda()

    # 2. 데이터 경로 설정
    base_data_dir = '/media/juneyonglee/My Book1/Preprocessed/GOCI_tiles_daily/band2/2021'
    base_result = '/home/juneyonglee/myhdd/GOCI_RRS/daily_results/band2'
    land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

    print("="*80)
    print("========== FINDING DATES WITH OCEAN MISSING DATA ==========")
    print("="*80)

    # 경로 유형 확인 및 처리
    if os.path.isdir(base_data_dir):
        # 특정 년도를 직접 지정한 경우 (예: .../band3/2021)
        if os.path.basename(base_data_dir).isdigit():  # 마지막 폴더가 년도인 경우
            year = os.path.basename(base_data_dir)
            print(f"[INFO] Processing specific year: {year}")
            print(f"[INFO] Year directory: {base_data_dir}")

            # 특정 년도의 모든 날짜/시간 처리
            valid_dates = []
            for date_item in sorted(os.listdir(base_data_dir)):
                date_path = os.path.join(base_data_dir, date_item)
                if not os.path.isdir(date_path):
                    continue

                print(f"[INFO] Checking date: {date_item}")
                for time_item in sorted(os.listdir(date_path)):
                    time_path = os.path.join(date_path, time_item)
                    if not os.path.isdir(time_path):
                        continue

                    # 시간 필터링 (00-07시만)
                    if time_item[:2] not in {'00', '01', '02', '03', '04', '05', '06', '07'}:
                        continue

                    # 해양 결측치 확인
                    has_missing, missing_count = check_ocean_missing_data(time_path, land_sea_mask_path)

                    if has_missing:
                        valid_dates.append((year, date_item, time_item))
                        print(f"  ✓ {date_item} {time_item}: {missing_count} ocean missing pixels - ADDED")
        else:
            # 여러 년도가 포함된 디렉토리인 경우 (예: .../band3)
            valid_dates = find_dates_with_ocean_missing(base_data_dir, land_sea_mask_path)
    else:
        print(f"[ERROR] Directory not found: {base_data_dir}")
        valid_dates = []

    if not valid_dates:
        print("\n[WARNING] No dates found with ocean missing data!")
        print("Exiting without processing any dates.")
        exit(0)

    print(f"\n[SUMMARY] Found {len(valid_dates)} time periods with ocean missing data:")
    for year, date, time in valid_dates:
        print(f"  - {year}/{date} {time}")

    print("\n" + "="*80)
    print("========== PROCESSING DATES WITH OCEAN MISSING DATA ==========")
    print("="*80)

    # 4. 데이터 처리 루프
    processed_count = 0
    for year, date, time in valid_dates:
        print(f"\n[Main] Processing: {year}/{date} {time} ({processed_count+1}/{len(valid_dates)})")

        # 입력 디렉토리 설정
        if os.path.basename(base_data_dir).isdigit():
            # base_data_dir가 이미 특정 년도 경로인 경우 (예: .../band3/2021)
            time_folder = os.path.join(base_data_dir, date, time)
        else:
            # base_data_dir가 band3인 경우 (예: .../band3)
            time_folder = os.path.join(base_data_dir, year, date, time)

        # 5. 평가용 데이터셋 생성
        dataset = GOCIEvalDataset(
            image_dir=time_folder,
            land_sea_mask_path=land_sea_mask_path,
            patch_size=256
        )
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True
        )

        # 6. 결과 저장 경로 설정
        result_save_path = os.path.join(base_result, year, date, time)
        os.makedirs(result_save_path, exist_ok=True)
        print(f"[Main] Saving results under: {result_save_path}")

        # 7. 모델 테스트 실행 (model.py 사용)
        model.test(
            test_loader=loader,
            result_save_path=result_save_path
        )

        processed_count += 1
        print(f"[Main] Completed {processed_count}/{len(valid_dates)}")

    print(f"\n[Main] All {len(valid_dates)} time periods with ocean missing data processed.")
