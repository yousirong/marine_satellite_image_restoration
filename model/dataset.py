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

        # Load all images from subdirectories
        self.data = self.load_list(image_path)
        print(f"Total training samples: {len(self.data)}")

        # Load all mask files
        self.mask_data = self.load_list(mask_path)
        print(f"Total masks: {len(self.mask_data)}")
        if len(self.mask_data) == 0:
            print(f"Warning: No mask files found in the directory: {mask_path}")

        # Load the land-sea mask (assuming this is a .npy file)
        self.land_sea_mask = self.load_land_sea_mask(land_sea_mask_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, mask = self.load_item(index, test_mode=not self.training)
        if img is None or mask is None:
            return self.__getitem__((index + 1) % len(self.data))

        # ====================================================================
        if np.isnan(img).any() or np.isinf(img).any() \
        or np.isnan(mask).any() or np.isinf(mask).any():
            print(f"[Dataset] NaN or inf detected — skipping: {self.data[index]}")
            return self.__getitem__((index + 1) % len(self.data))
        # ====================================================================

        filename = os.path.basename(self.data[index])
        img_tensor = self.to_tensor(img)
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return img_tensor, mask_tensor, filename


    def load_item(self, index, test_mode=False):
        max_attempts = 10
        for attempt in range(max_attempts):
            img = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            if img.ndim == 2:
                img = np.stack([img]*3, axis=2)
            elif img.shape[2] == 1:
                img = np.concatenate([img]*3, axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.shape[2] != 3:
                continue

            img = img.astype(np.float32)
            img[img == -999] = np.nan
            img = np.nan_to_num(img, 0.0)

            if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                img = cv2.resize(img, (self.target_size, self.target_size))
            img = np.transpose(img, (2, 0, 1))

            mask = self.load_mask(img, index)
            land_sea_mask_patch = self.get_land_sea_mask_patch(img, index, self.land_sea_mask)
            if land_sea_mask_patch.shape[1:] != (self.target_size, self.target_size):
                land_sea_mask_patch = cv2.resize(
                    land_sea_mask_patch.transpose(1, 2, 0),
                    (self.target_size, self.target_size)
                )
                land_sea_mask_patch = np.transpose(land_sea_mask_patch, (2, 0, 1))

            land_removed_mask = self.remove_land_from_mask(mask, land_sea_mask_patch)

            sea_mask = (land_sea_mask_patch == 0).astype(np.uint8)
            total_ocean = sea_mask.sum()
            if total_ocean == 0:
                continue

            ocean_holes = (land_removed_mask == 1) & (sea_mask == 1)
            if ocean_holes.sum() / total_ocean >= 0.01:
                return img, land_removed_mask

        return img, land_removed_mask

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
        mask_image: np.array [3,H,W], 1=hole,0=valid
        land_sea_mask_patch: np.array [3,H,W], 1=sea,0=land

        반환: np.array [3,H,W],  land(흰색=1), sea-hole(검은색=0), sea-valid(흰색=1)
        """
        # 1) 1채널로 축소
        sea = land_sea_mask_patch[0]    # 1=sea, 0=land
        hole = mask_image[0]            # 1=hole, 0=valid

        # 2) 기본값을 1(흰색)으로 세팅
        final = np.ones_like(hole, dtype=np.uint8)

        # 3) 바다 영역의 hole만 0(검은색)으로 변경
        final[(sea == 0) & (hole == 1)] = 0

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
        lm = np.where(lm == 999, 1, 0).astype(np.uint8)
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