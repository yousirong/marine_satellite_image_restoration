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
        # if load_item failed
        if img is None or mask is None:
            return self.__getitem__((index + 1) % len(self.data))

        filename = os.path.basename(self.data[index])
        img_tensor = self.to_tensor(img)
        # for mask, keep values 0 or 1 exactly
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        return img_tensor, mask_tensor, filename

    def load_item(self, index, test_mode=False):
        max_attempts = 10
        for attempt in range(max_attempts):
            # 1) Load GT image
            img = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # handle channels
            if img.ndim == 2:
                img = np.stack([img]*3, axis=2)
            elif img.shape[2] == 1:
                img = np.concatenate([img]*3, axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            elif img.shape[2] != 3:
                continue

            # clean special values
            img = img.astype(np.float32)
            img[img == -999] = np.nan
            img = np.nan_to_num(img, 0.0)

            # resize and transpose
            if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                img = cv2.resize(img, (self.target_size, self.target_size))
            img = np.transpose(img, (2, 0, 1))  # C, H, W

            # 2) Load random mask
            mask = self.load_mask(img, index)

            # 3) Extract land-sea patch
            land_sea_mask_patch = self.get_land_sea_mask_patch(img, index, self.land_sea_mask)
            if land_sea_mask_patch.shape[1:] != (self.target_size, self.target_size):
                land_sea_mask_patch = cv2.resize(
                    land_sea_mask_patch.transpose(1, 2, 0),
                    (self.target_size, self.target_size)
                )
                land_sea_mask_patch = np.transpose(land_sea_mask_patch, (2, 0, 1))

            # 4) Remove land from mask
            land_removed_mask = self.remove_land_from_mask(mask, land_sea_mask_patch)

            # 5) Ensure at least 1% ocean coverage
            sea_mask = (land_sea_mask_patch == 0).astype(np.uint8)
            total_ocean = sea_mask.sum()
            if total_ocean == 0:
                continue

            ocean_holes = (land_removed_mask == 1) & (sea_mask == 0)
            if ocean_holes.sum() / total_ocean >= 0.01:
                return img, land_removed_mask

        # fallback after attempts
        return img, land_removed_mask

    def load_mask(self, img, index):
        imgh, imgw = img.shape[1:]
        path = random.choice(self.mask_data)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((imgh, imgw), dtype=np.uint8)
        m = cv2.resize(m, (self.target_size, self.target_size))
        # black (0) → hole (1), white (255) → keep (0)
        m = (m == 0).astype(np.uint8)
        # expand to 3 channels
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
        # sea=1, land=0 after conversion
        sea_mask = (land_sea_mask_patch == 0).astype(np.uint8)
        return mask_image * sea_mask

    def extract_row_col(self, filename):
        m = re.search(r'r(\d+)_c(\d+)', filename)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    def load_land_sea_mask(self, path):
        lm = np.load(path)
        print("Before conversion:", np.unique(lm))
        # original: 999=land → want 1, others=sea → want 0
        lm = np.where(lm == 999, 1, 0).astype(np.uint8)
        print("After conversion:", np.unique(lm))
        return lm

    def to_tensor(self, arr):
        """
        Convert a HWC or CHW NumPy array in [0,255] to a [0,1] float tensor.
        """
        t = torch.from_numpy(arr.astype(np.float32))
        # only normalize if values exceed 1.0
        if t.max() > 1.0:
            t = t.clamp(0, 255) / 255.0
        return t

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
