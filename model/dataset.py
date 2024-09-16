import os
import glob
import torch
import random
import numpy as np
from skimage import io
import cv2
from PIL import Image
import re

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, land_sea_mask_path, mask_mode, target_size, augment=False, training=True, mask_reverse=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.target_size = target_size if isinstance(target_size, int) else target_size[0]
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse

        # Load all images from subdirectories (including perfect and loss-based folders)
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
        try:
            item = self.load_item(index)
        except Exception as e:
            print(f"Loading error for: {self.data[index]}")
            print(e)
            item = self.load_item(0)
        return item

    def load_item(self, index):
        # Load the image
        img = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
        img[np.isnan(img)] = 0  # Replace NaNs with 0

        # Stack the image to create a 3-channel image if it's grayscale
        img = np.stack([img, img, img], axis=0)

        # Load and apply the mask (removing land areas)
        mask = self.load_mask(img, index)
        land_sea_mask_patch = self.get_land_sea_mask(img, index)

        # Remove land from the mask (keep ocean only)
        land_removed_mask = self.remove_land_from_mask(mask, land_sea_mask_patch)

        # Combine both masks (apply ocean-only mask)
        combined_mask = self.apply_land_sea_mask(mask, land_removed_mask)

        # Optional data augmentation
        if self.training and self.augment:
            if np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]
                combined_mask = combined_mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(combined_mask)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[1:3]

        if len(self.mask_data) == 0:
            raise ValueError("Mask data is empty. Please check the mask directory and ensure that masks are available.")
        
        if self.mask_type == 0:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = io.imread(self.mask_data[mask_index])
            mask = (mask > 0).astype(np.uint8)  # Threshold due to interpolation
            mask = self.resize(mask, False)
            mask = np.stack([mask, mask, mask], axis=0)
            if self.mask_reverse:
                return (1 - mask)
            else:
                return mask

    def apply_land_sea_mask(self, mask, land_removed_mask):
        """
        Combine land-sea mask and input mask. Keep ocean and mask out land.
        """
        # Apply land-sea mask where land is excluded (land = 0)
        combined_mask = np.where(land_removed_mask == 1, mask, 0)
        return combined_mask

    def get_land_sea_mask(self, img, index):
        """
        Extract the land-sea mask patch corresponding to the image.
        """
        imgh, imgw = img.shape[1:3]
        filename = os.path.basename(self.data[index])
        row, col = self.extract_row_col(filename)

        if row is None or col is None or row + imgh > self.land_sea_mask.shape[0] or col + imgw > self.land_sea_mask.shape[1]:
            raise ValueError(f"Row or column is out of bounds for land-sea mask. row: {row}, col: {col}, mask shape: {self.land_sea_mask.shape}")

        # Extract corresponding land-sea mask patch
        land_sea_mask_patch = self.land_sea_mask[row:row+imgh, col:col+imgw]
        land_sea_mask_patch = np.stack([land_sea_mask_patch, land_sea_mask_patch, land_sea_mask_patch], axis=0)

        return land_sea_mask_patch

    def remove_land_from_mask(self, mask_image, land_sea_mask_patch):
        """
        Remove land areas from the mask and only keep ocean areas.
        육지 부분을 0으로 설정하고 해양 부분만 유지합니다.
        """
        # Remove land areas (999) and set sea areas (1) to 255 in the mask
        return np.where((land_sea_mask_patch == 1) & (mask_image == 0), 255, mask_image)

    def extract_row_col(self, filename):
        """
        Extract row and column from the filename using regex.
        """
        match = re.search(r'r(\d+)_c(\d+)', filename)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            return row, col
        return None, None

    def load_land_sea_mask(self, land_mask_path):
        """
        Load the land-sea mask from a .npy file and convert values accordingly.
        """
        land_sea_mask = np.load(land_mask_path)
        print(f"Loaded land-sea mask with shape: {land_sea_mask.shape}, dtype: {land_sea_mask.dtype}")
        # Convert land = 0 to 999, and ocean = 1
        land_sea_mask = np.where(land_sea_mask == 0, 999, 1)
        return land_sea_mask

    def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
        """
        Resize the image to the target size.
        """
        imgh, imgw = img.shape[:2]
        img = np.array(Image.fromarray(img).resize(size=(self.target_size, self.target_size)))
        return img

    def to_tensor(self, img):
        """
        Convert the numpy array to a PyTorch tensor.
        """
        img_t = torch.Tensor(img)
        return img_t

    def load_list(self, path):
        """
        Load the list of files from the directory or a single file path.
        """
        if isinstance(path, str):
            if os.path.isdir(path):
                all_images = []
                for root, _, files in os.walk(path):
                    images = [os.path.join(root, file) for file in files if file.endswith(('.tiff', '.png'))]
                    all_images.extend(images)

                if len(all_images) == 0:
                    print(f"No images found in the directory: {path}")

                all_images.sort()
                return all_images

            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
                except:
                    return [path]
        return []
