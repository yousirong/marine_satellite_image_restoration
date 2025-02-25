import os
import cv2
import numpy as np
import torch
import random
import re

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, land_sea_mask_path, mask_mode, target_size, augment=False, training=True, mask_reverse=False):
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
        filename = os.path.basename(self.data[index])  # Extract just the filename
        return self.to_tensor(img), self.to_tensor(mask), filename

    def load_item(self, index, test_mode=False):
        max_attempts = 10  # 최대 시도 횟수 설정

        img = None
        land_removed_mask = None

        for attempt in range(max_attempts):
            # Load the image (GT)
            img = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
            if img is None:
                # print(f"Attempt {attempt+1}/{max_attempts}: Failed to load image at {self.data[index]}")
                continue  # 다음 시도로 넘어감

            # Handle different channel configurations
            if len(img.shape) == 2:
                # Grayscale image: Convert to 3 channels
                img = np.stack([img, img, img], axis=2)  # Shape: (H, W, 3)
            elif img.shape[2] == 1:
                # Single channel: Convert to 3 channels
                img = np.concatenate([img, img, img], axis=2)  # Shape: (H, W, 3)
            elif img.shape[2] == 4:
                # 4 channels (e.g., RGBA): Convert to 3 channels by dropping the alpha channel
                img = img[:, :, :3]  # Shape: (H, W, 3)
            elif img.shape[2] != 3:
                # print(f"Attempt {attempt+1}/{max_attempts}: Unexpected number of channels ({img.shape[2]}) in image: {self.data[index]}")
                continue  # 다음 시도로 넘어감

            # Replace NaNs with 0
            img = img.astype(np.float32)
            img[np.isnan(img)] = 0

            # Resize the image to the target size if it's not the correct size
            if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                img = cv2.resize(img, (self.target_size, self.target_size))

            # Transpose the image to have channels first (C, H, W)
            img = np.transpose(img, (2, 0, 1))  # Shape: (3, H, W)

            # Step 2: Load mask image
            if test_mode:
                # # For testing, generate mask based on image data
                # # Assuming that NaN or 0 in image indicates missing data
                # mask = ((img == 0).any(axis=0)).astype(np.uint8)  # Shape: (H, W)
                # mask = 1 - mask  # Invert mask: 1 for known, 0 for missing
                # mask = np.expand_dims(mask, axis=0)  # Shape: (1, H, W)
                # mask = np.repeat(mask, 3, axis=0)  # Shape: (3, H, W)
                mask = self.load_mask(img, index)
            else:
                # For training, load mask from mask files
                mask = self.load_mask(img, index)

            # Step 3: Load the land-sea mask
            land_sea_mask_patch = self.get_land_sea_mask_patch(img, index, self.land_sea_mask)

            # Ensure the land-sea mask patch is resized to match the target size
            if land_sea_mask_patch.shape[1] != self.target_size or land_sea_mask_patch.shape[2] != self.target_size:
                land_sea_mask_patch = cv2.resize(land_sea_mask_patch.transpose(1, 2, 0), (self.target_size, self.target_size))
                land_sea_mask_patch = np.transpose(land_sea_mask_patch, (2, 0, 1))  # Shape: (3, H, W)

            # Step 4: Modify the mask (set land areas to 1, ocean areas to 0)
            land_removed_mask = self.remove_land_from_mask(mask, land_sea_mask_patch)

            # Step 5: Calculate mask coverage within ocean pixels
            # 해양 영역을 정의 (sea_mask == 1)
            sea_mask = (land_sea_mask_patch != 1).astype(np.uint8)  # 육지: 1, 해양: 0 -> 해양: 1
            total_ocean_pixels = sea_mask.sum()
            if total_ocean_pixels == 0:
                # print(f"Attempt {attempt+1}/{max_attempts}: No ocean pixels found for image: {self.data[index]}")
                continue  # 다음 시도로 넘어감

            # 해양 영역 내에서 마스크가 적용된 픽셀 수
            masked_ocean_pixels = (land_removed_mask == 1) & (sea_mask == 1)
            masked_ocean_pixels_count = masked_ocean_pixels.sum()

            # 해양 영역 내 마스크 비율
            ocean_mask_ratio = masked_ocean_pixels_count / total_ocean_pixels

            # 디버깅 정보 추가
            # print(f"Attempt {attempt+1}/{max_attempts}: Ocean Mask Ratio = {ocean_mask_ratio*100:.2f}% for image: {self.data[index]}")

            if ocean_mask_ratio >= 0.01:
                # Optional data augmentation
                if self.training and self.augment:
                    if np.random.binomial(1, 0.5) > 0:
                        img = img[:, ::-1, ...]  # Flip image horizontally
                        land_removed_mask = land_removed_mask[:, ::-1, ...]

                # Convert the images and masks to tensors
                img_tensor = self.to_tensor(img)
                mask_tensor = self.to_tensor(land_removed_mask)

                return img_tensor, mask_tensor
            else:
                # print(f"Attempt {attempt+1}/{max_attempts}: Mask ocean area too small ({ocean_mask_ratio*100:.2f}%) for image: {self.data[index]}")
                continue  # 다음 시도로 넘어감

        # After max_attempts, log and return the last mask (even if ocean area is small)
        # print(f"Could not find a mask with ocean area >=1% for image: {self.data[index]} after {max_attempts} attempts, using the last mask.")
        img_tensor = self.to_tensor(img)
        mask_tensor = self.to_tensor(land_removed_mask)
        return img_tensor, mask_tensor

    def load_mask(self, img, index):
        """
        Load the mask for a specific image.
        """
        imgh, imgw = img.shape[1:3]

        if len(self.mask_data) == 0:
            raise ValueError("Mask data is empty. Please check the mask directory and ensure that masks are available.")

        # Based on the mask type, randomly select a mask
        mask_index = random.randint(0, len(self.mask_data) - 1)
        mask = cv2.imread(self.mask_data[mask_index], cv2.IMREAD_GRAYSCALE)

        # Check if the mask is loaded successfully
        if mask is None:
            print(f"Failed to load mask for index {mask_index}, using a blank mask instead.")
            mask = np.zeros((imgh, imgw), dtype=np.uint8)

        # Resize the mask to match the image size
        mask = self.resize(mask, False)

        # Ensure the mask is binary
        mask = (mask > 0).astype(np.uint8)  # 0 or 1

        # Stack into three channels if needed
        mask = np.expand_dims(mask, axis=0)  # Shape: (1, H, W)
        mask = np.repeat(mask, 3, axis=0)  # Shape: (3, H, W)

        # Apply mask reversal if specified
        if self.mask_reverse:
            mask = 1 - mask

        return mask

    def get_land_sea_mask_patch(self, img, index, land_sea_mask):
        """
        Extract the land-sea mask patch that corresponds to the current image based on row and col.
        """
        imgh, imgw = img.shape[1:3]  # Get the height and width of the image
        filename = os.path.basename(self.data[index])
        row, col = self.extract_row_col(filename)

        if row is None or col is None:
            raise ValueError(f"Could not extract row and col from filename: {filename}")

        # Ensure the land-sea mask patch is within bounds
        if row + imgh > land_sea_mask.shape[0] or col + imgw > land_sea_mask.shape[1]:
            raise ValueError(f"Row or column is out of bounds for land-sea mask. row: {row}, col: {col}, mask shape: {land_sea_mask.shape}")

        # Extract the land-sea mask patch based on the row and column
        land_sea_mask_patch = land_sea_mask[row:row+imgh, col:col+imgw]

        # Stack the mask into three channels to match the image's channels
        land_sea_mask_patch = np.expand_dims(land_sea_mask_patch, axis=0)  # Shape: (1, H, W)
        land_sea_mask_patch = np.repeat(land_sea_mask_patch, 3, axis=0)  # Shape: (3, H, W)

        # Debugging: 확인
        assert land_sea_mask_patch.shape == (3, imgh, imgw), f"Unexpected land_sea_mask_patch shape: {land_sea_mask_patch.shape}"

        return land_sea_mask_patch

    def remove_land_from_mask(self, mask_image, land_sea_mask_patch):
        """
        Remove land areas from the mask and keep only ocean areas.
        육지 부분을 1로 설정하고 해양 부분은 mask_image 값을 0으로 설정합니다.
        """
        # 육지인 부분을 1로 설정하여 복원 불필요
        # 해양인 부분은 기존 mask_image 값을 유지 (복원)
        land_value = 1  # 육지를 나타내는 실제 값으로 수정
        sea_mask = (land_sea_mask_patch != land_value).astype(np.uint8)  # 해양: 1, 육지: 0

        # 해양 영역만 남기고 마스크 적용
        mask_modified = mask_image * sea_mask  # 해양 영역에서만 마스크 적용

        # Debugging: 확인
        assert mask_modified.shape == mask_image.shape, f"mask_modified shape {mask_modified.shape} does not match mask_image shape {mask_image.shape}"
        unique_vals = np.unique(mask_modified)
        assert set(unique_vals.tolist()).issubset({0, 1}), f"mask_modified should only contain 0 and 1: {unique_vals}"

        return mask_modified

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
        unique_vals = np.unique(land_sea_mask)
        print(f"Loaded land-sea mask with shape: {land_sea_mask.shape}, dtype: {land_sea_mask.dtype}")
        print(f"Unique values in land_sea_mask: {unique_vals}")
        return land_sea_mask

    def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
        """
        Resize the image to the target size.
        """
        img = cv2.resize(img, (self.target_size, self.target_size))
        return img

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
                    return np.genfromtxt(path, dtype=str, encoding='utf-8')
                except:
                    return [path]
        return []
