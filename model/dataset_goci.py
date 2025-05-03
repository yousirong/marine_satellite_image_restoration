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
        # 만약 load_item이 None을 반환했다면, 다음 인덱스(혹은 랜덤)로 넘어가서 재시도
        if img is None or mask is None:
            next_idx = (index + 1) % len(self.data)
            return self.__getitem__(next_idx)
        filename = os.path.basename(self.data[index])  # Extract just the filename
        return self.to_tensor(img), self.to_tensor(mask), filename

    def load_item(self, index, test_mode=False):
        max_attempts = 10  # 최대 시도 횟수 설정

        img = None
        land_removed_mask = None
        ocean_mask_ratio = 0.0
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

            # NaN 및 특수값 처리
            img = img.astype(np.float32)
            img[img == -999] = np.nan
            img[np.isnan(img)] = 0

            # Resize the image to the target size if it's not the correct size
            if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                img = cv2.resize(img, (self.target_size, self.target_size))

            # Transpose the image to have channels first (C, H, W)
            img = np.transpose(img, (2, 0, 1))  # Shape: (3, H, W)

            # Step 2: Load mask image
            if test_mode:

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
            sea_mask = (land_sea_mask_patch == 1).astype(np.uint8)  # 육지: 1, 해양: 0 -> 해양: 1
            total_ocean_pixels = sea_mask.sum()
            if total_ocean_pixels == 0:
                # print(f"Attempt {attempt+1}/{max_attempts}: No ocean pixels found for image: {self.data[index]}")
                continue  # 다음 시도로 넘어감

            # 해양 영역 내에서 마스크가 적용된 픽셀 수
            masked_ocean_pixels = (land_removed_mask == 1) & (sea_mask == 1)
            masked_ocean_pixels_count = masked_ocean_pixels.sum()

            # 해양 영역 내 마스크 비율
            ocean_mask_ratio = masked_ocean_pixels_count / total_ocean_pixels

            if ocean_mask_ratio >= 0.01:
                img_tensor = self.to_tensor(img)
                mask_tensor = self.to_tensor(land_removed_mask)
                return img_tensor, mask_tensor

        # After max_attempts, log and return the last mask (even if ocean area is small)
        # print(f"Could not find a mask with ocean area >=1% for image: {self.data[index]} after {max_attempts} attempts, using the last mask.")
        img_tensor = self.to_tensor(img)
        mask_tensor = self.to_tensor(land_removed_mask)
        return img_tensor, mask_tensor

    def load_mask(self, img, index):
        """
        Load the mask for a specific image.
        원본 PNG가 0/255 이므로,
        검은색(0)=복원 대상→1, 흰색(255)=복원 제외→0 으로 만듭니다.
        """
        imgh, imgw = img.shape[1:3]
        mask_path = self.mask_data[random.randint(0, len(self.mask_data)-1)]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((imgh, imgw), dtype=np.uint8)

        # 리사이즈
        mask = self.resize(mask, False)

        # 0→1, 255→0
        mask = (mask == 0).astype(np.uint8)

        # 채널 차원 추가
        mask = np.expand_dims(mask, 0)   # (1, H, W)
        mask = np.repeat(mask, 3, 0)     # (3, H, W)

        # mask_reverse 옵션
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
        mask_image: 1=복원 대상, 0=제외
        land_sea_mask_patch: 1=육지, 0=해양
        → 최종: 바다이면서 원래 복원 대상이었던 곳만 1, 나머진 0
        """
        # sea_mask: 바다=1, 육지=0
        sea_mask = (land_sea_mask_patch == 0).astype(np.uint8)

        # 최종 마스크: sea_mask * mask_image
        mask_mod = mask_image * sea_mask

        return mask_mod


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
        Load the land-sea mask from a .npy file and normalize values:
        원본 mask: 1=해양?, 999=육지  →  변환 mask: 0=해양, 1=육지
        """
        land_sea_mask = np.load(land_mask_path)
        print("Before conversion, unique values:", np.unique(land_sea_mask))
        # 999 는 육지 → 1, 나머지(1) 는 해양 → 0
        land_sea_mask = np.where(land_sea_mask == 999, 0, 1).astype(np.uint8)
        print("After conversion, unique values:", np.unique(land_sea_mask))
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
            # return torch.Tensor(img)
            t = torch.from_numpy(img.astype(np.float32)).clamp(0,255) / 255.0
            return t
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
