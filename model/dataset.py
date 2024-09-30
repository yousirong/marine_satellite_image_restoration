import os
import cv2
import numpy as np
from PIL import Image
import re
import torch
import random
from scipy.io import loadmat

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

    # def __getitem__(self, index):
    #     try:
    #         item = self.load_item(index)
    #     except Exception as e:
    #         print(f"Loading error for: {self.data[index]}")
    #         print(e)
    #         item = self.load_item(0)
    #     return item
    def __getitem__(self, index):
        try:
            img, mask = self.load_item(index)
        except Exception as e:
            print(f"Loading error for: {self.data[index]}")
            print(e)
            img, mask = self.load_item(0)
        
        # Return the image, mask, and the corresponding filename
        filename = os.path.basename(self.data[index])  # Extract just the filename
        return self.to_tensor(img), self.to_tensor(mask), filename

    def load_item(self, index):
        # Load the image (GT)
        img = cv2.imread(self.data[index], cv2.IMREAD_UNCHANGED)
        # test 할경우 주석 풀기 
        # print(f"Image loaded: {self.data[index]}, dtype: {img.dtype}, shape: {img.shape}")

        # Replace NaNs with 0
        img[np.isnan(img)] = 0

        # Convert dtype if necessary
        if img.dtype == object:
            print(f"Converting image dtype from object to float32 for: {self.data[index]}")
            img = img.astype(np.float32)

        # Resize the image to the target size if it's not the correct size
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size))

        # Ensure the image has 3 channels
        if len(img.shape) == 2:  # Grayscale image
            img = np.stack([img, img, img], axis=0)
        elif img.shape[2] == 1:  # Single channel but already in 3D shape
            img = np.concatenate([img, img, img], axis=2)

        # Step 2: Load mask image
        mask = self.load_mask(img, index)

        # Step 3: Load the land-sea mask
        land_sea_mask = self.land_sea_mask

        # Get the land-sea mask patch corresponding to the GT image
        land_sea_mask_patch = self.get_land_sea_mask_patch(img, index, land_sea_mask)

        # Ensure the land-sea mask patch is resized to match the target size
        if land_sea_mask_patch.shape[1] != self.target_size or land_sea_mask_patch.shape[2] != self.target_size:
            land_sea_mask_patch = cv2.resize(land_sea_mask_patch, (self.target_size, self.target_size))

        # Step 4: Modify the mask (set sea areas where mask is 0 to 255)
        land_removed_mask = self.remove_land_from_mask(mask, land_sea_mask_patch)

        # Optional data augmentation
        if self.training and self.augment:
            if np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]  # Flip image horizontally
                land_removed_mask = land_removed_mask[:, ::-1, ...]

        # Convert the images and masks to tensors
        return self.to_tensor(img), self.to_tensor(land_removed_mask)

    def load_mask(self, img, index):
        """
        Load the mask for a specific image.
        """
        imgh, imgw = img.shape[1:3]

        if len(self.mask_data) == 0:
            raise ValueError("Mask data is empty. Please check the mask directory and ensure that masks are available.")

        # Based on the mask type, randomly select a mask
        mask_index = random.randint(0, len(self.mask_data) - 1)
        mask = cv2.imread(self.mask_data[mask_index], cv2.IMREAD_UNCHANGED)

        # Resize the mask to match the image size
        mask = self.resize(mask, False)

        # Ensure the mask is binary
        mask = (mask > 0).astype(np.uint8)  * 255
        mask = np.stack([mask, mask, mask], axis=0)

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

        # Ensure the land-sea mask patch is within bounds
        if row is None or col is None or row + imgh > land_sea_mask.shape[0] or col + imgw > land_sea_mask.shape[1]:
            raise ValueError(f"Row or column is out of bounds for land-sea mask. row: {row}, col: {col}, mask shape: {land_sea_mask.shape}")

        # Extract the land-sea mask patch based on the row and column
        land_sea_mask_patch = land_sea_mask[row:row+imgh, col:col+imgw]

        # Stack the mask into three channels to match the image's channels
        land_sea_mask_patch = np.stack([land_sea_mask_patch, land_sea_mask_patch, land_sea_mask_patch], axis=0)
        return land_sea_mask_patch

    def remove_land_from_mask(self, mask_image, land_sea_mask_patch):
        """
        Remove land areas from the mask and keep only ocean areas.
        육지 부분을 0으로 설정하고 해양 부분만 유지합니다.
        """
        # Mask out the land: set land areas (land_sea_mask_patch == 999) to 0
        # Keep only ocean areas (land_sea_mask_patch == 1)
        mask_modified = np.where((land_sea_mask_patch == 1) & (mask_image == 0), 255, mask_image)
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
        # land_sea_mask = loadmat(land_mask_path)['Land']
        print(f"Loaded land-sea mask with shape: {land_sea_mask.shape}, dtype: {land_sea_mask.dtype}")
        # Convert land = 0 to 999, and ocean = 1
        # land_sea_mask = np.where(land_sea_mask == 0, 999, 1)
        return land_sea_mask

    # 기존 ust21 resize
    # def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
    #     """
    #     Resize the image to the target size.
    #     """
    #     imgh, imgw = img.shape[:2]
    #     img = np.array(Image.fromarray(img).resize(size=(self.target_size, self.target_size)))
    #     return img
    def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
        """
        Resize the image to the target size.
        """
        img = cv2.resize(img, (self.target_size, self.target_size))
        return img

    # def to_tensor(self, img):
    #     """
    #     Convert the numpy array to a PyTorch tensor.
    #     Ensures the image is in a valid format and data type.
    #     """
    #     if img.dtype == object:
    #         print(f"Converting image dtype from object to float32.")
    #         img = img.astype(np.float32)

    #     # If the image is in 2D, expand it to 3D with a single channel
    #     if len(img.shape) == 2:
    #         img = np.expand_dims(img, axis=0)

    #     # Ensure dtype is compatible with PyTorch tensor
    #     if img.dtype not in [np.float32, np.float64, np.uint16, np.uint8, np.int16, np.int32]:
    #         print(f"Unsupported dtype {img.dtype}. Converting to float32.")
    #         img = img.astype(np.float32)

    #     return torch.Tensor(img)
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
                    return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
                except:
                    return [path]
        return []
