import h5py
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy import io
from datetime import datetime
from tqdm import tqdm
import tifffile as tiff
import csv
import re
import multiprocessing as mp
import logging
import argparse
from skimage.transform import resize
import matplotlib.patches as patches

# ======================== Path Definitions =========================
data_base = '/media/juneyonglee/My Book/GOCI_RRS_Data'  # Update to your GOCI RRS data path
save_base = '/media/juneyonglee/My Book/Preprocessed/GOCI_RRS'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI.npy'  # Updated mask path

# ======================== Configure Logging =========================
logging.basicConfig(
    filename='preprocessing_goci_rrs.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ================== Load and Prepare Land-Sea Mask ===================
try:
    # Load the mask from the .npy file
    land_sea_mask_original = np.load(mask_path)
    logging.info("Land-sea mask loaded successfully from .npy file.")

    # 확인: 고유값 출력
    unique_values = np.unique(land_sea_mask_original)
    logging.info(f"Unique values in GOCI mask: {unique_values}")
    print(f"Unique values in GOCI mask: {unique_values}")

    # 올바른 마스킹 로직 설정
    # 'Land' is -1, 'Ocean' is 0
    if (-1 in unique_values) and (0 in unique_values):
        # Convert land (-1) to 999 and ocean (0) to 1
        land_sea_mask = np.where(land_sea_mask_original == -1, 999, 1)
        logging.info("Land-sea mask processed: Land=999, Ocean=1.")
    else:
        logging.error("Unexpected values in GOCI mask. Expected values [-1, 0].")
        exit(1)
except FileNotFoundError:
    logging.error(f"Mask file not found: {mask_path}")
    exit(1)
except Exception as e:
    logging.error(f"Failed to load land-sea mask: {e}")
    raise

# ======================== Define Regions ============================
# Example coordinates; adjust as needed
top_left_y, top_left_x = 2560, 1536  # Top-Left of the rectangle
bottom_right_y, bottom_right_x = 3072, 3072  # Bottom-Right of the rectangle

# ================== Create Necessary Directories =====================
pcts = [str(i) for i in range(0, 101, 10)]  # '0', '10', ..., '100'
pcts.append('perfect')

for phase in ['train', 'test']:
    for pct in pcts:
        temp = os.path.join(save_base, phase, pct)
        if not os.path.isdir(temp):
            try:
                os.makedirs(temp)
                logging.info(f"Created directory: {temp}")
            except Exception as e:
                logging.error(f"Failed to create directory {temp}: {e}")
                raise

# ======================== Define Helper Functions ====================

def check_pct(img, mask):
    """
    Calculates the percentage of ocean pixels that are NaN.

    Parameters:
    - img (numpy.ndarray): 2D array of RRS data.
    - mask (numpy.ndarray): 2D mask array where ocean pixels are 1 and land pixels are 999.

    Returns:
    - loss_pct (float): Percentage of ocean pixels with NaN values.
    """
    ocean_pixels = (mask == 1)
    valid_ocean_data = img[ocean_pixels]

    total_ocean_pixels = valid_ocean_data.size
    nan_count = np.isnan(valid_ocean_data).sum()

    if total_ocean_pixels > 0:
        loss_pct = (nan_count / total_ocean_pixels) * 100
    else:
        loss_pct = 100

    return loss_pct

def check_ocean_pct(patch, mask):
    """
    Calculates the percentage of valid ocean pixels within a patch.

    Parameters:
    - patch (numpy.ndarray): 2D array of RRS data for the patch.
    - mask (numpy.ndarray): 2D mask array for the patch.

    Returns:
    - ocean_data_pct (float): Percentage of valid ocean pixels.
    """
    ocean_pixels = (mask == 1)
    valid_ocean_pixels = np.sum((patch >= 0.01) & (patch <= 10) & ocean_pixels)
    total_ocean_pixels = np.sum(ocean_pixels)

    ocean_data_pct = (valid_ocean_pixels / total_ocean_pixels) * 100 if total_ocean_pixels > 0 else 0
    return ocean_data_pct

def calculate_daily_composite(files, land_sea_mask):
    """
    Calculates the daily composite of RRS data by averaging the 8 time slices.

    Parameters:
    - files (list): List of file paths for the 8 time slices.
    - land_sea_mask (numpy.ndarray): Mask array where land pixels are 999 and ocean pixels are 1.

    Returns:
    - avg_data (numpy.ndarray): 2D array representing the averaged RRS data.
    """
    data_list = []
    for file in files:
        try:
            with h5py.File(file, 'r') as f:
                # Navigate to the RRS data fields
                # Adjust the path based on actual .he5 file structure
                # Example path; modify as per your data structure
                rrs_dataset = f['HDFEOS']['GRIDS']['RRS Data']['Data Fields']['RRS_Image_Pixel_Values']
                rrs_data = np.array(rrs_dataset)

                # Replace sentinel values and outliers with NaN
                rrs_data = np.where(rrs_data == -999.0, np.nan, rrs_data)
                rrs_data[rrs_data > 10] = np.nan

                # Apply the land-sea mask
                rrs_data = rrs_data * land_sea_mask

                # Replace NaNs in ocean pixels with 0
                rrs_data = np.nan_to_num(rrs_data, nan=0)

                # Exclude land pixels by setting them to 0
                rrs_data = np.where(land_sea_mask == 999, 0, rrs_data)

                # Append to the data list if there's valid ocean data
                if not np.all(rrs_data == 0):
                    data_list.append(rrs_data)
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            continue

    if not data_list:
        # Return an array of zeros if no valid data is found
        return np.zeros(land_sea_mask.shape, dtype=np.float32)

    # Calculate the mean across the 8 time slices
    avg_data = np.mean(data_list, axis=0)
    return avg_data

def save_patch_image(patch, file_path):
    """
    Saves an RRS patch as a TIFF file.

    Parameters:
    - patch (numpy.ndarray): 2D array of RRS data for the patch.
    - file_path (str): Destination file path for the TIFF image.
    """
    # Create a copy to avoid modifying the original patch
    patch_visual = patch.copy()

    # Normalize or scale the data as needed before saving
    # For example, scaling to 16-bit unsigned integers
    if np.nanmax(patch_visual) - np.nanmin(patch_visual) > 0:
        patch_visual = (patch_visual - np.nanmin(patch_visual)) / (np.nanmax(patch_visual) - np.nanmin(patch_visual) + 1e-6)
    else:
        patch_visual = np.zeros_like(patch_visual)
    patch_visual = (patch_visual * 65535).astype(np.uint16)

    # Save as TIFF
    tiff.imwrite(file_path, patch_visual)

def extract_date_from_filename(filename):
    """
    Extracts an 8-digit date from the filename using regex.

    Parameters:
    - filename (str): Filename containing the date.

    Returns:
    - date (datetime): Extracted date as a datetime object.

    Raises:
    - ValueError: If no valid date is found in the filename.
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        date_format = '%Y%m%d'
        return datetime.strptime(date_str, date_format)
    else:
        raise ValueError(f"No valid date found in filename: {filename}")

def gather_files_by_date(data_base):
    """
    Gathers all '.he5' files from the data directory and groups them by date.

    Parameters:
    - data_base (str): Root directory containing '.he5' files.

    Returns:
    - grouped_files (dict): Dictionary with date keys and list of file paths as values.
    """
    grouped_files = {}
    for root, dirs, files in os.walk(data_base):
        for file in files:
            if "RRS" in file and file.endswith('.he5'):
                full_path = os.path.join(root, file)
                try:
                    file_date = extract_date_from_filename(file)
                    date_key = file_date.strftime('%Y%m%d')
                    if date_key not in grouped_files:
                        grouped_files[date_key] = []
                    grouped_files[date_key].append(full_path)
                except ValueError as e:
                    logging.warning(f"Skipping file due to error: {e}")
    # Sort the file lists for each date
    for date in grouped_files:
        grouped_files[date].sort()
    return grouped_files

# ==================== Patch Extraction Parameters ======================
train_ratio = 0.8  # train:test ratio 8:2
min_ocean_pct = 0.1  # Minimum ocean data percentage

# Define the target rectangle coordinates
# Already defined above

# Define the number of patches to extract per day
num_patches_per_day = 256  # Adjust as needed

# =================== Initialize Global Variables for Pool =================
grouped_files_global = None
land_sea_mask_global = None
pcts_global = None
top_left_coords_global = None
bottom_right_coords_global = None
num_patches_global = None

def pool_initializer(lsm, gf, p, tl, br, npatches):
    """
    Initializer for each worker process in the Pool.
    Sets global variables for shared data.
    """
    global land_sea_mask_global
    global grouped_files_global
    global pcts_global
    global top_left_coords_global
    global bottom_right_coords_global
    global num_patches_global
    land_sea_mask_global = lsm
    grouped_files_global = gf
    pcts_global = p
    top_left_coords_global = tl
    bottom_right_coords_global = br
    num_patches_global = npatches
    logging.info("Worker process initialized with shared data.")

def process_day(date_key):
    """
    Worker function to process a single day's composite and extract patches.

    Parameters:
    - date_key (str): Date key in 'YYYYMMDD' format.
    """
    try:
        files = grouped_files_global.get(date_key, [])
        if len(files) == 0:
            logging.warning(f"No files found for date {date_key}. Skipping.")
            return

        # Ensure there are exactly 8 time slices
        if len(files) != 8:
            logging.warning(f"Expected 8 time slices for date {date_key}, found {len(files)}. Skipping.")
            return

        # Calculate daily composite
        avg_data = calculate_daily_composite(files, land_sea_mask_global)

        # Check if avg_data contains all zeros (no valid ocean data)
        if np.all(avg_data == 0):
            logging.info(f"All ocean data is invalid for date {date_key}. Skipping.")
            return

        # Define the valid range for patch extraction
        min_y = top_left_coords_global[0]
        max_y = bottom_right_coords_global[0] - 256
        min_x = top_left_coords_global[1]
        max_x = bottom_right_coords_global[1] - 256

        # If max_y or max_x is less than min_y or min_x, adjust accordingly
        if max_y < min_y or max_x < min_x:
            logging.warning(f"Invalid rectangle dimensions for date {date_key}. Skipping.")
            return

        # Extract the date string
        date_str = date_key

        # Initialize a set to keep track of selected patch coordinates to prevent duplication
        selected_patches = set()
        attempts = 0
        max_attempts = num_patches_global * 10  # To prevent infinite loops

        while len(selected_patches) < num_patches_global and attempts < max_attempts:
            patch_y_start = random.randint(min_y, max_y)
            patch_x_start = random.randint(min_x, max_x)

            # Check if this patch has already been selected
            if (patch_y_start, patch_x_start) in selected_patches:
                attempts += 1
                continue

            patch_y_end = patch_y_start + 256
            patch_x_end = patch_x_start + 256

            # Extract the patch
            patch = avg_data[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            mask_patch = land_sea_mask_global[patch_y_start:patch_y_end, patch_x_start:patch_x_end]

            # Ensure the patch is 256x256
            if patch.shape != (256, 256):
                logging.warning(f"Skipping patch at ({patch_y_start}, {patch_x_start}) due to insufficient size: {patch.shape}")
                attempts += 1
                continue

            # Check if 100% of the patch contains zeros (invalid data)
            if np.all(patch == 0):
                logging.info(f"Patch at ({patch_y_start}, {patch_x_start}) contains 100% invalid ocean data. Skipping.")
                attempts += 1
                continue

            # Calculate ocean data percentage and NaN loss percentage
            ocean_pct = check_ocean_pct(patch, mask_patch)
            loss_pct = check_pct(patch, mask_patch)

            # Determine the phase (train or test)
            phase = 'train' if random.random() < train_ratio else 'test'

            # Determine the percentage folder
            if loss_pct == 0:
                pct_folder = 'perfect'
            elif 0 < loss_pct <= 100:
                pct_folder = str(int(loss_pct // 10) * 10)
                if pct_folder not in pcts_global:
                    pct_folder = 'perfect'  # Fallback if pct_folder not in pcts
            else:
                pct_folder = 'perfect'  # Handle any unexpected loss_pct values

            # Construct the filename
            patch_identifier = f"r{patch_y_start}_c{patch_x_start}"
            filename = f"{date_str}_{patch_identifier}.tiff"

            # Define the save path
            save_path = os.path.join(save_base, phase, pct_folder, filename)

            # Ensure the save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save the patch
            try:
                save_patch_image(patch, save_path)
                logging.info(f"Saved patch: {save_path}")
                # Add to selected patches to prevent duplication
                selected_patches.add((patch_y_start, patch_x_start))
            except Exception as e:
                logging.error(f"Failed to save patch {filename}: {e}")
                attempts += 1
                continue

        if len(selected_patches) < num_patches_global:
            logging.warning(f"Only {len(selected_patches)} patches were extracted for date {date_key}, less than the requested {num_patches_global} patches.")

    except Exception as e:
        logging.error(f"Error processing date {date_key}: {e}")
        pass

def main():
    """
    Main function to set up multiprocessing and start processing.
    """
    # Parse command-line arguments for maximum processes and other parameters
    parser = argparse.ArgumentParser(description="Process GOCI RRS daily composites and extract random patches using multiprocessing.")
    parser.add_argument('--max_processes', type=int, default=4, help='Maximum number of worker processes (default: 4)')
    parser.add_argument('--num_patches', type=int, default=256, help='Number of random patches to extract per day (default: 256)')
    parser.add_argument('--top_left', type=int, nargs=2, default=[2560, 1536], help='Top-left coordinates (y x) of the rectangle (default: 2560 1536)')
    parser.add_argument('--bottom_right', type=int, nargs=2, default=[3072, 3072], help='Bottom-right coordinates (y x) of the rectangle (default: 3072 3072)')
    args = parser.parse_args()

    MAX_PROCESSES = args.max_processes
    num_patches = args.num_patches
    top_left_coords = args.top_left
    bottom_right_coords = args.bottom_right

    print(f"Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"Rectangle Top-Left: ({top_left_coords[0]}, {top_left_coords[1]}), Bottom-Right: ({bottom_right_coords[0]}, {bottom_right_coords[1]})")
    logging.info(f"Number of patches per day: {num_patches}")

    try:
        # Gather all files grouped by date
        grouped_files = gather_files_by_date(data_base)
        logging.info(f"Total unique dates gathered: {len(grouped_files)}")
        print(f"Total unique dates gathered: {len(grouped_files)}")
    except Exception as e:
        logging.error(f"Failed to gather files: {e}")
        raise

    try:
        # Initialize the Pool with the initializer and shared data
        pool = mp.Pool(
            processes=MAX_PROCESSES,
            initializer=pool_initializer,
            initargs=(land_sea_mask, grouped_files, pcts, top_left_coords, bottom_right_coords, num_patches)
        )
    except Exception as e:
        logging.error(f"Failed to initialize multiprocessing pool: {e}")
        raise

    # Create an iterable of date keys
    date_keys = list(grouped_files.keys())

    # Use imap_unordered for efficient parallel processing
    # Wrap it with tqdm for progress indication
    try:
        for _ in tqdm(pool.imap_unordered(process_day, date_keys), total=len(date_keys), desc="Processing daily composites"):
            pass  # The worker function handles everything
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user.")
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        logging.error(f"An error occurred during multiprocessing: {e}")
        pool.terminate()
        pool.join()
        raise

    # Close the Pool and wait for all workers to finish
    pool.close()
    pool.join()

    print("Patch extraction and saving completed.")
    logging.info("Patch extraction and saving completed.")

if __name__ == "__main__":
    main()
