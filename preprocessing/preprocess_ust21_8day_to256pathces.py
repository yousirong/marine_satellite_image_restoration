# 필요한 라이브러리 임포트
import netCDF4 as nc
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy import io
from datetime import datetime
from tqdm import tqdm
import tifffile as tiff
import csv
import re  # Import the regular expressions module
import multiprocessing as mp
import logging
import argparse
from skimage.transform import resize
import matplotlib.patches as patches

########## Path Definitions ##########
data_base = '/media/juneyonglee/My Book/UST21/01_day'
save_base = '/media/juneyonglee/My Book/Preprocessed/UST21'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'
# Removed CSV file path as it's no longer needed
# csv_file = '/home/juneyonglee/Desktop/AY_ust/Notebook/146to149_boundary_coordinates_included_UST21.csv'  # Update to your CSV filename

########## Configure Logging ##########
logging.basicConfig(
    filename='preprocessing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

########## Load and Prepare Land-Sea Mask ##########
try:
    land_sea_mask_mat = io.loadmat(mask_path)
    land_sea_mask_original = land_sea_mask_mat['Land']  # MATLAB에서 'Land' 변수 가져오기
    logging.info("Land-sea mask loaded successfully.")

    # 확인: 고유값 출력
    unique_values = np.unique(land_sea_mask_original)
    logging.info(f"Unique values in 'Land' mask: {unique_values}")
    print(f"Unique values in 'Land' mask: {unique_values}")

    # 올바른 마스킹 로직 설정
    # 여기서는 '1'이 육지, '0'이 해양이라고 가정합니다.
    # 따라서 육지 (1)을 999로, 해양 (0)을 1로 설정합니다.
    if 1 in unique_values and 0 in unique_values:
        # '1'이 육지, '0'이 해양
        land_sea_mask = np.where(land_sea_mask_original == 1, 999, 1)
        logging.info("Land-sea mask processed: Land=999, Ocean=1.")
    else:
        logging.error("Unexpected values in Land mask. Expected values [0, 1].")
        exit(1)
except FileNotFoundError:
    logging.error(f"마스크 파일을 찾을 수 없습니다: {mask_path}")
    exit(1)
except KeyError:
    logging.error("마스크 파일 내 'Land' 변수를 찾을 수 없습니다.")
    exit(1)
except Exception as e:
    logging.error(f"Failed to load land-sea mask: {e}")
    raise

########## Define Regions ##########
# 좌표 정의 (필요에 따라 조정)
# These variables are defined but not used in this script. They can be removed or used as needed.
x_nak, y_nak = (3751, 4757)  # 낙동강 영역 시작점
x_sae, y_sae = (3505, 3920)  # 새만금 영역 시작점

########## Create Necessary Directories ##########
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

########## Define Helper Functions ##########

def check_pct(img, mask):
    """
    Calculates the percentage of ocean pixels that are NaN.

    Parameters:
    - img (numpy.ndarray): 2D array of chlorophyll-a data.
    - mask (numpy.ndarray): 2D mask array where ocean pixels are 1 and land pixels are 999.

    Returns:
    - loss_pct (float): Percentage of ocean pixels with NaN values.
    """
    ocean_pixels = (mask == 1)  # 해양 픽셀만 선택
    valid_ocean_data = img[ocean_pixels]  # 육지를 제외한 해양 픽셀

    total_ocean_pixels = valid_ocean_data.size  # 전체 해양 픽셀 수
    nan_count = np.isnan(valid_ocean_data).sum()  # NaN 값 수

    if total_ocean_pixels > 0:
        loss_pct = (nan_count / total_ocean_pixels) * 100  # NaN 비율을 퍼센트로 계산
    else:
        loss_pct = 100

    return loss_pct

def check_ocean_pct(patch, mask):
    """
    Calculates the percentage of valid ocean pixels within a patch.

    Parameters:
    - patch (numpy.ndarray): 2D array of chlorophyll-a data for the patch.
    - mask (numpy.ndarray): 2D mask array for the patch.

    Returns:
    - ocean_data_pct (float): Percentage of valid ocean pixels.
    """
    ocean_pixels = (mask == 1)  # 해양 픽셀 선택 (육지 제외)
    valid_ocean_pixels = np.sum((patch >= 0.01) & (patch <= 10) & ocean_pixels)  # 유효한 해양 데이터
    total_ocean_pixels = np.sum(ocean_pixels)  # 전체 해양 픽셀 수 (육지 제외)

    ocean_data_pct = (valid_ocean_pixels / total_ocean_pixels) * 100 if total_ocean_pixels > 0 else 0
    return ocean_data_pct

def calculate_8day_avg(files, data_dir, land_sea_mask):
    """
    Calculates the 8-day moving average of chlorophyll-a data.

    Parameters:
    - files (list): List of file paths for the 8-day window.
    - data_dir (str): Directory where the files are located.
    - land_sea_mask (numpy.ndarray): Mask array where land pixels are 999 and ocean pixels are 1.

    Returns:
    - avg_data (numpy.ndarray): 2D array representing the averaged chlorophyll-a data.
    """
    data_list = []
    for file in files:
        if file.endswith('.nc'):  # Process only '.nc' files
            path = os.path.join(data_dir, file)
            try:
                with nc.Dataset(path, 'r') as f:
                    # Extract the chlorophyll-a data
                    a = f['merged_daily_Chl'][:].data

                    # Ensure data is a NumPy array
                    np_a = np.array(a)

                    # Replace sentinel values and outliers with NaN
                    np_a = np.where(np_a == -999.0, np.nan, np_a)  # Handle missing data
                    np_a[np_a > 10] = np.nan  # Handle outliers

                    # Apply the land-sea mask: land pixels set to 999, ocean pixels remain as is
                    np_a = np_a * land_sea_mask

                    # Replace NaNs in ocean pixels with 0
                    np_a = np.nan_to_num(np_a, nan=0)

                    # Exclude land pixels by setting them to 0
                    np_a = np.where(land_sea_mask == 999, 0, np_a)

                    # Append to the data list if there's valid ocean data
                    if not np.all(np_a == 0):
                        data_list.append(np_a)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")
                continue

    if not data_list:
        # Return an array of zeros if no valid data is found
        return np.zeros(land_sea_mask.shape, dtype=np.float32)

    # Calculate the mean across the 8-day window
    avg_data = np.mean(data_list, axis=0)
    return avg_data

def save_patch_image(patch, file_path):
    """
    Saves a chlorophyll-a patch as a TIFF file.

    Parameters:
    - patch (numpy.ndarray): 2D array of chlorophyll-a data for the patch.
    - file_path (str): Destination file path for the TIFF image.
    """
    # Create a copy to avoid modifying the original patch
    patch_visual = patch.copy()

    # Cast to uint16 for TIFF format
    patch_visual = patch_visual.astype(np.uint16)

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

def gather_all_files(data_base):
    """
    Gathers all '.nc' files from the data directory and sorts them by date.

    Parameters:
    - data_base (str): Root directory containing '.nc' files.

    Returns:
    - all_files (list): Sorted list of tuples (file_path, date).
    """
    all_files = []
    for root, dirs, files in os.walk(data_base):
        for file in files:
            if file.endswith('.nc'):
                full_path = os.path.join(root, file)
                try:
                    file_date = extract_date_from_filename(file)
                    all_files.append((full_path, file_date))
                except ValueError as e:
                    logging.warning(f"Skipping file due to error: {e}")
    # Sort files by date
    all_files.sort(key=lambda x: x[1])
    return all_files

########## Gather All Data Files ##########
all_files = gather_all_files(data_base)
print(f"Total files gathered: {len(all_files)}")
logging.info(f"Total files gathered: {len(all_files)}")

########## Patch Extraction Parameters ##########
train_ratio = 0.8  # train:test 비율 8:2
min_ocean_pct = 0.1  # 최소 해양 데이터 비율을 설정

# Define the target rectangle coordinates
# Top-Left: (3584,3072)
# Bottom-Right: (4096,5120)
top_left_y, top_left_x = 3584, 3072
bottom_right_y, bottom_right_x = 4096, 5120

# Define the number of patches to extract per window
num_patches_per_window = 100  # Adjust as needed

########## Initialize Global Variables for Pool ##########
# These will be set in the Pool initializer
land_sea_mask_global = None
all_files_global = None
pcts_global = None
top_left = None
bottom_right = None
num_patches = None

def pool_initializer(lsm, af, p, tl, br, npatches):
    """
    Initializer for each worker process in the Pool.
    Sets global variables for shared data.
    """
    global land_sea_mask_global
    global all_files_global
    global pcts_global
    global top_left
    global bottom_right
    global num_patches
    land_sea_mask_global = lsm
    all_files_global = af
    pcts_global = p
    top_left = tl
    bottom_right = br
    num_patches = npatches
    logging.info("Worker process initialized with shared data.")

def process_window(i):
    """
    Worker function to process a single 8-day window.

    Parameters:
    - i (int): Index of the starting file for the 8-day window.
    """
    try:
        # Select the 8 consecutive files for the moving average
        selected_files = [all_files_global[i + k][0] for k in range(8)]
        avg_data = calculate_8day_avg(selected_files, os.path.dirname(selected_files[0]), land_sea_mask_global)

        # Check if avg_data contains all zeros (no valid ocean data)
        if np.all(avg_data == 0):
            logging.info(f"All ocean data is invalid for window {i}. Skipping.")
            return

        # Extract start and end dates
        start_date = all_files_global[i][1]
        end_date = all_files_global[i + 7][1]

        # Define the valid range for patch extraction
        # Ensure that the patch fits entirely within the rectangle
        min_y = top_left_y
        max_y = bottom_right_y - 256
        min_x = top_left_x
        max_x = bottom_right_x - 256

        # If max_y or max_x is less than min_y or min_x, adjust accordingly
        if max_y < min_y or max_x < min_x:
            logging.warning(f"Invalid rectangle dimensions for window {i}. Skipping.")
            return

        # Initialize a set to keep track of selected patch coordinates
        selected_patches = set()
        attempts = 0
        max_attempts = num_patches * 10  # To prevent infinite loops

        while len(selected_patches) < num_patches and attempts < max_attempts:
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

            # Check if 100% of the patch contains zeros (originally NaNs or land)
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
            date_str = start_date.strftime('%Y%m%d')
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

        if len(selected_patches) < num_patches:
            logging.warning(f"Only {len(selected_patches)} patches were extracted for window {i}, less than the requested {num_patches} patches.")

    except Exception as e:
        logging.error(f"Error processing window {i}: {e}")
        return  # Optionally, you can re-raise the exception or handle it as needed

def main():
    """
    Main function to set up multiprocessing and start processing.
    """
    # Parse command-line arguments for maximum processes and other parameters
    parser = argparse.ArgumentParser(description="Process 8-day moving average and extract random patches within a rectangle using multiprocessing.")
    parser.add_argument('--max_processes', type=int, default=4, help='Maximum number of worker processes (default: 4)')
    parser.add_argument('--num_patches', type=int, default=256, help='Number of random patches to extract per window (default: 256)')
    parser.add_argument('--top_left', type=int, nargs=2, default=[3584,3072], help='Top-left coordinates (y x) of the rectangle (default: 3584 3072)')
    parser.add_argument('--bottom_right', type=int, nargs=2, default=[4096,5120], help='Bottom-right coordinates (y x) of the rectangle (default: 4096 5120)')
    args = parser.parse_args()

    MAX_PROCESSES = args.max_processes
    num_patches = args.num_patches
    top_left_coords = args.top_left
    bottom_right_coords = args.bottom_right

    # Assign rectangle coordinates
    global top_left_y, top_left_x, bottom_right_y, bottom_right_x
    top_left_y, top_left_x = top_left_coords
    bottom_right_y, bottom_right_x = bottom_right_coords

    print(f"Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"Rectangle Top-Left: ({top_left_y}, {top_left_x}), Bottom-Right: ({bottom_right_y}, {bottom_right_x})")
    logging.info(f"Number of patches per window: {num_patches}")

    try:
        # Initialize the Pool with the initializer and shared data
        pool = mp.Pool(
            processes=MAX_PROCESSES,
            initializer=pool_initializer,
            initargs=(land_sea_mask, all_files, pcts, top_left_coords, bottom_right_coords, num_patches)
        )
    except Exception as e:
        logging.error(f"Failed to initialize multiprocessing pool: {e}")
        raise

    # Create an iterable of window indices
    window_indices = range(0, len(all_files) - 7)

    # Use imap_unordered for efficient parallel processing
    # Wrap it with tqdm for progress indication
    try:
        for _ in tqdm(pool.imap_unordered(process_window, window_indices), total=len(all_files)-7, desc="Processing 8-day windows"):
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
