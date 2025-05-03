import netCDF4 as nc
import numpy as np
import os
import random
import matplotlib.pyplot as plt
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
data_base = '/media/juneyonglee/GOCI_vol1/GOCI/L2_Rrs'  # GOCI RRS 데이터(.he5)가 있는 루트 경로
save_base = '/media/juneyonglee/My Book1/Preprocessed/GOCI_RRS'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'  # GOCI용 육지-해양 마스크 (.npy)

# 사용할 밴드 리스트
band_lst = [2, 3, 4]

# ======================== Configure Logging =========================
logging.basicConfig(
    filename='preprocessing_goci_rrs_43to45.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ================== Load and Prepare Land-Sea Mask ===================
try:
    land_sea_mask_original = np.load(mask_path)
    logging.info("Land-sea mask loaded successfully from .npy file.")

    unique_values = np.unique(land_sea_mask_original)
    logging.info(f"Unique values in GOCI mask: {unique_values}")
    print(f"Unique values in GOCI mask: {unique_values}")

    # GOCI 마스크가 -1(육지), 0(해양) 형태라면 => 육지=999, 해양=1 로 변환
    if (-1 in unique_values) and (0 in unique_values):
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
top_left_y, top_left_x = 2048, 1536   # (y, x)
bottom_right_y, bottom_right_x = 2560, 3072  # (y, x)

# ================== Create Necessary Directories =====================
pcts = [str(i) for i in range(0, 101, 10)]  # ['0','10','20',...,'100']
pcts.append('perfect')

for band in band_lst:
    band_folder = os.path.join(save_base, f"band{band}")
    for phase in ['train', 'test']:
        for pct in pcts:
            temp = os.path.join(band_folder, phase, pct)
            if not os.path.isdir(temp):
                try:
                    os.makedirs(temp)
                    logging.info(f"Created directory: {temp}")
                except Exception as e:
                    logging.error(f"Failed to create directory {temp}: {e}")
                    raise

# ======================== Define Helper Functions ====================

def extract_date_from_filename(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    else:
        raise ValueError(f"No valid date found in filename: {filename}")

def gather_files_by_date(data_base):
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

    for date in grouped_files:
        grouped_files[date].sort()
    return grouped_files

def calculate_daily_composite(files, land_sea_mask, band):
    """
    파일별 RRS 데이터를 읽은 뒤,
    - -999.0은 NaN으로 변환 (결측치)
    - 육지(land_sea_mask == 999)도 NaN 처리
    -> 이후 valid 데이터만 np.nanmean으로 일평균 합성
    """
    data_list = []
    for file_path in files:
        try:
            with nc.Dataset(file_path, 'r') as f:
                var_name = f'HDFEOS/GRIDS/Image Data/Data Fields/Band {band} RRS Image Pixel Values'
                try:
                    rrs_data = f[var_name][:]
                except KeyError:
                    logging.warning(f"Band {band} not found in file {os.path.basename(file_path)}.")
                    continue

                # -999.0인 값 → NaN
                rrs_data = np.where(rrs_data == -999.0, np.nan, rrs_data)
                # 육지 = 999인 부분 → NaN
                rrs_data = np.where(land_sea_mask == 999, np.nan, rrs_data)

                data_list.append(rrs_data)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            continue

    if not data_list:
        # 모든 파일이 유효치가 없으면 0배열 반환 (shape 동일)
        return np.zeros(land_sea_mask.shape, dtype=np.float32)

    # valid한 픽셀만 평균 내기 위해 nanmean 사용
    avg_data = np.nanmean(data_list, axis=0)
    # 결측이 전부인 경우 nanmean 결과가 NaN이 될 수 있으니, NaN → 0 처리 (또는 그대로 두어도 됨)
    avg_data = np.nan_to_num(avg_data, nan=0.0)
    return avg_data

def check_pct(img, mask):
    """
    img: 일평균 합성된 RRS (육지/결측 제외는 이미 0으로 대체된 상태)
    mask: ocean=1, land=999
    -> 해양 픽셀 중 실제 유효 데이터(>0) vs 0을 비교하여 손실률 계산
    만약 더 과학적으로 처리하려면 nan값 보존 로직에 맞춰 separate logic으로 가능.
    """
    ocean_pixels = (mask == 1)
    ocean_vals = img[ocean_pixels]

    if ocean_vals.size == 0:
        # 해양 픽셀이 전혀 없으면 손실 100%로 간주
        return 100

    # 결측(또는 육지) → 0으로 대체된 상태라면, 0인 픽셀은 곧 missing/invalid
    missing_count = np.count_nonzero(ocean_vals == 0)
    loss_pct = (missing_count / ocean_vals.size) * 100
    return loss_pct

def save_patch_image(patch, file_path):
    """
    Saves a chlorophyll-a patch as a TIFF file using the original float data.

    Parameters:
    - patch (numpy.ndarray): 2D array of chlorophyll-a data for the patch.
    - file_path (str): Destination file path for the TIFF image.
    """
    # Create a copy to avoid modifying the original patch
    patch_visual = patch.copy()
    # Save as TIFF without casting to integer (원본 데이터 타입 그대로 저장)
    tiff.imwrite(file_path, patch_visual)

def compute_ocean_ratio(mask_patch):
    """
    mask_patch: 256x256, where ocean=1, land=999
    해양(==1) 픽셀의 비율(0~1)을 계산
    """
    total_pixels = mask_patch.size  # 256*256
    ocean_pixels = np.count_nonzero(mask_patch == 1)
    return ocean_pixels / total_pixels

# ==================== Patch Extraction ======================

train_ratio = 0.8
num_patches_per_day = 256

grouped_files_global = None
land_sea_mask_global = None
pcts_global = None
top_left_coords_global = None
bottom_right_coords_global = None
num_patches_global = None

def pool_initializer(lsm, gf, p, tl, br, npatches):
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
    try:
        files = grouped_files_global.get(date_key, [])
        if len(files) == 0:
            logging.warning(f"No files found for date {date_key}. Skipping.")
            return

        # 하루 8장(시간대별)
        if len(files) != 8:
            logging.warning(f"Expected 8 time slices for date {date_key}, found {len(files)}. Skipping.")
            return

        for band in band_lst:
            avg_data = calculate_daily_composite(files, land_sea_mask_global, band)

            # 합성 결과가 전부 0이거나, 유효 데이터가 없는 경우
            if np.all(avg_data == 0):
                logging.info(f"(Band {band}) All ocean data is invalid (or zero) for date {date_key}. Skipping.")
                continue

            min_y = top_left_coords_global[0]
            max_y = bottom_right_coords_global[0] - 256
            min_x = top_left_coords_global[1]
            max_x = bottom_right_coords_global[1] - 256

            if max_y < min_y or max_x < min_x:
                logging.warning(f"(Band {band}) Invalid rectangle for date {date_key}. Skipping.")
                continue

            selected_patches = set()
            attempts = 0
            max_attempts = num_patches_global * 10

            while len(selected_patches) < num_patches_global and attempts < max_attempts:
                py_start = random.randint(min_y, max_y)
                px_start = random.randint(min_x, max_x)
                if (py_start, px_start) in selected_patches:
                    attempts += 1
                    continue

                py_end = py_start + 256
                px_end = px_start + 256

                patch = avg_data[py_start:py_end, px_start:px_end]
                mask_patch = land_sea_mask_global[py_start:py_end, px_start:px_end]
                attempts += 1

                # 크기 검사
                if patch.shape != (256, 256):
                    continue

                # 패치 전체가 0인지, 혹은 모두 NaN인지(위에서 nan->0했으므로 여기서는 0 검사 중심)
                if np.all(patch == 0):
                    continue

                # 최소 해양 비율 0.1(10%) 이상인지 검사
                ocean_ratio = compute_ocean_ratio(mask_patch)
                if ocean_ratio < 0.1:
                    # 해양이 너무 적으면 스킵
                    continue

                loss_pct = check_pct(patch, mask_patch)

                phase = 'train' if random.random() < train_ratio else 'test'

                if loss_pct == 0:
                    pct_folder = 'perfect'
                elif 0 < loss_pct <= 100:
                    pct_folder = str(int(loss_pct // 10) * 10)
                    if pct_folder not in pcts_global:
                        pct_folder = 'perfect'
                else:
                    pct_folder = 'perfect'

                patch_identifier = f"r{py_start}_c{px_start}"
                date_str = date_key
                filename = f"{date_str}_{patch_identifier}.tiff"

                final_save_path = os.path.join(
                    save_base, f"band{band}", phase, pct_folder, filename
                )

                os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

                try:
                    # float 형식으로 저장
                    save_patch_image(patch, final_save_path)
                    logging.info(f"(Band {band}) Saved patch: {final_save_path}")
                    selected_patches.add((py_start, px_start))
                except Exception as e:
                    logging.error(f"(Band {band}) Failed to save patch {filename}: {e}")
                    continue

            if len(selected_patches) < num_patches_global:
                logging.warning(
                    f"(Band {band}) Only {len(selected_patches)} patches for date {date_key} < requested {num_patches_global}."
                )

    except Exception as e:
        logging.error(f"Error processing date {date_key}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Process GOCI RRS daily composites (Band 2,3,4) with minimal ocean ratio=0.1, then extract random patches."
    )
    parser.add_argument('--max_processes', type=int, default=16,
                        help='Maximum number of worker processes (default: 4)')
    parser.add_argument('--num_patches', type=int, default=100,
                        help='Number of random patches per day per band (default: 256)')
    parser.add_argument('--top_left', type=int, nargs=2, default=[2048, 1536],
                        help='Top-left (y x) of patch region (default: 2560 1536)')
    parser.add_argument('--bottom_right', type=int, nargs=2, default=[3584, 3072],
                        help='Bottom-right (y x) of patch region (default: 3072 3072)')
    args = parser.parse_args()

    MAX_PROCESSES = args.max_processes
    num_patches = args.num_patches
    top_left_coords = args.top_left
    bottom_right_coords = args.bottom_right

    print(f"Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"Rectangle Top-Left: ({top_left_coords[0]}, {top_left_coords[1]}), "
                 f"Bottom-Right: ({bottom_right_coords[0]}, {bottom_right_coords[1]})")
    logging.info(f"Number of patches per day (per band): {num_patches}")

    try:
        grouped_files = gather_files_by_date(data_base)
        logging.info(f"Total unique dates gathered: {len(grouped_files)}")
        print(f"Total unique dates gathered: {len(grouped_files)}")
    except Exception as e:
        logging.error(f"Failed to gather files: {e}")
        raise

    try:
        pool = mp.Pool(
            processes=MAX_PROCESSES,
            initializer=pool_initializer,
            initargs=(land_sea_mask, grouped_files, pcts,
                      top_left_coords, bottom_right_coords, num_patches)
        )
    except Exception as e:
        logging.error(f"Failed to initialize multiprocessing pool: {e}")
        raise

    date_keys = list(grouped_files.keys())

    try:
        for _ in tqdm(pool.imap_unordered(process_day, date_keys),
                      total=len(date_keys),
                      desc="Processing daily composites (all bands)"):
            pass
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

    pool.close()
    pool.join()

    print("Patch extraction and saving completed.")
    logging.info("Patch extraction and saving completed.")

if __name__ == "__main__":
    main()
