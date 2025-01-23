import netCDF4 as nc
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy import io
from datetime import datetime
from tqdm import tqdm
import re
import multiprocessing as mp
import logging
import argparse

########## Path Definitions ##########
data_base = '/media/juneyonglee/My Book/UST21/01_day'   # 원본 .nc 파일 폴더
save_base = '/media/juneyonglee/My Book/Preprocessed/UST21/mask'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'

########## Configure Logging ##########
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('preprocessing.log', mode='a')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

########## Create Necessary Directories ##########
test_pcts = [str(i) for i in range(10, 51, 10)]  # '10','20','30','40','50'

for phase in ['train', 'test']:
    if phase == 'train':
        train_dir = os.path.join(save_base, phase)
        os.makedirs(train_dir, exist_ok=True)
    else:
        for pct in test_pcts:
            temp = os.path.join(save_base, phase, pct)
            os.makedirs(temp, exist_ok=True)

########## Helper Functions ##########

def load_land_sea_mask(mask_path):
    """
    MATLAB .mat 파일에서 육지/해양 마스크를 로드.
    'Land' 변수가 1=육지, 0=해양일 때,
    육지=999, 해양=1 형태로 변환한 뒤 반환.
    """
    mat_data = io.loadmat(mask_path)
    land_array = mat_data['Land']  # 1=육지, 0=해양
    # 육지(1)→999, 해양(0)→1
    land_sea_mask = np.where(land_array == 1, 999, 1).astype(np.uint16)
    return land_sea_mask

def extract_date_from_filename(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    else:
        raise ValueError(f"No valid date found in filename: {filename}")

def gather_all_files(data_base):
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
    all_files.sort(key=lambda x: x[1])
    return all_files

def calculate_loss_pct(patch):
    """
    patch 전체 픽셀 중 0(결측)의 비율(%).
    """
    total_pixels = patch.size
    if total_pixels == 0:
        return 100
    missing_count = np.count_nonzero(patch == 0)
    return (missing_count / total_pixels) * 100

########## Multiprocessing ##########
def pool_initializer(af, tl, br, npatches, land_mask):
    """
    각 프로세스가 시작할 때 호출되는 initializer.
    전역 변수에 공유 데이터를 등록.
    """
    global all_files_global
    global top_left_coords_global
    global bottom_right_coords_global
    global num_patches_global
    global land_sea_mask_global

    all_files_global = af
    top_left_coords_global = tl
    bottom_right_coords_global = br
    num_patches_global = npatches
    land_sea_mask_global = land_mask

    logging.info("Worker process initialized with shared data.")

def process_file(file_tuple):
    file_path, file_date = file_tuple
    process_single_day(file_path, num_patches_global, save_base, land_sea_mask_global)

########## Core Logic ##########
def process_single_day(file_path, num_patches, save_base, land_sea_mask):
    """
    (1) .nc 파일에서 chl 데이터 로드
    (2) -999 => NaN => 0 치환 (육지/무효)
    (3) 10 이상 => NaN => 0 치환 (이상값)
    (4) 256x256 랜덤 패치 추출(결측>50% 제외)
    (5) 8:2로 train:test 분할
    (6) test는 결측률(0~50)에 따라 폴더 분류
    (7) 최종 PNG:
       - 해양 중 유효(≠0) => 255
       - 해양 중 결측(=0) => 0
       - 육지(land_sea_mask=999) => 255
    """
    try:
        import netCDF4 as nc

        with nc.Dataset(file_path, 'r') as f:
            chl_data = f['merged_daily_Chl'][:].data
            np_a = np.array(chl_data, dtype=np.float32)

            # -999 => NaN => 0
            np_a = np.where(np_a == -999.0, np.nan, np_a)
            # 10 초과 => NaN
            np_a[np_a > 10] = np.nan

            # NaN => 0
            np_a = np.nan_to_num(np_a, nan=0.0)

            # 전부 0이면 스킵
            if np.all(np_a == 0):
                logging.info(f"[SKIP] All data is invalid for {file_path}.")
                return

            date_str = extract_date_from_filename(os.path.basename(file_path)).strftime('%Y%m%d')

            # 좌표 범위
            top_left_y, top_left_x = top_left_coords_global
            bottom_right_y, bottom_right_x = bottom_right_coords_global
            max_y = bottom_right_y - 256
            max_x = bottom_right_x - 256

            if max_y < top_left_y or max_x < top_left_x:
                logging.warning(f"[SKIP] Invalid rectangle range for {file_path}.")
                return

            selected_patches = []
            attempts = 0
            max_attempts = num_patches * 20

            while len(selected_patches) < num_patches and attempts < max_attempts:
                py_start = random.randint(top_left_y, max_y)
                px_start = random.randint(top_left_x, max_x)

                py_end = py_start + 256
                px_end = px_start + 256

                patch = np_a[py_start:py_end, px_start:px_end]
                attempts += 1

                # 크기 체크
                if patch.shape != (256, 256):
                    continue

                # 100% 0 => 스킵
                if np.all(patch == 0):
                    continue

                # 결측률
                loss_pct = calculate_loss_pct(patch)
                if loss_pct > 50:
                    continue

                # 중복 좌표 방지
                if any((py == py_start and px == px_start) for (py, px, _, _) in selected_patches):
                    continue

                selected_patches.append((py_start, px_start, patch, loss_pct))

            if len(selected_patches) < num_patches:
                logging.warning(f"[INFO] Only {len(selected_patches)} patches < {num_patches} for {file_path}.")

            # 최대 num_patches만큼만 사용
            selected_patches = selected_patches[:num_patches]
            patch_count = len(selected_patches)
            if patch_count == 0:
                logging.warning(f"[SKIP] No valid patches after filtering: {file_path}")
                return

            # 8:2 분할
            train_count = int(patch_count * 0.8)
            test_count = patch_count - train_count
            train_patches = selected_patches[:train_count]
            test_patches = selected_patches[train_count:]

            # ============ Train 저장 ============
            for (py, px, patch, loss_pct) in train_patches:
                # 육지 마스크 패치 추출
                land_patch = land_sea_mask[py:py+256, px:px+256]
                if land_patch.shape != (256,256):
                    continue

                # 해양(≠0) => 255, 결측(=0) => 0
                bw_image = np.where(patch != 0, 255, 0).astype(np.uint8)
                # 육지(=999) => 255
                bw_image[land_patch == 999] = 255

                save_path = os.path.join(save_base, 'train', f"{date_str}_r{py}_c{px}.png")
                try:
                    plt.imsave(save_path, bw_image, cmap='gray')
                except Exception as e:
                    logging.error(f"[Train] Failed to save: {save_path}, {e}")

            # ============ Test 저장 ============
            for (py, px, patch, loss_pct) in test_patches:
                # 육지 마스크 패치 추출
                land_patch = land_sea_mask[py:py+256, px:px+256]
                if land_patch.shape != (256,256):
                    continue

                bw_image = np.where(patch != 0, 255, 0).astype(np.uint8)
                # 육지(=999) => 255
                bw_image[land_patch == 999] = 255

                # 결측률 구간별 폴더
                if 1 < loss_pct <= 10:
                    pct_folder = '10'
                elif 10 < loss_pct <= 20:
                    pct_folder = '20'
                elif 20 < loss_pct <= 30:
                    pct_folder = '30'
                elif 30 < loss_pct <= 40:
                    pct_folder = '40'
                elif 40 < loss_pct <= 50:
                    pct_folder = '50'
                else:
                    continue

                save_path = os.path.join(save_base, 'test', pct_folder,
                                         f"{date_str}_r{py}_c{px}.png")
                try:
                    plt.imsave(save_path, bw_image, cmap='gray')
                except Exception as e:
                    logging.error(f"[Test] Failed to save: {save_path}, {e}")

    except Exception as e:
        logging.error(f"[ERROR] process_single_day: {file_path} => {e}")

########## Main Function ##########
def main():
    parser = argparse.ArgumentParser(
        description="Process 1-day chlorophyll-a data with land-sea mask, extract random patches, and save as PNG."
    )
    parser.add_argument('--max_processes', type=int, default=4,
                        help='Maximum number of worker processes (default: 4)')
    parser.add_argument('--num_patches', type=int, default=100,
                        help='Number of random patches to extract per day (default: 100)')
    parser.add_argument('--top_left', type=int, nargs=2, default=[3584, 3072],
                        help='Top-left coords (y x) for patch sampling (default: 3584 3072)')
    parser.add_argument('--bottom_right', type=int, nargs=2, default=[4096, 5120],
                        help='Bottom-right coords (y x) for patch sampling (default: 4096 5120)')
    args = parser.parse_args()

    MAX_PROCESSES = args.max_processes
    num_patches = args.num_patches
    top_left_coords = args.top_left
    bottom_right_coords = args.bottom_right

    global top_left_y, top_left_x, bottom_right_y, bottom_right_x
    top_left_y, top_left_x = top_left_coords
    bottom_right_y, bottom_right_x = bottom_right_coords

    print(f"[INFO] Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"[INFO] Starting processing with {MAX_PROCESSES} processes.")
    logging.info(f"[INFO] Rectangle Top-Left: ({top_left_y}, {top_left_x}), "
                 f"Bottom-Right: ({bottom_right_y}, {bottom_right_x})")
    logging.info(f"[INFO] Number of patches per day: {num_patches}")

    # === 육지-해양 마스크 로드 ===
    land_sea_mask = load_land_sea_mask(mask_path)

    try:
        all_files = gather_all_files(data_base)
        print(f"[INFO] Total files gathered: {len(all_files)}")
        logging.info(f"[INFO] Total files gathered: {len(all_files)}")

        if not all_files:
            logging.error("[ERROR] No '.nc' files found in the specified data directory.")
            print("[ERROR] No '.nc' files found in the specified data directory.")
            exit(1)

        # 멀티프로세스 Pool 초기화
        pool = mp.Pool(
            processes=MAX_PROCESSES,
            initializer=pool_initializer,
            initargs=(all_files, [top_left_y, top_left_x],
                      [bottom_right_y, bottom_right_x], num_patches, land_sea_mask)
        )
    except Exception as e:
        logging.error(f"[ERROR] Failed to initialize multiprocessing pool: {e}")
        raise

    file_tuples = all_files

    try:
        for _ in tqdm(pool.imap_unordered(process_file, file_tuples),
                      total=len(file_tuples),
                      desc="Processing daily files"):
            pass
    except KeyboardInterrupt:
        logging.warning("[WARNING] Processing interrupted by user.")
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        logging.error(f"[ERROR] An error occurred during multiprocessing: {e}")
        pool.terminate()
        pool.join()
        raise

    pool.close()
    pool.join()

    print("[INFO] Patch extraction and saving completed.")
    logging.info("[INFO] Patch extraction and saving completed.")

if __name__ == "__main__":
    main()
