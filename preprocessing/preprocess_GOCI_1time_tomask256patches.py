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

# ======================== Path Definitions =========================
data_base = '/media/juneyonglee/GOCI_vol2/GOCI/L2_Rrs'  # GOCI RRS 데이터(.he5)가 있는 루트 경로
save_base = '/media/juneyonglee/My Book/Preprocessed/GOCI_RRS/mask'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI.npy'  # GOCI용 육지-해양 마스크 (.npy)

# 사용할 밴드 리스트
band_lst = [2, 3, 4]

########## Configure Logging ##########
logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('preprocessing_rrs.log', mode='a')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

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
# 필요한 영역 잘라낼 때 사용할 좌표 (사용자 정의)
top_left_y, top_left_x = 2560, 1536   # (y, x)
bottom_right_y, bottom_right_x = 3072, 3072  # (y, x)

########## Create Necessary Directories ##########
# test 데이터를 결측률(0~50%) 별로 분류할 폴더
test_pcts = [str(i) for i in range(10, 51, 10)]  # '10','20','30','40','50'

# band별 폴더 + train/test/... 생성
for band in band_lst:
    band_dir = os.path.join(save_base, f"band{band}")
    # train 폴더
    train_dir = os.path.join(band_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    # test/\[pcts\] 폴더
    for pct in test_pcts:
        os.makedirs(os.path.join(band_dir, 'test', pct), exist_ok=True)

########## Helper Functions ##########
def extract_date_from_filename(filename):
    """
    파일명에서 8자리 날짜(YYYYMMDD) 추출
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    else:
        raise ValueError(f"No valid date found in filename: {filename}")

def gather_all_files(data_base):
    """
    data_base 폴더 아래 '.he5' 파일들을 전부 모아서,
    (full_path, date) 튜플 리스트로 정렬 반환
    """
    all_files = []
    for root, dirs, files in os.walk(data_base):
        for file in files:
            # GOCI가 .he5 확장자를 쓰면 아래를 '.he5'로 맞춤
            if file.endswith('.he5'):
                full_path = os.path.join(root, file)
                try:
                    # 파일명에서 날짜 추출
                    file_date = extract_date_from_filename(file)
                    all_files.append((full_path, file_date))
                except ValueError as e:
                    logging.warning(f"Skipping file due to error: {e}")
    # 날짜 기준 정렬
    all_files.sort(key=lambda x: x[1])
    return all_files

def calculate_loss_pct(patch):
    """
    patch 전체 픽셀 중 0(결측)인 픽셀 비율(%).
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
    """
    한 파일(= 한 시간대)을 처리하는 함수.
    여기서 band 2,3,4 각각에 대해 'RRS' 데이터를 읽고 랜덤 패치를 추출.
    """
    file_path, file_date = file_tuple
    process_time_rrs(file_path, num_patches_global, land_sea_mask_global)

########## Core Logic ##########
def process_time_rrs(file_path, num_patches, land_sea_mask):
    """
    (1) 한 파일(=한 시간대)에 대해 band_lst의 각 band에 대해 RRS 데이터 로드
    (2) -999 => NaN => 0 치환 (결측)
    (3) 256x256 랜덤 패치 추출(결측>50%는 제외)
    (4) 8:2로 train:test 분할
    (5) test는 결측률(0~50)에 따라 폴더 분류
    (6) 최종 PNG:
       - 해양 중 유효(≠0) => 255
       - 해양 중 결측(=0) => 0
       - 육지(land_sea_mask=999) => 255
    (7) band별 폴더에 저장
    """
    import os

    try:
        # 파일 열기
        with nc.Dataset(file_path, 'r') as f:
            # 파일명에서 날짜 추출
            date_str = extract_date_from_filename(os.path.basename(file_path)).strftime('%Y%m%d')

            # 만약 전체가 0인 데이터면 스킵 (하지만 band별로 달라질 수 있으니, 밑에서 처리)
            # 일단은 band별로 진행

            # 좌표 범위
            top_left_y, top_left_x = top_left_coords_global
            bottom_right_y, bottom_right_x = bottom_right_coords_global
            max_y = bottom_right_y - 256
            max_x = bottom_right_x - 256

            if max_y < top_left_y or max_x < top_left_x:
                logging.warning(f"[SKIP] Invalid rectangle range: {file_path}")
                return

            # ============ band 2,3,4 각각 처리 ============
            for band in band_lst:
                # GOCI 실제 구조에 맞춰 아래 var_name 수정
                # 예: 'HDFEOS/GRIDS/Image Data/Data Fields/Band 2 RRS Image Pixel Values'
                var_name = f'HDFEOS/GRIDS/Image Data/Data Fields/Band {band} RRS Image Pixel Values'
                try:
                    rrs_data = f[var_name][:]  # (H, W)
                except KeyError:
                    logging.warning(f"Band {band} not found in file {os.path.basename(file_path)}.")
                    continue

                np_a = np.array(rrs_data, dtype=np.float32)
                # -999 => NaN
                np_a = np.where(np_a == -999.0, np.nan, np_a)
                np_a = np.nan_to_num(np_a, nan=0.0)

                # 전부 0이면 스킵
                if np.all(np_a == 0):
                    logging.info(f"[SKIP] {file_path}, band{band}: all data invalid.")
                    continue

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

                    # 전부 0 => 스킵
                    if np.all(patch == 0):
                        continue

                    # 결측률
                    loss_pct = calculate_loss_pct(patch)
                    if loss_pct > 50:
                        continue

                    # 중복 좌표 방지
                    if any((p[0] == py_start and p[1] == px_start) for p in selected_patches):
                        continue

                    selected_patches.append((py_start, px_start, patch, loss_pct))

                if len(selected_patches) < num_patches:
                    logging.warning(f"[INFO] {file_path} (band{band}): only {len(selected_patches)} patches (< {num_patches})")

                # 최대 num_patches만큼만 사용
                selected_patches = selected_patches[:num_patches]
                patch_count = len(selected_patches)
                if patch_count == 0:
                    logging.warning(f"[SKIP] {file_path}, band{band}: No valid patches after filtering.")
                    continue

                # 8:2 분할
                train_count = int(patch_count * 0.8)
                test_count = patch_count - train_count
                train_patches = selected_patches[:train_count]
                test_patches = selected_patches[train_count:]

                # ============ Train 저장 ============
                for (py, px, patch, loss_pct) in train_patches:
                    land_patch = land_sea_mask[py:py+256, px:px+256]
                    if land_patch.shape != (256,256):
                        continue

                    # 해양(≠0) => 255, 결측(=0) => 0
                    bw_image = np.where(patch != 0, 255, 0).astype(np.uint8)
                    # 육지(=999) => 255
                    bw_image[land_patch == 999] = 255

                    # band별 train 폴더에 저장
                    save_path = os.path.join(save_base, f"band{band}", 'train',
                                             f"{date_str}_r{py}_c{px}.png")
                    try:
                        plt.imsave(save_path, bw_image, cmap='gray')
                    except Exception as e:
                        logging.error(f"[Train] Failed to save: {save_path}, {e}")

                # ============ Test 저장 ============
                for (py, px, patch, loss_pct) in test_patches:
                    land_patch = land_sea_mask[py:py+256, px:px+256]
                    if land_patch.shape != (256,256):
                        continue

                    bw_image = np.where(patch != 0, 255, 0).astype(np.uint8)
                    bw_image[land_patch == 999] = 255

                    # 결측률 구간별 폴더 분류
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
                        # 여기선 0% 미만, 50% 초과는 제외
                        continue

                    # band별 test/\[pct_folder\] 폴더에 저장
                    save_path = os.path.join(save_base, f"band{band}", 'test', pct_folder,
                                             f"{date_str}_r{py}_c{px}.png")
                    try:
                        plt.imsave(save_path, bw_image, cmap='gray')
                    except Exception as e:
                        logging.error(f"[Test] Failed to save: {save_path}, {e}")

    except Exception as e:
        logging.error(f"[ERROR] process_time_rrs: {file_path} => {e}")


########## Main Function ##########
def main():
    parser = argparse.ArgumentParser(
        description="Process GOCI RRS time-slice data for Band 2,3,4, extract random patches, and save as PNG."
    )
    parser.add_argument('--max_processes', type=int, default=4,
                        help='Maximum number of worker processes (default: 4)')
    parser.add_argument('--num_patches', type=int, default=10,
                        help='Number of random patches to extract per file/time (default: 100)')
    parser.add_argument('--top_left', type=int, nargs=2, default=[2560, 1536],
                        help='Top-left coords (y x) for patch sampling (default: 2560 1536)')
    parser.add_argument('--bottom_right', type=int, nargs=2, default=[3072, 3072],
                        help='Bottom-right coords (y x) for patch sampling (default: 3072 3072)')
    args = parser.parse_args()

    MAX_PROCESSES = args.max_processes
    num_patches = args.num_patches
    top_left_coords = args.top_left
    bottom_right_coords = args.bottom_right

    global top_left_y, top_left_x, bottom_right_y, bottom_right_x
    top_left_y, top_left_x = top_left_coords
    bottom_right_y, bottom_right_x = bottom_right_coords

    print(f"[INFO] Starting GOCI time-slice processing with {MAX_PROCESSES} processes.")
    logging.info(f"[INFO] Starting GOCI time-slice processing with {MAX_PROCESSES} processes.")
    logging.info(f"[INFO] Rectangle Top-Left: ({top_left_y}, {top_left_x}), "
                 f"Bottom-Right: ({bottom_right_y}, {bottom_right_x})")
    logging.info(f"[INFO] Number of patches per file: {num_patches}")

    try:
        all_files = gather_all_files(data_base)
        print(f"[INFO] Total files gathered: {len(all_files)}")
        logging.info(f"[INFO] Total files gathered: {len(all_files)}")

        if not all_files:
            logging.error("[ERROR] No '.he5' files found in the specified data directory.")
            print("[ERROR] No '.he5' files found in the specified data directory.")
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
                      desc="Processing time-slice files"):
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
