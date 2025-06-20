# -*- coding: utf-8 -*-
"""
GOCI L2A Rrs 데이터: 하루에 8회 촬영된 .he5 파일을
지정한 밴드별로 256×256 타일로 저장하는 스크립트
"""
import os
import re
import h5py
import numpy as np
from datetime import datetime
from tqdm import tqdm
import tifffile as tiff
import logging
import multiprocessing as mp
import argparse

# -------- 경로 및 기본 설정 --------
DATA_BASE    = '/media/juneyonglee/My Book/GOCI/L2_Rrs/2021'
SAVE_BASE    = '/media/juneyonglee/My Book/Preprocessed/GOCI_tiles_daily'
MASK_PATH    = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'
PATCH_SIZE   = 256
HDF_BASE_PATH = 'HDFEOS/GRIDS/Image Data/Data Fields'  # 밴드별 데이터 그룹

# -------- 로깅 설정 --------
logging.basicConfig(
    filename='goci_tile_preproc.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# -------- 전역 변수 (밴드 리스트는 main()에서 설정) --------
BAND_LIST = [2]  # 기본값

# -------- Land-Sea 마스크 로드 --------
land_mask = np.load(MASK_PATH).astype(np.uint8)
H, W = land_mask.shape
logging.info(f"Loaded land-sea mask: shape={land_mask.shape}")

# -------- 날짜+시간 파싱 --------
def extract_datetime(fname):
    m = re.search(r"(\d{14})", os.path.basename(fname))
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None

# -------- 파일 리스트 수집 --------
all_files = []
for root, _, files in os.walk(DATA_BASE):
    for F in files:
        if F.lower().endswith('.he5'):
            dt = extract_datetime(F)
            if dt:
                all_files.append((os.path.join(root, F), dt))
all_files.sort(key=lambda x: x[1])
logging.info(f"Found {len(all_files)} GOCI .he5 files under {DATA_BASE}")

# -------- 개별 파일 처리 --------
def process_file(item):
    file_path, dt = item
    date_str = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S')

    try:
        with h5py.File(file_path, 'r') as ds:
            for band in BAND_LIST:
                dset_path = f"{HDF_BASE_PATH}/Band {band} RRS Image Pixel Values"
                if dset_path not in ds:
                    logging.error(f"[{os.path.basename(file_path)}] Band {band} not found.")
                    continue

                data = ds[dset_path][:].astype(np.float32)
                # 음수 → NaN
                data = np.where(data < 0, np.nan, data)
                # 육지(mask==0) 제외 → NaN→0
                data = np.nan_to_num(data * land_mask)

                # 밴드별 저장 디렉토리
                out_dir = os.path.join(
                    SAVE_BASE,
                    f"band{band}",
                    str(dt.year),
                    date_str,
                    time_str
                )
                os.makedirs(out_dir, exist_ok=True)

                # 타일링
                y_starts = list(range(0, H - PATCH_SIZE + 1, PATCH_SIZE))
                x_starts = list(range(0, W - PATCH_SIZE + 1, PATCH_SIZE))
                if H % PATCH_SIZE and y_starts[-1] != H - PATCH_SIZE:
                    y_starts.append(H - PATCH_SIZE)
                if W % PATCH_SIZE and x_starts[-1] != W - PATCH_SIZE:
                    x_starts.append(W - PATCH_SIZE)

                for y0 in y_starts:
                    for x0 in x_starts:
                        patch = data[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
                        fname = f"y{y0:04d}_x{x0:04d}.tiff"
                        tiff.imwrite(os.path.join(out_dir, fname), patch)

                logging.info(f"[{date_str} {time_str}] band{band} → {out_dir}")

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

# -------- 메인 --------
def main():
    global BAND_LIST

    parser = argparse.ArgumentParser(
        description="GOCI L2A .he5 파일의 Rrs band 2/3/4 등을 256×256 타일로 저장"
    )
    parser.add_argument(
        '--bands', type=int, nargs='+', default=[2],
        help='처리할 Rrs 밴드 번호 리스트, 예) --bands 2 3 4'
    )
    parser.add_argument(
        '--workers', type=int, default=mp.cpu_count(),
        help='사용할 멀티프로세스 워커 수 (기본: CPU 코어 수)'
    )
    args = parser.parse_args()

    BAND_LIST = args.bands
    print(f"Processing bands: {BAND_LIST} with {args.workers} workers")
    logging.info(f"Start: bands={BAND_LIST}, workers={args.workers}")

    with mp.Pool(processes=args.workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_file, all_files),
            total=len(all_files),
            desc='GOCI tiling'
        ):
            pass

    print("모든 GOCI 타일 저장 완료.")
    logging.info("Finished all tiling jobs.")

if __name__ == '__main__':
    main()

# python /home/juneyonglee/Desktop/AY_ust/preprocessing/preprocess_goci_1day_fullpatch.py --bands 4 --workers 32
# /home/juneyonglee/Desktop/AY_ust/preprocessing/preprocess_goci_1day_fullpatch.py ->