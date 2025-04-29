# -*- coding: utf-8 -*-
"""
UST21 데이터셋: 원본 1일 단위 chlorophyll-a 데이터를 멀티프로세싱으로 처리,
전체 이미지를 256x256 타일로 분할하여
년도/날짜별 폴더에 TIFF 패치로 저장하는 스크립트
"""
import os
import re
import netCDF4 as nc
import numpy as np
from scipy import io
import tifffile as tiff
from datetime import datetime
from tqdm import tqdm
import logging
import multiprocessing as mp

# -------- 경로 설정 --------
DATA_BASE = '/media/juneyonglee/My Book/UST21/01_day'
SAVE_BASE = '/media/juneyonglee/My Book/Preprocessed/UST21_tiles_daily'
MASK_PATH = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'
PATCH_SIZE = 256

# -------- 로깅 설정 --------
logging.basicConfig(
    filename='tile_preprocessing_daily.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# -------- Land-Sea 마스크 로드 --------
mask_mat = io.loadmat(MASK_PATH)
land_mask_raw = mask_mat['Land']  # MATLAB: 0=바다, 1=육지
# 바다만 처리하기 위해: 바다=1, 육지=0
land_mask = np.where(land_mask_raw == 0, 1, 0).astype(np.uint8)
H, W = land_mask.shape
logging.info(f"Loaded land-sea mask: shape={land_mask.shape}")

# -------- 파일 수집 및 정렬 --------

def extract_date(fname):
    m = re.search(r"(\d{8})", os.path.basename(fname))
    return datetime.strptime(m.group(1), "%Y%m%d") if m else None

all_files = []
for root, _, files in os.walk(DATA_BASE):
    for F in files:
        if F.endswith('.nc'):
            date = extract_date(F)
            if date:
                all_files.append((os.path.join(root, F), date))
all_files.sort(key=lambda x: x[1])
logging.info(f"Total daily .nc files: {len(all_files)}")

# -------- 개별 파일 처리 함수 --------
def process_file(item):
    file_path, date = item
    try:
        # 데이터 로드
        with nc.Dataset(file_path, 'r') as ds:
            data = ds['merged_daily_Chl'][:].data.astype(np.float32)
        # 결측치 및 이상치 처리
        data = np.where(data == -999.0, np.nan, data)
        data[data > 10] = np.nan
        # 바다만 남기고 NaN->0
        data = np.nan_to_num(data * land_mask)

        # 저장 디렉토리 생성
        year_dir = os.path.join(SAVE_BASE, str(date.year))
        date_dir = os.path.join(year_dir, date.strftime('%Y%m%d'))
        os.makedirs(date_dir, exist_ok=True)

        # 타일 시작 좌표 리스트 (끝단 포함)
        y_starts = list(range(0, H - PATCH_SIZE + 1, PATCH_SIZE))
        x_starts = list(range(0, W - PATCH_SIZE + 1, PATCH_SIZE))
        if (H % PATCH_SIZE) != 0 and y_starts[-1] != H - PATCH_SIZE:
            y_starts.append(H - PATCH_SIZE)
        if (W % PATCH_SIZE) != 0 and x_starts[-1] != W - PATCH_SIZE:
            x_starts.append(W - PATCH_SIZE)

        # 타일 분할 및 저장
        for y0 in y_starts:
            for x0 in x_starts:
                patch = data[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
                fname = f"y{y0:04d}_x{x0:04d}.tiff"
                out_path = os.path.join(date_dir, fname)
                try:
                    tiff.imwrite(out_path, patch)
                except Exception as ex:
                    logging.error(f"Failed to write patch {out_path}: {ex}")
        logging.info(f"Saved daily patches for {date.strftime('%Y-%m-%d')} ({file_path})")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
    return

# -------- 메인 실행부 --------
if __name__ == '__main__':
    print(f"Starting multiprocessing with {mp.cpu_count()} workers...")
    with mp.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files), desc='Daily files'):
            pass
    print('모든 일별 패치 저장 완료.')
