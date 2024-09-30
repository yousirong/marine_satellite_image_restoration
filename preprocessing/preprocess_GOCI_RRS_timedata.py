import os
import shutil
import numpy as np
import tifffile as tiff  # OpenCV 대신 tifffile을 사용하여 TIFF 파일을 처리
from tqdm import tqdm
import math

# 경로 설정
source_base = '/home/juneyonglee/gocirrs_backup/home/pmilab/Documents/preprocessed_data/GOCI/Rrs'
dest_base = '/home/juneyonglee/gocirrs_backup/home/pmilab/Documents/preprocessed_data/GOCI/Chl-a_processed'

bands = [2, 3, 4]  # 사용할 밴드
pcts = [str(i) for i in range(0, 100, 10)]  # 0부터 90까지 결측치 폴더
pcts.append('perfect')

# train/loss별 폴더 생성
for band in bands:
    for pct in pcts:
        dest_dir = os.path.join(dest_base, 'train', f'band_{band}', pct)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)

# 결측치 비율 계산 함수
def calculate_missing_rate(img):
    total_pixels = img.size
    missing_pixels = np.sum(img == 0)  # 0을 결측치로 간주
    missing_rate = missing_pixels / total_pixels
    return missing_rate

# 데이터 처리
years = [str(i) for i in range(2013, 2022)]  # 2013~2021 년도

for band in bands:
    print(f"Processing band {band}")
    for year in tqdm(years, desc=f"Processing band {band}"):
        band_folder = os.path.join(source_base, year, str(band))
        if not os.path.isdir(band_folder):
            print(f"{band_folder} 경로가 존재하지 않습니다. 건너뜁니다.")
            continue

        # 0 폴더에서 결측치 0.001 미만인 경우는 perfect 폴더로 이동
        source_zero_folder = os.path.join(band_folder, '0')
        if os.path.isdir(source_zero_folder):
            files = os.listdir(source_zero_folder)
            for file in files:
                file_path = os.path.join(source_zero_folder, file)
                try:
                    # tifffile을 사용하여 TIFF 파일 읽기
                    img = tiff.imread(file_path)
                except Exception as e:
                    print(f"이미지 로드 실패: {file_path}, 오류: {str(e)}")
                    continue

                # 결측치 비율 계산
                missing_rate = calculate_missing_rate(img)

                # 결측치 비율이 0.001 미만인 경우 perfect 폴더로 이동
                if missing_rate < 0.001:
                    dest_perfect_folder = os.path.join(dest_base, 'train', f'band_{band}', 'perfect')
                    shutil.copy(file_path, os.path.join(dest_perfect_folder, file))
                else:
                    # 각 결측치 비율에 맞는 폴더로 복사
                    pct_folder = math.floor(missing_rate * 10) * 10
                    if pct_folder < 100:  # 100인 폴더는 제외
                        dest_folder = os.path.join(dest_base, 'train', f'band_{band}', str(pct_folder))
                        shutil.copy(file_path, os.path.join(dest_folder, file))

print("데이터 처리 완료.")
