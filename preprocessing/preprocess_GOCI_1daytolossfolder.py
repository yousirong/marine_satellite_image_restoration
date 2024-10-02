import os
import shutil
import numpy as np
from tqdm import tqdm
import cv2  # OpenCV for reading TIFF files

# 전처리된 파일 경로 설정
processed_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs'
save_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_patch_by_loss'
data_path = '/media/juneyonglee/My Book1/GOCI/L2_Rrs/2021'
save_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_new'
# 밴드 리스트
band_lst = [2, 3, 4]

# 손실률 계산 기준(0%~10%, 10%~20%, ..., 90%~100%)
loss_intervals = [i for i in range(0, 100, 10)]

# 저장 경로 없을 경우 생성
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# 모든 연도, 월, 일 폴더 순회
for root, dirs, files in os.walk(processed_path):
    for file in tqdm(files):
        # .tiff 파일만 처리
        if file.endswith(".tiff"):
            file_path = os.path.join(root, file)

            # 파일명에서 연도, 월, 일 추출
            file_parts = file.split('_')
            band_str = file_parts[2]  # band 정보 (예: band_4)
            sae_nak_info = file_parts[3]  # sae 또는 nak 정보
            row_info = file_parts[4]  # r1861 형식의 정보
            col_info = file_parts[5]  # c2659 형식의 정보

            # 파일 경로에서 폴더명을 통해 연도, 월, 일 정보를 추출
            folder_parts = root.split(os.sep)
            year = folder_parts[-3]
            month = folder_parts[-2]
            day = folder_parts[-1]

            # 새로운 파일명 구성
            new_file_name = f"RRS_{band_str}_{sae_nak_info}_{year}y_{month}m_{day}d_{row_info}_{col_info}"

            # 파일명에서 밴드 정보 추출
            for band in band_lst:
                if f"RRS_band_{band}_" in file:
                    # TIFF 파일 읽기
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        print(f"Failed to read image: {file_path}")
                        continue

                    # 0인 값의 비율(손실률) 계산
                    total_pixels = img.size
                    zero_pixels = np.count_nonzero(img == 0)
                    loss_percentage = (zero_pixels / total_pixels) * 100
                    
                    # 손실률이 0.001% 미만인 경우 perfect 폴더로 이동
                    if loss_percentage < 0.001:
                        save_folder = os.path.join(save_path, f'band_{band}', 'perfect')
                    else:
                        # 손실률에 따라 폴더 결정 (0~10%, 10~20%, ..., 90~100%)
                        loss_category = None
                        for loss in loss_intervals:
                            if loss_percentage <= loss + 10:  # 범위가 0~10%, 10~20% 이렇게 설정됨
                                loss_category = loss
                                break

                        if loss_category is None:
                            loss_category = 90  # 90% 이상의 손실률은 90 폴더에 저장

                        # 손실률에 따른 폴더 경로 설정
                        save_folder = os.path.join(save_path, f'band_{band}', f'loss_{loss_category}')
                    
                    # 폴더가 없으면 생성
                    if not os.path.isdir(save_folder):
                        os.makedirs(save_folder)

                    # 새로운 파일 경로 설정
                    save_file_path = os.path.join(save_folder, new_file_name)
                    
                    # 파일 복사
                    shutil.copy(file_path, save_file_path)

                    print(f"Copied {file} to {save_folder} as {new_file_name}")
