import netCDF4 as nc
import numpy as np
import os
import cv2
import math
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from scipy import io

# 수정된 check_pct 함수: 해양 데이터에 대해서만 loss를 계산
def check_pct(arr, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀만 선택
    arr_ocean = arr[ocean_pixels]  # 육지 부분을 제외하고 해양 부분만 분석
    
    # 육지(999)를 제외하고 유효한 해양 데이터 추출
    valid_ocean_data = arr_ocean[(arr_ocean != 999)]
    
    # 결측값 및 이상치 계산
    zeros = (valid_ocean_data == 0)
    neg_outlier = (valid_ocean_data < 0)
    pos_outlier = (valid_ocean_data > 20)
    
    count = np.sum(zeros) + np.sum(neg_outlier) + np.sum(pos_outlier)
    total_ocean_pixels = valid_ocean_data.size  # 유효한 해양 픽셀 수
    
    if total_ocean_pixels > 0:
        loss_pct = (count / total_ocean_pixels) * 100  # 비율을 퍼센트로 계산
    else:
        loss_pct = 100
    
    return loss_pct

# 해양 데이터 비율을 계산하는 함수
def check_ocean_pct(patch, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀 선택
    valid_ocean_pixels = np.sum(~np.isnan(patch) & (patch != 999) & (patch > 0) & (patch <= 20) & ocean_pixels)  # 유효한 해양 데이터 (육지 및 이상치 제외)
    total_ocean_pixels = np.sum(ocean_pixels)  # 전체 해양 픽셀 수
    
    ocean_data_pct = (valid_ocean_pixels / total_ocean_pixels) * 100 if total_ocean_pixels > 0 else 0
    return ocean_data_pct

########## path ##########
data_base = '/media/juneyonglee/My Book/UST21/Daily'
save_base = '/media/juneyonglee/My Book/Preprocessed/UST/Chl-a_8day'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'
ocean_idx = np.load('/home/juneyonglee/Desktop/AY_ust/preprocessing/ocean_idx_arr.npy', allow_pickle=True)
land_sea_mask = io.loadmat(mask_path)
land_sea_mask = land_sea_mask['Land']  # MATLAB에서 Land 변수 가져오기

# 육지 부분을 빨간색(999)로 설정, 해양은 1로 설정
land_sea_mask = np.where(land_sea_mask == 0, 999, 1)

######### point #########
x_nak, y_nak = (3377, 3664)
x_sae, y_sae = (3751, 4629)
##########################

# 필요한 디렉토리 생성
pcts = [str(i) for i in range(0, 100, 10)]
pcts.append('perfect')

for phase in ['train', 'test', 'ocean']:
    for pct in pcts:
        temp = os.path.join(save_base, phase, pct)
        if not os.path.isdir(temp):
            os.makedirs(temp)

# 8일 평균을 계산하는 함수
def calculate_8day_avg(files, data_dir, land_sea_mask):
    data_list = []
    for file in files:
        if file.endswith('.nc'):  # '.nc' 파일만 처리
            path = os.path.join(data_dir, file)
            f = nc.Dataset(path, 'r')
            a = f['merged_daily_Chl'][:].data
            np_a = np.array(a)
            np_a = np.where(np_a == -999.0, 0, np_a)  # 결측값을 0으로 처리
            np_a[np_a > 20] = 0  # 이상치를 0으로 처리
            np_a = np_a * land_sea_mask  # 육지-해양 마스크 적용 (육지는 999)
            data_list.append(np_a)
            f.close()
    
    # NaN을 무시하고 평균 계산
    avg_data = np.nanmean(data_list, axis=0)
    print(f"8일 평균 계산 완료: {files[0]} ~ {files[-1]}")
    
    return avg_data

# 패치 저장 함수 추가
def save_patch_image(patch, file_path):
    # 육지(빨간색 999)는 빨간색으로 시각화
    patch_visual = np.where(patch == 999, 255, patch)  # 빨간색을 RGB 255로 시각화
    cv2.imwrite(file_path, patch_visual)

# 전체 년도를 처리할 수 있도록 수정
years = os.listdir(data_base)

# 패치 생성 개수 설정 (예: 한 영역에서 5개의 256x256 패치 추출)
num_patches_per_region = 50

for year in years:
    print(f"Processing year: {year}")
    data_year = os.path.join(data_base, year)

    # 각 월에 대해서 처리
    for month in tqdm([f"{i:02}" for i in range(1, 13)]):
        data_month = os.path.join(data_year, str(month))
        if os.path.isdir(data_month):
            imgs = sorted(os.listdir(data_month))

            # 8일씩 그룹화하여 평균 계산
            idx = 0
            for i in range(0, len(imgs) - 7):
                avg_data = calculate_8day_avg(imgs[i:i + 8], data_month, land_sea_mask)

                # 중요한 해역(낙동강, 새만금)에서 512x512 영역 추출
                n_patch_512 = avg_data[x_nak:x_nak + 512, y_nak:y_nak + 512]
                s_patch_512 = avg_data[x_sae:x_sae + 512, y_sae:y_sae + 512]

                # 가능한 패치 좌표를 미리 생성
                n_possible_coords = [(row, col) for row in range(0, 512 - 256) for col in range(0, 512 - 256)]
                s_possible_coords = [(row, col) for row in range(0, 512 - 256) for col in range(0, 512 - 256)]

                # 여러 개의 256x256 랜덤 패치 추출 (좌표 중복 방지)
                for patch_num in range(num_patches_per_region):
                    # 낙동강에서 랜덤 좌표 선택 (중복 방지)
                    n_row, n_col = random.choice(n_possible_coords)
                    n_possible_coords.remove((n_row, n_col))  # 선택된 좌표 제거

                    # 새만금에서 랜덤 좌표 선택 (중복 방지)
                    s_row, s_col = random.choice(s_possible_coords)
                    s_possible_coords.remove((s_row, s_col))  # 선택된 좌표 제거

                    n_patch_256 = n_patch_512[n_row:n_row + 256, n_col:n_col + 256]
                    s_patch_256 = s_patch_512[s_row:s_row + 256, s_col:s_col + 256]

                    # 마스크 패치 추출
                    n_mask_patch = land_sea_mask[x_nak + n_row:x_nak + n_row + 256, y_nak + n_col:y_nak + n_col + 256]
                    s_mask_patch = land_sea_mask[x_sae + s_row:x_sae + s_row + 256, y_sae + s_col:y_sae + s_col + 256]

                    # 해양 데이터 비율 확인
                    min_ocean_pct = 0.1  # 최소 해양 데이터 비율을 설정
                    n_ocean_pct = check_ocean_pct(n_patch_256, n_mask_patch)
                    s_ocean_pct = check_ocean_pct(s_patch_256, s_mask_patch)

                    # 결측치 및 이상치 비율 계산
                    n_pct = check_pct(n_patch_256, n_mask_patch)
                    s_pct = check_pct(s_patch_256, s_mask_patch)

                    print(f"Patch {i}: n_ocean_pct = {n_ocean_pct:.2f}, s_ocean_pct = {s_ocean_pct:.2f}, n_pct = {n_pct:.2f}, s_pct = {s_pct:.2f}")

                    # 패치 저장 (조건을 변경해 ocean_idx 매칭 조건 제거)
                    if n_ocean_pct >= min_ocean_pct:
                        if n_pct == 0:
                            save_path = os.path.join(save_base, 'train', 'perfect', f"{year}_{month}_{i}_nak_r{n_row}_c{n_col}.tiff")
                        else:
                            save_path = os.path.join(save_base, 'train', str(int(n_pct // 10) * 10), f"{year}_{month}_{i}_nak_r{n_row}_c{n_col}.tiff")
                        save_patch_image(n_patch_256, save_path)

                    if s_ocean_pct >= min_ocean_pct:
                        if s_pct == 0:
                            save_path = os.path.join(save_base, 'train', 'perfect', f"{year}_{month}_{i}_sae_r{s_row}_c{s_col}.tiff")
                        else:
                            save_path = os.path.join(save_base, 'train', str(int(s_pct // 10) * 10), f"{year}_{month}_{i}_sae_r{s_row}_c{s_col}.tiff")
                        save_patch_image(s_patch_256, save_path)
