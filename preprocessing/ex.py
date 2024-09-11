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
    # 마스크가 1인 부분(해양)만 선택
    ocean_pixels = (mask == 1)
    arr_ocean = arr[ocean_pixels]  # 육지 부분을 제외하고 해양 부분만 분석
    
    # 육지(999)를 제외하고 유효한 해양 데이터 추출
    valid_ocean_data = arr_ocean[(arr_ocean != 999)]
    
    # 결측값과 이상치 계산
    # nans = np.isnan(valid_ocean_data)
    zeros = (valid_ocean_data == 0)
    neg_outlier = (valid_ocean_data < 0)
    pos_outlier = (valid_ocean_data > 20)
    
    # 결측값 및 이상치 개수 확인
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
    valid_ocean_pixels = np.sum(~np.isnan(patch) & (patch != 999) & (patch > 0) & (patch <= 20) & ocean_pixels)  # 유효한 해양 데이터 (육지, 이상치 제외)
    total_ocean_pixels = np.sum(ocean_pixels)  # 전체 해양 픽셀 수
    
    ocean_data_pct = (valid_ocean_pixels / total_ocean_pixels) * 100 if total_ocean_pixels > 0 else 0
    return ocean_data_pct

########## path ##########
data_base = '/media/juneyonglee/My Book/UST21/Daily/2012/01'
save_base = '/media/juneyonglee/My Book/Preprocessed/UST/Chl-a_8day'
ocean_base = '/media/juneyonglee/My Book/Preprocessed/UST21/Chl-a/ocean'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'
land_sea_mask = io.loadmat(mask_path)
land_sea_mask = land_sea_mask['Land']  # MATLAB에서 Land 변수 가져오기

# 육지 부분을 빨간색(999)로 설정, 해양은 1로 설정
land_sea_mask = np.where(land_sea_mask == 0, 999, 1)

######### point #########
x_nak, y_nak = (3377, 3664)
x_sae, y_sae = (3751, 4629)
##########################
years = os.listdir(data_base)
months = [f"{i:02}" for i in range(1, 13)]

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
    print(f"Saving patch to: {file_path}")
    
    # 육지(빨간색 999)는 빨간색으로 시각화
    patch_visual = np.where(patch == 999, 255, patch)  # 빨간색을 RGB 255로 시각화
    cv2.imwrite(file_path, patch_visual)
    print(f"Patch saved as {file_path}")

# 2012년 1월 데이터만 처리
data_month = data_base

if os.path.isdir(data_month):
    imgs = sorted(os.listdir(data_month))

    # 8일씩 그룹화하여 평균 계산
    for i in range(0, len(imgs) - 7):
        avg_data = calculate_8day_avg(imgs[i:i + 8], data_month, land_sea_mask)

        # 중요한 해역(낙동강, 새만금)에서 512x512 영역 추출
        n_patch_512 = avg_data[x_nak:x_nak + 512, y_nak:y_nak + 512]
        s_patch_512 = avg_data[x_sae:x_sae + 512, y_sae:y_sae + 512]

        # 512x512 영역에서 256x256 랜덤 패치 선택
        n_row = random.randint(0, 512 - 256)
        n_col = random.randint(0, 512 - 256)
        s_row = random.randint(0, 512 - 256)
        s_col = random.randint(0, 512 - 256)

        n_patch_256 = n_patch_512[n_row:n_row + 256, n_col:n_col + 256]
        s_patch_256 = s_patch_512[s_row:s_row + 256, s_col:s_col + 256]

        # 마스크 패치 추출
        n_mask_patch = land_sea_mask[x_nak + n_row:x_nak + n_row + 256, y_nak + n_col:y_nak + n_col + 256]
        s_mask_patch = land_sea_mask[x_sae + s_row:x_sae + s_row + 256, y_sae + s_col:y_sae + s_col + 256]

        # 해양 데이터 비율 확인
        min_ocean_pct = 0.1  # 최소 해양 데이터 비율을 낮춰서 설정
        n_ocean_pct = check_ocean_pct(n_patch_256, n_mask_patch)
        s_ocean_pct = check_ocean_pct(s_patch_256, s_mask_patch)

        # 결측치 및 이상치 비율 계산
        n_pct = check_pct(n_patch_256, n_mask_patch)
        s_pct = check_pct(s_patch_256, s_mask_patch)

        # 패치 로그 출력 및 저장 확인
        print(f"Patch {i}: n_ocean_pct = {n_ocean_pct:.2f}, s_ocean_pct = {s_ocean_pct:.2f}, n_pct = {n_pct:.2f}, s_pct = {s_pct:.2f}")

        # n_patch 저장 (loss_pct와 ocean_pct에 따라 다양한 폴더에 저장)
        if n_ocean_pct >= min_ocean_pct:
            if n_pct == 0:
                save_path = os.path.join(save_base, 'train', 'perfect', f"2012_01_{i}_nak.tiff")
            else:
                save_path = os.path.join(save_base, 'train', str(int(n_pct // 10) * 10), f"2012_01_{i}_nak.tiff")
            save_patch_image(n_patch_256, save_path)

        # s_patch 저장 (loss_pct와 ocean_pct에 따라 다양한 폴더에 저장)
        if s_ocean_pct >= min_ocean_pct:
            if s_pct == 0:
                save_path = os.path.join(save_base, 'train', 'perfect', f"2012_01_{i}_sae.tiff")
            else:
                save_path = os.path.join(save_base, 'train', str(int(s_pct // 10) * 10), f"2012_01_{i}_sae.tiff")
            save_patch_image(s_patch_256, save_path)
