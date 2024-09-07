import netCDF4 as nc
import numpy as np
import os
import cv2
import math
from tqdm import tqdm
import random

def check_pct(arr):
    nans = np.isnan(arr)
    zeros = (arr == 0)
    neg_outlier = (arr < 0)
    pos_outlier = (arr > 20)
    assert np.sum(pos_outlier) < 1
    count = np.sum(nans) + np.sum(zeros) + np.sum(neg_outlier) + np.sum(pos_outlier)
    # loss rate
    pct = count / (256 * 256) * 10
    temp_pct = pct * 10
    pct = math.floor(pct) * 10
    return temp_pct, pct

########## path ##########
data_base = '/media/juneyonglee/My Book/UST21/Daily'
save_base = '/media/juneyonglee/My Book/Preprocessed/UST/Chl-a_8day'
ocean_base = '/media/juneyonglee/My Book/Preprocessed/UST21/Chl-a/ocean'
ocean_idx = np.load("/home/juneyonglee/Desktop/AY_ust/preprocessing/ocean_idx_arr.npy")
######### point #########
x_nak, y_nak = (3377, 3664)
x_sae, y_sae = (3751, 4629)
##########################

# 연도와 월 리스트
years = [str(i) for i in range(2012, 2022)]
months = [f"{i:02}" for i in range(1, 13)]
pcts = [str(i) for i in range(0, 100, 10)]
pcts.append('perfect')

for phase in ['train', 'test', 'ocean']:
    for pct in pcts:
        temp = os.path.join(save_base, phase, pct)
        if not os.path.isdir(temp):
            os.makedirs(temp)

# 8일 평균을 계산하는 함수
def calculate_8day_avg(files, data_dir):
    data_list = []
    for file in files:
        path = os.path.join(data_dir, file)
        f = nc.Dataset(path, 'r')
        a = f['merged_daily_Chl'][:].data
        np_a = np.array(a)
        np_a = np.where(np_a == -999.0, 0, np_a)  # 결측값 처리
        np_a[np_a > 20] = 0  # 20 이상의 값은 이상치로 간주하여 제거
        data_list.append(np_a)
        f.close()
    
    # 8일 평균 계산
    avg_data = np.mean(data_list, axis=0)
    return avg_data

# 각 년도에 대해 처리
for year in years:
    phase = 'train' if int(year) < 2021 else 'test'
    data_year = os.path.join(data_base, year)

    # 각 월에 대해 처리
    for month in tqdm(months):
        data_month = os.path.join(data_year, str(month))
        if os.path.isdir(data_month):
            imgs = sorted(os.listdir(data_month))

            # 8일씩 그룹화하여 평균 계산
            for i in range(0, len(imgs) - 7):
                avg_data = calculate_8day_avg(imgs[i:i + 8], data_month)

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

                # 패치에 대한 결측치와 이상치 확인
                s_temp, s_pct = check_pct(s_patch_256)
                n_temp, n_pct = check_pct(n_patch_256)

                # 저장 경로 설정 및 저장
                if s_pct != 100:
                    if s_temp < 1:
                        save_path = os.path.join(ocean_base, 'perfect', f"{year}_{month}_{i}_sae.tiff")
                        cv2.imwrite(save_path, s_patch_256)
                    save_path = os.path.join(ocean_base, str(s_pct), f"{year}_{month}_{i}_sae.tiff")
                    cv2.imwrite(save_path, s_patch_256)

                if n_pct != 100:
                    if n_temp < 1:
                        save_path = os.path.join(ocean_base, 'perfect', f"{year}_{month}_{i}_nak.tiff")
                        cv2.imwrite(save_path, n_patch_256)
                    save_path = os.path.join(ocean_base, str(n_pct), f"{year}_{month}_{i}_nak.tiff")
                    cv2.imwrite(save_path, n_patch_256)
