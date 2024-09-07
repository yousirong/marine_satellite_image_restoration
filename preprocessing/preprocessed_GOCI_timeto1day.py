import netCDF4 as nc
import numpy as np
import os
import cv2
import math
from tqdm import tqdm

# 낙동강과 새만금 좌표
x_nak, y_nak = (3377, 3664)  # 낙동강
x_sae, y_sae = (3751, 4629)  # 새만금

#GOCI
# img_path = '/media/pmilab/My Book1/GOCI/L2_Chl-a'
# save_path = '/home/pmilab/Documents/GOCI/Chl-a'
# save_degree_path = '/home/pmilab/Documents/GOCI/Rrs_degree'

#GOCI-II
#img_path = '/media/pmilab/My Book/GOCI-II/Chl/5/1'
#save_path = '/home/pmilab/Documents/GOCI-II/Chl'
#img_path = '/media/pmilab/My Book1/GOCI-II/Rrs(AC)'


# 데이터 경로 설정
img_path = '/media/pmilab/My Book1/GOCI/L2_Chl-a'
save_path = '/home/pmilab/Documents/New_prep/GOCI/RRS_preprocessed'
ocean_idx = np.load("/home/juneyonglee/Desktop/AY_ust/preprocessing/ocean_idx_arr.npy")

if not os.path.isdir(save_path):
    os.makedirs(save_path)

satellite_type = "GOCI" #set satellite_type(GOCI, GOCI-II, ...) 

data_type = 'CHL' #set data_type(RRS, CHL, ...)

if "GOCI-II" in img_path:
    satellite_type = "GOCI-II"

if 'Chl' in img_path:
    data_type = 'CHL'
elif 'Rrs' in img_path:
    data_type = 'RRS'
elif 'SSC' in img_path:
    data_type = 'SSC'




if not os.path.isdir(save_path):
    os.makedirs(save_path)

def check_pct(arr):
    """결측치 및 이상치 비율 계산 함수"""
    nans = np.isnan(arr)
    zeros = (arr == 0)
    neg_outlier = (arr < 0)
    pos_outlier = (arr > 20)  # 상한선은 필요에 맞게 수정 가능
    count = np.sum(nans) + np.sum(zeros) + np.sum(neg_outlier) + np.sum(pos_outlier)
    pct = count / (256 * 256) * 10
    temp_pct = pct * 10
    pct = math.floor(pct) * 10
    return temp_pct, pct

def process_day_data(day_path, img_list, satellite_type, data_type, save_path, ocean_idx):
    """하루 동안의 데이터를 평균 내고 전처리하는 함수"""
    daily_rrs_sum = None
    count = 0

    # 하루에 찍힌 모든 데이터를 평균
    for img_file in img_list:
        file_path = os.path.join(day_path, img_file)
        f = nc.Dataset(file_path, 'r')
        rrs_data = f['geophysical_data']['Rrs']['Rrs_555'][:].data  # 여기서 555 밴드를 사용 (필요에 따라 변경 가능)
        np_rrs = np.array(rrs_data)
        np_rrs = np.where(np_rrs == -999.0, 0, np_rrs)

        if daily_rrs_sum is None:
            daily_rrs_sum = np_rrs
        else:
            daily_rrs_sum += np_rrs
        count += 1
        f.close()

    if count > 0:
        avg_rrs = daily_rrs_sum / count
        avg_rrs = avg_rrs * 255  # 필요한 경우 이미지 범위 조정

        # 낙동강 및 새만금 좌표에 대해 256x256 패치 추출 및 저장
        process_and_save_patch(avg_rrs, x_nak, y_nak, 'nak', day_path, img_file, save_path)
        process_and_save_patch(avg_rrs, x_sae, y_sae, 'sae', day_path, img_file, save_path)

        # 전체 데이터에 대해 256x256 패치 추출
        row, col = avg_rrs.shape
        idx = 0
        for k in range(0, row, 256):
            for r in range(0, col, 256):
                if idx in ocean_idx:
                    patch = avg_rrs[k:k + 256, r:r + 256]
                    if patch.shape != (256, 256):
                        continue

                    row_col = f'_r{k}_c{r}'
                    temp_pct, pct = check_pct(patch)

                    # 100% 결측치 패치는 제외
                    if pct < 100:
                        if temp_pct < 1:
                            save_dir = os.path.join(save_path, 'perfect')
                        else:
                            save_dir = os.path.join(save_path, str(pct))
                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)
                        save_filename = os.path.join(save_dir, img_file[:-3] + row_col + '.tiff')
                        cv2.imwrite(save_filename, patch)
                idx += 1

def process_and_save_patch(data, x, y, region, day_path, img_file, save_path):
    """특정 좌표에서 패치 추출 및 저장"""
    patch = data[x:x + 256, y:y + 256]
    temp_pct, pct = check_pct(patch)

    if pct < 100:
        if temp_pct < 1:
            save_dir = os.path.join(save_path, 'perfect')
        else:
            save_dir = os.path.join(save_path, str(pct))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_filename = os.path.join(save_dir, img_file[:-3] + f'_{region}.tiff')
        cv2.imwrite(save_filename, patch)

# GOCI 또는 GOCI-II 데이터 처리
year_list = os.listdir(img_path)
for year in year_list:
    if year in ["2021"]:
        year_path = os.path.join(img_path, year)
        months = os.listdir(year_path)
        
        for month in tqdm(months):
            month_path = os.path.join(year_path, month)
            if os.path.isdir(month_path):
                days = os.listdir(month_path)

                for day in days:
                    day_path = os.path.join(month_path, day)
                    img_list = [f for f in os.listdir(day_path) if f.endswith('.nc')]

                    if img_list:
                        process_day_data(day_path, img_list, satellite_type="GOCI", data_type="RRS", save_path=save_path, ocean_idx=ocean_idx)
