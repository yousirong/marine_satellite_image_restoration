# MODIS Chl-a 데이터의 전체 년도에 대한 정보 추출 및 전처리
import netCDF4 as nc
import numpy as np
import os
import cv2
import math
from tqdm import tqdm
import time

########## path ##########
data_base = '/media/juneyonglee/My Book/MODIS/MODIS-Aqua/OC'
save_base = '/media/juneyonglee/My Book/Preprocessed/MODIS/Chl-a'
ocean_idx = np.load("/home/juneyonglee/Desktop/AY_ust/preprocessing/ocean_idx_arr.npy", allow_pickle=True)
##########################

years = os.listdir(data_base)
months = [f"{i:02}" for i in range(1, 13)]
days = [f"{i:02}" for i in range(1, 32)]
pcts = [str(i) for i in range(0, 100, 10)]
pcts.append('perfect')

if '2021_old' in years:
    years.remove('2021_old')

for phase in ['Train', 'Test']:
    for pct in pcts:
        temp = os.path.join(save_base, phase, pct)
        if not os.path.isdir(temp):
            os.makedirs(temp)

for year in years:
    print(year)
    phase = 'Train' if int(year) < 2021 else 'Test'
    data_year = os.path.join(data_base, year)
    for month in tqdm(months):
        data_month = os.path.join(data_year, str(month))
        if os.path.isdir(data_month):
            for day in days:
                data_day = os.path.join(data_month, str(day))
                if os.path.isdir(data_day):
                    imgs = os.listdir(data_day)
                    for img in imgs:
                        if not 'nc' in img:
                            continue
                        path = os.path.join(data_day, img)
                        
                        # 예외 처리 및 재시도 로직 추가
                        attempts = 3
                        for attempt in range(attempts):
                            try:
                                f = nc.Dataset(path, 'r')
                                break
                            except Exception as e:
                                print(f"Error opening {path}: {e}")
                                if attempt < attempts - 1:
                                    print(f"Retrying... ({attempt+1}/{attempts})")
                                    time.sleep(1)  # 잠시 대기 후 재시도
                                else:
                                    print(f"Failed to open {path} after {attempts} attempts. Skipping.")
                                    continue

                        # 파일 열기 실패 시 스킵
                        if not f.isopen():
                            continue

                        try:
                            a = f['geophysical_data']['chlor_a'][:]
                        except KeyError:
                            print(f"chlor_a not found in {path}")
                            f.close()
                            continue
                        except Exception as e:
                            print(f"Error reading chlor_a from {path}: {e}")
                            f.close()
                            continue

                        np_a = np.array(a)
                        np_a = np.where(np_a == -999.0, 0, np_a)
                        np_a[np_a > 20] = 0
                        idx = 0
                        row, col = np_a.shape
                        for k in range(0, row, 256):
                            for r in range(0, col, 256):
                                if idx in ocean_idx:
                                    new_arr = np_a[k:k+256, r+r+256]

                                    if new_arr.shape != (256, 256):
                                        continue
                                    row_col = '_r' + str(k) + '_c' + str(r)

                                    # outliers
                                    nans = np.isnan(new_arr)
                                    zeros = (new_arr == 0)
                                    neg_outlier = (new_arr < 0)
                                    pos_outlier = (new_arr > 20)
                                    assert np.sum(pos_outlier) < 1
                                    count = np.sum(nans) + np.sum(zeros) + np.sum(neg_outlier) + np.sum(pos_outlier)

                                    # loss rate
                                    pct = count / (256 * 256) * 100
                                    temp_pct = pct
                                    pct = math.floor(pct)

                                    # save file
                                    if pct == 100:
                                        continue
                                    if temp_pct < 0.01:
                                        save_path = os.path.join(save_base, phase, 'perfect', img[:-7] + row_col + '.tiff')
                                        cv2.imwrite(save_path, new_arr)  # save a .tiff file
                                    save_path = os.path.join(save_base, phase, str(pct), img[:-7] + row_col + '.tiff')
                                    cv2.imwrite(save_path, new_arr)  # save a .tiff file
                                idx = idx + 1
                        f.close()
