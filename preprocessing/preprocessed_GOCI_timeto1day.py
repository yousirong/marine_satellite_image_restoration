# 234 밴드만 사용 
import netCDF4 as nc
import numpy as np
import os
import cv2
from tqdm import tqdm

# 데이터 경로 설정
img_path = '/media/juneyonglee/GOCI_vol1/GOCI/L2_Rrs'
save_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs'

# 낙동강, 새만금 좌표 설정
lat_min = 35.1  # 위도
lat_max = 36.2
long_min = 129.1  # 경도
long_max = 130.25

# 동쪽 범위 설정
x_min = 2702
x_max = 2958
y_min = 2397
y_max = 2653

# 저장 경로 없을 경우 생성
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# 위성 및 데이터 유형 설정
satellite_type = "GOCI"
data_type = 'RRS'
band_lst = [1,2,3,4,5,6,7,8]

# GOCI 데이터 경로에 있는 파일 확인
file_list = os.listdir(img_path)

# 파일명에서 연도, 월, 일 추출 및 처리
for file in tqdm(file_list):
    if "RRS" in file and file.endswith(".he5"):  # RRS 데이터를 포함하는 파일만 처리
        file_path = os.path.join(img_path, file)

        # 파일명에서 연도, 월, 일 추출 (예: 'COMS_GOCI_L2A_GA_20110401001641.RRS.he5')
        #                                  0123456789012345678901234567
        file_name = os.path.basename(file)
        year = file_name[19:23]
        month = file_name[23:25]
        day = file_name[25:27]

        # 각 밴드에 대해 일별 평균 RRS를 저장할 배열과 파일 개수 초기화
        daily_rrs_sum = {band: np.zeros((y_max - y_min + 1, x_max - x_min + 1)) for band in band_lst}
        file_count = 0

        try:
            f = nc.Dataset(file_path, 'r')  # netCDF 파일 열기
        except:
            print(f"Failed to open file: {file_path}")
            continue

        file_count += 1

        # 각 밴드에 대해 데이터 읽기 및 합산
        for band in band_lst:
            try:
                # netCDF 형식으로 데이터를 읽고 처리
                rrs_data = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Band '+str(band)+ ' RRS Image Pixel Values']
            except KeyError:
                print(f"Band {band} not found in file {file}")
                continue
            
            np_rrs = np.array(rrs_data)
            np_rrs = np.where(np_rrs == -999.0, 0, np_rrs)  # 결측치 처리
            np_rrs = np_rrs[y_min:y_max + 1, x_min:x_max + 1]  # 지정된 좌표 범위 내 데이터만 선택

            # 일별 합산
            daily_rrs_sum[band] += np_rrs

        f.close()

        # 파일 수만큼 나눠 평균값 계산 후 저장
        if file_count > 0:
            for band in band_lst:
                daily_rrs_avg = daily_rrs_sum[band] / file_count  # 평균 계산
                daily_rrs_avg = np.where(np.isnan(daily_rrs_avg), 0, daily_rrs_avg)  # NaN 처리
                daily_rrs_avg = daily_rrs_avg * 255  # 모델 입력을 위해 255 스케일로 변환

                # 일별 TIFF 파일로 저장
                save_file_path = os.path.join(save_path, year, month, day)
                if not os.path.isdir(save_file_path):
                    os.makedirs(save_file_path)

                save_file = os.path.join(save_file_path, f"RRS_band_{band}.tiff")
                cv2.imwrite(save_file, daily_rrs_avg)
