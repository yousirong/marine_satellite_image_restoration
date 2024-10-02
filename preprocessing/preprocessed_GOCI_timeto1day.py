import netCDF4 as nc
import numpy as np
import os
import cv2
from tqdm import tqdm

# 데이터 경로 설정
img_path = '/media/juneyonglee/My Book1/GOCI/L2_Rrs/2021'
save_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_new'
# # 낙동강 좌표 설정 (픽셀 좌표, 중간좌표 기준)
# region1_center_x, region1_center_y = (2336 + 2592) // 2, (3053 + 3309) // 2
# # 새만금 좌표 설정 (픽셀 좌표, 중간좌표 기준)
# region2_center_x, region2_center_y = (1851 + 2107) // 2, (2639 + 2895) // 2
# 낙동강 좌표 설정 (픽셀 좌표)
region1_x_min, region1_x_max = 2336, 2592
region1_y_min, region1_y_max = 3053, 3309

# 새만금 좌표 설정 (픽셀 좌표)
region2_x_min, region2_x_max = 1851, 2107
region2_y_min, region2_y_max = 2639, 2895

# 저장 경로 없을 경우 생성
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# 위성 및 데이터 유형 설정
satellite_type = "GOCI"
data_type = 'RRS'
band_lst = [2, 3, 4]

# GOCI 데이터 경로에 있는 파일 확인
file_list = os.listdir(img_path)

# 파일명에서 연도, 월, 일 추출 및 처리
for file in tqdm(file_list):
    if "RRS" in file and file.endswith(".he5"):  # RRS 데이터를 포함하는 파일만 처리
        file_path = os.path.join(img_path, file)

        # 파일명에서 연도, 월, 일 추출 (예: 'COMS_GOCI_L2A_GA_20110401001641.RRS.he5')
        file_name = os.path.basename(file)
        year = file_name[17:21]
        month = file_name[21:23]
        day = file_name[23:25]
        # time =file_name[25:27]
        # 각 밴드에 대해 일별 평균 RRS를 저장할 배열과 파일 개수 초기화
        daily_rrs_sum_region1 = {band: np.zeros((region1_y_max - region1_y_min + 1, region1_x_max - region1_x_min + 1)) for band in band_lst}
        daily_rrs_sum_region2 = {band: np.zeros((region2_y_max - region2_y_min + 1, region2_x_max - region2_x_min + 1)) for band in band_lst}
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
                rrs_data = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Band ' + str(band) + ' RRS Image Pixel Values']
            except KeyError:
                print(f"Band {band} not found in file {file}")
                continue
            
            np_rrs = np.array(rrs_data)
            np_rrs = np.where(np_rrs == -999.0, 0, np_rrs)  # 결측치 처리
            
            # 낙동강 지역 데이터 선택
            np_rrs_region1 = np_rrs[region1_y_min:region1_y_max + 1, region1_x_min:region1_x_max + 1]
            # 새만금 지역 데이터 선택
            np_rrs_region2 = np_rrs[region2_y_min:region2_y_max + 1, region2_x_min:region2_x_max + 1]

            # 일별 합산
            daily_rrs_sum_region1[band] += np_rrs_region1
            daily_rrs_sum_region2[band] += np_rrs_region2

        f.close()

        # 파일 수만큼 나눠 평균값 계산 후 저장
        if file_count > 0:
            # 동일한 폴더에 저장할 경로 설정
            save_file_path = os.path.join(save_path, year, month, day)
            if not os.path.isdir(save_file_path):
                os.makedirs(save_file_path)

            for band in band_lst:
                # 낙동강 지역 평균 계산 및 저장
                daily_rrs_avg_region1 = daily_rrs_sum_region1[band] / file_count
                daily_rrs_avg_region1 = np.where(np.isnan(daily_rrs_avg_region1), 0, daily_rrs_avg_region1)
                daily_rrs_avg_region1 = daily_rrs_avg_region1 * 255  # 255 스케일로 변환

                # 새만금 지역 평균 계산 및 저장
                daily_rrs_avg_region2 = daily_rrs_sum_region2[band] / file_count
                daily_rrs_avg_region2 = np.where(np.isnan(daily_rrs_avg_region2), 0, daily_rrs_avg_region2)
                daily_rrs_avg_region2 = daily_rrs_avg_region2 * 255  # 255 스케일로 변환

                # 낙동강 지역 파일 저장 (같은 폴더 안에)
                save_file_region1 = os.path.join(save_file_path, f"RRS_band_{band}_nak_r{region1_x_min}_c{region1_y_min}.tiff")
                cv2.imwrite(save_file_region1, daily_rrs_avg_region1.astype(np.uint8))

                # 새만금 지역 파일 저장 (같은 폴더 안에)
                save_file_region2 = os.path.join(save_file_path, f"RRS_band_{band}_sae_r{region2_x_min}_c{region2_y_min}.tiff")
                cv2.imwrite(save_file_region2, daily_rrs_avg_region2.astype(np.uint8))
