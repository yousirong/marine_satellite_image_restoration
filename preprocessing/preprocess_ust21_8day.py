import netCDF4 as nc
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy import io
from datetime import datetime
from tqdm import tqdm
import tifffile as tiff

########## path ##########
data_base = '/media/juneyonglee/My Book/UST21/Daily'
save_base = '/media/juneyonglee/My Book/Preprocessed/UST/Chl-a_8day'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'
land_sea_mask = io.loadmat(mask_path)
land_sea_mask = land_sea_mask['Land']  # MATLAB에서 Land 변수 가져오기

# 육지 부분을 빨간색(999)로 설정, 해양은 1로 설정
land_sea_mask = np.where(land_sea_mask == 0, 999, 1)

######### point #########
x_nak, y_nak =  (3751, 4757)  # 낙동강 영역 시작점
x_sae, y_sae = (3505, 3920)  # 새만금 영역 시작점
##########################

# 필요한 디렉토리 생성
pcts = [str(i) for i in range(0, 100, 10)]
pcts.append('perfect')

for phase in ['train', 'test']:
    for pct in pcts:
        temp = os.path.join(save_base, phase, pct)
        if not os.path.isdir(temp):
            os.makedirs(temp)

# 수정된 check_pct 함수: NaN 비율로 loss 계산, 육지(999) 제외
def check_pct(img, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀만 선택
    valid_ocean_data = img[ocean_pixels]  # 육지를 제외한 해양 픽셀

    total_ocean_pixels = valid_ocean_data.size  # 전체 해양 픽셀 수
    nan_count = np.isnan(valid_ocean_data).sum()  # NaN 값 수
    
    if total_ocean_pixels > 0:
        loss_pct = (nan_count / total_ocean_pixels) * 100  # NaN 비율을 퍼센트로 계산
    else:
        loss_pct = 100
    
    return loss_pct

# 해양 데이터 비율을 계산하는 함수
def check_ocean_pct(patch, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀 선택 (육지 제외)
    valid_ocean_pixels = np.sum((patch > 0) & (patch <= 20) & ocean_pixels)  # 유효한 해양 데이터
    total_ocean_pixels = np.sum(ocean_pixels)  # 전체 해양 픽셀 수 (육지 제외)

    ocean_data_pct = (valid_ocean_pixels / total_ocean_pixels) * 100 if total_ocean_pixels > 0 else 0
    return ocean_data_pct

# 8일 평균을 계산하는 함수
def calculate_8day_avg(files, data_dir, land_sea_mask):
    data_list = []
    for file in files:
        if file.endswith('.nc'):  # '.nc' 파일만 처리
            path = os.path.join(data_dir, file)
            f = nc.Dataset(path, 'r')
            a = f['merged_daily_Chl'][:].data
            np_a = np.array(a)
            np_a = np.where(np_a == -999.0, np.nan, np_a)  # 결측값을 NaN으로 처리
            np_a[np_a > 20] = np.nan  # 이상치를 NaN으로 처리
            np_a = np_a * land_sea_mask  # 육지-해양 마스크 적용 (육지는 999)
            data_list.append(np_a)
            f.close()
    
    # NaN을 무시하고 평균 계산
    avg_data = np.nanmean(data_list, axis=0)
    return avg_data

# 패치 저장 함수 (tifffile로 저장하도록 수정)
def save_patch_image(patch, file_path):
    # 육지(999)는 그대로 유지하며 TIFF 파일로 저장 (uint16 형식으로 저장)
    patch_visual = patch.astype(np.uint16)
    tiff.imwrite(file_path, patch_visual)

# 날짜 형식을 추출하는 함수
def extract_date_from_filename(filename):
    date_str = filename.split('_')[-1].split('.')[0]  # 파일 이름에서 날짜 부분 추출
    date_format = '%Y%m%d'  # 날짜 형식
    return datetime.strptime(date_str, date_format)

# 전체 데이터를 하나의 리스트로 정리하여 날짜 순으로 정렬하는 함수
def gather_all_files(data_base):
    all_files = []
    for root, dirs, files in os.walk(data_base):
        for file in files:
            if file.endswith('.nc'):
                full_path = os.path.join(root, file)
                file_date = extract_date_from_filename(file)
                all_files.append((full_path, file_date))
    # 날짜 순으로 정렬
    all_files.sort(key=lambda x: x[1])
    return all_files

# 전체 데이터를 처리할 수 있도록 수정
all_files = gather_all_files(data_base)

# 패치 생성 개수 설정 (예: 한 영역에서 50개의 256x256 패치 추출)
num_patches_per_region = 50
train_ratio = 0.8  # train:test 비율 8:2

# 8일씩 그룹화하여 평균 계산
for i in tqdm(range(0, len(all_files) - 7), desc="8-day moving avg"):
    selected_files = [f[0] for f in all_files[i:i + 8]]
    avg_data = calculate_8day_avg(selected_files, os.path.dirname(selected_files[0]), land_sea_mask)

    # 날짜 정보 추출
    start_date = all_files[i][1]
    end_date = all_files[i + 7][1]
    print(f"8일 이동평균 합성 완료: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

    # 낙동강과 새만금에서 지정된 512x512 영역에서 패치 추출
    n_patch_512 = avg_data[x_nak:x_nak+512, y_nak-256:y_nak+256]
    s_patch_512 = avg_data[x_sae-256:x_sae+256, y_sae-512:y_sae]

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

        # 해양 데이터 비율 확인 (육지를 제외한 전체 해양 비율 계산)
        min_ocean_pct = 0.1  # 최소 해양 데이터 비율을 설정
        n_ocean_pct = check_ocean_pct(n_patch_256, n_mask_patch)
        s_ocean_pct = check_ocean_pct(s_patch_256, s_mask_patch)

        # NaN 비율로 loss 계산
        n_pct = check_pct(n_patch_256, n_mask_patch)
        s_pct = check_pct(s_patch_256, s_mask_patch)

        # 패치 저장 (NaN 비율을 기준으로 폴더 선택)
        # 데이터가 train에 저장될지 test에 저장될지 결정
        phase = 'train' if random.random() < train_ratio else 'test'
        
        if n_ocean_pct >= min_ocean_pct:
            actual_n_row = x_nak + n_row  # 실제 좌표
            actual_n_col = y_nak - 256 + n_col  # 실제 좌표
            if n_pct == 0:
                save_path = os.path.join(save_base, phase, 'perfect', f"{start_date.strftime('%Y%m%d')}_nak_r{actual_n_row}_c{actual_n_col}.tiff")
            else:
                save_path = os.path.join(save_base, phase, str(int(n_pct // 10) * 10), f"{start_date.strftime('%Y%m%d')}_nak_r{actual_n_row}_c{actual_n_col}.tiff")
            save_patch_image(n_patch_256, save_path)

        if s_ocean_pct >= min_ocean_pct:
            actual_s_row = x_sae - 256 + s_row  # 실제 좌표
            actual_s_col = y_sae - 512 + s_col  # 실제 좌표
            if s_pct == 0:
                save_path = os.path.join(save_base, phase, 'perfect', f"{start_date.strftime('%Y%m%d')}_sae_r{actual_s_row}_c{actual_s_col}.tiff")
            else:
                save_path = os.path.join(save_base, phase, str(int(s_pct // 10) * 10), f"{start_date.strftime('%Y%m%d')}_sae_r{actual_s_row}_c{actual_s_col}.tiff")
            save_patch_image(s_patch_256, save_path)
