import numpy as np
import os
import h5py
import tifffile as tiff
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# 데이터 경로 설정
data_path = '/media/juneyonglee/GOCI_vol1/GOCI/L2_Rrs'
save_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_new'
land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI.npy'
land_sea_mask = np.load(land_sea_mask_path)

# 육지(-1)와 해양(0) 구분 마스크에서 해양을 1로, 육지를 0으로 변환
ocean_mask = np.where(land_sea_mask == 0, 1, 0)

# 낙동강 좌표 설정 (픽셀 좌표, 중간좌표 기준)
region1_center_x, region1_center_y = (2336 + 2592) // 2, (3053 + 3309) // 2
# 새만금 좌표 설정 (픽셀 좌표, 중간좌표 기준)
region2_center_x, region2_center_y = (1851 + 2107) // 2, (2639 + 2895) // 2

# 패치 설정
patch_size = 256
large_patch_size = 512

# 밴드 리스트 (2, 3, 4 밴드 처리)
band_lst = [2, 3, 4]

# 결측치 비율로 분류할 퍼센티지 폴더 생성
pcts = [str(i) for i in range(0, 100, 10)]  # 0부터 90까지
pcts.append('perfect')

# 각 밴드에 대해 train/test 폴더 생성
for band in band_lst:
    for phase in ['train', 'test']:
        for pct in pcts:
            temp = os.path.join(save_path, f'band_{band}', phase, pct)
            if not os.path.isdir(temp):
                os.makedirs(temp)

# 결측치 (NaN) 비율 계산 함수 (해양 영역에서만 NaN 값 비율을 계산)
def check_nan_pct(img, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀만 선택
    valid_ocean_data = img[ocean_pixels]  # 육지를 제외한 해양 픽셀

    total_ocean_pixels = valid_ocean_data.size  # 전체 해양 픽셀 수
    nan_count = np.isnan(valid_ocean_data).sum()  # NaN 값 수
    
    if total_ocean_pixels > 0:
        loss_pct = (nan_count / total_ocean_pixels) * 100  # NaN 비율을 퍼센트로 계산
    else:
        loss_pct = 100
    
    return loss_pct

# 육지 비율 계산 함수 (해양 영역에서만 육지(0) 비율을 계산)
def check_land_pct(mask):
    total_pixels = mask.size  # 전체 픽셀 수
    land_pixels = np.sum(mask == 0)  # 육지 픽셀 수 (mask == 0인 경우)
    
    if total_pixels > 0:
        land_pct = (land_pixels / total_pixels) * 100  # 육지 비율을 퍼센트로 계산
    else:
        land_pct = 100
    
    return land_pct

# 데이터가 모두 NaN인지 확인하는 함수
def is_all_nans(img, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀만 선택 (mask == 1)
    valid_ocean_data = img[ocean_pixels]
    return np.all(np.isnan(valid_ocean_data))  # 모든 값이 NaN인지 확인

# 랜덤 패치 추출 함수 (해양 비율이 90% 이상인 패치만 추출)
def extract_random_patches(large_patch, large_patch_mask, patch_size, num_patches):
    patches = []
    possible_coords = [(row, col) for row in range(large_patch.shape[0] - patch_size)
                                  for col in range(large_patch.shape[1] - patch_size)]
    random.shuffle(possible_coords)
    used_coords = set()  # 사용된 좌표 저장
    
    for row, col in possible_coords:
        if (row, col) in used_coords:
            continue
        
        patch = large_patch[row:row + patch_size, col:col + patch_size]
        patch_mask = large_patch_mask[row:row + patch_size, col:col + patch_size]
        
        # 육지 비율을 계산하여 육지 비율이 10% 이하인 패치만 선택
        land_pct = check_land_pct(patch_mask)
        if land_pct > 10:  # 육지가 10% 이상이면 건너뜀
            continue
        
        used_coords.update([(row + i, col + j) for i in range(patch_size) for j in range(patch_size)])
        patches.append((patch, patch_mask, row, col))
        
        if len(patches) >= num_patches:
            break
    
    return patches

# 파일들을 날짜와 시간별로 그룹화
def group_files_by_date_time(file_list):
    file_groups = defaultdict(list)
    for file in file_list:
        if "RRS" in file and file.endswith(".he5"):
            try:
                year = file[17:21]
                month = file[21:23]
                day = file[23:25]
                time = file[25:27]  # 시간 추출
                date = f"{year}-{month}-{day}"
                file_groups[date].append((time, file))
            except IndexError:
                print(f"Skipping file with unexpected format: {file}")
    return file_groups

# 메인 실행
file_list = os.listdir(data_path)
file_groups = group_files_by_date_time(file_list)


def process_file_group(file_group, band, date, region1_center_x, region1_center_y, region2_center_x, region2_center_y, ocean_mask, data_path, save_path):
    time_groups = defaultdict(list)

    for time, file in file_group:
        time_groups[time].append(file)

    daily_rrs_sum_region1 = np.zeros((512, 512))
    daily_rrs_sum_region2 = np.zeros((512, 512))

    # Add tqdm to track the progress of processing time groups (1-day synthesis)
    for time, files_in_time in tqdm(time_groups.items(), desc=f"Processing time groups for date {date}, band {band}"):
        for file_path in files_in_time:
            try:
                f = h5py.File(os.path.join(data_path, file_path), 'r')
            except Exception as e:
                print(f"Failed to open file: {file_path}, Error: {str(e)}")
                continue

            try:
                rrs_data = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Band ' + str(band) + ' RRS Image Pixel Values']
                np_rrs = np.array(rrs_data)
                np_rrs = np.where(np_rrs == -999.0, np.nan, np_rrs)

                # 해양 영역은 NaN 처리, 육지 영역은 0으로 처리
                np_rrs_region1 = np_rrs[region1_center_x - 256:region1_center_x + 256, region1_center_y - 256:region1_center_y + 256]
                np_rrs_region2 = np_rrs[region2_center_x - 256:region2_center_x + 256, region2_center_y - 256:region2_center_y + 256]

                daily_rrs_sum_region1 += np_rrs_region1
                daily_rrs_sum_region2 += np_rrs_region2

            except KeyError:
                print(f"Band {band} not found in file {file_path}")
                continue

            f.close()

    daily_rrs_avg_region1 = daily_rrs_sum_region1 / 8
    daily_rrs_avg_region2 = daily_rrs_sum_region2 / 8

    mask_region1 = ocean_mask[region1_center_x - 256:region1_center_x + 256, region1_center_y - 256:region1_center_y + 256]
    mask_region2 = ocean_mask[region2_center_x - 256:region2_center_x + 256, region2_center_y - 256:region2_center_y + 256]

    # 해양은 NaN으로, 육지는 0으로 처리
    daily_rrs_avg_region1 = np.where(mask_region1 == 1, daily_rrs_avg_region1, 0)
    daily_rrs_avg_region2 = np.where(mask_region2 == 1, daily_rrs_avg_region2, 0)

    # 패치 추출 시 육지 비율이 10% 이하인 패치만 선택
    region1_patches = extract_random_patches(daily_rrs_avg_region1, mask_region1, 256, 50)
    region2_patches = extract_random_patches(daily_rrs_avg_region2, mask_region2, 256, 50)

    combined_patches = region1_patches + region2_patches
    random.shuffle(combined_patches)

    split_index = int(0.8 * len(combined_patches))
    train_patches = combined_patches[:split_index]
    test_patches = combined_patches[split_index:]

    # 패치 저장 로직에서 결측치 (NaN) 비율에 따라 폴더 선택
    # Save train patches with date in the filename
    for patch_num, (patch, patch_mask, row, col) in enumerate(train_patches):
        nan_pct = check_nan_pct(patch, patch_mask)  # NaN 비율 계산
        if nan_pct == 100 or is_all_nans(patch, patch_mask):
            continue  # Skip if all values are NaN
        if nan_pct < 0.001:
            dest_folder = os.path.join(save_path, f'band_{band}', 'train', 'perfect')
        else:
            pct_folder = int(nan_pct // 10) * 10  # NaN 비율에 따라 폴더 선택
            dest_folder = os.path.join(save_path, f'band_{band}', 'train', str(pct_folder))
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder)

        actual_row = region1_center_x - 256 + row
        actual_col = region1_center_y - 256 + col

        # Adding the date to the filename
        patch_save_file = os.path.join(dest_folder, f"RRS_band_{band}_nak_{date}_r{actual_row}_c{actual_col}.tiff")
        tiff.imwrite(patch_save_file, patch.astype(np.float32))

        # Print the path where the patch is saved
        print(f"Saved train patch at: {patch_save_file}")

    # Save test patches with date in the filename
    for patch_num, (patch, patch_mask, row, col) in enumerate(test_patches):
        nan_pct = check_nan_pct(patch, patch_mask)  # NaN 비율 계산
        if nan_pct == 100 or is_all_nans(patch, patch_mask):
            continue  # Skip if all values are NaN
        if nan_pct < 0.001:
            dest_folder = os.path.join(save_path, f'band_{band}', 'test', 'perfect')
        else:
            pct_folder = int(nan_pct // 10) * 10  # NaN 비율에 따라 폴더 선택
            dest_folder = os.path.join(save_path, f'band_{band}', 'test', str(pct_folder))
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder)

        actual_row = region2_center_x - 256 + row
        actual_col = region2_center_y - 256 + col

        # Adding the date to the filename
        patch_save_file = os.path.join(dest_folder, f"RRS_band_{band}_sae_{date}_r{actual_row}_c{actual_col}.tiff")
        tiff.imwrite(patch_save_file, patch.astype(np.float32))

        # Print the path where the patch is saved
        print(f"Saved test patch at: {patch_save_file}")

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        futures = []
        
        # Add tqdm to track the progress of band processing
        for band in tqdm(band_lst, desc="Processing bands"):
            for date, file_group in file_groups.items():
                futures.append(executor.submit(process_file_group, file_group, band, date, region1_center_x, region1_center_y, region2_center_x, region2_center_y, ocean_mask, data_path, save_path))

        for future in as_completed(futures):
            print(future.result())

    print("Preprocessing completed.")
