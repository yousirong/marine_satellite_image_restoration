import numpy as np
import os
import h5py
import tifffile as tiff
from collections import defaultdict
from tqdm import tqdm
import random

# 데이터 경로 설정
data_path = '/media/juneyonglee/My Book1/GOCI/L2_Rrs/2020'
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

# 결측치 비율 계산 함수
def check_pct(img, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀만 선택 (mask == 1)
    valid_ocean_data = img[ocean_pixels]  # 육지를 제외한 해양 픽셀

    total_ocean_pixels = valid_ocean_data.size
    if total_ocean_pixels == 0:
        return 100  # All ocean pixels missing

    nan_count = np.isnan(valid_ocean_data).sum()  # NaN 값 수
    loss_pct = (nan_count / total_ocean_pixels) * 100  # NaN 비율을 퍼센트로 계산
    
    return loss_pct

# 데이터가 모두 0인지 확인하는 함수
def is_all_zeros(img, mask):
    ocean_pixels = (mask == 1)  # 해양 픽셀만 선택 (mask == 1)
    valid_ocean_data = img[ocean_pixels]
    return np.all(valid_ocean_data == 0)  # 모든 값이 0인지 확인

# 랜덤 패치 추출 함수
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

# 1-day 합성을 한 뒤 각 밴드별로 처리
for band in band_lst:  # 밴드를 하나씩 처리
    for date, files in tqdm(file_groups.items(), desc=f"Processing dates for Band {band}"):
        time_groups = defaultdict(list)
    
        for time, file in files:
            time_groups[time].append(file)

        # 512x512 크기의 패치를 합산
        daily_rrs_sum_region1 = np.zeros((512, 512))
        daily_rrs_sum_region2 = np.zeros((512, 512))

        # 1-day 합성 (시간별 파일 8개 사용)
        for time, files_in_time in tqdm(time_groups.items(), desc=f"Processing time for date {date}, band {band}", leave=False):
            for file_path in files_in_time:
                try:
                    f = h5py.File(os.path.join(data_path, file_path), 'r')  # he5 파일 열기
                except Exception as e:
                    print(f"Failed to open file: {file_path}, Error: {str(e)}")
                    continue

                try:
                    rrs_data = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Band ' + str(band) + ' RRS Image Pixel Values']
                    np_rrs = np.array(rrs_data)
                    np_rrs = np.where(np_rrs == -999.0, np.nan, np_rrs)  # 결측치 처리

                    # 512x512 크기로 패치
                    np_rrs_region1 = np_rrs[region1_center_x - 256:region1_center_x + 256, region1_center_y - 256:region1_center_y + 256]
                    np_rrs_region2 = np_rrs[region2_center_x - 256:region2_center_x + 256, region2_center_y - 256:region2_center_y + 256]

                    daily_rrs_sum_region1 += np_rrs_region1
                    daily_rrs_sum_region2 += np_rrs_region2

                except KeyError:
                    print(f"Band {band} not found in file {file_path}")
                    continue

                f.close()

        # 하루 동안의 데이터를 합산 후 평균 계산 (고정된 8개의 시간)
        daily_rrs_avg_region1 = daily_rrs_sum_region1 / 8
        daily_rrs_avg_region2 = daily_rrs_sum_region2 / 8

        # Extract the corresponding mask regions
        mask_region1 = ocean_mask[region1_center_x - 256:region1_center_x + 256, region1_center_y - 256:region1_center_y + 256]
        mask_region2 = ocean_mask[region2_center_x - 256:region2_center_x + 256, region2_center_y - 256:region2_center_y + 256]

        # Apply land-sea mask by setting land pixels to NaN
        daily_rrs_avg_region1 = np.where(mask_region1 == 1, daily_rrs_avg_region1, np.nan)
        daily_rrs_avg_region2 = np.where(mask_region2 == 1, daily_rrs_avg_region2, np.nan)

        # 결측치(NaN)를 0으로 대체 (필요한 경우)
        daily_rrs_avg_region1 = np.where(np.isnan(daily_rrs_avg_region1), 0, daily_rrs_avg_region1)
        daily_rrs_avg_region2 = np.where(np.isnan(daily_rrs_avg_region2), 0, daily_rrs_avg_region2)

        # 512x512 패치 추출 후 256x256 패치를 랜덤하게 추출
        region1_patches = extract_random_patches(daily_rrs_avg_region1, mask_region1, patch_size, 10)
        region2_patches = extract_random_patches(daily_rrs_avg_region2, mask_region2, patch_size, 10)

        # Combine both region patches
        combined_patches = region1_patches + region2_patches
        random.shuffle(combined_patches)  # Shuffle the combined patches

        # Split into 80% train and 20% test
        split_index = int(0.8 * len(combined_patches))
        train_patches = combined_patches[:split_index]
        test_patches = combined_patches[split_index:]

        # Save train patches
        for patch_num, (patch, patch_mask, row, col) in enumerate(train_patches):
            missing_rate = check_pct(patch, patch_mask)
            
            # 결측치가 100%이거나 데이터가 모두 0인 경우 저장하지 않음
            if missing_rate == 100 or is_all_zeros(patch, patch_mask):
                continue
            
            if missing_rate < 0.001:
                dest_folder = os.path.join(save_path, f'band_{band}', 'train', 'perfect')
            else:
                pct_folder = int(missing_rate // 10) * 10  
                dest_folder = os.path.join(save_path, f'band_{band}', 'train', str(pct_folder))
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)
            
            actual_row = region1_center_x - 256 + row
            actual_col = region1_center_y - 256 + col
            patch_save_file = os.path.join(dest_folder, f"RRS_band_{band}_nak_r{actual_row}_c{actual_col}.tiff")
            tiff.imwrite(patch_save_file, patch.astype(np.float32))  # float32로 저장
            print(f"Saved train patch at: {patch_save_file}")

        # Save test patches
        for patch_num, (patch, patch_mask, row, col) in enumerate(test_patches):
            missing_rate = check_pct(patch, patch_mask)
            
            # 결측치가 100%이거나 데이터가 모두 0인 경우 저장하지 않음
            if missing_rate == 100 or is_all_zeros(patch, patch_mask):
                continue
            
            if missing_rate < 0.001:
                dest_folder = os.path.join(save_path, f'band_{band}', 'test', 'perfect')
            else:
                pct_folder = int(missing_rate // 10) * 10 
                dest_folder = os.path.join(save_path, f'band_{band}', 'test', str(pct_folder))
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)
            
            actual_row = region2_center_x - 256 + row
            actual_col = region2_center_y - 256 + col
            patch_save_file = os.path.join(dest_folder, f"RRS_band_{band}_sae_r{actual_row}_c{actual_col}.tiff")
            tiff.imwrite(patch_save_file, patch.astype(np.float32))  # float32로 저장
            print(f"Saved test patch at: {patch_save_file}")

print("데이터 처리 완료.")
