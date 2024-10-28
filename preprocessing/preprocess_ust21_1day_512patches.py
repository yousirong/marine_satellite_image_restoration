import os
import re
import numpy as np
import netCDF4 as nc
from tqdm import tqdm
import tifffile as tiff
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import io

########## UST21 데이터 경로 ##########
data_base = '/media/juneyonglee/My Book/UST21/Daily'
save_base = '/media/juneyonglee/My Book/Preprocessed/UST21/chl-a_512'
mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'
land_sea_mask = io.loadmat(mask_path)['Land']  # MATLAB에서 Land 변수 가져오기
land_sea_mask = np.where(land_sea_mask == 0, 999, 1)  # 육지는 999, 해양은 1로 설정

######### UST21 패치 좌표 #########
x_nak, y_nak = (3751, 4757)  # 낙동강 영역 시작점
x_sae, y_sae = (3505, 3920)  # 새만금 영역 시작점

# 패치 저장 함수 (tifffile로 저장)
def save_patch_image(patch, file_path):
    """TIFF 파일로 512x512 크기의 패치를 저장"""
    patch_visual = patch.astype(np.uint16)
    tiff.imwrite(file_path, patch_visual)

# Color bar와 title이 있는 패치를 저장하는 함수
def save_resized_patch_with_colorbar(patch, region_name, output_path, target_size=512):
    """512x512 크기로 리사이즈된 패치를 저장하고, color bar와 title을 추가"""
    resized_patch = resize(patch, (target_size, target_size), anti_aliasing=True)
    # 이미지를 0~20의 범위로 스케일링
    scaled_patch = resized_patch * 20

    plt.imshow(scaled_patch, cmap='jet', vmin=0, vmax=20)
    plt.colorbar(label='Chlorophyll-a (mg m^-3)')
    plt.title(f'Chlorophyll-a Concentration ({region_name})')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# Color bar 없이 저장하는 함수
def save_resized_patch_no_colorbar(patch, output_path, target_size=512):
    """512x512 크기로 리사이즈된 패치를 저장 (color bar와 title 없이)"""
    resized_patch = resize(patch, (target_size, target_size), anti_aliasing=True)
    scaled_patch = resized_patch * 20

    plt.imshow(scaled_patch, cmap='jet', vmin=0, vmax=20)
    plt.axis('off')  # Remove axis for cleaner output
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# 날짜 형식을 추출하는 함수
def extract_date_from_filename(filename):
    """파일 이름에서 날짜 정보 추출"""
    date_str = filename.split('_')[-1].split('.')[0]
    date_format = '%Y%m%d'
    return datetime.strptime(date_str, date_format)

# 전체 파일을 리스트로 정리하고 날짜 순으로 정렬하는 함수
def gather_all_files(data_base):
    all_files = []
    for root, dirs, files in os.walk(data_base):
        for file in files:
            if file.endswith('.nc'):
                full_path = os.path.join(root, file)
                file_date = extract_date_from_filename(file)
                all_files.append((full_path, file_date))
    all_files.sort(key=lambda x: x[1])
    return all_files

# 각 파일에서 패치를 추출하고 저장하는 함수
def process_and_save_patches(files, land_sea_mask):
    for file_path, file_date in tqdm(files, desc="Processing UST21 files"):
        try:
            # .nc 파일 열기
            with nc.Dataset(file_path, 'r') as f:
                # Chlorophyll-a 데이터 로드
                data = f['merged_daily_Chl'][:].data
                # 결측값과 이상치 처리
                data = np.where(data == -999.0, np.nan, data)  # 결측값을 NaN으로 처리
                data[data > 20] = np.nan  # 이상치를 NaN으로 처리
                data = data * land_sea_mask  # 육지-해양 마스크 적용

                # 데이터 정규화: 0~1 사이로 스케일링
                valid_data = data[~np.isnan(data)]  # NaN이 아닌 값들만 추출
                min_val, max_val = np.nanmin(valid_data), np.nanmax(valid_data)
                data = (data - min_val) / (max_val - min_val)
                data = np.clip(data, 0, 1)  # 0~1 범위로 제한

                # 낙동강과 새만금에서 512x512 패치 추출
                n_patch_512 = data[x_nak:x_nak + 512, y_nak - 256:y_nak + 256]
                s_patch_512 = data[x_sae - 256:x_sae + 256, y_sae - 512:y_sae]

                # 년도 및 월 폴더 생성
                year_folder = os.path.join(save_base, f"{file_date.year}")
                month_folder = os.path.join(year_folder, f"{file_date.month:02d}")
                os.makedirs(month_folder, exist_ok=True)

                # 낙동강 패치 저장 (Color bar 포함 및 미포함)
                nak_colorbar_path = os.path.join(month_folder, f"nak_{file_date.strftime('%Y%m%d')}_r{x_nak}_c{y_nak - 256}_bar.png")
                nak_no_colorbar_path = os.path.join(month_folder, f"nak_{file_date.strftime('%Y%m%d')}_r{x_nak}_c{y_nak - 256}.png")
                save_resized_patch_with_colorbar(n_patch_512, "Nakdong River", nak_colorbar_path)
                save_resized_patch_no_colorbar(n_patch_512, nak_no_colorbar_path)

                # 새만금 패치 저장 (Color bar 포함 및 미포함)
                sae_colorbar_path = os.path.join(month_folder, f"sae_{file_date.strftime('%Y%m%d')}_r{x_sae - 256}_c{y_sae - 512}_bar.png")
                sae_no_colorbar_path = os.path.join(month_folder, f"sae_{file_date.strftime('%Y%m%d')}_r{x_sae - 256}_c{y_sae - 512}.png")
                save_resized_patch_with_colorbar(s_patch_512, "Saemangeum", sae_colorbar_path)
                save_resized_patch_no_colorbar(s_patch_512, sae_no_colorbar_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# 전체 파일을 수집하고 처리
all_files = gather_all_files(data_base)
process_and_save_patches(all_files, land_sea_mask)
