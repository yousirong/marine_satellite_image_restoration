import os
import glob
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

# 육지 마스크 파일 경로 (적절히 수정)
land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy'

def scale_data(data, target_min=0.01, target_max=10):
    """
    NaN을 제외한 data의 최소/최대값을 이용해 선형 스케일링을 수행합니다.
    값이 하나로 고정된 경우 target_min 값을 반환합니다.
    """
    valid = ~np.isnan(data)
    if np.sum(valid) == 0:
        return data.copy()
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if dmax - dmin == 0:
        return np.full_like(data, target_min)
    return target_min + (data - dmin) * (target_max - target_min) / (dmax - dmin)

def create_composite_1row4(gt_file, mask_file, recon_file, save_path, land_sea_mask_path):
    """
    gt, mask, recon CSV 파일을 읽어들여,
      - GT: 원본 ground truth (255는 NaN 처리 후 선형 스케일링)
      - Mask: mask 파일 (255는 NaN 처리)
      - Masked: GT에서 mask가 1인 부분만 표시하고 나머지는 NaN 처리한 이미지
      - Recon: 복원(recon) 결과 (255는 NaN 처리 후 선형 스케일링)
    에 대해, 육지 영역은 검은색으로 표시하고,
    1행 4열 subplot (순서: GT, Mask, Masked, Recon)으로 하나의 이미지 파일로 저장합니다.

    단, 파일명 내에 'r{row}_c{col}' 패턴으로 crop 좌표를 추출하여
    land mask 파일에서 해당 영역(256×256)을 사용합니다.
    """
    # CSV 파일 읽기
    gt_data = np.loadtxt(gt_file, delimiter=',', dtype='float32')
    mask_data = np.loadtxt(mask_file, delimiter=',', dtype='float32')
    recon_data = np.loadtxt(recon_file, delimiter=',', dtype='float32')

    # 255 값을 NaN 처리 (예: 육지 또는 결측)
    gt_data = np.where(gt_data == 255, np.nan, gt_data)
    mask_data = np.where(mask_data == 255, np.nan, mask_data)
    recon_data = np.where(recon_data == 255, np.nan, recon_data)

    # Masked: mask가 1인 부분만 GT 값을 사용, 그렇지 않으면 NaN 처리
    masked_data = np.where(mask_data == 1, gt_data, np.nan)

    # 각각 선형 스케일링 (0.01 ~ 10)
    gt_scaled     = scale_data(gt_data)
    masked_scaled = scale_data(masked_data)
    recon_scaled  = scale_data(recon_data)

    # 파일명에서 crop 좌표 (row, col) 추출 (예: ..._r256_c128.csv)
    basename = os.path.basename(gt_file)
    match = re.search(r'r(\d+)_c(\d+)', basename)
    if match:
        row, col = int(match.group(1)), int(match.group(2))
    else:
        row, col = 0, 0

    # 육지 마스크 로드 및 crop (256x256)
    land_mask_full = np.load(land_sea_mask_path)
    land_mask_cropped = land_mask_full[row:row+256, col:col+256]
    # 육지 영역은 값이 1인 부분으로 가정 (True이면 육지)
    land_bool = (land_mask_cropped == 1)

    # GT, Masked, Recon에 대해 육지 영역을 마스킹 처리 (masked array로 변경)
    # imshow 시 colormap의 bad 값(color for NaN)을 검정색으로 지정합니다.
    gt_masked    = np.ma.array(gt_scaled, mask=land_bool)
    masked_mask  = np.ma.array(masked_scaled, mask=land_bool)
    recon_masked = np.ma.array(recon_scaled, mask=land_bool)

    # figure 생성 (1행 4열)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 공통 colormap (jet) 및 정규화: 0.01 ~ 10 범위 고정
    cmap = plt.get_cmap('jet')
    cmap.set_bad(color='black')  # NaN (육지) 영역은 검정색으로 표시
    norm = plt.Normalize(vmin=0.01, vmax=10)

    # GT 이미지 (컬러맵 적용, 육지 부분은 검정)
    axes[0].imshow(gt_masked, cmap=cmap, norm=norm)
    axes[0].set_title('GT', fontsize=16)
    axes[0].axis('off')

    # Mask 이미지 (gray colormap; mask 데이터는 그대로 표시)
    axes[1].imshow(mask_data, cmap='gray')
    axes[1].set_title('Mask', fontsize=16)
    axes[1].axis('off')

    # Masked 이미지 (컬러맵 적용, 육지 부분은 검정)
    axes[2].imshow(masked_mask, cmap=cmap, norm=norm)
    axes[2].set_title('Masked', fontsize=16)
    axes[2].axis('off')

    # Recon 이미지 (컬러맵 적용, 육지 부분은 검정)
    axes[3].imshow(recon_masked, cmap=cmap, norm=norm)
    axes[3].set_title('Recon', fontsize=16)
    axes[3].axis('off')

    # 오른쪽에 공통 컬러바 추가 (GT, Masked, Recon과 동일 scale)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label='Value')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def process_random_samples(gt_dir, mask_dir, recon_dir, num_samples, output_dir, land_sea_mask_path):
    """
    각 폴더 내의 CSV 파일들을 정렬한 후, 전체 파일 중 num_samples 개만 랜덤 선택하여
    각 샘플에 대해 create_composite_1row4 함수를 멀티프로세싱으로 실행합니다.
    """
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.csv')))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.csv')))
    recon_files = sorted(glob.glob(os.path.join(recon_dir, '*.csv')))

    if not (len(gt_files) == len(mask_files) == len(recon_files)):
        print("GT, Mask, Recon 파일 수가 일치하지 않습니다.")
        return

    if len(gt_files) < num_samples:
        print("요청한 랜덤 샘플 수보다 파일 수가 적습니다.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample_indices = random.sample(range(len(gt_files)), num_samples)
    tasks = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for idx in sample_indices:
            gt_file = gt_files[idx]
            mask_file = mask_files[idx]
            recon_file = recon_files[idx]
            base_name = os.path.splitext(os.path.basename(gt_file))[0]
            output_path = os.path.join(output_dir, f'composite_{base_name}.png')
            print(f"Processing sample: {base_name}")
            tasks.append(executor.submit(create_composite_1row4,
                                         gt_file, mask_file, recon_file,
                                         output_path, land_sea_mask_path))
        for future in concurrent.futures.as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print("Error processing a sample:", e)

# 예제 사용
if __name__ == '__main__':
    # GT, Mask, Recon CSV 파일들이 있는 폴더 경로 (원하는 경로로 변경)
    gt_directory = '/media/juneyonglee/My Book/results/50/degree/gt'
    mask_directory = '/media/juneyonglee/My Book/results/50/degree/mask'
    recon_directory = '/media/juneyonglee/My Book/results/50/degree/recon'

    # 결과 composite 이미지들을 저장할 폴더 경로 설정
    output_directory = '/home/juneyonglee/Documents/visual50'

    # 랜덤으로 처리할 샘플 수 지정
    num_samples_to_process = 5

    process_random_samples(gt_directory, mask_directory, recon_directory,
                           num_samples_to_process, output_directory, land_sea_mask_path)
