import os
import glob
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')    # <-- 반드시 pyplot import 전에 호출
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from sklearn.metrics import r2_score as r2_
from matplotlib import cm
from matplotlib.colors import Normalize
import re
import random

# Pillow (16비트 TIFF 저장용)
from PIL import Image

# ust21
land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy'
# goci
# land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_parity(filename, loss_rate, true, pred, rmse_, mae_,
                kind="scatter",  # scatter로 기본 설정 변경
                xlabel="true (mg/m$^3$)", ylabel="predict (mg/m$^3$)",
                title="Loss 50-60%",
                hist2d_kws=None, scatter_kws=None, kde_kws=None,
                equal=True, metrics=True, metrics_position="lower right",
                figsize=(8, 8), ax=None, save_file=True):

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Data range, constrained between 0.01 and 10
    val_min = 0.01
    val_max = 10

    # Data plot
    if "scatter" in kind:
        if not scatter_kws:
            scatter_kws = {'s': 1, 'alpha': 0.1}
        ax.scatter(true, pred, **scatter_kws)
    elif "hist2d" in kind:
        if not hist2d_kws:
            hist2d_kws = {'bins': 300, 'cmap': 'Greens', 'vmin': 1}
        ax.hist2d(true, pred, **hist2d_kws)
    elif "kde" in kind:
        if not kde_kws:
            kde_kws = {'cmap': 'viridis', 'levels': 5}
        sns.kdeplot(x=true, y=pred, **kde_kws, ax=ax)

    # x, y bounds
    ax.set_xlim([val_min, val_max])
    ax.set_ylim([val_min, val_max])

    ticks = np.arange(0, 11, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=15)

    # Grid
    ax.grid(True)

    # Diagonal reference line
    ax.plot([val_min, val_max], [val_min, val_max], c="k", alpha=0.3)

    # x, y labels
    font_label = {"color": "gray", "fontsize": 20}
    ax.set_xlabel(xlabel, fontdict=font_label, labelpad=8)
    ax.set_ylabel(ylabel, fontdict=font_label, labelpad=8)

    # Title
    font_title = {"color": "gray", "fontsize": 20, "fontweight": "bold"}
    ax.set_title(title, fontdict=font_title, pad=16)

    # Metrics
    if metrics:
        r2 = r2_(true, pred)
        font_metrics = {'color': 'k', 'fontsize': 14}

        if metrics_position == "lower right":
            text_pos_x = 0.98
            text_pos_y = 0.3
            ha = "right"
        elif metrics_position == "upper left":
            text_pos_x = 0.1
            text_pos_y = 0.9
            ha = "left"
        else:
            text_pos_x, text_pos_y = 0.1, 0.9
            ha = "left"

        ax.text(text_pos_x, text_pos_y, f"RMSE = {rmse_:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.1, f"MAE = {mae_:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.2, f"R2 = {r2:.3f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)

    # Save to file
    fig = ax.figure
    fig.tight_layout()
    if save_file:
        os.makedirs(filename, exist_ok=True)
        fig.savefig(os.path.join(filename, f'{loss_rate}.png'))
    else:
        print("Check save file path, saving failed.")
    # plt.show()
    return ax

def convert_raw_to_color(data, vmin=0.01, vmax=10, cmap_name='jet'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap_name)
    colored_img = colormap(norm(data))[:, :, :3]
    return colored_img

def save_16bit_grayscale_tiff(data, save_path):
    data_clipped = np.clip(data, 0, 10)
    data_16 = (data_clipped / 10.0 * 65535.0).astype(np.uint16)
    im = Image.fromarray(data_16, mode='I;16')
    if not (save_path.lower().endswith('.tif') or save_path.lower().endswith('.tiff')):
        save_path += '.tif'
    im.save(save_path, format='TIFF')

def save_colormap_image_with_land_mask(data, land_sea_mask_path, row, col,
                                       save_path, global_min, global_max,
                                       land_color=[0, 0, 0],
                                       recon_file_name=None):

    date_str = None
    if recon_file_name:
        match = re.search(r'(\d{8})', recon_file_name)
        if match:
            date_str = match.group(1)

    land_mask_full = np.load(land_sea_mask_path)
    land_mask_cropped = land_mask_full[row:row + 256, col:col + 256]

    data_clipped = data.copy()
    data_clipped = np.where(data_clipped == 255, np.nan, data_clipped)

    # 육지 마스크 생성 (999 = 육지)
    land_mask = (land_mask_cropped == 999)
    sea_mask = (land_mask_cropped == 1)

    # 해양 영역만 있는지 확인
    if not np.any(sea_mask):
        print(f"Warning: No ocean pixels found in {recon_file_name}, skipping colormap generation")
        return

    # 해양 영역의 데이터만 사용하여 스케일링
    sea_data = data_clipped[sea_mask]
    if len(sea_data) == 0 or np.all(np.isnan(sea_data)):
        print(f"Warning: No valid ocean data in {recon_file_name}, skipping colormap generation")
        return

    # 해양 영역의 유효한 데이터 범위 확인
    valid_sea_data = sea_data[~np.isnan(sea_data)]
    if len(valid_sea_data) == 0:
        print(f"Warning: No valid ocean data after NaN removal in {recon_file_name}")
        return

    # 실제 해양 데이터 범위를 고려한 스케일링
    sea_min = np.percentile(valid_sea_data, 1)   # 1% percentile
    sea_max = np.percentile(valid_sea_data, 99)  # 99% percentile

    # 범위가 너무 작으면 전역 범위 사용
    if (sea_max - sea_min) < 0.1:
        sea_min = global_min
        sea_max = global_max

    # 스케일링 (0.01 ~ 10 범위로)
    scaled_data = np.full_like(data_clipped, np.nan)
    scaled_data[sea_mask] = 0.01 + (data_clipped[sea_mask] - sea_min) * (10 - 0.01) / (sea_max - sea_min)

    # 육지 영역은 NaN으로 설정하여 컬러맵에서 제외
    scaled_data[land_mask] = np.nan

    # 컬러맵 적용 (NaN은 자동으로 마스킹됨)
    colored_img = convert_raw_to_color(scaled_data, vmin=0.01, vmax=10, cmap_name='jet')

    # 육지 영역을 명시적으로 검은색으로 설정
    colored_img[land_mask] = land_color

    if not save_path.lower().endswith('.png'):
        save_path_with_extension = save_path.replace('.csv', '')
        if date_str:
            save_path_with_extension += f'_{date_str}.png'
        else:
            save_path_with_extension += '.png'
    else:
        save_path_with_extension = save_path

    # (A) 컬러 PNG(8비트) 저장
    plt.imsave(save_path_with_extension, colored_img)

    # (B) 16비트 TIFF 저장 예시 (주석 해제 시 사용 가능)
    # save_16bit_grayscale_tiff(scaled_data, save_path_with_extension.replace('.png','_16bit.tif'))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(colored_img)
    if recon_file_name:
        ax.set_title(f"Restored Chl-a\n{recon_file_name}\n(r{row}_c{col})\nOcean pixels only", fontsize=16)
    else:
        ax.set_title("Restored Chlorophyll-a Concentration (Ocean Only)", fontsize=16)
    ax.axis('off')
    sm = cm.ScalarMappable(norm=Normalize(vmin=0.01, vmax=10), cmap='jet')
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Chlorophyll-a concentration (mg/m³)',
                 ticks=np.linspace(0.01, 10, num=5))
    fig.tight_layout()
    fig.savefig(save_path_with_extension.replace('.png', '_bar.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def calculate_global_min_max(recon_files, gt_files, sample_count=10000):
    """
    전체 데이터셋에서 global_min과 global_max를 계산
    """
    print(f"\n=== Calculating Global Min/Max from {len(recon_files)} files ===")

    # 샘플링으로 계산 속도 향상
    if sample_count < len(recon_files):
        random.seed(42)
        sample_indices = random.sample(range(len(recon_files)), sample_count)
        sampled_recon = [recon_files[i] for i in sample_indices]
        sampled_gt = [gt_files[i] for i in sample_indices]
        print(f"Using random sample of {sample_count} files for min/max calculation")
    else:
        sampled_recon = recon_files
        sampled_gt = gt_files
        print(f"Using all {len(recon_files)} files for min/max calculation")

    all_values = []

    for i, (recon_f, gt_f) in enumerate(tqdm(zip(sampled_recon, sampled_gt),
                                              desc="Calculating global min/max",
                                              total=len(sampled_recon))):
        try:
            arr_rec = np.loadtxt(recon_f, delimiter=',', dtype='float32')
            arr_gt = np.loadtxt(gt_f, delimiter=',', dtype='float32')

            # 255 값을 nan으로 변환
            arr_rec = np.where(arr_rec == 255, np.nan, arr_rec)
            arr_gt = np.where(arr_gt == 255, np.nan, arr_gt)

            # 유효한 값들만 수집
            valid_rec = arr_rec[~np.isnan(arr_rec)]
            valid_gt = arr_gt[~np.isnan(arr_gt)]

            if len(valid_rec) > 0:
                all_values.extend(valid_rec.tolist())
            if len(valid_gt) > 0:
                all_values.extend(valid_gt.tolist())

        except Exception as e:
            print(f"Error reading {os.path.basename(recon_f)}: {e}")
            continue

    if not all_values:
        print("⚠️ No valid values found, using default values")
        return -0.6930203437805176, 11.1470947265625

    global_min = float(np.min(all_values))
    global_max = float(np.max(all_values))

    print(f"✅ Global Min: {global_min:.6f}")
    print(f"✅ Global Max: {global_max:.6f}")
    print(f"✅ Range: {global_max - global_min:.6f}")
    print(f"✅ Calculated from {len(all_values):,} valid pixels")

    return global_min, global_max

def calculate_file_performance(recon_f, gt_f, mask_f, global_min, global_max, land_sea_mask_path):
    """
    개별 파일의 성능 지표 계산 (RMSE, MAE, valid pixel count)
    육지/해양 마스크를 적용하여 해양 영역만 처리
    """
    try:
        arr_rec = np.loadtxt(recon_f, delimiter=',', dtype='float32')
        arr_gt  = np.loadtxt(gt_f,    delimiter=',', dtype='float32')
        arr_m   = np.loadtxt(mask_f,  delimiter=',', dtype='float32')

        # 파일명에서 row, col 추출
        name = os.path.basename(recon_f)
        row, col = None, None
        patterns = [
            r'y(\d+)_x(\d+)',
            r'r(\d+)_c(\d+)',
            r'img_\d+_y(\d+)_x(\d+)',
            r'(\d+)_(\d+)',
        ]
        for pattern in patterns:
            m = re.search(pattern, name)
            if m:
                row, col = int(m.group(1)), int(m.group(2))
                break

        if row is None or col is None:
            print(f"Could not extract row/col from filename: {name}")
            return float('inf'), float('inf'), 0

        # 육지/해양 마스크 로드 및 적용
        try:
            land_sea_mask_full = np.load(land_sea_mask_path)
            land_sea_mask_crop = land_sea_mask_full[row:row + 256, col:col + 256]

            # 해양 영역 마스크 (1 = 해양, 999 = 육지)
            sea_mask = (land_sea_mask_crop == 1)

            # 해양 영역이 충분하지 않으면 건너뛰기
            sea_pixel_count = np.sum(sea_mask)
            total_pixel_count = sea_mask.size
            sea_ratio = sea_pixel_count / total_pixel_count

            if sea_pixel_count == 0 or sea_ratio < 0.1:  # 해양 영역이 10% 미만
                return float('inf'), float('inf'), 0

        except Exception as e:
            print(f"Error loading land-sea mask for {name}: {e}")
            return float('inf'), float('inf'), 0

        # nan 처리
        for arr in (arr_rec, arr_gt, arr_m):
            np.place(arr, arr == 255, np.nan)

        # 스케일링
        arr_rec = 0.01 + (arr_rec - global_min)*(10-0.01)/(global_max-global_min)
        arr_gt  = 0.01 + (arr_gt  - global_min)*(10-0.01)/(global_max-global_min)

        # 유효한 픽셀 마스크 (해양 영역만 + 기존 조건들)
        valid = sea_mask & (~np.isnan(arr_m)) & (~np.isnan(arr_gt)) & (~np.isnan(arr_rec)) & (arr_gt != 0)

        if not np.any(valid):
            return float('inf'), float('inf'), 0  # 유효한 해양 데이터 없음

        diff = arr_gt[valid] - arr_rec[valid]
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        valid_count = np.sum(valid)

        return rmse, mae, valid_count

    except Exception as e:
        print(f"Error calculating performance for {os.path.basename(recon_f)}: {e}")
        return float('inf'), float('inf'), 0

def validate(loss_rate, data_path, save_path, land_sea_mask_path):
    """
    상위 80% 성능 샘플들을 선택하여 검증하는 함수
    """
    import os, glob, numpy as np, math, re, random
    from tqdm import tqdm

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    def find_all_date_folders(base_path):
        date_folders = []
        date_pattern = re.compile(r'^\d{8}$')  # YYYYMMDD 패턴

        print(f"Looking for date folders in: {base_path}")

        if not os.path.exists(base_path):
            print(f"Base path does not exist: {base_path}")
            return date_folders

        if not os.path.isdir(base_path):
            print(f"Base path is not a directory: {base_path}")
            return date_folders

        try:
            items = os.listdir(base_path)
            print(f"Items found in {base_path}: {items}")

            for item in items:
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and date_pattern.match(item):
                    print(f"Found date folder: {item} -> {item_path}")
                    date_folders.append(item_path)
                elif os.path.isdir(item_path):
                    print(f"Found non-date folder: {item}")
                else:
                    print(f"Found file: {item}")

        except PermissionError:
            print(f"Permission denied accessing: {base_path}")
        except Exception as e:
            print(f"Error accessing {base_path}: {e}")

        return sorted(date_folders)

    def find_csvs_in_folder(folder_path):
        if os.path.basename(folder_path) == "degree":
            degree_path = folder_path
        else:
            degree_path = os.path.join(folder_path, "degree")

        pattern_recon = os.path.join(degree_path, 'recon', '*.csv')
        pattern_gt    = os.path.join(degree_path, 'gt', '*.csv')
        pattern_mask  = os.path.join(degree_path, 'mask', '*.csv')

        recon_files = sorted(glob.glob(pattern_recon), key=natural_sort_key)
        gt_files    = sorted(glob.glob(pattern_gt), key=natural_sort_key)
        mask_files  = sorted(glob.glob(pattern_mask), key=natural_sort_key)

        return recon_files, gt_files, mask_files

    def extract_date_from_path(file_path):
        date_pattern = re.compile(r'(\d{8})')
        matches = date_pattern.findall(file_path)
        return matches[-1] if matches else None

    def calculate_mask_loss_rate(mask_files, sample_count=1000, land_sea_mask_path=None):
        if not mask_files or land_sea_mask_path is None:
            print("⚠️ mask 파일 또는 해양 마스크 경로 누락")
            return 50

        try:
            land_sea_mask_full = np.load(land_sea_mask_path)
        except Exception as e:
            print(f"❌ 해양 마스크 로딩 실패: {e}")
            return 50

        sample_size = min(sample_count, len(mask_files))
        random.seed(42)
        sampled_files = random.sample(mask_files, sample_size)

        total_sea_pixels = 0
        sea_hole_pixels = 0
        processed_files = 0
        skipped_land_only = 0
        skipped_errors = 0

        print(f"Calculating loss rate using land-sea mask for {sample_size} mask files...")

        for mask_file in sampled_files:
            try:
                mask_data = np.loadtxt(mask_file, delimiter=',', dtype='float32')

                name = os.path.basename(mask_file)
                row, col = None, None
                patterns = [
                    r'y(\d+)_x(\d+)',
                    r'r(\d+)_c(\d+)',
                    r'img_\d+_y(\d+)_x(\d+)',
                    r'(\d+)_(\d+)',
                ]
                for pattern in patterns:
                    m = re.search(pattern, name)
                    if m:
                        row, col = int(m.group(1)), int(m.group(2))
                        break

                if row is None or col is None:
                    print(f"⚠️ row/col 추출 실패: {name}")
                    skipped_errors += 1
                    continue

                sea_mask_crop = land_sea_mask_full[row:row + 256, col:col + 256]
                if sea_mask_crop.shape != mask_data.shape:
                    print(f"⚠️ Shape mismatch: {name}")
                    skipped_errors += 1
                    continue

                # 해양 영역과 육지 영역 구분
                sea_area = (sea_mask_crop == 1)  # 해양
                land_area = (sea_mask_crop == 999)  # 육지

                # 해양 영역이 없거나 매우 적은 경우 건너뛰기
                sea_pixel_count = np.sum(sea_area)
                total_pixel_count = sea_area.size
                sea_ratio = sea_pixel_count / total_pixel_count

                if sea_pixel_count == 0:
                    # 완전히 육지만 있는 경우
                    skipped_land_only += 1
                    continue
                elif sea_ratio < 0.1:  # 해양 영역이 10% 미만인 경우도 건너뛰기
                    skipped_land_only += 1
                    continue

                # 해양 영역에서만 마스크 비율 계산
                # mask_data에서 1 = hole(손실), 0 = valid pixel
                holes_in_sea = np.sum((mask_data == 1) & sea_area)

                total_sea_pixels += sea_pixel_count
                sea_hole_pixels += holes_in_sea
                processed_files += 1

            except Exception as e:
                print(f"❌ Error reading mask file {mask_file}: {e}")
                skipped_errors += 1
                continue

        print(f"Mask loss rate calculation summary:")
        print(f"  - Total sampled files: {sample_size}")
        print(f"  - Successfully processed: {processed_files}")
        print(f"  - Skipped (land-only/low sea ratio): {skipped_land_only}")
        print(f"  - Skipped (errors): {skipped_errors}")

        if total_sea_pixels > 0 and processed_files > 0:
            loss_rate = (sea_hole_pixels / total_sea_pixels) * 100
            print(f"✅ Calculated loss rate: {loss_rate:.1f}%")
            print(f"   - Sea holes: {sea_hole_pixels:,}")
            print(f"   - Total sea pixels: {total_sea_pixels:,}")
            print(f"   - Based on {processed_files} files with sufficient ocean coverage")
            return round(loss_rate)
        else:
            print("⚠️ 유효한 해양 픽셀이 없습니다. 기본값 50 사용")
            return 50

    def check_pixel_diversity(data, min_range=0.5, min_std=0.1):
        """
        픽셀값의 다양성을 확인하는 함수
        min_range: 최소 범위 (max - min)
        min_std: 최소 표준편차
        """
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < 10:  # 유효한 픽셀이 너무 적으면 False
            return False

        data_range = np.max(valid_data) - np.min(valid_data)
        data_std = np.std(valid_data)

        return (data_range >= min_range) and (data_std >= min_std)

    # Debug: Print the incoming data_path
    print(f"=== DEBUG: validate() called with data_path: {data_path} ===")

    # Check if data_path accidentally has /degree appended
    if data_path.endswith('/degree'):
        print(f"WARNING: data_path ends with '/degree'. Removing it...")
        data_path = data_path[:-7]
        print(f"Corrected data_path: {data_path}")

    # 모든 날짜 폴더 찾기
    print(f"Searching for date folders under: {data_path}")
    date_folders = find_all_date_folders(data_path)

    if not date_folders:
        print(f"No date folders (YYYYMMDD format) found under {data_path}")
        return

    print(f"Found {len(date_folders)} date folders:")
    for folder in date_folders:
        print(f"  - {folder}")

    # 모든 날짜 폴더에서 CSV 파일 수집
    all_recon_files = []
    all_gt_files = []
    all_mask_files = []
    date_file_mapping = {}

    for date_folder in date_folders:
        date_name = os.path.basename(date_folder)
        print(f"\nProcessing date folder: {date_folder}")

        degree_path = os.path.join(date_folder, 'degree')
        if not os.path.exists(degree_path):
            print(f"  No 'degree' folder found in {date_folder}")
            continue

        print(f"  Found degree folder: {degree_path}")

        try:
            degree_contents = os.listdir(degree_path)
            print(f"  Contents of degree folder: {degree_contents}")
        except Exception as e:
            print(f"  Error reading degree folder: {e}")
            continue

        recon_files, gt_files, mask_files = find_csvs_in_folder(date_folder)

        print(f"  CSV files found:")
        print(f"    - Recon: {len(recon_files)} files")
        print(f"    - GT: {len(gt_files)} files")
        print(f"    - Mask: {len(mask_files)} files")

        if recon_files and gt_files and mask_files:
            print(f"  ✅ Date {date_name}: Successfully found CSV files")

            all_recon_files.extend(recon_files)
            all_gt_files.extend(gt_files)
            all_mask_files.extend(mask_files)

            date_file_mapping[date_name] = {
                'recon': recon_files,
                'gt': gt_files,
                'mask': mask_files
            }
        else:
            print(f"  ❌ Date {date_name}: Incomplete CSV files found")

    if not (all_recon_files and all_gt_files and all_mask_files):
        print(f"No valid recon/gt/mask CSVs found in any date folders under {data_path}")
        return

    print(f"\nTotal files across all dates:")
    print(f"  - GT files: {len(all_gt_files)}")
    print(f"  - Recon files: {len(all_recon_files)}")
    print(f"  - Mask files: {len(all_mask_files)}")

    # 파일 개수 일치 확인
    if not (len(all_recon_files) == len(all_gt_files) == len(all_mask_files)):
        print(f"Warning: File count mismatch - recon:{len(all_recon_files)}, gt:{len(all_gt_files)}, mask:{len(all_mask_files)}")

    # 마스크 파일들로부터 실제 loss rate 계산
    print(f"\n=== Calculating Actual Loss Rate ===")
    calculated_loss_rate = calculate_mask_loss_rate(all_mask_files, sample_count=100, land_sea_mask_path=land_sea_mask_path)
    print(f"Using calculated loss rate: {calculated_loss_rate}%")

    loss_rate = calculated_loss_rate

    # ===== Global Min/Max 동적 계산 =====
    global_min, global_max = calculate_global_min_max(all_recon_files, all_gt_files, sample_count=1000)

    # ===== 성능 기반 샘플 선택 =====
    print(f"\n=== Calculating Performance for All Files ===")

    # 1단계: 모든 파일의 성능 계산
    file_performances = []

    for i in tqdm(range(len(all_recon_files)), desc="Calculating performance", unit="file"):
        recon_f = all_recon_files[i]
        gt_f    = all_gt_files[i]
        mask_f  = all_mask_files[i]

        rmse, mae, valid_count = calculate_file_performance(recon_f, gt_f, mask_f, global_min, global_max, land_sea_mask_path)

        file_performances.append({
            'index': i,
            'recon_file': recon_f,
            'gt_file': gt_f,
            'mask_file': mask_f,
            'rmse': rmse,
            'mae': mae,
            'valid_count': valid_count,
            'filename': os.path.basename(recon_f)
        })

    # 2단계: 유효한 파일들만 필터링 (RMSE가 무한대가 아닌 것들)
    valid_performances = [p for p in file_performances if p['rmse'] != float('inf')]

    print(f"Valid files for performance calculation: {len(valid_performances)}/{len(file_performances)}")

    if not valid_performances:
        print("No valid files found for performance calculation!")
        return

    # 3단계: RMSE 기준으로 정렬 (오름차순 - 낮은 RMSE가 더 좋음)
    valid_performances.sort(key=lambda x: x['rmse'])

    # 4단계: 상위 80% 선택 (변수명 수정)
    top_80_percent_count = int(len(valid_performances) * 0.8)
    if top_80_percent_count < 1:
        top_80_percent_count = 1

    selected_performances = valid_performances[:top_80_percent_count]

    print(f"Selected top 80% ({top_80_percent_count}/{len(valid_performances)}) best performing files")
    print(f"Performance ratio: {top_80_percent_count/len(valid_performances)*100:.1f}%")

    # 성능 분포 출력
    print(f"\n=== Performance Distribution of Selected Files (Top 80%) ===")
    rmse_values = [p['rmse'] for p in selected_performances]
    mae_values = [p['mae'] for p in selected_performances]

    print(f"RMSE - Min: {min(rmse_values):.6f}, Max: {max(rmse_values):.6f}, Mean: {np.mean(rmse_values):.6f}")
    print(f"MAE  - Min: {min(mae_values):.6f}, Max: {max(mae_values):.6f}, Mean: {np.mean(mae_values):.6f}")

    # 상위 10개 파일 출력
    print(f"\n=== Top 10 Best Performing Files ===")
    for i, perf in enumerate(selected_performances[:10]):
        print(f"{i+1:2d}. {perf['filename']} - RMSE: {perf['rmse']:.6f}, MAE: {perf['mae']:.6f}, Valid: {perf['valid_count']}")

    # 선택된 파일들로 파일 리스트 구성 (상위 80%만)
    recon_list = [p['recon_file'] for p in selected_performances]
    gt_list = [p['gt_file'] for p in selected_performances]
    mask_list = [p['mask_file'] for p in selected_performances]

    # ===== 컬러맵 생성용 다양성 체크 및 저장 =====
    print(f"\n=== Processing Selected Files for Color Maps and Plots ===")

    # 날짜별 결과 저장 디렉터리 생성
    base_color_path = os.path.join(save_path, f'color_{loss_rate}_top80_performance')
    os.makedirs(base_color_path, exist_ok=True)

    # 날짜별 통계 저장
    date_stats = {}

    total_rmse = total_mae = 0.0
    count = 0
    accu_true, accu_pred = [], []

    # 컬러맵 생성 카운터
    colormap_created = 0
    colormap_skipped = 0

    # 선택된 파일들만 처리 (상위 80%)
    for i in tqdm(range(len(recon_list)), desc="Processing top 80% selected files", unit="file"):
        recon_f = recon_list[i]
        gt_f    = gt_list[i]
        mask_f  = mask_list[i]
        name    = os.path.basename(recon_f)

        # 파일에서 날짜 추출
        file_date = extract_date_from_path(recon_f)
        if not file_date:
            file_date = "unknown"

        # 날짜별 컬러 저장 폴더 생성
        date_color_path = os.path.join(base_color_path, file_date)
        os.makedirs(date_color_path, exist_ok=True)

        try:
            arr_rec = np.loadtxt(recon_f, delimiter=',', dtype='float32')
            arr_gt  = np.loadtxt(gt_f,    delimiter=',', dtype='float32')
            arr_m   = np.loadtxt(mask_f,  delimiter=',', dtype='float32')
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue

        # nan 처리 및 스케일링
        for arr in (arr_rec, arr_gt, arr_m):
            np.place(arr, arr == 255, np.nan)

        arr_rec = 0.01 + (arr_rec - global_min)*(10-0.01)/(global_max-global_min)
        arr_gt  = 0.01 + (arr_gt  - global_min)*(10-0.01)/(global_max-global_min)

        # row/col 파싱: 여러 패턴 지원
        row, col = None, None
        patterns = [
            r'y(\d+)_x(\d+)',
            r'r(\d+)_c(\d+)',
            r'img_\d+_y(\d+)_x(\d+)',
            r'(\d+)_(\d+)',
        ]

        for pattern in patterns:
            m = re.search(pattern, name)
            if m:
                row, col = int(m.group(1)), int(m.group(2))
                break

        if row is None or col is None:
            print(f"Could not extract row/col from filename: {name}")
            continue

        # 픽셀 분포 다양성 확인 후 컬러맵 이미지 저장
        if check_pixel_diversity(arr_rec, min_range=0.5, min_std=0.1):
            try:
                save_colormap_image_with_land_mask(
                    arr_rec, land_sea_mask_path, row, col,
                    os.path.join(date_color_path, name),
                    global_min=global_min, global_max=global_max,
                    recon_file_name=name
                )
                colormap_created += 1
            except Exception as e:
                print(f"Error saving colormap for {name}: {e}")
        else:
            colormap_skipped += 1


        # 통계 계산 (모든 선택된 파일들에 대해)
        valid = (~np.isnan(arr_m)) & (~np.isnan(arr_gt)) & (~np.isnan(arr_rec)) & (arr_gt != 0)
        if not np.any(valid):
            continue

        # Get valid values directly using boolean mask
        valid_gt = arr_gt[valid]
        valid_rec = arr_rec[valid]

        diff = valid_gt - valid_rec
        file_mae = np.mean(np.abs(diff))
        file_rmse = np.sqrt(np.mean(diff**2))
        n = len(valid_gt)

        # 전체 통계에 누적
        total_mae  += np.sum(np.abs(diff))
        total_rmse += np.sum(diff**2)
        count += n
        MAX_ACCUMULATE = 1_000_000

        # Sample for accumulation if needed
        remaining = MAX_ACCUMULATE - len(accu_true)
        if remaining > 0 and len(valid_gt) > 0:
            select_count = min(remaining, len(valid_gt))
            if select_count < len(valid_gt):
                # Sample randomly
                indices = np.random.choice(len(valid_gt), size=select_count, replace=False)
                accu_true.extend(valid_gt[indices].tolist())
                accu_pred.extend(valid_rec[indices].tolist())
            else:
                # Use all valid values
                accu_true.extend(valid_gt.tolist())
                accu_pred.extend(valid_rec.tolist())

        # 날짜별 통계 누적
        if file_date not in date_stats:
            date_stats[file_date] = {
                'total_mae': 0.0,
                'total_rmse': 0.0,
                'count': 0,
                'files': 0,
                'true_vals': [],
                'pred_vals': []
            }

        date_stats[file_date]['total_mae'] += np.sum(np.abs(diff))
        date_stats[file_date]['total_rmse'] += np.sum(diff**2)
        date_stats[file_date]['count'] += n
        date_stats[file_date]['files'] += 1
        date_stats[file_date]['true_vals'].extend(valid_gt.tolist())
        date_stats[file_date]['pred_vals'].extend(valid_rec.tolist())

    if count == 0:
        print("No valid data for plotting.")
        return

    # 전체 통계 계산 및 출력
    rmse_val = math.sqrt(total_rmse / count)
    mae_val  = total_mae / count

    print(f"\n=== Overall Statistics (Top 80% Performance Samples) ===")
    print(f"Actual loss rate used: {loss_rate}%")
    print(f"Selected sample size: {len(selected_performances)} (top 80%)")
    print(f"Total valid pixels: {count:,}")
    print(f"Overall RMSE: {rmse_val:.6f}")
    print(f"Overall MAE: {mae_val:.6f}")
    print(f"Color maps created: {colormap_created}")
    print(f"Color maps skipped (low diversity): {colormap_skipped}")

    # 데이터 분포 확인
    print(f"\n=== Data Distribution Check ===")
    print(f"Total pixels in plot: {len(accu_true):,}")

    # 안전한 min/max 계산을 위한 체크
    if accu_true and accu_pred:
        # NaN이나 inf 값을 제거한 유효한 값들만 필터링
        valid_true = [x for x in accu_true if not (math.isnan(x) or math.isinf(x))]
        valid_pred = [x for x in accu_pred if not (math.isnan(x) or math.isinf(x))]

        if valid_true and valid_pred:
            print(f"True value range: {min(valid_true):.6f} - {max(valid_true):.6f}")
            print(f"Pred value range: {min(valid_pred):.6f} - {max(valid_pred):.6f}")
            print(f"True value std: {np.std(valid_true):.6f}")
            print(f"Pred value std: {np.std(valid_pred):.6f}")
            print(f"Valid true values: {len(valid_true):,}/{len(accu_true):,}")
            print(f"Valid pred values: {len(valid_pred):,}/{len(accu_pred):,}")
        else:
            print("Warning: No valid numeric values found in accumulated data")
            print(f"accu_true sample: {accu_true[:5] if accu_true else 'empty'}")
            print(f"accu_pred sample: {accu_pred[:5] if accu_pred else 'empty'}")
    else:
        print("Warning: No data accumulated for plotting")
        print(f"accu_true length: {len(accu_true)}")
        print(f"accu_pred length: {len(accu_pred)}")

    # 날짜별 통계 출력 및 개별 parity plot 생성
    print(f"\n=== Date-wise Statistics ===")
    for date, stats in sorted(date_stats.items()):
        if stats['count'] > 0:
            date_rmse = math.sqrt(stats['total_rmse'] / stats['count'])
            date_mae = stats['total_mae'] / stats['count']

            print(f"Date {date}:")
            print(f"  - Files: {stats['files']}")
            print(f"  - Valid pixels: {stats['count']:,}")
            print(f"  - RMSE: {date_rmse:.6f}")
            print(f"  - MAE: {date_mae:.6f}")

            # 날짜별 parity plot 생성
            date_save_path = os.path.join(save_path, f'date_plots_{loss_rate}_top80_performance')
            os.makedirs(date_save_path, exist_ok=True)

            try:
                plot_parity(
                    filename=date_save_path,
                    loss_rate=f"{loss_rate}_{date}_top80",
                    true=np.array(stats['true_vals']),
                    pred=np.array(stats['pred_vals']),
                    rmse_=date_rmse,
                    mae_=date_mae,
                    title=f"Date {date} - Loss {loss_rate}% (Top 80%)",
                    kind="scatter",
                    scatter_kws={'s': 1, 'alpha': 0.05}
                )
            except Exception as e:
                print(f"  - Error creating parity plot for {date}: {e}")

    # 전체 parity plot 생성
    print(f"\n=== Creating Overall Parity Plot ===")
    try:
        plot_parity(
            filename=save_path,
            loss_rate=f"{loss_rate}_overall_top80_performance",
            true=np.array(accu_true),
            pred=np.array(accu_pred),
            rmse_=rmse_val,
            mae_=mae_val,
            title=f"Overall - Loss {loss_rate}% (Top 80% Performance)",
            kind="scatter",
            scatter_kws={'s': 1, 'alpha': 0.01}
        )
        print(f"Overall parity plot saved successfully")
    except Exception as e:
        print(f"Error creating overall parity plot: {e}")

    print(f"\n=== Summary ===")
    print(f"Selection method: Top 80% best performing files (by RMSE)")
    print(f"Total files available: {len(valid_performances)}")
    print(f"Files selected: {len(selected_performances)} ({len(selected_performances)/len(valid_performances)*100:.1f}%)")
    print(f"Processed {len(date_stats)} different dates")
    print(f"Total files processed: {sum(stats['files'] for stats in date_stats.values())}")
    print(f"Calculated and used loss rate: {loss_rate}%")
    print(f"Results saved to: {save_path}")
    print(f"Color images saved to: {base_color_path} (organized by date)")
    print(f"Date-wise plots saved to: {os.path.join(save_path, f'date_plots_{loss_rate}_top80_performance')}")
    print(f"RMSE range of top 80%: {min(rmse_values):.6f} - {max(rmse_values):.6f}")
    print(f"MAE range of top 80%: {min(mae_values):.6f} - {max(mae_values):.6f}")
    print(f"Color diversity filter: Created {colormap_created}, Skipped {colormap_skipped}")