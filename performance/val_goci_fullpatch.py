import os
import glob
import re
import numpy as np
import tifffile as tiff
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ================== 설정 ==================
# 기본 경로 설정
BASE_RESULTS_DIR = '/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/band2/2021'
BASE_PERFORMANCE_DIR = '/home/juneyonglee/myhdd/GOCI_RRS/performance/band2/2021'

# 고정 설정
LAND_MASK_NPY = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'
PATCH_SIZE = 256
CMAP_NAME = 'jet'
# ========================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]

def check_ocean_mask_exists(mask_dir):
    """
    마스크 디렉토리에서 해양 영역에 결측치가 있는지 확인
    Returns: (has_ocean_mask, ocean_hole_count)
    """
    if not os.path.isdir(mask_dir):
        return False, 0

    land_mask = np.load(LAND_MASK_NPY)
    # 수정: 999=육지를 0으로, 나머지(바다)를 1로 설정 (eval과 동일)
    land_sea_mask = np.where(land_mask == 999, 0, 1).astype(np.uint8)  # 0=육지, 1=바다

    csv_files = sorted(glob.glob(os.path.join(mask_dir, '*.csv')), key=natural_sort_key)
    if not csv_files:
        return False, 0

    total_ocean_holes = 0
    valid_patches_with_ocean = 0

    # 몇 개 샘플 패치만 확인하여 성능 최적화 (균등하게 선택)
    step = max(1, len(csv_files) // 10)
    sample_files = csv_files[::step][:10]

    for fpath in sample_files:
        try:
            arr = np.loadtxt(fpath, delimiter=',', dtype=np.float32)
            if arr.shape != (PATCH_SIZE, PATCH_SIZE):
                continue

            # 패치 좌표 추출
            m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
            if not m:
                continue
            y0, x0 = map(int, m.groups())

            # 해당 영역의 육지-해양 마스크 추출
            try:
                patch_land_sea_mask = land_sea_mask[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            except IndexError:
                continue

            # 해양 영역 확인
            ocean_mask = (patch_land_sea_mask == 1)  # 1=바다
            total_ocean_pixels = np.sum(ocean_mask)
            total_pixels = PATCH_SIZE * PATCH_SIZE
            ocean_ratio = total_ocean_pixels / total_pixels

            # 해양 영역이 30% 미만이면 건너뜀 (eval과 동일 기준)
            if ocean_ratio < 0.3:
                continue

            valid_patches_with_ocean += 1

            # 해양 영역에서 결측치(0) 확인
            ocean_holes = ocean_mask & (arr == 0)
            ocean_hole_count = np.sum(ocean_holes)
            ocean_hole_ratio = ocean_hole_count / total_ocean_pixels if total_ocean_pixels > 0 else 0

            # 해양 결측치 비율이 1-95% 범위에 있는 경우만 카운트 (eval과 동일 기준)
            if 0.01 <= ocean_hole_ratio <= 0.95:
                total_ocean_holes += ocean_hole_count
                print(f"    ✓ {os.path.basename(fpath)}: Ocean {ocean_ratio:.1%}, Holes {ocean_hole_ratio:.1%} ({ocean_hole_count} pixels)")

        except Exception:
            continue

    print(f"    Valid ocean patches checked: {valid_patches_with_ocean}/{len(sample_files)}")
    return total_ocean_holes > 0, total_ocean_holes

def find_dates_with_ocean_masks():
    """
    해양 마스크가 있는 날짜들을 자동으로 찾아서 반환
    """
    dates_with_masks = []

    if not os.path.exists(BASE_RESULTS_DIR):
        print(f"[ERROR] Base results directory not found: {BASE_RESULTS_DIR}")
        return dates_with_masks

    # 모든 날짜 폴더 검색
    for date_item in sorted(os.listdir(BASE_RESULTS_DIR)):
        date_path = os.path.join(BASE_RESULTS_DIR, date_item)
        if not os.path.isdir(date_path):
            continue

        print(f"[INFO] Checking date: {date_item}")
        date_has_masks = False
        total_ocean_holes = 0

        # 각 날짜의 시간대별 폴더 검색
        for time_item in sorted(os.listdir(date_path)):
            time_path = os.path.join(date_path, time_item)
            if not os.path.isdir(time_path):
                continue

            mask_dir = os.path.join(time_path, 'degree', 'mask')
            has_mask, hole_count = check_ocean_mask_exists(mask_dir)

            if has_mask:
                date_has_masks = True
                total_ocean_holes += hole_count
                print(f"  ✓ Found ocean masks in {time_item}: {hole_count} holes")

        if date_has_masks:
            dates_with_masks.append(date_item)
            print(f"[FOUND] {date_item} has {total_ocean_holes} total ocean holes - ADDED to processing list")
        else:
            print(f"[SKIP] {date_item} has no ocean masks")

    return dates_with_masks

def convert_raw_to_color(data, vmin, vmax, cmap_name='jet'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap_name)
    return colormap(norm(data))[:, :, :3]  # RGB

def load_full_image_from_patches(patch_dir):
    land_mask = np.load(LAND_MASK_NPY)
    H, W = land_mask.shape
    full_img = np.zeros((H, W), dtype=np.float32)
    csv_files = sorted(glob.glob(os.path.join(patch_dir, '*.csv')), key=natural_sort_key)

    if not csv_files:
        print(f"[WARN] No CSV found in {patch_dir}. Returning empty image.")
        return None

    for fpath in csv_files:
        try:
            arr = np.loadtxt(fpath, delimiter=',', dtype=np.float32)
            if arr.shape != (PATCH_SIZE, PATCH_SIZE):
                print(f"[SKIP] Invalid shape {arr.shape} in {fpath}")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to load {fpath}: {e}")
            continue

        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if not m:
            continue
        y0, x0 = map(int, m.groups())
        full_img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = arr

    return full_img

def save_image_with_details(full_img, out_dir, file_prefix, label_text, cmap_name='jet', global_vmin=None, global_vmax=None):
    land_mask = np.load(LAND_MASK_NPY)  # 999: land
    # 수정: eval과 동일하게 land-sea mask 설정
    land_sea_mask = np.where(land_mask == 999, 0, 1).astype(np.uint8)  # 0=육지(999), 1=바다
    land_bool = (land_sea_mask == 0)  # 육지 영역
    ocean_bool = (land_sea_mask == 1)  # 해양 영역

    os.makedirs(out_dir, exist_ok=True)
    # out_tiff = os.path.join(out_dir, f'{file_prefix}.tiff')
    out_png  = os.path.join(out_dir, f'{file_prefix}.png')
    out_bar  = os.path.join(out_dir, f'{file_prefix}_bar.png')

    # tiff.imwrite(out_tiff, full_img)
    # print(f"[{file_prefix.upper()}] TIFF saved → {out_tiff}")

    # Determine invalid pixels more carefully
    if 'mask' in file_prefix:
        # For mask, create 3-value visualization: 0=missing_ocean, 0.5=land, 1=valid_ocean
        unique_vals = np.unique(full_img)
        print(f"[{file_prefix.upper()}] Mask unique values: {unique_vals}")

        # Create 3-category mask for better visualization
        display_img = full_img.copy()

        # 0 = missing ocean pixels (red)
        # 0.5 = land pixels (black)
        # 1 = valid ocean pixels (white)
        display_img[land_bool] = 0.5  # Land = middle value (will be black)

        # 해양 영역에서만 결측치를 처리하도록 수정 (eval과 동일 로직)
        missing_ocean_mask = ocean_bool & (full_img == 0)  # 해양 영역의 결측치만
        valid_ocean_mask = ocean_bool & (full_img == 1)    # 해양 영역의 유효값만
        missing_land_mask = land_bool & (full_img == 0)     # 육지 영역의 결측치 (표시 안함)

        print(f"[DEBUG] Land pixels: {np.sum(land_bool)}")
        print(f"[DEBUG] Ocean pixels: {np.sum(ocean_bool)}")
        print(f"[DEBUG] Missing ocean pixels (red): {np.sum(missing_ocean_mask)}")
        print(f"[DEBUG] Valid ocean pixels (white): {np.sum(valid_ocean_mask)}")
        print(f"[DEBUG] Missing land pixels (not shown): {np.sum(missing_land_mask)}")

        # eval과 동일한 기준 적용: 해양 영역 비율 확인
        total_pixels = full_img.size
        ocean_ratio = np.sum(ocean_bool) / total_pixels
        print(f"[DEBUG] Ocean ratio in full image: {ocean_ratio:.1%}")

        # 해양 영역의 결측치만 빨간색으로 표시
        display_img[missing_ocean_mask] = 0.0   # 해양 결측 = red
        display_img[valid_ocean_mask] = 1.0     # 유효 해양 = white
        # 육지는 0.5 (이미 위에서 설정됨) = black

        vmin, vmax = 0, 1
        invalid_data_mask = np.zeros_like(full_img, dtype=bool)  # No invalid mask needed
        print(f"[{file_prefix.upper()}] Using 3-category mask: 0=missing_ocean(red), 0.5=land(black), 1=valid_ocean(white)")
    elif 'recon' in file_prefix or 'gt' in file_prefix:
        # Use same colorbar range calculation for both GT and recon data
        if 'recon' in file_prefix:
            invalid_data_mask = (full_img == -999) | (full_img == 0)
        else:  # GT data
            # For GT, treat 1000 as invalid data (seems to be a fill value)
            invalid_data_mask = (full_img == -999) | (full_img == 0) | (full_img == 1000)

        # Use global range if provided, otherwise calculate from current image
        if global_vmin is not None and global_vmax is not None:
            vmin, vmax = global_vmin, global_vmax
            print(f"[{file_prefix.upper()}] Using global range [{vmin:.6f}, {vmax:.6f}]")
        else:
            valid_data = full_img[~invalid_data_mask & ~land_bool]
            if valid_data.size == 0:
                print(f"[{file_prefix.upper()}] WARN: No valid data. Using default [0, 1].")
                vmin, vmax = 0, 1
            else:
                # Use percentile-based range for both GT and recon to ensure same colorbar scale
                vmin = np.percentile(valid_data, 1)
                vmax = np.percentile(valid_data, 99)
                print(f"[{file_prefix.upper()}] Using calculated range [{vmin:.6f}, {vmax:.6f}]")
                print(f"[{file_prefix.upper()}] Valid pixels: {valid_data.size}, Range: [{np.min(valid_data):.6f}, {np.max(valid_data):.6f}]")

    # Handle display image
    if 'mask' in file_prefix:
        # For masks, display_img was already created above with 3 categories
        pass
    else:
        # For both GT and recon, use the original data without normalization
        display_img = full_img.copy()

    # Set invalid pixels to vmin for proper clipping (but not for masks)
    if not 'mask' in file_prefix:
        all_invalid_mask = invalid_data_mask | land_bool
        display_img[all_invalid_mask] = vmin

    # Clip and colorize
    clipped = np.clip(display_img, vmin, vmax)
    colored = convert_raw_to_color(clipped, vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    # Apply final masks for visualization
    if 'mask' in file_prefix:
        # For masks, create completely custom colors for 3 categories
        # First initialize as black
        colored = np.zeros((display_img.shape[0], display_img.shape[1], 3), dtype=np.float32)

        # Missing ocean pixels only = red (eval과 동일 로직)
        missing_ocean_only = (display_img == 0.0) & ocean_bool  # 해양 영역의 결측치만
        colored[missing_ocean_only] = [1.0, 0.0, 0.0]  # Red
        print(f"[DEBUG] Missing ocean pixels set to red: {np.sum(missing_ocean_only)}")

        # Valid ocean pixels = white
        valid_ocean_pixels = ocean_bool & (display_img == 1.0)
        colored[valid_ocean_pixels] = [1.0, 1.0, 1.0]  # White
        print(f"[DEBUG] Valid ocean pixels set to white: {np.sum(valid_ocean_pixels)}")

        # All land pixels = black (regardless of missing status)
        colored[land_bool] = [0.0, 0.0, 0.0]  # Black
        print(f"[DEBUG] All land pixels set to black: {np.sum(land_bool)}")

    else:
        colored[land_bool] = [0.0, 0.0, 0.0]  # Land = black
        # Invalid ocean data = jet colormap minimum (dark blue)
        jet_min_color = convert_raw_to_color(np.array([[vmin]]), vmin, vmax, cmap_name)[0, 0]
        colored[invalid_data_mask & ~land_bool] = jet_min_color

    plt.imsave(out_png, colored, origin='upper')
    print(f"[{file_prefix.upper()}] Saved colored PNG: {out_png}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(colored, origin='upper')
    ax.axis('off')

    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap(cmap_name))
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(label_text, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_bar, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[{file_prefix.upper()}] Saved colored PNG with colorbar: {out_bar}")

def calculate_global_range(gt_img, recon_img):
    """
    Calculate a common colorbar range for GT and recon images.
    """
    land_mask = np.load(LAND_MASK_NPY)
    land_sea_mask = np.where(land_mask == 999, 0, 1).astype(np.uint8)
    land_bool = (land_sea_mask == 0)

    all_valid_data = []

    # Process GT data
    if gt_img is not None:
        gt_invalid_mask = (gt_img == -999) | (gt_img == 0) | (gt_img == 1000)
        gt_valid = gt_img[~gt_invalid_mask & ~land_bool]
        if gt_valid.size > 0:
            all_valid_data.append(gt_valid)

    # Process recon data
    if recon_img is not None:
        recon_invalid_mask = (recon_img == -999) | (recon_img == 0)
        recon_valid = recon_img[~recon_invalid_mask & ~land_bool]
        if recon_valid.size > 0:
            all_valid_data.append(recon_valid)

    if not all_valid_data:
        print("[GLOBAL RANGE] No valid data found. Using default [0, 1].")
        return 0, 1

    # Combine all valid data
    combined_data = np.concatenate(all_valid_data)

    # Calculate global percentile range
    global_vmin = np.percentile(combined_data, 1)
    global_vmax = np.percentile(combined_data, 99)

    print(f"[GLOBAL RANGE] Combined range [{global_vmin:.6f}, {global_vmax:.6f}] from {combined_data.size} valid pixels")

    return global_vmin, global_vmax

def process_and_average(image_list, data_type='default'):
    if not image_list:
        return None

    print(f"[AVERAGE] Processing {len(image_list)} images for {data_type}")
    stacked_images = np.stack(image_list, axis=0)

    # More careful handling of masked data
    masked_stacked_images = np.ma.masked_where((stacked_images == -999) | (stacked_images == 0), stacked_images)

    if data_type == 'mask':
        # For masks, use majority vote instead of average
        valid_mask = ~masked_stacked_images.mask
        sum_valid = np.sum(valid_mask, axis=0)
        sum_ones = np.sum(stacked_images * valid_mask, axis=0)
        # Use safe division to avoid warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            averaged_img = np.where(sum_valid > 0, sum_ones / sum_valid, 0)
    else:
        # For GT and recon data, use mean of valid pixels
        averaged_img = np.ma.mean(masked_stacked_images, axis=0)
        averaged_img = averaged_img.filled(0)  # Fill masked values with 0

    # Print statistics for debugging
    non_zero = averaged_img[averaged_img != 0]
    if non_zero.size > 0:
        print(f"[AVERAGE] {data_type} - Non-zero pixels: {non_zero.size}, "
              f"Range: [{np.min(non_zero):.6f}, {np.max(non_zero):.6f}], "
              f"Mean: {np.mean(non_zero):.6f}")
    else:
        print(f"[AVERAGE] {data_type} - No non-zero pixels found!")

    return averaged_img

def process_date(base_date_path, out_root_dir):
    """Main processing logic for a single date."""
    if not os.path.isdir(base_date_path):
        print(f"\n[ERROR] Base date path not found: {base_date_path}. Skipping.")
        return

    time_subdirs = [d for d in os.listdir(base_date_path) if os.path.isdir(os.path.join(base_date_path, d))]

    # Process each time subdirectory individually instead of averaging
    for time_idx, time_subdir in enumerate(sorted(time_subdirs)):
        degree_path = os.path.join(base_date_path, time_subdir, 'degree')
        print(f"\n[Main] Processing time subdir: {time_subdir}")

        # Create output directory for this specific time
        time_out_dir = os.path.join(out_root_dir, f"time_{time_idx:02d}_{time_subdir}")

        # Load all images first
        gt_img = None
        recon_img = None
        mask_img = None

        for data_type in ['gt', 'recon', 'mask']:
            patch_dir = os.path.join(degree_path, data_type)
            if os.path.isdir(patch_dir):
                print(f"  - Loading {data_type} data from: {patch_dir}")
                full_img = load_full_image_from_patches(patch_dir)
                if full_img is not None:
                    if data_type == 'gt':
                        gt_img = full_img
                    elif data_type == 'recon':
                        recon_img = full_img
                    elif data_type == 'mask':
                        mask_img = full_img
            else:
                print(f"  - '{data_type}' directory not found in {degree_path}. Skipping.")

        # Calculate global colorbar range for GT and recon
        global_vmin, global_vmax = calculate_global_range(gt_img, recon_img)

        # Save images with common colorbar range
        if gt_img is not None:
            save_image_with_details(gt_img, time_out_dir, f'gt_{time_subdir}', 'GT Chlorophyll-a (mg/m³)',
                                   global_vmin=global_vmin, global_vmax=global_vmax)
        if recon_img is not None:
            save_image_with_details(recon_img, time_out_dir, f'recon_{time_subdir}', 'Recon Chlorophyll-a (mg/m³)',
                                   global_vmin=global_vmin, global_vmax=global_vmax)
        if mask_img is not None:
            save_image_with_details(mask_img, time_out_dir, f'mask_{time_subdir}', 'Mask', cmap_name='RdYlBu_r')

def process_multiple_dates(base_results_dir, base_performance_dir, land_sea_mask_path, target_dates):
    """
    여러 날짜를 일괄 처리하는 함수 (GOCI용)

    Args:
        base_results_dir: 결과 기본 경로 (예: '/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/band2/2021')
        base_performance_dir: 성능 저장 기본 경로 (예: '/home/juneyonglee/myhdd/GOCI_RRS/performance/band2/2021')
        land_sea_mask_path: GOCI 육지-해양 마스크 파일 경로
        target_dates: 처리할 날짜 리스트 (예: ['20210101', '20210102'])
    """
    print(f"=== Processing Multiple GOCI Dates ===")
    print(f"Base results dir: {base_results_dir}")
    print(f"Base performance dir: {base_performance_dir}")
    print(f"Target dates: {target_dates}")

    success_count = 0
    failed_dates = []

    for date_str in target_dates:
        if not date_str.strip():  # 빈 문자열 건너뛰기
            continue

        print(f"\n{'='*60}")
        print(f"Processing GOCI Date: {date_str}")
        print(f"{'='*60}")

        try:
            # 입력 및 출력 경로 구성
            date_result_path = os.path.join(base_results_dir, date_str)
            date_output_path = os.path.join(base_performance_dir, date_str)

            print(f"Input path: {date_result_path}")
            print(f"Output path: {date_output_path}")

            # 경로 존재 확인
            if not os.path.exists(date_result_path):
                print(f"❌ Input path does not exist: {date_result_path}")
                failed_dates.append((date_str, "Input path not found"))
                continue

            # 시간대별 하위 디렉토리 확인
            time_subdirs = [d for d in os.listdir(date_result_path)
                           if os.path.isdir(os.path.join(date_result_path, d))]

            if not time_subdirs:
                print(f"⚠️  No time subdirectories found in: {date_result_path}")
                print(f"Available items: {os.listdir(date_result_path) if os.path.exists(date_result_path) else 'None'}")
                failed_dates.append((date_str, "No time subdirectories found"))
                continue

            print(f"Found {len(time_subdirs)} time subdirectories: {time_subdirs}")

            # 해양 마스크 존재 확인 (간단한 버전)
            has_ocean_data = False
            for time_subdir in time_subdirs[:3]:  # 처음 몇 개만 확인
                degree_path = os.path.join(date_result_path, time_subdir, 'degree')
                if os.path.exists(degree_path):
                    mask_dir = os.path.join(degree_path, 'mask')
                    if os.path.exists(mask_dir):
                        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.csv')]
                        if mask_files:
                            has_ocean_data = True
                            break

            if not has_ocean_data:
                print(f"⚠️  No ocean mask data found for {date_str}")
                failed_dates.append((date_str, "No ocean mask data found"))
                continue

            # 처리 실행
            process_date(date_result_path, date_output_path)

            success_count += 1
            print(f"✅ Successfully processed {date_str}")

        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")
            failed_dates.append((date_str, str(e)))
            continue

    # 최종 결과 요약
    print(f"\n{'='*60}")
    print(f"GOCI PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total dates requested: {len([d for d in target_dates if d.strip()])}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_dates)}")

    if failed_dates:
        print(f"\nFailed dates:")
        for date_str, reason in failed_dates:
            print(f"  - {date_str}: {reason}")

    if success_count > 0:
        print(f"\nResults saved to: {base_performance_dir}")

# 메인 루프
if __name__ == '__main__':
    # 설정 변수들
    base_results_dir = '/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/band2/2021'
    base_performance_dir = '/home/juneyonglee/myhdd/GOCI_RRS/performance/band2/2021'
    land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

    # 처리할 날짜 리스트 (원하는 날짜들을 여기에 추가)
    target_dates = ['20210101', '20210108','20210115','20210122','20210129']

    # 선택한 날짜들만 처리
    process_multiple_dates(base_results_dir, base_performance_dir, land_sea_mask_path, target_dates)

    # === 기존 자동 탐색 방식 (주석 처리됨) ===
    # print("="*60)
    # print("========== FINDING DATES WITH OCEAN MASKS ==========")
    # print("="*60)
    #
    # # 해양 마스크가 있는 날짜들을 자동으로 찾기
    # dates_with_ocean_masks = find_dates_with_ocean_masks()
    #
    # if not dates_with_ocean_masks:
    #     print("\n[WARNING] No dates found with ocean masks!")
    #     print("Exiting without processing any dates.")
    # else:
    #     print(f"\n[SUMMARY] Found {len(dates_with_ocean_masks)} dates with ocean masks:")
    #     for date in dates_with_ocean_masks:
    #         print(f"  - {date}")
    #
    #     print("\n" + "="*60)
    #     print("========== PROCESSING DATES WITH OCEAN MASKS ==========")
    #     print("="*60)
    #
    #     for date_str in dates_with_ocean_masks:
    #         print(f"\n======================================================")
    #         print(f"========== PROCESSING DATE: {date_str} ==========")
    #         print(f"======================================================")
    #
    #         base_path = os.path.join(BASE_RESULTS_DIR, date_str)
    #         out_path = os.path.join(BASE_PERFORMANCE_DIR, date_str)
    #
    #         process_date(base_path, out_path)
    #
    #     print(f"\n[Main] All {len(dates_with_ocean_masks)} dates with ocean masks processed.")
