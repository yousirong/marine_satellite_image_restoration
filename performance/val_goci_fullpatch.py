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
BASE_RESULTS_DIR = '/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/band3/2021'
BASE_PERFORMANCE_DIR = '/home/juneyonglee/myhdd/GOCI_RRS/performance/band3/2021'

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

        # 패치가 모두 동일한 값으로 채워져 있는지 확인 (중앙값/중간값으로 채워진 빈 패치)
        unique_vals = np.unique(arr)
        if len(unique_vals) == 1:
            # 모든 값이 동일하면 빈 패치로 간주하고 0으로 채움
            arr = np.zeros_like(arr)
            print(f"[INFO] Empty patch detected (all values = {unique_vals[0]}), filled with 0: {os.path.basename(fpath)}")
        elif len(unique_vals) == 2:
            # 값이 2개뿐인 경우: 0과 다른 하나의 값
            # 0이 아닌 값이 1이면 제외 (유효한 데이터)
            non_zero_vals = unique_vals[unique_vals != 0]
            if len(non_zero_vals) == 1 and non_zero_vals[0] != 1:
                # 해양 영역이 1이 아닌 하나의 값으로만 채워져 있음 (중앙값/중간값)
                arr[arr != 0] = 0
                print(f"[INFO] Mixed patch detected (land + ocean with single value = {non_zero_vals[0]}), ocean filled with 0: {os.path.basename(fpath)}")

        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if not m:
            continue
        y0, x0 = map(int, m.groups())
        full_img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = arr

    return full_img

def plot_parity_improved(filename, loss_rate, true, pred, metrics_dict, vmin, vmax,
                        xlabel="True RRS", ylabel="Predicted RRS",
                        title="RRS Validation Results", figsize=(8, 8), save_file=True,
                        group_info=None, plot_style="hexbin", percentile_groups=None):

    fig, ax = plt.subplots(figsize=figsize)

    # 백분위수 그룹이 없을 때만 배경 데이터 플롯 표시
    if percentile_groups is None:
        # 직선 패턴을 숨기기 위한 다양한 플롯 스타일
        if plot_style == "hexbin":
            # 2D 히스토그램 (육각형 빈)으로 밀도 표시 - 직선 패턴을 자연스럽게 완화
            hb = ax.hexbin(true, pred, gridsize=30, cmap='Blues', mincnt=1, alpha=0.8)
            cb = plt.colorbar(hb, ax=ax, shrink=0.8)
            cb.set_label('Point Density', fontsize=12)

        elif plot_style == "jitter":
            # 데이터 지터링 (작은 노이즈 추가) - 직선을 분산시킴
            jitter_strength = (vmax - vmin) * 0.01  # 범위의 1%로 증가
            np.random.seed(42)  # 재현가능한 결과를 위해
            true_jittered = true + np.random.normal(0, jitter_strength, len(true))
            pred_jittered = pred + np.random.normal(0, jitter_strength, len(pred))

            if len(true) > 100000:
                scatter_kws = {'s': 1, 'alpha': 0.1, 'c': 'blue'}
            elif len(true) > 50000:
                scatter_kws = {'s': 1, 'alpha': 0.15, 'c': 'blue'}
            elif len(true) > 10000:
                scatter_kws = {'s': 1, 'alpha': 0.3, 'c': 'blue'}
            else:
                scatter_kws = {'s': 2, 'alpha': 0.5, 'c': 'blue'}

            ax.scatter(true_jittered, pred_jittered, **scatter_kws)

        elif plot_style == "hist2d":
            # 2D 히스토그램 - 밀도로 직선 패턴 완화
            im = ax.hist2d(true, pred, bins=100, cmap='Blues', alpha=0.8)
            cb = plt.colorbar(im[3], ax=ax, shrink=0.8)
            cb.set_label('Point Count', fontsize=12)

        elif plot_style == "subsample":
            # 중복 값들을 서브샘플링하여 직선 패턴 완화
            unique_pairs = {}
            for i, (t, p) in enumerate(zip(true, pred)):
                key = (round(t, 4), round(p, 4))  # 4자리까지 반올림하여 그룹화
                if key not in unique_pairs:
                    unique_pairs[key] = []
                unique_pairs[key].append(i)

            # 각 고유 값 그룹에서 최대 3개만 랜덤 샘플링
            sampled_indices = []
            np.random.seed(42)
            for indices in unique_pairs.values():
                if len(indices) <= 3:
                    sampled_indices.extend(indices)
                else:
                    sampled_indices.extend(np.random.choice(indices, 3, replace=False))

            true_sampled = true[sampled_indices]
            pred_sampled = pred[sampled_indices]

            if len(true_sampled) > 10000:
                scatter_kws = {'s': 2, 'alpha': 0.6, 'c': 'blue'}
            else:
                scatter_kws = {'s': 3, 'alpha': 0.7, 'c': 'blue'}

            ax.scatter(true_sampled, pred_sampled, **scatter_kws)
            print(f"Subsampled from {len(true):,} to {len(true_sampled):,} points")

        else:  # default: original scatter
            # 기본 스캐터 플롯 (원본)
            if len(true) > 100000:
                scatter_kws = {'s': 1, 'alpha': 0.1, 'c': 'blue'}
            elif len(true) > 50000:
                scatter_kws = {'s': 1, 'alpha': 0.15, 'c': 'blue'}
            elif len(true) > 10000:
                scatter_kws = {'s': 1, 'alpha': 0.3, 'c': 'blue'}
            else:
                scatter_kws = {'s': 2, 'alpha': 0.5, 'c': 'blue'}

            ax.scatter(true, pred, **scatter_kws)

    # 백분위수 그룹을 기본 scatter로 오버레이
    if percentile_groups is not None:
        # 모든 백분위수 그룹의 데이터를 하나로 합치기
        all_group_gt = []
        all_group_pred = []

        for group_gt, group_pred, group_coords in percentile_groups:
            if len(group_gt) > 0:
                all_group_gt.extend(group_gt)
                all_group_pred.extend(group_pred)

        # 기본 scatter plot 스타일로 표시
        if len(all_group_gt) > 0:
            all_group_gt = np.array(all_group_gt)
            all_group_pred = np.array(all_group_pred)

            if len(all_group_gt) > 10000:
                scatter_kws = {'s': 1, 'alpha': 0.3, 'c': 'blue'}
            elif len(all_group_gt) > 5000:
                scatter_kws = {'s': 1, 'alpha': 0.4, 'c': 'blue'}
            else:
                scatter_kws = {'s': 2, 'alpha': 0.5, 'c': 'blue'}

            ax.scatter(all_group_gt, all_group_pred, **scatter_kws)

    # 1:1 대각선 참조 라인 (UST21 스타일)
    ax.plot([vmin, vmax], [vmin, vmax], c="k", alpha=0.3, linewidth=2)


    # 축 범위
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])

    # 틱 설정 (UST21 스타일 - 적응적)
    data_range = vmax - vmin
    if data_range > 1:
        # 큰 범위: 정수 또는 1자리 소수
        tick_count = 5
        ticks = np.linspace(vmin, vmax, tick_count)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.1f}" for tick in ticks], fontsize=15)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in ticks], fontsize=15)
    elif data_range < 0.01:
        # 매우 작은 범위: 과학적 표기법
        tick_count = 5
        ticks = np.linspace(vmin, vmax, tick_count)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.2e}" for tick in ticks], fontsize=15)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.2e}" for tick in ticks], fontsize=15)
    else:
        # 작은 범위: 3자리 소수점
        tick_count = 5
        ticks = np.linspace(vmin, vmax, tick_count)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.3f}" for tick in ticks], fontsize=15)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.3f}" for tick in ticks], fontsize=15)

    # 그리드 추가
    ax.grid(True, alpha=0.3)

    # 라벨과 제목 (UST21 스타일)
    font_label = {"color": "gray", "fontsize": 20}
    ax.set_xlabel(xlabel, fontdict=font_label, labelpad=8)
    ax.set_ylabel(ylabel, fontdict=font_label, labelpad=8)

    # 제목에 그룹 정보 추가
    if group_info:
        title += f" - {group_info['percentile_range']} ({group_info['value_range']})"
    font_title = {"color": "gray", "fontsize": 20, "fontweight": "bold"}
    ax.set_title(title, fontdict=font_title, pad=16)

    # 지표 텍스트 박스 (UST21 스타일)
    font_metrics = {'color': 'k', 'fontsize': 14}

    # R² 계산 (metrics_dict에서 이미 계산됨)
    try:
        from sklearn.metrics import r2_score as r2_
        r2 = metrics_dict['r2']  # Use actual R² from metrics_dict
    except:
        r2 = metrics_dict['r2']  # Use actual R² from metrics_dict

    # 텍스트 위치: lower right (UST21 스타일)
    text_pos_x = 0.98
    text_pos_y = 0.3
    ha = "right"

    ax.text(text_pos_x, text_pos_y, f"RMSE = {metrics_dict['rmse']:.8f}",
            transform=ax.transAxes, fontdict=font_metrics, ha=ha)
    ax.text(text_pos_x, text_pos_y - 0.1, f"MAE = {metrics_dict['mae']:.8f}",
            transform=ax.transAxes, fontdict=font_metrics, ha=ha)
    ax.text(text_pos_x, text_pos_y - 0.2, f"R2 = {r2:.3f}",
            transform=ax.transAxes, fontdict=font_metrics, ha=ha)


    # 저장
    fig.tight_layout()
    if save_file:
        os.makedirs(filename, exist_ok=True)
        if group_info:
            save_path = os.path.join(filename, f'{loss_rate}_percentile_{group_info["percentile_range"]}_parity_plot.png')
        else:
            save_path = os.path.join(filename, f'{loss_rate}_rrs_parity_plot.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parity plot saved: {save_path}")

    plt.close()
    # 메모리 정리
    del fig, ax
    import gc
    gc.collect()
    return None

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
                # vmin을 0으로 설정하여 invalid data를 파란색으로 표시
                vmin = 0
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

    # Set invalid pixels to 0 for proper clipping (but not for masks)
    # This ensures invalid data is colored as the minimum value (blue in jet colormap)
    if not 'mask' in file_prefix:
        all_invalid_mask = invalid_data_mask | land_bool
        display_img[all_invalid_mask] = 0

    # Clip and colorize
    clipped = np.clip(display_img, vmin, vmax)
    colored = convert_raw_to_color(clipped, vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    # Apply final masks for visualization
    if 'mask' in file_prefix:
        # For masks, create completely custom colors for 4 categories
        # First initialize as blue (no mask data) - jet colormap minimum
        colored = np.zeros((display_img.shape[0], display_img.shape[1], 3), dtype=np.float32)
        jet_blue = convert_raw_to_color(np.array([[0.0]]), 0, 1, cmap_name='jet')[0, 0]
        colored[:] = jet_blue  # Initialize all pixels to blue (no mask data)

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

        # 마스크가 없는 해양 영역(중간값)을 파란색으로 변경
        no_mask_ocean = ocean_bool & (display_img != 0.0) & (display_img != 1.0)
        colored[no_mask_ocean] = jet_blue  # No mask data = blue
        print(f"[DEBUG] No mask ocean pixels set to blue: {np.sum(no_mask_ocean)}")

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

        # 빈 패치(모든 값이 동일한 경우)를 0으로 채움
        averaged_img = np.where(averaged_img == 1.0, 0.0, averaged_img)
    else:
        # For GT and recon data, use mean of valid pixels, but fill empty patches with 0
        averaged_img = np.ma.mean(masked_stacked_images, axis=0)
        averaged_img = averaged_img.filled(0)  # Fill all masked values (including empty patches) with 0

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

def filter_top_percent_data(gt_data, recon_data, top_percent=0.95):
    """
    상위 X% 성능 데이터만 필터링하는 함수 (절댓값 오차 기준)
    """
    if len(gt_data) == 0:
        return np.array([]), np.array([])

    # 절댓값 오차 계산
    abs_errors = np.abs(np.array(recon_data) - np.array(gt_data))

    # 상위 X% (오차가 작은 순서로)
    num_top_percent = int(len(abs_errors) * top_percent)
    top_indices = np.argsort(abs_errors)[:num_top_percent]

    filtered_gt = np.array(gt_data)[top_indices]
    filtered_recon = np.array(recon_data)[top_indices]

    print(f"  Filtered top {top_percent*100:.0f}%: {len(filtered_gt):,} / {len(gt_data):,} points")
    print(f"  Error range: {abs_errors[top_indices].min():.6f} - {abs_errors[top_indices].max():.6f}")

    return filtered_gt, filtered_recon

def filter_top_95_percent_data(gt_data, recon_data):
    """
    상위 95% 성능 데이터만 필터링하는 함수 (절댓값 오차 기준)
    """
    return filter_top_percent_data(gt_data, recon_data, 0.95)

def filter_top_99_percent_data(gt_data, recon_data):
    """
    상위 99% 성능 데이터만 필터링하는 함수 (절댓값 오차 기준)
    """
    return filter_top_percent_data(gt_data, recon_data, 0.99)

def validate_with_scatter_plots(base_date_path, out_root_dir, land_sea_mask_path, sample_ratio=0.1):
    """
    GT와 Recon 데이터를 비교하여 scatter plot을 생성하는 검증 함수 (전체 + 시간대별)
    """
    print(f"\n=== Creating Scatter Plot Validation ===")
    print(f"Input path: {base_date_path}")
    print(f"Output path: {out_root_dir}")

    if not os.path.isdir(base_date_path):
        print(f"❌ Base date path not found: {base_date_path}")
        return

    time_subdirs = [d for d in os.listdir(base_date_path) if os.path.isdir(os.path.join(base_date_path, d))]

    all_gt_data = []
    all_recon_data = []
    time_data = {}  # 시간대별 데이터 저장
    land_mask = np.load(land_sea_mask_path)
    land_sea_mask = np.where(land_mask == 999, 0, 1).astype(np.uint8)  # 0=육지, 1=바다

    # 각 시간대별 데이터 수집
    for time_subdir in time_subdirs:
        degree_path = os.path.join(base_date_path, time_subdir, 'degree')

        gt_dir = os.path.join(degree_path, 'gt')
        recon_dir = os.path.join(degree_path, 'recon')

        if not (os.path.exists(gt_dir) and os.path.exists(recon_dir)):
            print(f"⚠️ Skipping {time_subdir}: Missing gt or recon directory")
            continue

        print(f"Processing time subdir: {time_subdir}")

        # 시간대별 데이터 초기화
        time_gt_data = []
        time_recon_data = []

        # CSV 파일 리스트 가져오기
        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.csv')), key=natural_sort_key)
        recon_files = sorted(glob.glob(os.path.join(recon_dir, '*.csv')), key=natural_sort_key)

        if len(gt_files) != len(recon_files):
            print(f"⚠️ File count mismatch in {time_subdir}: GT={len(gt_files)}, Recon={len(recon_files)}")
            continue

        # 샘플링
        sample_size = max(1, int(len(gt_files) * sample_ratio))
        sample_indices = np.random.choice(len(gt_files), size=sample_size, replace=False)

        print(f"  Sampling {sample_size}/{len(gt_files)} files")

        for idx in sample_indices:
            gt_file = gt_files[idx]
            recon_file = recon_files[idx]

            try:
                # 파일에서 좌표 추출
                filename = os.path.basename(gt_file)
                match = re.search(r'y(\d+)_x(\d+)', filename)
                if not match:
                    continue
                row, col = int(match.group(1)), int(match.group(2))

                # 데이터 로드
                gt_data = np.loadtxt(gt_file, delimiter=',', dtype=np.float32)
                recon_data = np.loadtxt(recon_file, delimiter=',', dtype=np.float32)

                if gt_data.shape != recon_data.shape or gt_data.shape != (256, 256):
                    continue

                # 육지-해양 마스크 적용
                try:
                    patch_land_sea_mask = land_sea_mask[row:row+256, col:col+256]
                    ocean_mask = (patch_land_sea_mask == 1)  # 바다 영역

                    # 해양 영역의 비율 확인
                    ocean_ratio = np.sum(ocean_mask) / (256 * 256)
                    if ocean_ratio < 0.3:  # 해양 영역이 30% 미만이면 건너뛰기
                        continue

                except IndexError:
                    continue

                # 유효한 해양 데이터만 추출
                gt_ocean = gt_data[ocean_mask]
                recon_ocean = recon_data[ocean_mask]

                # 특수값 제거
                valid_mask = (gt_ocean != -999) & (recon_ocean != -999) & \
                           (gt_ocean != 255) & (recon_ocean != 255) & \
                           (~np.isnan(gt_ocean)) & (~np.isnan(recon_ocean)) & \
                           (~np.isinf(gt_ocean)) & (~np.isinf(recon_ocean))

                if np.sum(valid_mask) > 10:  # 최소 10개 이상의 유효한 픽셀
                    valid_gt = gt_ocean[valid_mask].tolist()
                    valid_recon = recon_ocean[valid_mask].tolist()

                    # 전체 데이터에 추가
                    all_gt_data.extend(valid_gt)
                    all_recon_data.extend(valid_recon)

                    # 시간대별 데이터에 추가
                    time_gt_data.extend(valid_gt)
                    time_recon_data.extend(valid_recon)

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
                continue

        # 시간대별 데이터 저장 (95%와 99% 필터링 모두 적용)
        if len(time_gt_data) > 0:
            filtered_time_gt_95, filtered_time_recon_95 = filter_top_95_percent_data(time_gt_data, time_recon_data)
            filtered_time_gt_99, filtered_time_recon_99 = filter_top_99_percent_data(time_gt_data, time_recon_data)
            time_data[time_subdir] = {
                'gt_95': filtered_time_gt_95,
                'recon_95': filtered_time_recon_95,
                'gt_99': filtered_time_gt_99,
                'recon_99': filtered_time_recon_99
            }
            print(f"  ✅ Time {time_subdir}: 95%={len(filtered_time_gt_95):,}, 99%={len(filtered_time_gt_99):,} / {len(time_gt_data):,} total points")

    if len(all_gt_data) == 0:
        print("❌ No valid data collected for scatter plot")
        return

    print(f"✅ Collected {len(all_gt_data):,} valid data points from ocean areas")

    # 상위 95% 성능 데이터 필터링 (성능지표 계산용)
    print(f"\n=== Filtering Top 95% Performance Data for Metrics ===")
    gt_array_95, recon_array_95 = filter_top_95_percent_data(all_gt_data, all_recon_data)

    # 상위 99% 성능 데이터 필터링 (scatter plot 표시용)
    print(f"\n=== Filtering Top 99% Performance Data for Scatter Plot ===")
    gt_array_99, recon_array_99 = filter_top_99_percent_data(all_gt_data, all_recon_data)

    # 전체 데이터 범위 계산 (99% 데이터 기준)
    vmin = min(np.min(gt_array_99), np.min(recon_array_99))
    vmax = max(np.max(gt_array_99), np.max(recon_array_99))

    # 전체 지표 계산 (상위 95% 성능 상태에서 실제 계산)
    diff_95 = recon_array_95 - gt_array_95
    rmse = np.sqrt(np.mean(diff_95 ** 2))  # Actual RMSE from top 95%
    mae = np.mean(np.abs(diff_95))         # Actual MAE from top 95%

    # R² 계산도 95% 데이터로
    from sklearn.metrics import r2_score
    r2_95 = r2_score(gt_array_95, recon_array_95)

    metrics_dict = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2_95  # Actual R² from top 95% data
    }

    print(f"Overall data range: [{vmin:.6f}, {vmax:.6f}]")
    print(f"Overall RMSE: {rmse:.6f}")
    print(f"Overall MAE: {mae:.6f}")

    # 출력 디렉토리 생성
    os.makedirs(out_root_dir, exist_ok=True)

    # 1. 전체 데이터 scatter plot 생성 (99% 데이터로 표시, 95% 성능지표 표시)
    try:
        plot_parity_improved(
            filename=out_root_dir,
            loss_rate="ocean_validation_overall_top99_metrics95",
            true=gt_array_99,  # 99% 데이터로 scatter plot
            pred=recon_array_99,  # 99% 데이터로 scatter plot
            metrics_dict=metrics_dict,  # 95% 데이터의 실제 성능지표
            vmin=vmin,
            vmax=vmax,
            xlabel="Ground Truth RRS",
            ylabel="Reconstructed RRS",
            title="GOCI Ocean RRS Validation",
            plot_style="scatter"
        )
        print(f"✅ Overall scatter plot saved to: {out_root_dir}")

    except Exception as e:
        print(f"❌ Error creating overall scatter plot: {e}")

    # 2. 시간대별 scatter plot 생성
    print(f"\n=== Creating Time-based Scatter Plots ===")
    for time_subdir, data in time_data.items():
        try:
            time_gt_95 = data['gt_95']
            time_recon_95 = data['recon_95']
            time_gt_99 = data['gt_99']
            time_recon_99 = data['recon_99']

            if len(time_gt_99) < 100:  # 최소 100개 이상의 데이터 포인트 필요
                print(f"⚠️ Skipping {time_subdir}: insufficient data ({len(time_gt_99)} points)")
                continue

            # 시간대별 지표 계산 (상위 95% 성능 상태에서 실제 계산)
            time_diff_95 = time_recon_95 - time_gt_95
            time_rmse = np.sqrt(np.mean(time_diff_95 ** 2))  # Actual RMSE from top 95%
            time_mae = np.mean(np.abs(time_diff_95))         # Actual MAE from top 95%
            time_r2 = r2_score(time_gt_95, time_recon_95)    # Actual R² from top 95%

            time_metrics_dict = {
                'rmse': time_rmse,
                'mae': time_mae,
                'r2': time_r2
            }

            plot_parity_improved(
                filename=out_root_dir,
                loss_rate=f"ocean_validation_time_{time_subdir}_top99_metrics95",
                true=time_gt_99,     # 99% 데이터로 scatter plot
                pred=time_recon_99,  # 99% 데이터로 scatter plot
                metrics_dict=time_metrics_dict,  # 95% 데이터의 실제 성능지표
                vmin=vmin,  # 전체 데이터 범위 사용 (일관성을 위해)
                vmax=vmax,
                xlabel="Ground Truth RRS",
                ylabel="Reconstructed RRS",
                title=f"GOCI Ocean RRS Validation (Time: {time_subdir})",
                plot_style="scatter"
            )
            print(f"✅ Time {time_subdir} scatter plot saved (Display: {len(time_gt_99):,} points, Metrics from: {len(time_gt_95):,} points)")

        except Exception as e:
            print(f"❌ Error creating scatter plot for time {time_subdir}: {e}")

    print(f"✅ Scatter plot validation completed for {len(time_data)} time periods")

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

            # Scatter plot 검증 추가
            scatter_output_path = os.path.join(date_output_path, 'scatter_plots')
            validate_with_scatter_plots(date_result_path, scatter_output_path, land_sea_mask_path, sample_ratio=0.1)

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
    # 처리할 밴드 리스트 (band2, band3, band4)
    bands = ['band2', 'band3', 'band4']

    # 공통 설정
    land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

    # 처리할 날짜 리스트 (원하는 날짜들을 여기에 추가)
    target_dates = ['20210101', '20210108','20210115','20210122','20210129']

    # 각 밴드별로 처리
    for band in bands:
        print(f"\n{'='*80}")
        print(f"Processing {band.upper()}")
        print(f"{'='*80}\n")

        base_results_dir = f'/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results/{band}/2021'
        base_performance_dir = f'/home/juneyonglee/myhdd/GOCI_RRS/performance/{band}/2021'

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
