import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.metrics import r2_score as r2_
from matplotlib.colors import Normalize
import re
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc  # 가비지 컬렉션 추가
import psutil  # 메모리 모니터링 추가

def get_memory_usage():
    """현재 메모리 사용량 반환 (GB)"""
    return psutil.Process().memory_info().rss / 1024 / 1024 / 1024

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def preprocess_rrs_data_consistent(data, missing_value=-999, valid_min=-1000, valid_max=1000):
    """
    훈련 코드와 일치하는 RRS 데이터 전처리
    """
    processed_data = data.copy()

    # 1. 극값 제거
    extreme_mask = (processed_data < -2000) | (processed_data > 2000)
    processed_data[extreme_mask] = missing_value

    # 2. 유효한 픽셀 마스크 생성
    valid_mask = (processed_data != missing_value) & \
                 (processed_data >= valid_min) & \
                 (processed_data <= valid_max) & \
                 (~np.isnan(processed_data))

    return processed_data, valid_mask

def filter_rrs_data_basic(gt_data, pred_data, coordinates=None):
    """
    RRS 검증용 데이터 필터링 (육지값만 제거, 유효한 해양 데이터는 모두 유지)
    """
    print(f"Initial data: GT={len(gt_data):,}, Pred={len(pred_data):,}")

    # 1. 기본 필터링: NaN, Inf 제거
    valid_mask = (~np.isnan(gt_data)) & (~np.isnan(pred_data)) & \
                 (~np.isinf(gt_data)) & (~np.isinf(pred_data))
    print(f"After NaN/Inf removal: {np.sum(valid_mask):,} ({np.sum(valid_mask)/len(gt_data)*100:.1f}%)")

    # 2. 기본 특수값 제거 (255는 육지 마킹, -999는 결측값)
    valid_mask = valid_mask & (gt_data != 255) & (pred_data != 255)
    valid_mask = valid_mask & (gt_data != -999) & (pred_data != -999)
    print(f"After special values removal: {np.sum(valid_mask):,} ({np.sum(valid_mask)/len(gt_data)*100:.1f}%)")

    filtered_gt = gt_data[valid_mask]
    filtered_pred = pred_data[valid_mask]

    if coordinates is not None:
        filtered_coords = coordinates[valid_mask]
        print(f"Final filtered data: {len(filtered_gt):,}")
        return filtered_gt, filtered_pred, filtered_coords

    print(f"Final filtered data: {len(filtered_gt):,}")
    return filtered_gt, filtered_pred

def process_single_file_memory_efficient(args):
    """
    메모리 효율적인 단일 파일 처리 함수 - 통계만 수집
    """
    recon_file, gt_file, mask_file, land_sea_mask_path, collect_data = args

    try:
        recon_file_name = os.path.basename(recon_file)

        # 육지-해양 마스크를 프로세스 내에서 로드 (메모리 절약)
        try:
            land_sea_mask_full = np.load(land_sea_mask_path)
            # 올바른 해석: 1=육지, 999=해양
            # 육지(1)는 0으로, 해양(999)는 1로 변환
            land_sea_mask_full = np.where(land_sea_mask_full == 999, 0, 1).astype(np.uint8)
        except:
            return None

        # 데이터 로드
        restored_np = np.loadtxt(recon_file, delimiter=',', dtype='float32')
        mask_np = np.loadtxt(mask_file, delimiter=',', dtype='float32')
        gt_np = np.loadtxt(gt_file, delimiter=',', dtype='float32')

        # 좌표 추출
        match = re.search(r'r(\d+)_c(\d+)', recon_file_name)
        if not match:
            return None
        row, col = int(match.group(1)), int(match.group(2))

        # 육지-해양 마스크 패치
        try:
            land_mask_patch = land_sea_mask_full[row:row + 256, col:col + 256]
        except IndexError:
            return None

        # 결과 저장용 (collect_data가 True일 때만)
        file_gt = []
        file_pred = []
        file_coordinates = []

        # 파일별 통계
        file_stats = {
            'total_pixels': 0,
            'land_pixels': 0,
            'ocean_pixels': 0,
            'masked_pixels': 0,
            'valid_pixels': 0,
            'missing_999': 0,
            'special_255': 0,
            'nan_inf': 0,
            'extreme_values': 0
        }

        # 픽셀별 검사 및 데이터 수집
        W, H = gt_np.shape
        for w in range(W):
            for h in range(H):
                file_stats['total_pixels'] += 1

                # 육지 체크 (값이 0이면 육지)
                if land_mask_patch[w, h] == 0:
                    file_stats['land_pixels'] += 1
                    continue

                file_stats['ocean_pixels'] += 1

                # 모든 해양 픽셀 검사 (마스킹 조건 제거)
                gt_val = gt_np[w, h]
                pred_val = restored_np[w, h]

                # 마스킹 정보 기록 (통계용)
                if mask_np[w, h] == 0:
                    file_stats['masked_pixels'] += 1

                # 모든 해양 픽셀에 대한 상세 필터링 및 통계
                if gt_val == -999 or pred_val == -999:
                    file_stats['missing_999'] += 1
                    continue

                if gt_val == 255 or pred_val == 255:
                    file_stats['special_255'] += 1
                    continue


                if (np.isnan(gt_val) or np.isnan(pred_val) or
                    np.isinf(gt_val) or np.isinf(pred_val)):
                    file_stats['nan_inf'] += 1
                    continue

                # 극값 필터링을 더 관대하게 (극도로 큰 값만 제거)
                if abs(gt_val) > 1000 or abs(pred_val) > 1000:
                    file_stats['extreme_values'] += 1
                    continue

                file_stats['valid_pixels'] += 1

                # 데이터 수집 여부에 따라
                if collect_data:
                    file_gt.append(gt_val)
                    file_pred.append(pred_val)
                    file_coordinates.append((row + w, col + h))

        # 메모리 정리
        del land_sea_mask_full, restored_np, mask_np, gt_np, land_mask_patch
        gc.collect()

        result = {
            'file_name': recon_file_name,
            'stats': file_stats
        }

        if collect_data:
            result.update({
                'gt_data': np.array(file_gt),
                'pred_data': np.array(file_pred),
                'coordinates': np.array(file_coordinates),
            })

        return result

    except Exception as e:
        print(f"⚠️ Error processing {recon_file}: {e}")
        return None

def calculate_rrs_metrics_improved(gt_data, pred_data):
    """
    RRS에 특화된 개선된 지표 계산
    """
    if len(gt_data) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan, 'bias': np.nan, 'count': 0}

    # RRS용 기본 지표 (마이너스 값 고려)
    diff = pred_data - gt_data
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)  # 양수면 과대예측, 음수면 과소예측

    # RRS 스케일에 맞는 상대 오차 (Relative Error)
    relative_rmse = rmse / (np.nanmax(gt_data) - np.nanmin(gt_data)) * 100

    try:
        r2 = r2_(gt_data, pred_data)
    except:
        r2 = np.nan

    # RRS 마이너스 값을 고려한 개선된 MAPE 계산
    # RRS가 마이너스 값이므로 절댓값 기준으로 유의미한 값만 선택
    abs_gt = np.abs(gt_data)

    # RRS 특성상 0에 가까운 값들은 해양의 특정 상태를 나타내므로 제외하지 않음
    # 대신 매우 작은 절댓값 (< 0.001)만 제외
    significant_mask = abs_gt >= 0.001

    if np.sum(significant_mask) > 0:
        # RRS 마이너스 값 처리: 절댓값으로 MAPE 계산
        mape = np.mean(np.abs((gt_data[significant_mask] - pred_data[significant_mask]) /
                             gt_data[significant_mask])) * 100
        mape_count = np.sum(significant_mask)
    else:
        mape = np.nan
        mape_count = 0

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'bias': bias,
        'relative_rmse': relative_rmse,
        'count': len(gt_data),
        'mape_count': mape_count,
        'data_range': np.nanmax(gt_data) - np.nanmin(gt_data)
    }

def determine_plot_range_improved(gt_data, pred_data, method='robust'):
    """
    개선된 플롯 범위 결정
    """
    combined_data = np.concatenate([gt_data, pred_data])

    if method == 'robust':
        # IQR 기반 robust 방법
        q1, q3 = np.percentile(combined_data, [25, 75])
        iqr = q3 - q1
        vmin = q1 - 1.5 * iqr
        vmax = q3 + 1.5 * iqr

        # 실제 데이터 범위로 제한
        vmin = max(vmin, combined_data.min())
        vmax = min(vmax, combined_data.max())

    elif method == 'percentile':
        vmin = np.percentile(combined_data, 2.5)
        vmax = np.percentile(combined_data, 97.5)
    else:
        vmin = combined_data.min()
        vmax = combined_data.max()

    # 범위가 너무 작으면 확장
    if (vmax - vmin) < 1e-6:
        center = (vmax + vmin) / 2
        vmin = center - 1e-3
        vmax = center + 1e-3

    return vmin, vmax

def plot_parity_improved(filename, loss_rate, true, pred, metrics_dict, vmin, vmax,
                        xlabel="True RRS", ylabel="Predicted RRS",
                        title="RRS Validation Results", figsize=(8, 8), save_file=True,
                        group_info=None):

    fig, ax = plt.subplots(figsize=figsize)

    # UST21 스타일 scatter plot (val_chl_new.py 참조)
    # 데이터 크기에 따라 점 크기와 투명도 조절
    if len(true) > 100000:
        # 매우 많은 데이터: 작은 점, 낮은 투명도
        scatter_kws = {'s': 1, 'alpha': 0.01}
    elif len(true) > 50000:
        # 많은 데이터: 작은 점, 중간 투명도
        scatter_kws = {'s': 1, 'alpha': 0.05}
    elif len(true) > 10000:
        # 중간 크기: 중간 점, 중간 투명도
        scatter_kws = {'s': 1, 'alpha': 0.1}
    else:
        # 작은 데이터: 큰 점, 높은 투명도
        scatter_kws = {'s': 2, 'alpha': 0.3}

    ax.scatter(true, pred, **scatter_kws)

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

    # R² 계산
    try:
        r2 = r2_(true, pred)
    except:
        r2 = metrics_dict['r2']

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

    # 추가 RRS 지표들은 상단에 표시
    ax.text(0.02, 0.98, f"N = {metrics_dict['count']:,}",
            transform=ax.transAxes, fontdict=font_metrics, ha="left", va="top")
    ax.text(0.02, 0.93, f"Bias = {metrics_dict['bias']:.6f}",
            transform=ax.transAxes, fontdict=font_metrics, ha="left", va="top")
    if 'relative_rmse' in metrics_dict:
        ax.text(0.02, 0.88, f"Rel.RMSE = {metrics_dict['relative_rmse']:.2f}%",
                transform=ax.transAxes, fontdict=font_metrics, ha="left", va="top")

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
    gc.collect()
    return None

def process_files_in_batches(file_args, land_sea_mask_path, n_processes=4, batch_size=1000):
    """
    배치 단위로 파일을 처리하여 메모리 사용량 제어
    """
    total_stats = {
        'total_pixels': 0,
        'land_pixels': 0,
        'ocean_pixels': 0,
        'masked_pixels': 0,
        'valid_pixels': 0,
        'missing_999': 0,
        'special_255': 0,
        'nan_inf': 0,
        'extreme_values': 0
    }

    print(f"Processing {len(file_args)} files in batches of {batch_size}")

    # 통계 수집을 위한 첫 번째 패스
    with tqdm(total=len(file_args), desc="Collecting statistics") as pbar:
        for i in range(0, len(file_args), batch_size):
            batch_args = [(f[0], f[1], f[2], land_sea_mask_path, False)
                         for f in file_args[i:i+batch_size]]

            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                future_to_file = {executor.submit(process_single_file_memory_efficient, args): args
                                 for args in batch_args}

                for future in as_completed(future_to_file):
                    result = future.result()
                    if result is not None:
                        for key in total_stats:
                            total_stats[key] += result['stats'][key]
                    pbar.update(1)

            # 배치 후 메모리 정리
            gc.collect()

            current_memory = get_memory_usage()
            print(f"Batch {i//batch_size + 1} completed. Memory usage: {current_memory:.1f} GB")

    return total_stats

def process_sample_for_plotting(file_args, land_sea_mask_path, sample_ratio=0.1, n_processes=4):
    """
    플롯팅을 위한 샘플 데이터 수집
    """
    # 샘플링
    sample_size = max(1, int(len(file_args) * sample_ratio))
    sample_indices = np.random.choice(len(file_args), size=sample_size, replace=False)
    sample_args = [(file_args[i][0], file_args[i][1], file_args[i][2], land_sea_mask_path, True)
                   for i in sample_indices]

    print(f"Processing sample of {len(sample_args)} files for plotting")

    all_gt = []
    all_pred = []
    all_coordinates = []

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        future_to_file = {executor.submit(process_single_file_memory_efficient, args): args
                         for args in sample_args}

        with tqdm(total=len(sample_args), desc="Collecting sample data") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None and 'gt_data' in result:
                    if len(result['gt_data']) > 0:
                        all_gt.extend(result['gt_data'])
                        all_pred.extend(result['pred_data'])
                        all_coordinates.extend(result['coordinates'])
                pbar.update(1)

    return np.array(all_gt), np.array(all_pred), np.array(all_coordinates)

def divide_data_by_percentiles(gt_data, pred_data, coordinates, n_groups=10):
    """
    데이터를 백분위수로 나누는 함수
    """
    # GT 데이터를 기준으로 백분위수 계산
    percentiles = np.linspace(0, 100, n_groups + 1)
    percentile_values = np.percentile(gt_data, percentiles)

    groups = []
    group_info = []

    for i in range(n_groups):
        if i == 0:
            # 첫 번째 그룹: 최솟값 이상
            mask = gt_data >= percentile_values[i]
        else:
            # 나머지 그룹: 이전 백분위수 초과
            mask = gt_data > percentile_values[i]

        if i < n_groups - 1:
            # 마지막 그룹이 아니면: 다음 백분위수 이하
            mask = mask & (gt_data <= percentile_values[i + 1])

        group_gt = gt_data[mask]
        group_pred = pred_data[mask]
        group_coords = coordinates[mask]

        groups.append((group_gt, group_pred, group_coords))
        group_info.append({
            'percentile_range': f"{percentiles[i]:.0f}-{percentiles[i+1]:.0f}%",
            'value_range': f"[{percentile_values[i]:.3f}, {percentile_values[i+1]:.3f}]",
            'count': len(group_gt)
        })

        print(f"Group {i+1}: {percentiles[i]:.0f}-{percentiles[i+1]:.0f}% "
              f"({percentile_values[i]:.3f} to {percentile_values[i+1]:.3f}), "
              f"N={len(group_gt):,}")

    return groups, group_info


def create_colormap_plots_efficient(filename, loss_rate, gt_data, pred_data,
                                    coordinates, title_prefix="RRS", group_info=None,
                                    land_sea_mask_path=None):
    """
    Create efficient colormap plots for spatial visualization of GT vs Predicted data
    Only shows ocean areas (land areas are masked out)
    """
    try:
        if len(gt_data) == 0 or len(coordinates) == 0:
            print(f"⚠️ No data to plot colormap for {loss_rate}")
            return

        print(f"Creating colormap plots for {len(gt_data):,} pixels")

        # Load land-sea mask if not provided
        if land_sea_mask_path is None:
            land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

        # Load land-sea mask
        try:
            land_sea_mask_full = np.load(land_sea_mask_path)
            # 육지(999가 아닌 값)는 0으로, 바다(999)는 1로 변환
            land_sea_mask_full = np.where(land_sea_mask_full == 999, 0, 1).astype(np.uint8)
        except Exception as e:
            print(f"⚠️ Could not load land-sea mask: {e}")
            land_sea_mask_full = None

        # Extract row and column coordinates
        rows = coordinates[:, 0]
        cols = coordinates[:, 1]

        # Determine grid size
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())

        # Create sparse grids for GT and Pred
        grid_height = max_row - min_row + 1
        grid_width = max_col - min_col + 1

        # Initialize grids with NaN
        gt_grid = np.full((grid_height, grid_width), np.nan)
        pred_grid = np.full((grid_height, grid_width), np.nan)

        # Fill grids with data (only ocean pixels)
        for i, (row, col) in enumerate(coordinates):
            grid_row = int(row) - min_row
            grid_col = int(col) - min_col

            # Double-check if this pixel is ocean before adding to grid
            if land_sea_mask_full is not None:
                try:
                    if land_sea_mask_full[int(row), int(col)] == 0:  # 육지이면 skip
                        continue
                except IndexError:
                    continue

            # GT=0.0은 유효한 해양 데이터이므로 제거하지 않음
            gt_val = gt_data[i]
            pred_val = pred_data[i]

            gt_grid[grid_row, grid_col] = gt_val
            pred_grid[grid_row, grid_col] = pred_val

        # Explicitly mask out any remaining land areas on the final grid
        if land_sea_mask_full is not None:
            try:
                land_mask_view = land_sea_mask_full[min_row:max_row + 1, min_col:max_col + 1]
                # 육지 지역(값이 99)을 NaN으로 설정 - 1=해양, 999=육지
                gt_grid[land_mask_view == 999] = np.nan
                pred_grid[land_mask_view == 999] = np.nan
                print(f"Applied land mask: {np.sum(land_mask_view == 1)} land pixels masked out")
            except Exception as e:
                print(f"⚠️ Could not apply land mask to grid: {e}")




        print(f"Final ocean pixels in grid: GT={np.sum(~np.isnan(gt_grid))}, Pred={np.sum(~np.isnan(pred_grid))}")

        # Determine color range using combined GT and Pred data for consistency
        combined_data = np.concatenate([gt_data, pred_data])
        vmin = np.nanmin(combined_data)
        vmax = np.nanmax(combined_data)
        print(f"Color scale range: [{vmin:.6f}, {vmax:.6f}]")

        # Create masked arrays to handle NaN values properly (land areas)
        gt_masked = np.ma.masked_invalid(gt_grid)
        pred_masked = np.ma.masked_invalid(pred_grid)
        diff_grid = pred_grid - gt_grid
        diff_masked = np.ma.masked_invalid(diff_grid)

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # GT colormap with masked array (land areas will be transparent)
        im1 = ax1.imshow(gt_masked, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax1.set_title(f'{title_prefix} - Ground Truth', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # Predicted colormap with masked array (land areas will be transparent)
        im2 = ax2.imshow(pred_masked, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax2.set_title(f'{title_prefix} - Predicted', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        # Difference map with percentile-based scaling to avoid extreme outliers
        diff_values = diff_grid[~np.isnan(diff_grid)]
        if len(diff_values) > 0:
            # Use percentile-based range to avoid extreme outliers
            diff_p5 = np.percentile(diff_values, 5)
            diff_p95 = np.percentile(diff_values, 95)
            diff_max = max(abs(diff_p5), abs(diff_p95))
        else:
            diff_max = 1.0

        print(f"Difference map range: [{-diff_max:.6f}, {diff_max:.6f}]")
        print(f"Actual difference range: [{np.nanmin(diff_values):.6f}, {np.nanmax(diff_values):.6f}]")

        im3 = ax3.imshow(diff_masked, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, aspect='auto')
        ax3.set_title(f'{title_prefix} - Difference (Pred - GT)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Column')
        ax3.set_ylabel('Row')
        plt.colorbar(im3, ax=ax3, shrink=0.8)

        # Add group info to main title if available
        if group_info:
            fig.suptitle(f'{title_prefix} - {group_info["percentile_range"]} ({group_info["value_range"]})',
                        fontsize=14, fontweight='bold')
            save_suffix = f'percentile_{group_info["percentile_range"].replace("-", "_").replace("%", "")}'
        else:
            fig.suptitle(f'{title_prefix} - Spatial Distribution', fontsize=14, fontweight='bold')
            save_suffix = 'overall'

        # Save plot
        os.makedirs(filename, exist_ok=True)
        save_path = os.path.join(filename, f'{loss_rate}_{save_suffix}_colormap.png')
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Colormap saved: {save_path}")

        plt.close(fig)

        # Memory cleanup
        del gt_grid, pred_grid, diff_grid, gt_masked, pred_masked, diff_masked, fig
        if land_sea_mask_full is not None:
            del land_sea_mask_full
        gc.collect()

    except Exception as e:
        print(f"❌ Error creating colormap for {loss_rate}: {e}")
        if 'fig' in locals():
            plt.close(fig)
        gc.collect()

def validate(loss_rate, data_path, save_path, land_sea_mask_path=None,
                             sample_size=None, create_colormaps=False, n_processes=None):
    """
    메모리 효율적인 RRS 검증 함수
    """
    if n_processes is None:
        n_processes = min(mp.cpu_count()//2, 4)

    if land_sea_mask_path is None:
        land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

    print(f"=== Memory-Efficient RRS Validation ===")
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")
    print(f"Sample size: {sample_size if sample_size else 'All'}")
    print(f"Processes: {n_processes}")
    print(f"Initial memory usage: {get_memory_usage():.1f} GB")

    # 경로 설정
    recon_path = os.path.join(data_path, 'recon')
    gt_path = os.path.join(data_path, 'gt')
    mask_path = os.path.join(data_path, 'mask')

    # 경로 확인
    for path_name, path in [("recon", recon_path), ("gt", gt_path), ("mask", mask_path)]:
        if not os.path.isdir(path):
            print(f"❌ Missing directory: {path_name}")
            return

    os.makedirs(save_path, exist_ok=True)

    # 파일 리스트
    recon_files = sorted(glob.glob(os.path.join(recon_path, '*.csv')), key=natural_sort_key)
    gt_files = sorted(glob.glob(os.path.join(gt_path, '*.csv')), key=natural_sort_key)
    mask_files = sorted(glob.glob(os.path.join(mask_path, '*.csv')), key=natural_sort_key)

    if len(recon_files) == 0:
        print("❌ No CSV files found")
        return

    print(f"Found {len(recon_files)} files")

    if sample_size is not None:
        recon_files = recon_files[:sample_size]
        gt_files = gt_files[:sample_size]
        mask_files = mask_files[:sample_size]
        print(f"Using {len(recon_files)} files")

    # 파일 인자 준비
    file_args = [(recon_files[i], gt_files[i], mask_files[i])
                 for i in range(len(recon_files))]

    # 1단계: 배치 단위로 통계 수집
    print(f"\n=== Step 1: Collecting Statistics ===")
    start_time = time.time()

    total_stats = process_files_in_batches(
        file_args, land_sea_mask_path, n_processes, batch_size=1000
    )

    print(f"Statistics collection completed in {time.time() - start_time:.1f} seconds")
    print(f"Memory usage after statistics: {get_memory_usage():.1f} GB")

    # 통계 출력
    print(f"\n=== Detailed Processing Statistics ===")
    total = total_stats['total_pixels']
    for key, val in total_stats.items():
        pct = val/total*100 if total > 0 else 0
        print(f"{key}: {val:,} ({pct:.1f}%)")

    # 2단계: 플롯팅을 위한 샘플 데이터 수집
    print(f"\n=== Step 2: Collecting Sample Data for Plotting ===")

    # 샘플 비율 결정 (메모리에 맞게 조정)
    valid_pixels = total_stats['valid_pixels']
    max_sample_pixels = 50_000_000  # 5천만 픽셀로 제한

    if valid_pixels > max_sample_pixels:
        sample_ratio = max_sample_pixels / valid_pixels
        print(f"Large dataset detected. Using sample ratio: {sample_ratio:.3f}")
    else:
        sample_ratio = 1.0
        print("Dataset size manageable. Using all data.")

    plt_gt, plt_pred, coordinates = process_sample_for_plotting(
        file_args, land_sea_mask_path, sample_ratio, n_processes
    )

    print(f"Sample data collection completed. Memory usage: {get_memory_usage():.1f} GB")

    if len(plt_gt) == 0:
        print("❌ No valid data collected")
        return

    print(f"\n=== Sample Data Analysis ===")
    print(f"Collected {len(plt_gt):,} sample pixels")
    print(f"GT: min={plt_gt.min():.3f}, max={plt_gt.max():.3f}, mean={plt_gt.mean():.3f}")
    print(f"Pred: min={plt_pred.min():.3f}, max={plt_pred.max():.3f}, mean={plt_pred.mean():.3f}")

    # 기본 필터링
    print(f"\n=== Basic Data Filtering Process ===")
    filtered_gt, filtered_pred, filtered_coords = filter_rrs_data_basic(plt_gt, plt_pred, coordinates)

    if len(filtered_gt) == 0:
        print("❌ No data after basic filtering")
        return

    print(f"Memory usage after filtering: {get_memory_usage():.1f} GB")

    # 전체 데이터에 대한 지표 계산
    overall_metrics = calculate_rrs_metrics_improved(filtered_gt, filtered_pred)
    print(f"\n=== Overall RRS Validation Results (Sample) ===")
    print(f"Sample pixels: {overall_metrics['count']:,}")
    print(f"RMSE: {overall_metrics['rmse']:.6f}")
    print(f"MAE: {overall_metrics['mae']:.6f}")
    print(f"R²: {overall_metrics['r2']:.4f}")
    print(f"Bias: {overall_metrics['bias']:.6f}")
    print(f"Relative RMSE: {overall_metrics['relative_rmse']:.2f}%")
    print(f"Data Range: {overall_metrics['data_range']:.6f}")

    # 전체 데이터 플롯
    vmin_overall, vmax_overall = determine_plot_range_improved(filtered_gt, filtered_pred, method='robust')
    print(f"Overall plot range: [{vmin_overall:.3f}, {vmax_overall:.3f}]")

    plot_parity_improved(
        filename=save_path,
        loss_rate=str(loss_rate) + "_overall",
        true=filtered_gt,
        pred=filtered_pred,
        metrics_dict=overall_metrics,
        vmin=vmin_overall,
        vmax=vmax_overall,
        title=f"RRS Validation Results (Sample): {loss_rate}"
    )

    # 전체 컬러맵 생성
    print(f"\n=== Creating Overall Colormap ===")
    create_colormap_plots_efficient(
        filename=save_path,
        loss_rate=str(loss_rate) + "_overall",
        gt_data=filtered_gt,
        pred_data=filtered_pred,
        coordinates=filtered_coords,
        title_prefix=f"RRS {loss_rate} (Overall)",
        group_info=None,
        land_sea_mask_path=land_sea_mask_path
    )

    # 백분위수별로 데이터 나누기
    print(f"\n=== Dividing Data by Percentiles ===")
    groups, group_info = divide_data_by_percentiles(filtered_gt, filtered_pred, filtered_coords, n_groups=10)

    # 백분위수 그룹들을 순차적으로 처리
    print(f"\n=== Creating Percentile Plots ===")

    for i, (group_data, info) in enumerate(zip(groups, group_info)):
        group_gt, group_pred, group_coords = group_data

        if len(group_gt) == 0:
            print(f"⚠️ Group {i+1} is empty, skipping...")
            continue

        try:
            print(f"Processing Group {i+1}: {info['percentile_range']}")

            # 그룹별 지표 계산
            group_metrics = calculate_rrs_metrics_improved(group_gt, group_pred)

            # 그룹별 플롯 범위 결정
            vmin_group, vmax_group = determine_plot_range_improved(group_gt, group_pred, method='robust')

            # 그룹별 플롯 생성
            plot_parity_improved(
                filename=save_path,
                loss_rate=str(loss_rate),
                true=group_gt,
                pred=group_pred,
                metrics_dict=group_metrics,
                vmin=vmin_group,
                vmax=vmax_group,
                title=f"RRS Validation Results (Sample): {loss_rate}",
                group_info=info
            )

            # 그룹별 컬러맵 생성
            create_colormap_plots_efficient(
                filename=save_path,
                loss_rate=str(loss_rate),
                gt_data=group_gt,
                pred_data=group_pred,
                coordinates=group_coords,
                title_prefix=f"RRS {loss_rate}",
                group_info=info,
                land_sea_mask_path=land_sea_mask_path
            )

            # 각 그룹 처리 후 즉시 메모리 정리
            del group_gt, group_pred, group_coords, group_metrics
            gc.collect()

        except Exception as e:
            print(f"❌ Error processing group {i+1}: {e}")
            try:
                del group_gt, group_pred, group_coords
            except:
                pass
            gc.collect()
            continue

    # 전체 데이터 정리
    del filtered_gt, filtered_pred, filtered_coords, plt_gt, plt_pred, coordinates, groups, group_info
    gc.collect()

    total_time = time.time() - start_time
    print(f"\n✅ Validation completed successfully in {total_time:.1f} seconds!")
    print(f"Created 11 parity plots: 1 overall + 10 percentile groups")
    print(f"Results saved to: {save_path}")
    print(f"Final memory usage: {get_memory_usage():.1f} GB")

    # 최종 메모리 정리
    matplotlib.pyplot.close('all')
    gc.collect()

def run_rrs_validation_improved(test_result_path, loss_rate="rrs_test", sample_size=None, n_processes=None):
    """
    메모리 효율적인 RRS 검증 실행 함수
    """
    data_path = os.path.join(test_result_path, 'degree')

    validate(
        loss_rate=loss_rate,
        data_path=data_path,
        save_path=test_result_path,
        sample_size=sample_size,
        create_colormaps=True,  # 컬러맵 생성 활성화
        n_processes=n_processes
    )

def run_quick_test(test_result_path, sample_count=100, loss_rate="quick_test"):
    """
    빠른 테스트를 위한 함수 - 지정된 샘플 수만 처리하여 육지/해양 마스킹 확인

    Args:
        test_result_path: 테스트 결과 경로
        sample_count: 처리할 샘플 수 (기본 100개)
        loss_rate: 테스트 식별자
    """
    print(f"\n🧪 === Quick Test Mode ===")
    print(f"Processing {sample_count} samples for land/ocean masking verification")
    print(f"Test path: {test_result_path}")

    data_path = os.path.join(test_result_path, 'degree')

    # 빠른 테스트를 위한 설정
    validate(
        loss_rate=loss_rate,
        data_path=data_path,
        save_path=test_result_path,
        sample_size=sample_count,  # 샘플 수 제한
        create_colormaps=True,
        n_processes=2  # 테스트용으로 프로세스 수 줄임
    )

    print(f"\n✅ Quick test completed!")
    print(f"Check the results in: {test_result_path}")
    print(f"Look for files starting with '{loss_rate}_'")
    print(f"Verify that land areas are transparent and ocean areas show RRS values.")

def run_from_yaml_config(yaml_path, quick_test=True, sample_count=100):
    """
    YAML 설정 파일에서 경로를 읽어와서 실행
    """
    import yaml
    import os

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    data_path = config['data_path']  # 실제 데이터가 있는 경로
    save_path = config.get('save_path', data_path + '_results')  # 결과 저장 경로

    print(f"Using config from: {yaml_path}")
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")

    # save_path 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)

    if quick_test:
        # run_quick_test_from_paths를 새로 만들어서 데이터와 저장 경로를 분리
        run_quick_test_from_paths(
            data_path=data_path,
            save_path=save_path,
            sample_count=sample_count,
            loss_rate="yaml_quick_test"
        )
    else:
        run_validation_from_paths(
            data_path=data_path,
            save_path=save_path,
            sample_size=config.get('sample_size', None),
            loss_rate="yaml_validation",
            n_processes=4
        )

def run_quick_test_from_paths(data_path, save_path, sample_count=100, loss_rate="quick_test"):
    """
    데이터 경로와 저장 경로를 분리하여 빠른 테스트 실행
    """
    print(f"\n🧪 === Quick Test Mode ===")
    print(f"Processing {sample_count} samples for land/ocean masking verification")
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")

    degree_data_path = os.path.join(data_path, 'degree')

    # 빠른 테스트를 위한 설정
    validate(
        loss_rate=loss_rate,
        data_path=degree_data_path,
        save_path=save_path,
        sample_size=sample_count,  # 샘플 수 제한
        create_colormaps=True,
        n_processes=2  # 테스트용으로 프로세스 수 줄임
    )

    print(f"\n✅ Quick test completed!")
    print(f"Check the results in: {save_path}")
    print(f"Look for files starting with '{loss_rate}_'")
    print(f"Verify that land areas are transparent and ocean areas show RRS values.")

def run_validation_from_paths(data_path, save_path, sample_size=None, loss_rate="validation", n_processes=4):
    """
    데이터 경로와 저장 경로를 분리하여 전체 검증 실행
    """
    degree_data_path = os.path.join(data_path, 'degree')

    validate(
        loss_rate=loss_rate,
        data_path=degree_data_path,
        save_path=save_path,
        sample_size=sample_size,
        create_colormaps=True,
        n_processes=n_processes
    )

# 실행 예시
if __name__ == "__main__":
    import sys

    # 명령행 인수 체크
    if len(sys.argv) > 1 and '--yaml' in sys.argv:
        yaml_idx = sys.argv.index('--yaml') + 1
        if yaml_idx < len(sys.argv):
            yaml_path = sys.argv[yaml_idx]
            quick_test = '--quick' in sys.argv
            run_from_yaml_config(yaml_path, quick_test=quick_test)
        else:
            print("❌ Please provide yaml path after --yaml")
    else:
        # === 빠른 테스트 모드 (100개 샘플) ===
        # 육지/해양 마스킹이 제대로 작동하는지 확인하는 빠른 테스트
        run_quick_test(
            test_result_path="/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/test/band2/20",
            sample_count=100,
            loss_rate="quick_test_100"
        )

        # === 전체 검증 모드 (주석 처리됨) ===
        # 전체 데이터로 완전한 검증을 실행하려면 아래 주석을 해제하세요
        # run_rrs_validation_improved(
        #     test_result_path="/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/test/band2/20",
        #     loss_rate="band2_ocean_only",
        #     sample_size=None,  # 전체 데이터 사용
        #     n_processes=4      # 프로세스 수 줄임
        # )