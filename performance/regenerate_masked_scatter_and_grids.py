#!/usr/bin/env python3
"""
Regenerate scatter plots and combined grids using gt_daily_masked.png and
recon_daily_clean.png for the GOCI daily performance outputs.
"""

import os
import re
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt


PERFORMANCE_ROOT = Path('/home/juneyonglee/Desktop/AY_ust/My_Book1/GOCI_5years/performance')
DAILY_RESULTS_ROOT = Path('/home/juneyonglee/Desktop/AY_ust/My_Book1/GOCI_5years/daily_results')
LAND_MASK_NPY = Path('/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy')

BANDS = ('band2_daily', 'band3_daily', 'band4_daily')
SCATTER_FILENAME = 'ocean_validation_daily_top99_metrics95_rrs_parity_plot.png'
SAMPLE_RATIO = 0.1
MAX_WORKERS = max(1, min(6, os.cpu_count() or 1))
OCEAN_MASK = np.where(np.load(LAND_MASK_NPY) == 999, 0, 1).astype(bool)


def natural_sort_key(text: str):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r'([0-9]+)', text)]


def filter_top_percent_data(gt_data: np.ndarray,
                            recon_data: np.ndarray,
                            top_percent: float) -> Tuple[np.ndarray, np.ndarray]:
    if gt_data.size == 0:
        return np.array([]), np.array([])

    abs_errors = np.abs(recon_data - gt_data)
    count = max(1, int(len(abs_errors) * top_percent))
    indices = np.argsort(abs_errors)[:count]
    return gt_data[indices], recon_data[indices]


def compute_r2_score(gt_data: np.ndarray, recon_data: np.ndarray) -> float:
    if gt_data.size == 0:
        return float('nan')

    gt_mean = np.mean(gt_data)
    ss_total = np.sum((gt_data - gt_mean) ** 2)
    if ss_total == 0:
        return float('nan')

    ss_residual = np.sum((gt_data - recon_data) ** 2)
    return 1.0 - (ss_residual / ss_total)


def plot_parity(scatter_dir: Path,
                gt_data: np.ndarray,
                recon_data: np.ndarray,
                metrics: Dict[str, float],
                title: str) -> Path:
    scatter_dir.mkdir(parents=True, exist_ok=True)
    output_path = scatter_dir / SCATTER_FILENAME

    if gt_data.size == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, 'No valid\ndata', ha='center', va='center', fontsize=18, color='gray')
        ax.axis('off')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path

    vmin = float(min(np.min(gt_data), np.min(recon_data)))
    vmax = float(max(np.max(gt_data), np.max(recon_data)))
    if vmin == vmax:
        vmax = vmin + 1e-6

    display_gt = gt_data
    display_recon = recon_data
    max_display_points = 200000
    if len(gt_data) > max_display_points:
        rng = np.random.default_rng(42)
        display_indices = rng.choice(len(gt_data), size=max_display_points, replace=False)
        display_gt = gt_data[display_indices]
        display_recon = recon_data[display_indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    if len(display_gt) > 100000:
        scatter_kws = {'s': 1, 'alpha': 0.1, 'c': 'blue', 'rasterized': True}
    elif len(display_gt) > 50000:
        scatter_kws = {'s': 1, 'alpha': 0.15, 'c': 'blue', 'rasterized': True}
    elif len(display_gt) > 10000:
        scatter_kws = {'s': 1, 'alpha': 0.3, 'c': 'blue', 'rasterized': True}
    else:
        scatter_kws = {'s': 2, 'alpha': 0.5, 'c': 'blue', 'rasterized': True}

    ax.scatter(display_gt, display_recon, **scatter_kws)
    ax.plot([vmin, vmax], [vmin, vmax], c='k', alpha=0.3, linewidth=2)
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    ticks = np.linspace(vmin, vmax, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    data_range = vmax - vmin
    if data_range > 1:
        labels = [f'{tick:.1f}' for tick in ticks]
    elif data_range < 0.01:
        labels = [f'{tick:.2e}' for tick in ticks]
    else:
        labels = [f'{tick:.3f}' for tick in ticks]
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Ground Truth RRS', color='gray', fontsize=20, labelpad=8)
    ax.set_ylabel('Reconstructed RRS', color='gray', fontsize=20, labelpad=8)
    ax.set_title(title, color='gray', fontsize=20, fontweight='bold', pad=16)
    ax.text(0.98, 0.3, f"RMSE = {metrics['rmse']:.8f}", transform=ax.transAxes,
            ha='right', color='k', fontsize=14)
    ax.text(0.98, 0.2, f"MAE = {metrics['mae']:.8f}", transform=ax.transAxes,
            ha='right', color='k', fontsize=14)
    ax.text(0.98, 0.1, f"R2 = {metrics['r2']:.3f}", transform=ax.transAxes,
            ha='right', color='k', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert('RGB'))


def build_scene_valid_mask(gt_masked_path: Path,
                           recon_clean_path: Path,
                           ocean_mask: np.ndarray) -> np.ndarray:
    gt_rgb = load_rgb_image(gt_masked_path)
    recon_rgb = load_rgb_image(recon_clean_path)

    if gt_rgb.shape[:2] != ocean_mask.shape or recon_rgb.shape[:2] != ocean_mask.shape:
        raise ValueError(
            f'Image/mask shape mismatch: gt={gt_rgb.shape[:2]}, recon={recon_rgb.shape[:2]}, '
            f'ocean_mask={ocean_mask.shape}'
        )

    gt_white = np.all(gt_rgb == 255, axis=2)
    recon_white = np.all(recon_rgb == 255, axis=2)
    gt_black = np.all(gt_rgb == 0, axis=2)
    recon_black = np.all(recon_rgb == 0, axis=2)

    return ocean_mask & (~gt_white) & (~recon_white) & (~gt_black) & (~recon_black)


def parse_patch_coords(filename: str) -> Optional[Tuple[int, int]]:
    match = re.search(r'y(\d+)_x(\d+)', filename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def crop_mask(mask: np.ndarray, y0: int, x0: int, shape: Tuple[int, int]) -> np.ndarray:
    patch_h, patch_w = shape
    if y0 >= mask.shape[0] or x0 >= mask.shape[1]:
        return np.zeros(shape, dtype=bool)

    y1 = min(y0 + patch_h, mask.shape[0])
    x1 = min(x0 + patch_w, mask.shape[1])
    patch_mask = np.zeros(shape, dtype=bool)
    patch_mask[:y1 - y0, :x1 - x0] = mask[y0:y1, x0:x1]
    return patch_mask


def collect_scatter_data(result_date_dir: Path,
                         scene_valid_mask: np.ndarray,
                         ocean_mask: np.ndarray,
                         sample_ratio: float,
                         rng_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    gt_dir = result_date_dir / 'degree' / 'gt'
    recon_dir = result_date_dir / 'degree' / 'recon'
    gt_files = sorted(gt_dir.glob('*.csv'), key=lambda path: natural_sort_key(path.name))
    recon_files = sorted(recon_dir.glob('*.csv'), key=lambda path: natural_sort_key(path.name))

    if not gt_files or len(gt_files) != len(recon_files):
        return np.array([]), np.array([])

    rng = np.random.default_rng(rng_seed)
    sample_size = max(1, int(len(gt_files) * sample_ratio))
    sample_indices = rng.choice(len(gt_files), size=sample_size, replace=False)

    collected_gt: List[np.ndarray] = []
    collected_recon: List[np.ndarray] = []

    for idx in sample_indices:
        gt_file = gt_files[idx]
        recon_file = recon_files[idx]
        coords = parse_patch_coords(gt_file.name)
        if coords is None:
            continue

        try:
            gt_patch = np.loadtxt(gt_file, delimiter=',', dtype=np.float32)
            recon_patch = np.loadtxt(recon_file, delimiter=',', dtype=np.float32)
        except Exception:
            continue

        if gt_patch.ndim == 1:
            gt_patch = gt_patch[np.newaxis, :]
        if recon_patch.ndim == 1:
            recon_patch = recon_patch[np.newaxis, :]

        if gt_patch.shape != recon_patch.shape or gt_patch.ndim != 2:
            continue

        y0, x0 = coords
        patch_shape = gt_patch.shape
        patch_scene_mask = crop_mask(scene_valid_mask, y0, x0, patch_shape)
        patch_ocean_mask = crop_mask(ocean_mask, y0, x0, patch_shape)

        raw_valid_mask = (
            (gt_patch != -999) & (recon_patch != -999) &
            (gt_patch != 255) & (recon_patch != 255) &
            (gt_patch != 1000) &
            (~np.isnan(gt_patch)) & (~np.isnan(recon_patch)) &
            (~np.isinf(gt_patch)) & (~np.isinf(recon_patch))
        )

        valid_mask = patch_scene_mask & patch_ocean_mask & raw_valid_mask
        if np.sum(valid_mask) <= 10:
            continue

        collected_gt.append(gt_patch[valid_mask])
        collected_recon.append(recon_patch[valid_mask])

    if not collected_gt:
        return np.array([]), np.array([])

    return np.concatenate(collected_gt), np.concatenate(collected_recon)


def create_combined_grid_from_pngs(gt_path: Path,
                                   mask_path: Path,
                                   recon_path: Path,
                                   scatter_path: Path,
                                   output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = [gt_path, mask_path, recon_path, scatter_path]
    titles = ['Input Image', 'Mask', 'Reconstruction', 'Validation']

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    for axis, image_path, title in zip(axes, image_paths, titles):
        if image_path.exists():
            axis.imshow(plt.imread(image_path))
        else:
            axis.text(0.5, 0.5, 'Image\nNot Available', ha='center', va='center',
                      fontsize=16, color='gray')
        axis.set_title(title, fontsize=18, fontweight='bold')
        axis.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_date(band_name: str,
                 date_dir: Path,
                 sample_ratio: float) -> Dict[str, str]:
    date = date_dir.name
    result_date_dir = DAILY_RESULTS_ROOT / band_name / '2021' / date

    gt_masked_path = date_dir / 'gt_daily_masked.png'
    recon_clean_path = date_dir / 'recon_daily_clean.png'
    mask_path = date_dir / 'mask_daily.png'
    scatter_dir = date_dir / 'scatter_plots'
    scatter_path = scatter_dir / SCATTER_FILENAME
    grid_path = date_dir / 'combined_grids' / 'combined_grid_daily.png'

    scene_valid_mask = build_scene_valid_mask(gt_masked_path, recon_clean_path, OCEAN_MASK)
    seed = zlib.crc32(f'{band_name}:{date}'.encode()) & 0xffffffff
    gt_values, recon_values = collect_scatter_data(
        result_date_dir,
        scene_valid_mask,
        OCEAN_MASK,
        sample_ratio,
        seed,
    )

    gt_95, recon_95 = filter_top_percent_data(gt_values, recon_values, 0.95)
    gt_99, recon_99 = filter_top_percent_data(gt_values, recon_values, 0.99)

    if gt_95.size == 0 or gt_99.size == 0:
        metrics = {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    else:
        diff_95 = recon_95 - gt_95
        metrics = {
            'rmse': float(np.sqrt(np.mean(diff_95 ** 2))),
            'mae': float(np.mean(np.abs(diff_95))),
            'r2': float(compute_r2_score(gt_95, recon_95)),
        }

    scatter_output = plot_parity(
        scatter_dir,
        gt_99,
        recon_99,
        metrics,
        title=f'{band_name} Daily Averaged RRS Validation',
    )
    create_combined_grid_from_pngs(gt_masked_path, mask_path, recon_clean_path, scatter_output, grid_path)

    return {
        'band': band_name,
        'date': date,
        'scatter': str(scatter_output),
        'grid': str(grid_path),
        'points': str(int(gt_values.size)),
    }


def process_task(task: Tuple[str, str]) -> Dict[str, str]:
    band_name, date_dir_str = task
    return process_date(band_name, Path(date_dir_str), SAMPLE_RATIO)


def discover_dates(band_name: str) -> List[Path]:
    band_perf_dir = PERFORMANCE_ROOT / band_name / '2021_daily'
    band_result_dir = DAILY_RESULTS_ROOT / band_name / '2021'
    if not band_perf_dir.exists() or not band_result_dir.exists():
        return []

    dates = []
    for date_dir in sorted(band_perf_dir.glob('20*')):
        if not date_dir.is_dir():
            continue

        required_files = [
            date_dir / 'gt_daily_masked.png',
            date_dir / 'recon_daily_clean.png',
            date_dir / 'mask_daily.png',
            band_result_dir / date_dir.name / 'degree' / 'gt',
            band_result_dir / date_dir.name / 'degree' / 'recon',
        ]
        if all(path.exists() for path in required_files):
            dates.append(date_dir)

    return dates


def main() -> int:
    tasks: List[Tuple[str, str]] = []
    failures: List[str] = []

    for band_name in BANDS:
        date_dirs = discover_dates(band_name)
        print(f'=== {band_name}: {len(date_dirs)} dates ===', flush=True)
        tasks.extend((band_name, str(date_dir)) for date_dir in date_dirs)

    total_processed = 0
    max_workers = min(MAX_WORKERS, len(tasks))
    print(f'\nRunning with {max_workers} workers', flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}

        for index, future in enumerate(as_completed(future_to_task), start=1):
            band_name, date_dir_str = future_to_task[future]
            date = Path(date_dir_str).name
            print(f'[{index}/{len(tasks)}] {band_name}/{date}', flush=True)
            try:
                result = future.result()
                print(f"  scatter -> {result['scatter']}", flush=True)
                print(f"  grid    -> {result['grid']}", flush=True)
                print(f"  points  -> {result['points']}", flush=True)
                total_processed += 1
            except Exception as exc:
                failures.append(f'{band_name}/{date}: {exc}')
                print(f'  ERROR   -> {exc}', flush=True)

    print(f'\nProcessed dates: {total_processed}', flush=True)
    if failures:
        print('Failures:', flush=True)
        for failure in failures:
            print(f'  - {failure}', flush=True)
        return 1

    print('All scatter plots and combined grids regenerated successfully.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
