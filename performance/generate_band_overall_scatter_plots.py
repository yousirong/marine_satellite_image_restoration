#!/usr/bin/env python3
"""
Generate one overall scatter plot per band by aggregating all daily masked/clean
scatter inputs across the 2021 GOCI performance outputs.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import zlib

import numpy as np

from regenerate_masked_scatter_and_grids import (
    BANDS,
    DAILY_RESULTS_ROOT,
    MAX_WORKERS,
    OCEAN_MASK,
    PERFORMANCE_ROOT,
    SAMPLE_RATIO,
    build_scene_valid_mask,
    collect_scatter_data,
    compute_r2_score,
    filter_top_percent_data,
    plot_parity,
)


OUTPUT_DIRNAME = 'overall_scatter_plots'
OUTPUT_FILENAME = 'overall_ocean_validation_daily_top99_metrics95_rrs_parity_plot.png'
METRICS_FILENAME = 'overall_scatter_metrics.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-performance-root',
        type=Path,
        default=PERFORMANCE_ROOT,
        help='Root used to read per-date performance images.',
    )
    parser.add_argument(
        '--output-performance-root',
        type=Path,
        default=PERFORMANCE_ROOT,
        help='Root used to save the overall scatter outputs.',
    )
    parser.add_argument(
        '--min-date-r2',
        type=float,
        default=None,
        help='Only include dates whose per-date overall R2 is at least this threshold.',
    )
    return parser.parse_args()


def discover_dates(performance_root: Path, band_name: str) -> List[Path]:
    band_perf_dir = performance_root / band_name / '2021_daily'
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


def filter_dates_by_min_r2(performance_root: Path,
                           band_name: str,
                           date_dirs: List[Path],
                           min_date_r2: float | None) -> Tuple[List[Path], List[str]]:
    if min_date_r2 is None:
        return date_dirs, []

    summary_path = performance_root / band_name / '2021_daily' / 'overall_metrics_summary.csv'
    if not summary_path.is_file():
        raise RuntimeError(f'missing overall metrics summary: {summary_path}')

    r2_by_date: Dict[str, float] = {}
    with summary_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = row['Date']
            if date == 'Average':
                continue
            r2_by_date[date] = float(row['R2'])

    kept: List[Path] = []
    dropped: List[str] = []
    for date_dir in date_dirs:
        date = date_dir.name
        if date not in r2_by_date:
            raise RuntimeError(f'missing R2 metric for {band_name}/{date} in {summary_path}')
        if r2_by_date[date] >= min_date_r2:
            kept.append(date_dir)
        else:
            dropped.append(date)

    return kept, dropped


def collect_band_date(task: Tuple[str, str]) -> Dict[str, object]:
    band_name, date_dir_str = task
    date_dir = Path(date_dir_str)
    date = date_dir.name
    result_date_dir = DAILY_RESULTS_ROOT / band_name / '2021' / date

    gt_masked_path = date_dir / 'gt_daily_masked.png'
    recon_clean_path = date_dir / 'recon_daily_clean.png'

    scene_valid_mask = build_scene_valid_mask(gt_masked_path, recon_clean_path, OCEAN_MASK)
    gt_values, recon_values = collect_scatter_data(
        result_date_dir,
        scene_valid_mask,
        OCEAN_MASK,
        SAMPLE_RATIO,
        rng_seed=zlib.crc32(f'{band_name}:{date}'.encode()) & 0xffffffff,
    )

    return {
        'band': band_name,
        'date': date,
        'gt': gt_values,
        'recon': recon_values,
    }


def write_metrics_csv(output_dir: Path,
                      point_count: int,
                      metrics: Dict[str, float],
                      filtered_point_count: int,
                      displayed_point_count: int,
                      selected_dates: List[str],
                      dropped_dates: List[str],
                      min_date_r2: float | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / METRICS_FILENAME
    lines = [
        'metric,value',
        f'selected_date_count,{len(selected_dates)}',
        f'dropped_date_count,{len(dropped_dates)}',
        f'raw_point_count,{point_count}',
        f'top95_point_count,{filtered_point_count}',
        f'top99_display_point_count,{displayed_point_count}',
        f'rmse,{metrics["rmse"]:.8f}',
        f'mae,{metrics["mae"]:.8f}',
        f'r2,{metrics["r2"]:.8f}',
    ]
    if min_date_r2 is not None:
        lines.append(f'min_date_r2,{min_date_r2:.8f}')
    if selected_dates:
        lines.append(f'selected_dates,{";".join(selected_dates)}')
    if dropped_dates:
        lines.append(f'dropped_dates,{";".join(dropped_dates)}')
    metrics_path.write_text('\n'.join(lines) + '\n', encoding='ascii')
    return metrics_path


def process_band(input_performance_root: Path,
                 output_performance_root: Path,
                 band_name: str,
                 min_date_r2: float | None) -> Dict[str, str]:
    date_dirs = discover_dates(input_performance_root, band_name)
    if not date_dirs:
        raise RuntimeError(f'no valid dates found for {band_name}')
    date_dirs, dropped_dates = filter_dates_by_min_r2(
        input_performance_root,
        band_name,
        date_dirs,
        min_date_r2,
    )
    if not date_dirs:
        raise RuntimeError(f'no dates remain for {band_name} after min-date-r2 filtering')

    tasks = [(band_name, str(date_dir)) for date_dir in date_dirs]
    collected_gt: List[np.ndarray] = []
    collected_recon: List[np.ndarray] = []
    selected_dates = sorted(date_dir.name for date_dir in date_dirs)

    max_workers = min(MAX_WORKERS, len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(collect_band_date, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task_band, date_dir_str = future_to_task[future]
            date = Path(date_dir_str).name
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f'{task_band}/{date}: {exc}') from exc

            gt_values = result['gt']
            recon_values = result['recon']
            if gt_values.size == 0:
                continue

            collected_gt.append(gt_values)
            collected_recon.append(recon_values)

    if not collected_gt:
        raise RuntimeError(f'no valid scatter data found for {band_name}')

    gt_values = np.concatenate(collected_gt)
    recon_values = np.concatenate(collected_recon)
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

    output_dir = output_performance_root / band_name / '2021_daily' / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_plot_path = plot_parity(
        output_dir,
        gt_99,
        recon_99,
        metrics,
        title=f'{band_name} Overall Daily Averaged RRS Validation',
    )
    plot_path = output_dir / OUTPUT_FILENAME
    temp_plot_path.replace(plot_path)
    metrics_path = write_metrics_csv(
        output_dir,
        point_count=int(gt_values.size),
        metrics=metrics,
        filtered_point_count=int(gt_95.size),
        displayed_point_count=int(gt_99.size),
        selected_dates=selected_dates,
        dropped_dates=dropped_dates,
        min_date_r2=min_date_r2,
    )

    return {
        'band': band_name,
        'plot': str(plot_path),
        'metrics': str(metrics_path),
        'raw_points': str(int(gt_values.size)),
        'top95_points': str(int(gt_95.size)),
        'top99_points': str(int(gt_99.size)),
        'rmse': f'{metrics["rmse"]:.8f}',
        'mae': f'{metrics["mae"]:.8f}',
        'r2': f'{metrics["r2"]:.8f}',
        'selected_date_count': str(len(selected_dates)),
        'dropped_date_count': str(len(dropped_dates)),
    }


def main() -> int:
    args = parse_args()
    failures: List[str] = []
    processed = 0

    for band_name in BANDS:
        print(f'=== {band_name} ===', flush=True)
        try:
            result = process_band(
                args.input_performance_root,
                args.output_performance_root,
                band_name,
                args.min_date_r2,
            )
        except Exception as exc:
            failures.append(f'{band_name}: {exc}')
            print(f'  ERROR -> {exc}', flush=True)
            continue

        print(f"  plot        -> {result['plot']}", flush=True)
        print(f"  metrics     -> {result['metrics']}", flush=True)
        print(f"  dates       -> kept {result['selected_date_count']}, dropped {result['dropped_date_count']}", flush=True)
        print(f"  raw points  -> {result['raw_points']}", flush=True)
        print(f"  top95 pts   -> {result['top95_points']}", flush=True)
        print(f"  top99 pts   -> {result['top99_points']}", flush=True)
        print(f"  RMSE/MAE/R2 -> {result['rmse']} / {result['mae']} / {result['r2']}", flush=True)
        processed += 1

    print(f'\nProcessed bands: {processed}', flush=True)
    if failures:
        print('Failures:', flush=True)
        for failure in failures:
            print(f'  - {failure}', flush=True)
        return 1

    print('Overall scatter plots generated successfully.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
