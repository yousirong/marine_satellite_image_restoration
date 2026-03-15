#!/usr/bin/env python3
"""
Regenerate daily OC3 and daily difference maps using recon_daily_clean masks.
"""

import csv
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from Differencemap_OC3GOCIvsUST21 import DailyAveraging
from oc3_algorithm import OC3Processor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


DATA_ROOT = Path('/home/juneyonglee/Desktop/AY_ust/My_Book1/GOCI_5years/daily_results')
PERFORMANCE_ROOT = Path('/home/juneyonglee/Desktop/AY_ust/My_Book1/GOCI_5years/performance')
OC3_OUTPUT_ROOT = Path('/home/juneyonglee/Desktop/AY_ust/My_Book1/GOCI_5years/oc3_batch_results_daily')
DIFFERENCE_OUTPUT_ROOT = Path('/home/juneyonglee/Desktop/AY_ust/My_Book1/GOCI_5years/daily_differencemap_results')
KHOA_DIR = Path('/home/juneyonglee/Desktop/AY_ust/My_Book/UST21/01_day/2021/01')
GOCI_LAND_MASK = Path('/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy')
UST_LAND_MASK = Path('/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy')
PERFORMANCE_BANDS = ('band2_daily', 'band3_daily', 'band4_daily')
MAX_OC3_WORKERS = max(1, min(6, os.cpu_count() or 1))


def discover_processing_dates() -> List[str]:
    band_date_sets = []

    for band_dir_name in PERFORMANCE_BANDS:
        year_dir = PERFORMANCE_ROOT / band_dir_name / '2021_daily'
        if not year_dir.exists():
            raise FileNotFoundError(f"Missing performance year directory: {year_dir}")

        valid_dates = set()
        for date_dir in sorted(year_dir.glob('20*')):
            if not date_dir.is_dir():
                continue

            recon_path = date_dir / 'recon_daily.png'
            clean_path = date_dir / 'recon_daily_clean.png'
            if recon_path.exists() and clean_path.exists():
                valid_dates.add(date_dir.name)

        band_date_sets.append(valid_dates)

    if not band_date_sets:
        return []

    return sorted(set.intersection(*band_date_sets))


def summarize_results(date: str, results: Sequence[Dict]) -> Dict:
    total_valid = sum(int(result['valid_pixels']) for result in results)
    total_pixels = sum(int(result['total_pixels']) for result in results)
    coverage = (total_valid / total_pixels) * 100 if total_pixels > 0 else 0.0

    valid_values = []
    for result in results:
        chlorophyll = result['chlorophyll_data']
        valid_mask = ~np.isnan(chlorophyll) & ~np.isinf(chlorophyll)
        if np.any(valid_mask):
            valid_values.append(chlorophyll[valid_mask])

    if valid_values:
        all_values = np.concatenate(valid_values).astype(np.float32, copy=False)
        mean_value = float(np.mean(all_values))
        median_value = float(np.median(all_values))
        std_value = float(np.std(all_values))
        min_value = float(np.min(all_values))
        max_value = float(np.max(all_values))
    else:
        mean_value = np.nan
        median_value = np.nan
        std_value = np.nan
        min_value = np.nan
        max_value = np.nan

    return {
        'date': date,
        'time': 'daily',
        'status': 'success',
        'tiles_processed': len(results),
        'total_valid_pixels': total_valid,
        'total_pixels': total_pixels,
        'coverage_percent': coverage,
        'mean_chlorophyll': mean_value,
        'median_chlorophyll': median_value,
        'std_chlorophyll': std_value,
        'min_chlorophyll': min_value,
        'max_chlorophyll': max_value,
        'processing_time': datetime.now().isoformat(),
    }


def write_batch_summary(summary_rows: Sequence[Dict]) -> None:
    OC3_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OC3_OUTPUT_ROOT / 'batch_processing_summary.csv'
    fieldnames = [
        'date',
        'time',
        'status',
        'tiles_processed',
        'total_valid_pixels',
        'total_pixels',
        'coverage_percent',
        'mean_chlorophyll',
        'median_chlorophyll',
        'std_chlorophyll',
        'min_chlorophyll',
        'max_chlorophyll',
        'error',
        'processing_time',
    ]

    with summary_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def clear_oc3_outputs(dates: Sequence[str]) -> None:
    OC3_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for date in dates:
        date_dir = OC3_OUTPUT_ROOT / f'{date}_daily'
        if date_dir.exists():
            shutil.rmtree(date_dir)

    batch_summary_path = OC3_OUTPUT_ROOT / 'batch_processing_summary.csv'
    if batch_summary_path.exists():
        batch_summary_path.unlink()


def clear_difference_outputs(dates: Sequence[str]) -> None:
    DIFFERENCE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for date in dates:
        comparison_path = DIFFERENCE_OUTPUT_ROOT / f'OC3_KHOA_comparison_{date}.png'
        difference_path = DIFFERENCE_OUTPUT_ROOT / f'OC3_KHOA_difference_{date}.png'
        if comparison_path.exists():
            comparison_path.unlink()
        if difference_path.exists():
            difference_path.unlink()

    rmse_path = DIFFERENCE_OUTPUT_ROOT / 'rmse_results.csv'
    if rmse_path.exists():
        rmse_path.unlink()


def run_oc3_regeneration(dates: Sequence[str]) -> Tuple[List[Dict], List[str]]:
    processor = OC3Processor(
        str(DATA_ROOT),
        daily_mode=True,
        performance_root=str(PERFORMANCE_ROOT),
    )
    if not processor.validate_band_directories():
        raise RuntimeError("Failed to validate OC3 input directories")

    max_workers = min(MAX_OC3_WORKERS, len(dates))
    summaries: List[Dict] = []
    failed_dates: List[str] = []

    logger.info("Running OC3 regeneration with %d workers", max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {
            executor.submit(process_single_oc3_date, date): date
            for date in dates
        }

        for index, future in enumerate(as_completed(future_to_date), start=1):
            date = future_to_date[future]
            logger.info("OC3 [%d/%d] completed %s", index, len(dates), date)
            summary = future.result()
            summaries.append(summary)
            if summary.get('status') != 'success':
                failed_dates.append(date)

    summaries.sort(key=lambda row: row['date'])
    write_batch_summary(summaries)
    return summaries, failed_dates


def process_single_oc3_date(date: str) -> Dict:
    output_dir = OC3_OUTPUT_ROOT / f'{date}_daily'
    processor = OC3Processor(
        str(DATA_ROOT),
        daily_mode=True,
        performance_root=str(PERFORMANCE_ROOT),
    )

    try:
        results = processor.process_date_time(date, None, str(output_dir))
        if not results:
            raise RuntimeError("No tiles processed successfully")

        return summarize_results(date, results)
    except Exception as exc:
        logger.exception("OC3 regeneration failed for %s", date)
        return {
            'date': date,
            'time': 'daily',
            'status': 'error',
            'tiles_processed': 0,
            'error': str(exc),
            'processing_time': datetime.now().isoformat(),
        }


def run_difference_regeneration(dates: Sequence[str]) -> List[str]:
    processor = DailyAveraging(
        str(OC3_OUTPUT_ROOT),
        str(KHOA_DIR),
        str(DIFFERENCE_OUTPUT_ROOT),
        goci_land_mask_path=str(GOCI_LAND_MASK),
        ust_land_mask_path=str(UST_LAND_MASK),
        daily_mode=True,
    )

    failed_dates: List[str] = []

    for index, date in enumerate(dates, start=1):
        logger.info("Difference [%d/%d] %s", index, len(dates), date)
        try:
            if not processor.process_date(date):
                failed_dates.append(date)
        except Exception:
            failed_dates.append(date)
            logger.exception("Difference regeneration failed for %s", date)

    processor.calculate_and_save_average_rmse()
    return failed_dates


def main() -> int:
    dates = discover_processing_dates()
    if not dates:
        logger.error("No dates found with recon_daily.png and recon_daily_clean.png in all three bands")
        return 1

    logger.info("Discovered %d dates: %s", len(dates), ', '.join(dates))

    clear_oc3_outputs(dates)
    oc3_summaries, oc3_failed = run_oc3_regeneration(dates)

    successful_oc3_dates = [row['date'] for row in oc3_summaries if row.get('status') == 'success']

    clear_difference_outputs(dates)
    difference_failed = run_difference_regeneration(successful_oc3_dates)

    logger.info("OC3 success: %d, OC3 failed: %d", len(successful_oc3_dates), len(oc3_failed))
    logger.info(
        "Difference success: %d, Difference failed: %d",
        len(successful_oc3_dates) - len(difference_failed),
        len(difference_failed),
    )

    if oc3_failed or difference_failed:
        logger.error("Failures detected. OC3 failed dates: %s", ', '.join(oc3_failed) or 'none')
        logger.error("Difference failed dates: %s", ', '.join(difference_failed) or 'none')
        return 1

    logger.info("Successfully regenerated OC3 and difference outputs for %d dates", len(dates))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
