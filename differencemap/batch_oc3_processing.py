#!/usr/bin/env python3
"""
Batch OC3 Processing Script
Processes multiple dates and times for OC3 algorithm application
"""

import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple
import argparse
import multiprocessing as mp
from functools import partial

# Import the OC3 processor
from oc3_algorithm import OC3Processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchOC3Processor:
    """
    Batch processor for OC3 algorithm across multiple dates and times
    """

    def __init__(self, data_root: str, output_root: str):
        """
        Initialize batch processor

        Args:
            data_root: Root directory containing GOCI band data
            output_root: Root directory for output results
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.processor = OC3Processor(str(data_root))

    def get_available_dates_times(self) -> List[Tuple[str, str]]:
        """
        Get all available date/time combinations from the data structure (recon directory)

        Returns:
            List of (date, time) tuples
        """
        logger.info("Scanning for available dates and times...")

        band2_dir = self.data_root / 'band2'
        date_time_pairs = set()

        # Scan through years
        for year_dir in sorted(band2_dir.glob('20*')):
            if not year_dir.is_dir():
                continue

            # Scan through dates
            for date_dir in sorted(year_dir.glob('20*')):
                if not date_dir.is_dir():
                    continue

                date = date_dir.name

                # Scan through times
                for time_dir in sorted(date_dir.glob('*')):
                    if not time_dir.is_dir():
                        continue

                    time = time_dir.name

                    # Check if degree/recon directory exists
                    recon_dir = time_dir / 'degree' / 'recon'
                    if recon_dir.exists() and any(recon_dir.glob('*.csv')):
                        date_time_pairs.add((date, time))

        date_time_list = sorted(list(date_time_pairs))
        logger.info(f"Found {len(date_time_list)} date/time combinations")

        return date_time_list

    def filter_date_range(self, date_time_pairs: List[Tuple[str, str]],
                         start_date: str = None, end_date: str = None) -> List[Tuple[str, str]]:
        """
        Filter date/time pairs by date range

        Args:
            date_time_pairs: List of (date, time) tuples
            start_date: Start date in YYYYMMDD format (inclusive)
            end_date: End date in YYYYMMDD format (inclusive)

        Returns:
            Filtered list of date/time pairs
        """
        if start_date is None and end_date is None:
            return date_time_pairs

        filtered_pairs = []
        for date, time in date_time_pairs:
            if start_date is not None and date < start_date:
                continue
            if end_date is not None and date > end_date:
                continue
            filtered_pairs.append((date, time))

        logger.info(f"Filtered to {len(filtered_pairs)} date/time combinations")
        return filtered_pairs

    def process_single_datetime(self, date_time: Tuple[str, str]) -> Dict:
        """
        Process a single date/time combination

        Args:
            date_time: Tuple of (date, time)

        Returns:
            Processing result dictionary
        """
        date, time = date_time
        logger.info(f"Processing {date} {time}")

        try:
            # Create output directory for this date/time
            output_dir = self.output_root / f"{date}_{time}"

            # Process the data
            results = self.processor.process_date_time(date, time, str(output_dir))

            if results:
                # Calculate summary statistics
                total_valid = sum(r['valid_pixels'] for r in results)
                total_pixels = sum(r['total_pixels'] for r in results)
                coverage = (total_valid / total_pixels) * 100 if total_pixels > 0 else 0

                # Collect all valid chlorophyll values
                all_chl_values = []
                for result in results:
                    valid_mask = ~np.isnan(result['chlorophyll_data']) & ~np.isinf(result['chlorophyll_data'])
                    all_chl_values.extend(result['chlorophyll_data'][valid_mask])

                summary = {
                    'date': date,
                    'time': time,
                    'status': 'success',
                    'tiles_processed': len(results),
                    'total_valid_pixels': total_valid,
                    'total_pixels': total_pixels,
                    'coverage_percent': coverage,
                    'mean_chlorophyll': np.mean(all_chl_values) if all_chl_values else np.nan,
                    'median_chlorophyll': np.median(all_chl_values) if all_chl_values else np.nan,
                    'std_chlorophyll': np.std(all_chl_values) if all_chl_values else np.nan,
                    'min_chlorophyll': np.min(all_chl_values) if all_chl_values else np.nan,
                    'max_chlorophyll': np.max(all_chl_values) if all_chl_values else np.nan,
                    'processing_time': datetime.now().isoformat()
                }

                logger.info(f"Successfully processed {date} {time}: {len(results)} tiles, {coverage:.2f}% coverage")

            else:
                summary = {
                    'date': date,
                    'time': time,
                    'status': 'failed',
                    'tiles_processed': 0,
                    'error': 'No tiles processed successfully',
                    'processing_time': datetime.now().isoformat()
                }
                logger.error(f"Failed to process {date} {time}")

        except Exception as e:
            summary = {
                'date': date,
                'time': time,
                'status': 'error',
                'tiles_processed': 0,
                'error': str(e),
                'processing_time': datetime.now().isoformat()
            }
            logger.error(f"Error processing {date} {time}: {e}")

        return summary

    def process_batch(self, date_time_pairs: List[Tuple[str, str]],
                     max_workers: int = None) -> List[Dict]:
        """
        Process multiple date/time combinations

        Args:
            date_time_pairs: List of (date, time) tuples to process
            max_workers: Maximum number of parallel workers (None for auto)

        Returns:
            List of processing summaries
        """
        logger.info(f"Starting batch processing of {len(date_time_pairs)} date/time combinations")

        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(date_time_pairs))

        if max_workers == 1:
            # Sequential processing
            summaries = []
            for i, dt_pair in enumerate(date_time_pairs):
                logger.info(f"Progress: {i+1}/{len(date_time_pairs)}")
                summary = self.process_single_datetime(dt_pair)
                summaries.append(summary)
        else:
            # Parallel processing
            logger.info(f"Using {max_workers} parallel workers")
            with mp.Pool(max_workers) as pool:
                summaries = pool.map(self.process_single_datetime, date_time_pairs)

        # Save batch summary
        self.save_batch_summary(summaries)

        return summaries

    def save_batch_summary(self, summaries: List[Dict]):
        """
        Save batch processing summary to CSV

        Args:
            summaries: List of processing summaries
        """
        summary_df = pd.DataFrame(summaries)
        summary_file = self.output_root / 'batch_processing_summary.csv'
        summary_df.to_csv(summary_file, index=False)

        logger.info(f"Saved batch summary to {summary_file}")

        # Print overall statistics
        successful = len([s for s in summaries if s['status'] == 'success'])
        failed = len(summaries) - successful

        print(f"\n=== Batch Processing Summary ===")
        print(f"Total date/time combinations: {len(summaries)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")

        if successful > 0:
            success_summaries = [s for s in summaries if s['status'] == 'success']
            total_tiles = sum(s['tiles_processed'] for s in success_summaries)
            avg_coverage = np.mean([s['coverage_percent'] for s in success_summaries if 'coverage_percent' in s])

            print(f"Total tiles processed: {total_tiles}")
            print(f"Average coverage: {avg_coverage:.2f}%")

    def create_combined_dataset(self) -> pd.DataFrame:
        """
        Create a combined dataset from all processed results

        Returns:
            Combined DataFrame with all OC3 results
        """
        logger.info("Creating combined dataset...")

        combined_data = []

        # Scan through all processing results
        for result_dir in self.output_root.glob('*_*'):
            if not result_dir.is_dir():
                continue

            parts = result_dir.name.split('_')
            if len(parts) != 2:
                continue

            date, time = parts

            # Load summary file if exists
            summary_file = result_dir / f'oc3_summary_{date}_{time}.csv'
            if summary_file.exists():
                try:
                    summary_df = pd.read_csv(summary_file)
                    summary_df['date'] = date
                    summary_df['time'] = time
                    combined_data.append(summary_df)
                except Exception as e:
                    logger.warning(f"Failed to load {summary_file}: {e}")

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)

            # Save combined dataset
            combined_file = self.output_root / 'combined_oc3_dataset.csv'
            combined_df.to_csv(combined_file, index=False)

            logger.info(f"Saved combined dataset to {combined_file}")
            logger.info(f"Combined dataset shape: {combined_df.shape}")

            return combined_df
        else:
            logger.warning("No data found for combined dataset")
            return pd.DataFrame()


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Batch process OC3 algorithm on GOCI data')
    parser.add_argument('--data_root',
                       default='/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results',
                       help='Root directory containing GOCI band data (default: /home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/daily_results)')
    parser.add_argument('--output_root',
                       default='/home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/oc3_batch_results',
                       help='Root directory for output results (default: /home/juneyonglee/Desktop/AY_ust/myhdd/GOCI_RRS/oc3_batch_results)')
    parser.add_argument('--start_date', help='Start date in YYYYMMDD format (inclusive)')
    parser.add_argument('--end_date', help='End date in YYYYMMDD format (inclusive)')
    parser.add_argument('--max_workers', type=int, help='Maximum number of parallel workers')
    parser.add_argument('--create_combined', action='store_true', help='Create combined dataset after processing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize batch processor
    batch_processor = BatchOC3Processor(args.data_root, args.output_root)

    # Validate directories
    if not batch_processor.processor.validate_band_directories():
        logger.error("Failed to validate band directories")
        return 1

    # Get available date/time combinations
    date_time_pairs = batch_processor.get_available_dates_times()

    if not date_time_pairs:
        logger.error("No date/time combinations found")
        return 1

    # Filter by date range if specified
    if args.start_date or args.end_date:
        date_time_pairs = batch_processor.filter_date_range(
            date_time_pairs, args.start_date, args.end_date
        )

    if not date_time_pairs:
        logger.error("No date/time combinations found in specified range")
        return 1

    # Process batch
    summaries = batch_processor.process_batch(date_time_pairs, args.max_workers)

    # Create combined dataset if requested
    if args.create_combined:
        combined_df = batch_processor.create_combined_dataset()

    logger.info("Batch processing completed successfully")
    return 0


if __name__ == '__main__':
    exit(main())