#!/usr/bin/env python3
"""
OC3 Algorithm Implementation for GOCI data
Calculates chlorophyll-a concentration using band 2, 3, and 4 reflectance data
"""

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OC3Processor:
    """
    OC3 Algorithm processor for GOCI satellite data
    """

    def __init__(self,
                 data_root: str,
                 daily_mode: bool = False,
                 performance_root: Optional[str] = None):
        """
        Initialize OC3 processor

        Args:
            data_root: Root directory containing GOCI band data
            daily_mode: If True, expects daily-averaged data structure (band2_daily, etc.)
            performance_root: Optional performance root containing recon_daily and
                recon_daily_clean PNGs used to build per-date exclusion masks
        """
        self.data_root = Path(data_root)
        self.daily_mode = daily_mode
        self.performance_root = Path(performance_root) if performance_root else None
        self._exclusion_mask_cache: Dict[str, np.ndarray] = {}

        if daily_mode:
            self.band_dirs = {
                'band2': self.data_root / 'band2_daily',
                'band3': self.data_root / 'band3_daily',
                'band4': self.data_root / 'band4_daily'
            }
        else:
            self.band_dirs = {
                'band2': self.data_root / 'band2',
                'band3': self.data_root / 'band3',
                'band4': self.data_root / 'band4'
            }

        # OC3 coefficients for chlorophyll-a calculation
        # Based on standard OC3 algorithm coefficients
        self.oc3_coeffs = {
            'a0': 0.283,
            'a1': -2.753,
            'a2': 1.457,
            'a3': 0.659,
            'a4': -1.403
        }

    def validate_band_directories(self) -> bool:
        """
        Validate that all required band directories exist

        Returns:
            True if all directories exist, False otherwise
        """
        for band_name, band_path in self.band_dirs.items():
            if not band_path.exists():
                logger.error(f"Band directory not found: {band_path}")
                return False
            logger.info(f"Found {band_name} directory: {band_path}")

        if self.performance_root is not None:
            if not self.performance_root.exists():
                logger.error(f"Performance directory not found: {self.performance_root}")
                return False

            if self.daily_mode:
                for band_name in self.band_dirs:
                    performance_band_dir = self.performance_root / f"{band_name}_daily"
                    if not performance_band_dir.exists():
                        logger.error(f"Performance band directory not found: {performance_band_dir}")
                        return False

        return True

    def _get_performance_date_dir(self, band_name: str, date: str) -> Path:
        if self.performance_root is None:
            raise ValueError("Performance root is not configured")

        if not self.daily_mode:
            raise ValueError("Performance masking is only supported in daily mode")

        return self.performance_root / f"{band_name}_daily" / f"{date[:4]}_daily" / date

    def _load_rgba_image(self, image_path: Path) -> np.ndarray:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing required image: {image_path}")

        try:
            with Image.open(image_path) as image:
                return np.array(image.convert('RGBA'))
        except Exception as exc:
            raise RuntimeError(f"Unreadable PNG: {image_path}: {exc}") from exc

    def _load_exclusion_mask_for_date(self, date: str) -> Optional[np.ndarray]:
        if self.performance_root is None:
            return None

        if date in self._exclusion_mask_cache:
            return self._exclusion_mask_cache[date]

        combined_mask: Optional[np.ndarray] = None
        expected_shape: Optional[Tuple[int, int]] = None

        for band_name in ('band2', 'band3', 'band4'):
            performance_date_dir = self._get_performance_date_dir(band_name, date)
            recon_path = performance_date_dir / 'recon_daily.png'
            clean_path = performance_date_dir / 'recon_daily_clean.png'

            recon_image = self._load_rgba_image(recon_path)
            clean_image = self._load_rgba_image(clean_path)

            if recon_image.shape != clean_image.shape:
                raise ValueError(
                    f"Image shape mismatch for {band_name} {date}: "
                    f"{recon_path} {recon_image.shape} vs {clean_path} {clean_image.shape}"
                )

            band_shape = recon_image.shape[:2]
            if expected_shape is None:
                expected_shape = band_shape
                combined_mask = np.zeros(expected_shape, dtype=bool)
            elif band_shape != expected_shape:
                raise ValueError(
                    f"Cross-band image shape mismatch for {date}: "
                    f"expected {expected_shape}, got {band_shape} from {recon_path}"
                )

            band_mask = (
                np.all(clean_image[:, :, :3] == 255, axis=2) &
                np.any(recon_image[:, :, :3] != 255, axis=2)
            )
            combined_mask |= band_mask

        if combined_mask is None:
            raise ValueError(f"Failed to build exclusion mask for {date}")

        self._exclusion_mask_cache[date] = combined_mask
        return combined_mask

    def _parse_tile_coordinates(self, tile_id: str) -> Tuple[int, int]:
        match = re.search(r'_y(\d+)_x(\d+)$', tile_id)
        if match is None:
            raise ValueError(f"Could not parse tile coordinates from tile id: {tile_id}")

        return int(match.group(1)), int(match.group(2))

    def _get_tile_exclusion_mask(self,
                                 date: str,
                                 tile_id: str,
                                 tile_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        full_mask = self._load_exclusion_mask_for_date(date)
        if full_mask is None:
            return None

        y_start, x_start = self._parse_tile_coordinates(tile_id)
        tile_height, tile_width = tile_shape

        if y_start >= full_mask.shape[0] or x_start >= full_mask.shape[1]:
            raise ValueError(
                f"Tile start is outside exclusion mask bounds for {date} {tile_id}: "
                f"mask shape {full_mask.shape}"
            )

        y_end = min(y_start + tile_height, full_mask.shape[0])
        x_end = min(x_start + tile_width, full_mask.shape[1])

        tile_mask = np.zeros(tile_shape, dtype=bool)
        mask_slice = full_mask[y_start:y_end, x_start:x_end]
        tile_mask[:mask_slice.shape[0], :mask_slice.shape[1]] = mask_slice

        return tile_mask

    def load_band_data(self, band_path: Path, date: str, time: str, tile_id: str) -> Optional[np.ndarray]:
        """
        Load band data from CSV file in recon directory

        Args:
            band_path: Path to band directory
            date: Date in YYYYMMDD format
            time: Time in HHMMSS format (ignored if daily_mode=True)
            tile_id: Tile identifier (e.g., img_504_y5429_x4864)

        Returns:
            Numpy array of band data or None if file not found
        """
        year = date[:4]

        if self.daily_mode:
            # For daily-averaged data: band_path/year/date/degree/recon/tile_id.csv
            csv_path = band_path / year / date / 'degree' / 'recon' / f'{tile_id}.csv'
        else:
            # For time-based data: band_path/year/date/time/degree/recon/tile_id.csv
            csv_path = band_path / year / date / time / 'degree' / 'recon' / f'{tile_id}.csv'

        if not csv_path.exists():
            logger.warning(f"File not found: {csv_path}")
            return None

        try:
            data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
            if data.ndim == 1:
                data = data[np.newaxis, :]

            if data.shape[1] < 2:
                raise ValueError(f"Unexpected CSV shape {data.shape}")

            # Skip first column (row indices)
            data = data[:, 1:]

            # Normalize data if it's in 0-255 range
            if np.isfinite(data).any() and np.nanmax(data) > 1.5:
                data = data / 255.0
                logger.debug("Normalized data from 0-255 to 0-1 range")

            logger.debug(f"Loaded data shape: {data.shape} from {csv_path}")
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
            return None

    def calculate_oc3_ratio(self, band2: np.ndarray, band3: np.ndarray, band4: np.ndarray) -> np.ndarray:
        """
        Calculate OC3 ratio (band ratio for chlorophyll estimation)

        Args:
            band2: Reflectance data for band 2 (443nm)
            band3: Reflectance data for band 3 (490nm)
            band4: Reflectance data for band 4 (555nm)

        Returns:
            OC3 ratio array
        """
        # OC3 uses max(Rrs443, Rrs490) / Rrs555
        max_blue_green = np.maximum(band2, band3)

        # Avoid division by zero
        ratio = np.where(band4 > 0, max_blue_green / band4, np.nan)

        # Apply log10 to the ratio for OC3 calculation
        log_ratio = np.where(ratio > 0, np.log10(ratio), np.nan)

        return log_ratio

    def calculate_chlorophyll_oc3(self, log_ratio: np.ndarray) -> np.ndarray:
        """
        Calculate chlorophyll-a concentration using OC3 algorithm

        Args:
            log_ratio: Log10 of band ratio

        Returns:
            Chlorophyll-a concentration in mg/m³
        """
        coeffs = self.oc3_coeffs

        # OC3 polynomial: log10(Chl) = a0 + a1*X + a2*X² + a3*X³ + a4*X⁴
        # where X = log10(max(Rrs443, Rrs490) / Rrs555)
        log_chl = (coeffs['a0'] +
                   coeffs['a1'] * log_ratio +
                   coeffs['a2'] * log_ratio**2 +
                   coeffs['a3'] * log_ratio**3 +
                   coeffs['a4'] * log_ratio**4)

        # Convert back from log space
        chlorophyll = 10**log_chl

        # Apply reasonable bounds for chlorophyll concentration
        chlorophyll = np.clip(chlorophyll, 0.01, 100.0)

        return chlorophyll

    def process_single_tile(self, date: str, time: str, tile_id: str) -> Optional[Dict]:
        """
        Process a single tile for OC3 calculation

        Args:
            date: Date in YYYYMMDD format
            time: Time in HHMMSS format
            tile_id: Tile identifier

        Returns:
            Dictionary containing results or None if processing failed
        """
        logger.info(f"Processing tile {tile_id} for {date} {time}")

        # Load band data
        band2_data = self.load_band_data(self.band_dirs['band2'], date, time, tile_id)
        band3_data = self.load_band_data(self.band_dirs['band3'], date, time, tile_id)
        band4_data = self.load_band_data(self.band_dirs['band4'], date, time, tile_id)

        if band2_data is None or band3_data is None or band4_data is None:
            logger.error(f"Failed to load all band data for tile {tile_id}")
            return None

        # Verify shapes match
        if not (band2_data.shape == band3_data.shape == band4_data.shape):
            logger.error(f"Band data shapes don't match for tile {tile_id}")
            return None

        # Apply val_goci_fullpatch.py logic: check for empty/fake patches
        # Check band2 first (representative)
        unique_vals = np.unique(band2_data)

        if len(unique_vals) == 1:
            # All values are identical - empty patch (filled with median/mean)
            logger.info(f"Empty patch detected in {tile_id} (all values = {unique_vals[0]}), filling with 0")
            band2_data = np.zeros_like(band2_data)
            band3_data = np.zeros_like(band3_data)
            band4_data = np.zeros_like(band4_data)
        elif len(unique_vals) == 2:
            # Two values only: 0 and one other value
            non_zero_vals = unique_vals[unique_vals != 0]
            if len(non_zero_vals) == 1 and non_zero_vals[0] != 1:
                # Ocean area filled with single value (median/mean) - fake data
                logger.info(f"Mixed patch detected in {tile_id} (ocean with single value = {non_zero_vals[0]}), filling ocean with 0")
                band2_data[band2_data != 0] = 0
                band3_data[band3_data != 0] = 0
                band4_data[band4_data != 0] = 0

        # Calculate OC3 ratio
        log_ratio = self.calculate_oc3_ratio(band2_data, band3_data, band4_data)

        # Calculate chlorophyll concentration
        chlorophyll = self.calculate_chlorophyll_oc3(log_ratio)

        tile_exclusion_mask = self._get_tile_exclusion_mask(date, tile_id, chlorophyll.shape)
        if tile_exclusion_mask is not None and np.any(tile_exclusion_mask):
            chlorophyll = chlorophyll.copy()
            log_ratio = log_ratio.copy()
            chlorophyll[tile_exclusion_mask] = np.nan
            log_ratio[tile_exclusion_mask] = np.nan

        # Calculate statistics
        valid_mask = ~np.isnan(chlorophyll) & ~np.isinf(chlorophyll)
        valid_chl = chlorophyll[valid_mask]

        if len(valid_chl) == 0:
            logger.info(f"No valid chlorophyll values remain for tile {tile_id}")
            return {
                'tile_id': tile_id,
                'date': date,
                'time': time,
                'shape': chlorophyll.shape,
                'valid_pixels': 0,
                'total_pixels': chlorophyll.size,
                'mean_chl': np.nan,
                'median_chl': np.nan,
                'std_chl': np.nan,
                'min_chl': np.nan,
                'max_chl': np.nan,
                'chlorophyll_data': chlorophyll,
                'log_ratio_data': log_ratio
            }

        stats = {
            'tile_id': tile_id,
            'date': date,
            'time': time,
            'shape': chlorophyll.shape,
            'valid_pixels': len(valid_chl),
            'total_pixels': chlorophyll.size,
            'mean_chl': np.mean(valid_chl),
            'median_chl': np.median(valid_chl),
            'std_chl': np.std(valid_chl),
            'min_chl': np.min(valid_chl),
            'max_chl': np.max(valid_chl),
            'chlorophyll_data': chlorophyll,
            'log_ratio_data': log_ratio
        }

        logger.info(f"Successfully processed tile {tile_id}: {len(valid_chl)}/{chlorophyll.size} valid pixels")

        return stats

    def get_available_tiles(self, date: str, time: str = None) -> List[str]:
        """
        Get list of available tiles for given date and time from recon directory

        Args:
            date: Date in YYYYMMDD format
            time: Time in HHMMSS format (ignored if daily_mode=True)

        Returns:
            List of available tile IDs
        """
        year = date[:4]

        if self.daily_mode:
            # For daily-averaged data: band2_daily/year/date/degree/recon
            recon_dir = self.band_dirs['band2'] / year / date / 'degree' / 'recon'
        else:
            # For time-based data: band2/year/date/time/degree/recon
            recon_dir = self.band_dirs['band2'] / year / date / time / 'degree' / 'recon'

        if not recon_dir.exists():
            logger.warning(f"Recon directory not found: {recon_dir}")
            return []

        # Find all CSV files in recon directory
        csv_files = list(recon_dir.glob('*.csv'))
        tile_ids = [f.stem for f in csv_files]

        if self.daily_mode:
            logger.info(f"Found {len(tile_ids)} tiles for {date} (daily-averaged)")
        else:
            logger.info(f"Found {len(tile_ids)} tiles for {date} {time}")
        return sorted(tile_ids)

    def process_date_time(self, date: str, time: str = None, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Process all tiles for a specific date and time

        Args:
            date: Date in YYYYMMDD format
            time: Time in HHMMSS format (ignored if daily_mode=True)
            output_dir: Directory to save results (optional)

        Returns:
            List of processing results for each tile
        """
        if self.daily_mode:
            logger.info(f"Processing date {date} (daily-averaged)")
            time = 'daily'  # Use 'daily' as placeholder for output naming
        else:
            logger.info(f"Processing date {date} time {time}")

        # Get available tiles
        tile_ids = self.get_available_tiles(date, time if not self.daily_mode else None)

        if not tile_ids:
            if self.daily_mode:
                logger.error(f"No tiles found for {date} (daily-averaged)")
            else:
                logger.error(f"No tiles found for {date} {time}")
            return []

        results = []
        successful_tiles = 0

        for tile_id in tile_ids:
            result = self.process_single_tile(date, time if not self.daily_mode else '', tile_id)
            if result is not None:
                results.append(result)
                successful_tiles += 1

        logger.info(f"Successfully processed {successful_tiles}/{len(tile_ids)} tiles")

        # Save results if output directory specified
        if output_dir and results:
            self.save_results(results, output_dir, date, time)

        return results

    def save_results(self, results: List[Dict], output_dir: str, date: str, time: str):
        """
        Save processing results to files

        Args:
            results: List of processing results
            output_dir: Output directory
            date: Date string
            time: Time string
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary statistics
        summary_data = []
        for result in results:
            summary_data.append({
                'tile_id': result['tile_id'],
                'date': result['date'],
                'time': result['time'],
                'valid_pixels': result['valid_pixels'],
                'total_pixels': result['total_pixels'],
                'coverage_percent': (result['valid_pixels'] / result['total_pixels']) * 100,
                'mean_chl': result['mean_chl'],
                'median_chl': result['median_chl'],
                'std_chl': result['std_chl'],
                'min_chl': result['min_chl'],
                'max_chl': result['max_chl']
            })

        summary_file = output_path / f'oc3_summary_{date}_{time}.csv'
        fieldnames = [
            'tile_id', 'date', 'time', 'valid_pixels', 'total_pixels',
            'coverage_percent', 'mean_chl', 'median_chl', 'std_chl',
            'min_chl', 'max_chl'
        ]
        with summary_file.open('w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        logger.info(f"Saved summary to {summary_file}")

        # Save individual tile chlorophyll data
        for result in results:
            tile_output_dir = output_path / f"{date}_{time}" / result['tile_id']
            tile_output_dir.mkdir(parents=True, exist_ok=True)

            # Save chlorophyll data
            chl_file = tile_output_dir / 'chlorophyll_oc3.csv'
            np.savetxt(chl_file, result['chlorophyll_data'], delimiter=',', fmt='%.6f')

            # Save log ratio data
            ratio_file = tile_output_dir / 'log_ratio.csv'
            np.savetxt(ratio_file, result['log_ratio_data'], delimiter=',', fmt='%.6f')

        logger.info(f"Saved individual tile data to {output_path}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Apply OC3 algorithm to GOCI data')
    parser.add_argument('--data_root', required=True, help='Root directory containing GOCI band data')
    parser.add_argument('--date', required=True, help='Date in YYYYMMDD format')
    parser.add_argument('--time', required=True, help='Time in HHMMSS format')
    parser.add_argument('--output_dir', help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize processor
    processor = OC3Processor(args.data_root)

    # Validate directories
    if not processor.validate_band_directories():
        logger.error("Failed to validate band directories")
        return 1

    # Process data
    results = processor.process_date_time(args.date, args.time, args.output_dir)

    if results:
        logger.info(f"Processing completed successfully. Processed {len(results)} tiles.")

        # Print summary statistics
        total_valid = sum(r['valid_pixels'] for r in results)
        total_pixels = sum(r['total_pixels'] for r in results)
        overall_coverage = (total_valid / total_pixels) * 100 if total_pixels > 0 else 0

        all_valid_chl = []
        for result in results:
            valid_mask = ~np.isnan(result['chlorophyll_data']) & ~np.isinf(result['chlorophyll_data'])
            all_valid_chl.extend(result['chlorophyll_data'][valid_mask])

        if all_valid_chl:
            print(f"\n=== OC3 Processing Summary ===")
            print(f"Date: {args.date}")
            print(f"Time: {args.time}")
            print(f"Total tiles processed: {len(results)}")
            print(f"Overall coverage: {overall_coverage:.2f}%")
            print(f"Overall chlorophyll statistics:")
            print(f"  Mean: {np.mean(all_valid_chl):.4f} mg/m³")
            print(f"  Median: {np.median(all_valid_chl):.4f} mg/m³")
            print(f"  Std: {np.std(all_valid_chl):.4f} mg/m³")
            print(f"  Range: {np.min(all_valid_chl):.4f} - {np.max(all_valid_chl):.4f} mg/m³")

        return 0
    else:
        logger.error("Processing failed")
        return 1


if __name__ == '__main__':
    exit(main())
