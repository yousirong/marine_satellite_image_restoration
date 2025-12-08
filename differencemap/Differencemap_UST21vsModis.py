#!/usr/bin/env python3
"""
UST21 vs MODIS 8-day Average Chlorophyll Difference Map Calculator
Processes 8-day averaged chlorophyll data from:
1. UST21 performance directory (01~08, 02~09, 03~10, etc.)
2. MODIS Aqua 8-day averages
Calculates difference map with proper spatial alignment and cropping
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict
import glob
import re
from scipy.ndimage import zoom
from datetime import datetime, timedelta
import netCDF4 as nc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Disable DEBUG logging from matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class UST21ModisDifferenceMap:
    """
    Calculate difference maps between UST21 and MODIS chlorophyll data
    """

    def __init__(self,
                 ust21_perf_dir: str,
                 modis_dir: str,
                 output_dir: str,
                 ust_land_mask_path: str = None):
        """
        Initialize processor

        Args:
            ust21_perf_dir: UST21 performance directory
            modis_dir: MODIS Aqua 8-day directory
            output_dir: Output directory for results
            ust_land_mask_path: Path to UST21 land mask file
        """
        self.ust21_perf_dir = Path(ust21_perf_dir)
        self.modis_dir = Path(modis_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load land mask
        self.ust_land_mask = None
        if ust_land_mask_path and Path(ust_land_mask_path).exists():
            self.ust_land_mask = np.load(ust_land_mask_path)
            logger.info(f"Loaded UST21 land mask: {self.ust_land_mask.shape}")

    def load_ust21_single_day(self, date_str: str) -> Optional[np.ndarray]:
        """
        Load single day chlorophyll from UST21 test/degree directory

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            Single day chlorophyll array (8000x10500) or None
        """
        try:
            year = date_str[:4]
            test_base_path = Path(str(self.ust21_perf_dir).replace('/performance/', '/test/'))
            recon_dir = test_base_path.parent / year / date_str / 'degree' / 'recon'

            if not recon_dir.exists():
                logger.warning(f"UST21 test/degree/recon directory not found: {recon_dir}")
                return None

            csv_files = sorted(glob.glob(str(recon_dir / '*.csv')))
            if not csv_files:
                logger.warning(f"No CSV files found in {recon_dir}")
                return None

            # Dictionary to store tiles by position
            tile_data_by_position = {}

            for csv_file in csv_files:
                try:
                    filename = Path(csv_file).stem
                    parts = filename.split('_')

                    if len(parts) >= 4:
                        y_pos = int(parts[2][1:])
                        x_pos = int(parts[3][1:])
                        data = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)

                        if data.shape != (256, 256):
                            continue

                        pos_key = (y_pos, x_pos)
                        if pos_key not in tile_data_by_position:
                            tile_data_by_position[pos_key] = []
                        tile_data_by_position[pos_key].append(data)

                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
                    continue

            if not tile_data_by_position:
                return None

            # Find grid dimensions
            y_positions = [pos[0] for pos in tile_data_by_position.keys()]
            x_positions = [pos[1] for pos in tile_data_by_position.keys()]
            max_y = max(y_positions) + 256
            max_x = max(x_positions) + 256

            # Initialize full grid
            full_grid = np.full((max_y, max_x), np.nan, dtype=np.float32)

            # Fill in tiles
            for (y_pos, x_pos), tile_list in tile_data_by_position.items():
                y_end = y_pos + 256
                x_end = x_pos + 256
                stacked_tiles = np.stack(tile_list, axis=0)
                averaged_tile = np.nanmean(stacked_tiles, axis=0)
                valid_mask = ~np.isnan(averaged_tile)
                full_grid[y_pos:y_end, x_pos:x_end][valid_mask] = averaged_tile[valid_mask]

            return full_grid.astype(np.float32)

        except Exception as e:
            logger.error(f"Error loading UST21 data for {date_str}: {e}")
            return None

    def load_ust21_8day_average(self, start_date: str) -> Optional[np.ndarray]:
        """
        Load 8-day averaged chlorophyll from UST21 test/degree directory
        Loads data from start_date to start_date+7 days and averages them pixel-wise

        Args:
            start_date: Start date in YYYYMMDD format

        Returns:
            8-day averaged chlorophyll array (8000x10500) or None
        """
        try:
            from datetime import datetime, timedelta

            # Generate 8 consecutive dates
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            date_list = [(start_dt + timedelta(days=i)).strftime('%Y%m%d') for i in range(8)]

            logger.info(f"Loading UST21 data for 8-day average: {date_list[0]} to {date_list[-1]}")

            # Load all 8 days
            daily_grids = []
            for date_str in date_list:
                daily_grid = self.load_ust21_single_day(date_str)
                if daily_grid is not None:
                    daily_grids.append(daily_grid)
                    logger.info(f"  Loaded {date_str}: shape={daily_grid.shape}, valid pixels={np.sum(~np.isnan(daily_grid))}")
                else:
                    logger.warning(f"  Failed to load {date_str}")

            if not daily_grids:
                logger.error(f"No valid daily data loaded for 8-day period starting {start_date}")
                return None

            logger.info(f"Loaded {len(daily_grids)}/{len(date_list)} days for 8-day average")

            # Stack and average across days
            stacked_grids = np.stack(daily_grids, axis=0)  # Shape: (n_days, height, width)
            averaged_grid = np.nanmean(stacked_grids, axis=0)  # Average across days

            logger.info(f"UST21 8-day averaged data shape: {averaged_grid.shape}")
            logger.info(f"Valid data pixels: {np.sum(~np.isnan(averaged_grid))}")
            logger.info(f"Data range: min={np.nanmin(averaged_grid):.4f}, max={np.nanmax(averaged_grid):.4f}, mean={np.nanmean(averaged_grid):.4f}")

            return averaged_grid.astype(np.float32)

        except Exception as e:
            logger.error(f"Error loading UST21 8-day average for {start_date}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_modis_8day_average(self, start_date: str) -> Optional[np.ndarray]:
        """
        Load MODIS Aqua 8-day average chlorophyll

        Args:
            start_date: Start date in YYYYMMDD format

        Returns:
            MODIS chlorophyll array or None
        """
        try:
            # Generate MODIS filename based on date pattern
            # MODIS 8-day products typically use YYYYDDD format
            from datetime import datetime
            date_obj = datetime.strptime(start_date, '%Y%m%d')
            doy = date_obj.timetuple().tm_yday

            # Try multiple possible MODIS filename patterns
            patterns = [
                f"*{start_date}*.nc",
                f"*{date_obj.strftime('%Y%j')}*.nc",  # YYYYDDD format
                f"*{date_obj.strftime('%Y%m%d')}*.nc",
            ]

            modis_file = None
            for pattern in patterns:
                search_result = glob.glob(str(self.modis_dir / pattern))
                if search_result:
                    modis_file = search_result[0]
                    break

            if not modis_file:
                logger.warning(f"MODIS file not found for {start_date}")
                return None

            # Load MODIS NetCDF file
            logger.info(f"Loading MODIS file: {modis_file}")
            ds = nc.Dataset(modis_file, 'r')

            # Try common variable names in MODIS files
            var_names = ['chlor_a', 'Rrs_443', 'Rrs_490', 'Rrs_555', 'merged_daily_Chl']
            chl_data = None

            for var_name in var_names:
                if var_name in ds.variables:
                    chl_data = ds.variables[var_name][:].astype(np.float32)
                    logger.info(f"Loaded variable '{var_name}' from MODIS")
                    break

            if chl_data is None:
                logger.error(f"Could not find chlorophyll variable in {modis_file}")
                ds.close()
                return None

            # Handle 3D data (time, lat, lon) -> take first time slice
            if chl_data.ndim == 3:
                chl_data = chl_data[0, :, :]
                logger.info(f"Extracted first time slice from 3D MODIS data")

            # Remove fill values and invalid data
            # Set values outside valid range (0-10 mg/m³) to 0
            chl_data[chl_data < 0] = 0.0  # Negative values → 0
            chl_data[chl_data > 100] = 0.0  # Extreme outliers (>100) → 0
            logger.info(f"Masked invalid values: negative values and extreme outliers set to 0")

            ds.close()

            # Log detailed shape information
            logger.info(f"Loaded MODIS data with shape: {chl_data.shape}")
            logger.info(f"  Interpreted as: height={chl_data.shape[0]} (pixels), width={chl_data.shape[1]} (pixels)")
            logger.info(f"  Aspect ratio (height/width): {chl_data.shape[0] / chl_data.shape[1]:.3f}")

            # Check if data might be stored as (lon, lat) instead of (lat, lon)
            # MODIS 4km global grid: typically stored as (lat, lon) = (4320, 8640)
            # But sometimes it's (lon, lat) = (8640, 4320) or needs transpose
            if chl_data.shape[0] == 4320 and chl_data.shape[1] == 8640:
                logger.info("MODIS data shape (4320, 8640) - appears to be (lat, lon) order")
                # This is the expected format: height=4320, width=8640
                # But we need to check if it actually represents lat/lon correctly
                # by testing with our geographic coordinates
            elif chl_data.shape[0] == 8640 and chl_data.shape[1] == 4320:
                logger.warning("MODIS data appears to be in (lon, lat) order - transposing to (lat, lon)")
                chl_data = chl_data.T  # Transpose to (lat, lon)
                logger.info(f"After transpose: shape = {chl_data.shape}")

            return chl_data

        except Exception as e:
            logger.error(f"Error loading MODIS data for {start_date}: {e}")
            return None

    def find_ust21_ocean_bounds(self) -> Tuple[int, int, int, int]:
        """
        Find ocean region bounds from UST21 land mask

        Returns:
            Tuple of (y_min, y_max, x_min, x_max) for ocean regions
        """
        if self.ust_land_mask is None:
            logger.warning("No land mask available, using default bounds")
            return (0, 8000, 0, 10500)

        try:
            # Find ocean pixels (non-999)
            ocean_mask = (self.ust_land_mask != 999)

            # Find bounds of ocean region
            ocean_y, ocean_x = np.where(ocean_mask)
            y_min, y_max = ocean_y.min(), ocean_y.max()
            x_min, x_max = ocean_x.min(), ocean_x.max()

            logger.info(f"UST21 ocean bounds: y=[{y_min}, {y_max}], x=[{x_min}, {x_max}]")
            logger.info(f"UST21 ocean region size: {y_max - y_min} x {x_max - x_min}")

            return (y_min, y_max, x_min, x_max)

        except Exception as e:
            logger.error(f"Error finding ocean bounds: {e}")
            return (0, 8000, 0, 10500)

    def estimate_modis_crop_from_ust21_bounds(self,
                                             modis_shape: Tuple[int, int],
                                             ust21_ocean_bounds: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Estimate MODIS crop region based on UST21 ocean bounds
        Maps UST21 ocean region coordinates to global MODIS coordinates

        Args:
            modis_shape: Shape of MODIS data (height, width)
            ust21_ocean_bounds: UST21 ocean bounds (y_min, y_max, x_min, x_max)

        Returns:
            MODIS crop region (y_start, y_end, x_start, x_end)
        """
        try:
            y_min_ust, y_max_ust, x_min_ust, x_max_ust = ust21_ocean_bounds
            h_modis, w_modis = modis_shape

            # UST21 is 8000x10500 (height x width) covering East Asia
            # UST21 coordinate system (pixel indices):
            # - y: 0-8000 (top to bottom)
            # - x: 0-10500 (left to right)

            # Full UST21 domain mapping to geographic coordinates
            # Based on actual UST21 data: 100°E-130°E (longitude), 10°N-50°N (latitude)
            ust21_h, ust21_w = 8000, 10500
            # lon_min_ust = 113.2  # 경도: 35°
            # lon_max_ust = 148.2
            # lat_max_ust = 47.70   # 위도: 26.32°
            # lat_min_ust = 20.78
            lon_min_ust = 114.25   # (146.75 - 114.25 = 32.5°)
            lon_max_ust = 146.75
            lat_max_ust = 47.65
            lat_min_ust = 20.73
            logger.info(f"UST21 ocean bounds from land mask: y=[{y_min_ust}, {y_max_ust}], x=[{x_min_ust}, {x_max_ust}]")
            logger.info(f"UST21 ocean region size: {y_max_ust - y_min_ust} (세로) x {x_max_ust - x_min_ust} (가로) pixels")

            # Map UST21 pixel coordinates to geographic coordinates
            lon_per_pixel_ust = (lon_max_ust - lon_min_ust) / ust21_w
            lat_per_pixel_ust = (lat_max_ust - lat_min_ust) / ust21_h

            # Geographic bounds of actual ocean region
            # x increases from left (100°E) to right (130°E)
            lon_start = lon_min_ust + x_min_ust * lon_per_pixel_ust
            lon_end = lon_min_ust + x_max_ust * lon_per_pixel_ust

            # y increases from top (50°N) to bottom (10°N)
            lat_start = lat_max_ust - y_min_ust * lat_per_pixel_ust
            lat_end = lat_max_ust - y_max_ust * lat_per_pixel_ust

            logger.info(f"Ocean region geographic bounds: lon=[{lon_start:.2f}°E, {lon_end:.2f}°E] ({lon_end - lon_start:.2f}° wide), lat=[{lat_end:.2f}°N, {lat_start:.2f}°N] ({lat_start - lat_end:.2f}° tall)")

            # Map geographic coordinates to MODIS pixel indices
            # MODIS global grid: -180° to 180° (longitude), 90°N to -90°S (latitude)
            # MODIS shape: (h_modis, w_modis) = (4320, 8640)

            # Calculate pixel indices based on geographic range
            # Longitude: -180° at x=0, 180° at x=w_modis
            lon_scale = w_modis / 360.0
            x_start_modis = int((lon_start + 180.0) * lon_scale)
            x_end_modis = int((lon_end + 180.0) * lon_scale)

            # Latitude: 90°N at y=0, -90°S at y=h_modis
            lat_scale = h_modis / 180.0
            y_start_modis = int((90.0 - lat_start) * lat_scale)
            y_end_modis = int((90.0 - lat_end) * lat_scale)

            logger.info(f"MODIS pixel scale: lon_scale={lon_scale:.2f} px/°, lat_scale={lat_scale:.2f} px/°")

            # Ensure y_start < y_end (in case of latitude reversal)
            if y_start_modis > y_end_modis:
                y_start_modis, y_end_modis = y_end_modis, y_start_modis

            crop_height = y_end_modis - y_start_modis
            crop_width = x_end_modis - x_start_modis
            crop_ratio = crop_height / crop_width

            logger.info(f"MODIS crop region: y=[{y_start_modis}, {y_end_modis}] ({crop_height} pixels 세로), x=[{x_start_modis}, {x_end_modis}] ({crop_width} pixels 가로)")
            logger.info(f"비율: 세로/가로 = {crop_ratio:.3f}")

            # Verify ratio matches UST21
            ust21_crop_height = y_max_ust - y_min_ust
            ust21_crop_width = x_max_ust - x_min_ust
            ust21_crop_ratio = ust21_crop_height / ust21_crop_width
            logger.info(f"UST21 비율: {ust21_crop_ratio:.3f}, MODIS 비율: {crop_ratio:.3f}")

            if abs(crop_ratio - ust21_crop_ratio) > 0.01:
                logger.warning(f"비율 불일치! UST21: {ust21_crop_ratio:.3f}, MODIS: {crop_ratio:.3f}")
                logger.warning(f"좌표가 바뀌었을 가능성이 있습니다. y와 x를 swap하면: {crop_width / crop_height:.3f}")

            return (y_start_modis, y_end_modis, x_start_modis, x_end_modis)

        except Exception as e:
            logger.error(f"Error estimating MODIS crop: {e}")
            import traceback
            traceback.print_exc()
            return (3000, 8400, 14400, 21600)

    def crop_and_align_modis_to_ust21(self,
                                      modis_data: np.ndarray,
                                      ust21_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Crop and interpolate global MODIS data to match UST21 dimensions exactly
        Aligns MODIS to UST21 using ocean/land boundaries from UST21 land mask

        Args:
            modis_data: Global MODIS chlorophyll array
            ust21_data: UST21 reconstructed data (to match dimensions)

        Returns:
            Tuple of (Cropped and interpolated MODIS data matching UST21 shape, crop info dict)
        """
        try:
            logger.info(f"Original MODIS shape: {modis_data.shape}")

            # Step 1: Find UST21 ocean bounds from land mask
            y_min_ust, y_max_ust, x_min_ust, x_max_ust = self.find_ust21_ocean_bounds()

            # Step 2: Estimate corresponding MODIS crop region
            y_start, y_end, x_start, x_end = self.estimate_modis_crop_from_ust21_bounds(
                modis_data.shape, (y_min_ust, y_max_ust, x_min_ust, x_max_ust)
            )

            # Step 3: Crop MODIS
            # Note: If MODIS data has lat/lon swapped in actual grid vs shape interpretation,
            # we might need to swap y and x indices

            # First, try with original order
            y_start = max(0, y_start)
            y_end = min(modis_data.shape[0], y_end)
            x_start = max(0, x_start)
            x_end = min(modis_data.shape[1], x_end)

            modis_cropped = modis_data[y_start:y_end, x_start:x_end]
            logger.info(f"Cropped MODIS shape: {modis_cropped.shape}")
            logger.info(f"Crop region: y=[{y_start}, {y_end}], x=[{x_start}, {x_end}]")


            # Step 4: Interpolate to match UST21 shape exactly
            target_shape = ust21_data.shape
            logger.info(f"Interpolating MODIS from {modis_cropped.shape} to target UST21 shape {target_shape}")

            # Use scipy zoom to interpolate
            if modis_cropped.shape != target_shape:
                zoom_y = target_shape[0] / modis_cropped.shape[0]
                zoom_x = target_shape[1] / modis_cropped.shape[1]

                logger.info(f"Zoom factors: y={zoom_y:.4f}, x={zoom_x:.4f}")
                modis_interpolated = zoom(modis_cropped, (zoom_y, zoom_x), order=1, prefilter=False)
                logger.info(f"Interpolated MODIS shape: {modis_interpolated.shape}")
            else:
                modis_interpolated = modis_cropped

            # Store crop info for visualization
            crop_info = {
                'y_start': y_start,
                'y_end': y_end,
                'x_start': x_start,
                'x_end': x_end,
                'original_shape': modis_data.shape
            }

            return modis_interpolated.astype(np.float32), crop_info

        except Exception as e:
            logger.error(f"Error cropping MODIS data: {e}")
            import traceback
            traceback.print_exc()
            return None, {}

    def interpolate_modis_to_ust21(self,
                                   modis_data: np.ndarray,
                                   ust21_shape: Tuple[int, int] = (8000, 10500)) -> np.ndarray:
        """
        Interpolate MODIS data to UST21 resolution

        Args:
            modis_data: MODIS cropped data
            ust21_shape: Target UST21 shape

        Returns:
            MODIS data interpolated to UST21 resolution
        """
        try:
            if modis_data.shape == ust21_shape:
                logger.info(f"MODIS data already matches UST21 shape")
                return modis_data

            zoom_y = ust21_shape[0] / modis_data.shape[0]
            zoom_x = ust21_shape[1] / modis_data.shape[1]

            logger.info(f"Interpolating MODIS from {modis_data.shape} to {ust21_shape}")
            logger.info(f"Zoom factors: y={zoom_y:.4f}, x={zoom_x:.4f}")

            interpolated = zoom(modis_data, (zoom_y, zoom_x), order=1, prefilter=False)
            logger.info(f"Interpolated shape: {interpolated.shape}")

            return interpolated.astype(np.float32)

        except Exception as e:
            logger.error(f"Error interpolating MODIS data: {e}")
            return None

    def apply_land_mask(self,
                       data: np.ndarray,
                       land_mask: np.ndarray = None) -> np.ndarray:
        """
        Apply land mask to data

        Args:
            data: Data array
            land_mask: Land mask where 999 = land

        Returns:
            Masked data array
        """
        if land_mask is None:
            return data

        try:
            masked_data = data.copy()
            land_pixels = (land_mask == 999)
            masked_data[land_pixels] = np.nan
            logger.info(f"Applied land mask: {np.sum(land_pixels)} land pixels masked")
            return masked_data

        except Exception as e:
            logger.error(f"Error applying land mask: {e}")
            return data

    def calculate_difference_map(self,
                                ust21_data: np.ndarray,
                                modis_data: np.ndarray,
                                threshold: float = 0.1) -> Optional[np.ndarray]:
        """
        Calculate difference map between UST21 and MODIS

        Args:
            ust21_data: UST21 chlorophyll array
            modis_data: MODIS chlorophyll array (must be same shape as UST21)
            threshold: Threshold for treating values as zero (default: 0.1 mg/m³)
                      Values with absolute value < threshold are set to exactly 0

        Returns:
            Difference map (UST21 - MODIS) or None
        """
        try:
            # Ensure shapes match
            if ust21_data.shape != modis_data.shape:
                logger.error(f"Shape mismatch: UST21={ust21_data.shape}, MODIS={modis_data.shape}")
                return None

            # Apply land masks
            if self.ust_land_mask is not None and self.ust_land_mask.shape == ust21_data.shape:
                # Land mask matches the data shape directly
                ust21_masked = self.apply_land_mask(ust21_data.copy(), self.ust_land_mask)
                modis_masked = self.apply_land_mask(modis_data.copy(), self.ust_land_mask)
            elif self.ust_land_mask is not None:
                # Land mask needs to be scaled/cropped to match data shape
                logger.warning(f"Land mask shape {self.ust_land_mask.shape} doesn't match data shape {ust21_data.shape}")
                logger.info("Skipping land mask application")
                ust21_masked = ust21_data.copy()
                modis_masked = modis_data.copy()
            else:
                ust21_masked = ust21_data.copy()
                modis_masked = modis_data.copy()

            # Clean up near-zero values caused by interpolation artifacts
            # Set values very close to zero to exactly zero
            ust21_clean = ust21_masked.copy()
            modis_clean = modis_masked.copy()

            # Find near-zero values before cleaning
            valid_ust21 = ust21_clean[~np.isnan(ust21_clean)]
            valid_modis = modis_clean[~np.isnan(modis_clean)]

            if len(valid_ust21) > 0:
                ust21_near_zero = valid_ust21[(np.abs(valid_ust21) < 0.1) & (np.abs(valid_ust21) > 0)]
                if len(ust21_near_zero) > 0:
                    logger.info(f"UST21 near-zero values (0 < |val| < 0.1): min={np.min(np.abs(ust21_near_zero)):.2e}, max={np.max(np.abs(ust21_near_zero)):.2e}")

            if len(valid_modis) > 0:
                modis_near_zero = valid_modis[(np.abs(valid_modis) < 0.1) & (np.abs(valid_modis) > 0)]
                if len(modis_near_zero) > 0:
                    logger.info(f"MODIS near-zero values (0 < |val| < 0.1): min={np.min(np.abs(modis_near_zero)):.2e}, max={np.max(np.abs(modis_near_zero)):.2e}")

            ust21_clean[(np.abs(ust21_clean) < threshold) & ~np.isnan(ust21_clean)] = 0.0
            modis_clean[(np.abs(modis_clean) < threshold) & ~np.isnan(modis_clean)] = 0.0

            # Calculate difference
            difference = np.where(
                (~np.isnan(ust21_clean)) & (~np.isnan(modis_clean)),
                ust21_clean - modis_clean,
                np.nan
            )

            # Count zero differences
            zero_diffs = np.sum(difference == 0.0)
            near_zero_diffs = np.sum((np.abs(difference) < 0.1) & (difference != 0.0) & ~np.isnan(difference))

            logger.info(f"Calculated difference map with shape {difference.shape}")
            logger.info(f"Threshold for removing interpolation artifacts: {threshold}")
            logger.info(f"Zero differences (0.0): {zero_diffs}")
            logger.info(f"Near-zero differences (0 < |diff| < 0.1): {near_zero_diffs}")

            return difference.astype(np.float32)

        except Exception as e:
            logger.error(f"Error calculating difference map: {e}")
            return None

    def save_difference_map_as_png(self,
                                   period_str: str,
                                   difference: np.ndarray,
                                   vmin: float = -10,
                                   vmax: float = 10):
        """
        Save difference map as PNG and return RMSE

        Args:
            period_str: Period string (e.g., "20201201_20201208" for 01~08)
            difference: Difference map array
            vmin: Minimum colormap value (default: -10 mg/m³)
            vmax: Maximum colormap value (default: 10 mg/m³)

        Returns:
            RMSE value or None
        """
        try:
            png_filename = f"UST21_MODIS_difference_{period_str}.png"
            png_path = self.output_dir / png_filename

            # Calculate statistics (excluding 0 values for cleaner stats)
            valid_data = difference[~np.isnan(difference)]
            if len(valid_data) == 0:
                logger.warning(f"No valid data in difference map for {period_str}")
                return None

            actual_min = np.nanmin(difference)
            actual_max = np.nanmax(difference)
            actual_mean = np.nanmean(difference)
            actual_std = np.nanstd(difference)

            # Calculate RMSE
            rmse = np.sqrt(np.nanmean(difference ** 2))

            logger.info(f"Difference map statistics:")
            logger.info(f"  Min: {actual_min:.4f}, Max: {actual_max:.4f}")
            logger.info(f"  Mean: {actual_mean:.4f}, Std: {actual_std:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")

            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10), dpi=150)

            # Prepare data for visualization
            # 0 values → NaN (will be white)
            # NaN values (land) → -999 (will be gray)
            vis_data = difference.copy()

            # Mark land (NaN) with special value -999
            land_mask = np.isnan(vis_data)
            vis_data[land_mask] = -999

            # Mark 0 values as NaN (will appear white)
            zero_mask = (vis_data == 0.0)
            vis_data[zero_mask] = np.nan

            # Create masked array
            # Mask both NaN values (0-difference) and land pixels
            masked_data = np.ma.array(vis_data, mask=(land_mask | zero_mask))

            # Create custom colormap: RdBu_r with gray for land
            from matplotlib.colors import ListedColormap
            import matplotlib.cm as cm

            # Get RdBu_r colormap
            cmap = cm.get_cmap('RdBu_r')

            # Create extended colormap with gray for land (-999 values)
            # We'll handle this by plotting in two steps

            # UST21 coordinate bounds (from NetCDF metadata)
            # Upper left: 23.17°N, 150.855°E
            # Lower right: 49.1579°N, 111.604°E
            lat_min, lat_max = 23.17, 49.1579
            lon_min, lon_max = 111.604, 150.855

            # Get image dimensions
            height, width = difference.shape

            # extent = [left, right, bottom, top] for imshow with origin='upper'
            # left = lon_min, right = lon_max, bottom = lat_min, top = lat_max
            extent = [lon_min, lon_max, lat_min, lat_max]

            # Calculate aspect ratio to preserve image dimensions
            # aspect = (extent_width / extent_height) / (pixel_width / pixel_height)
            extent_width = lon_max - lon_min  # degrees longitude
            extent_height = lat_max - lat_min  # degrees latitude
            pixel_aspect = width / height
            geo_aspect = extent_width / extent_height
            aspect_ratio = geo_aspect / pixel_aspect

            # First, plot the land as gray
            im = ax.imshow(vis_data, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          origin='upper', interpolation='none', extent=extent, aspect=aspect_ratio)

            # Overlay gray for land pixels
            gray_data = np.where(land_mask, 1, np.nan)
            ax.imshow(gray_data, cmap='gray', vmin=0, vmax=1,
                     origin='upper', interpolation='none', extent=extent, aspect=aspect_ratio, alpha=0.7)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label='Chlorophyll difference (mg/m³)', shrink=0.8)

            # Labels and title
            ax.set_xlabel('Longitude (°E)', fontsize=12)
            ax.set_ylabel('Latitude (°N)', fontsize=12)
            ax.set_title(f'UST21 - MODIS Chlorophyll Difference Map\nPeriod: {period_str}\n' +
                        f'RMSE: {rmse:.4f} mg/m³, Mean: {actual_mean:.4f} mg/m³, Std: {actual_std:.4f} mg/m³\n' +
                        f'(White: 0 difference, Gray: land)', fontsize=12)

            # Save
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved difference map PNG to {png_path}")
            logger.info(f"Color scheme: Red-Blue for difference values, White for 0-difference, Gray for land")

            return rmse

        except Exception as e:
            logger.error(f"Error saving difference map as PNG: {e}")
            return None

    def save_modis_crop_visualization(self,
                                     period_str: str,
                                     modis_data: np.ndarray,
                                     crop_info: Dict):
        """
        Save MODIS full image with crop region marked in red box

        Args:
            period_str: Period string
            modis_data: Full MODIS data (shape: height x width)
            crop_info: Dictionary containing crop region information
        """
        try:
            if not crop_info or 'y_start' not in crop_info:
                logger.warning("No crop info available for visualization")
                return

            fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

            # Plot full MODIS data
            # modis_data shape: (height, width) = (y_pixels, x_pixels)
            masked_modis = np.ma.masked_invalid(modis_data)
            im = ax.imshow(masked_modis, cmap='viridis', vmin=0, vmax=10,
                          origin='upper', interpolation='none')

            # Draw red rectangle for crop region
            # crop_info contains (y_start, y_end, x_start, x_end) in pixel indices
            y_start = crop_info['y_start']
            y_end = crop_info['y_end']
            x_start = crop_info['x_start']
            x_end = crop_info['x_end']

            # For imshow, Rectangle coordinate is (x_pixel, y_pixel) for the top-left corner
            # width = x_end - x_start, height = y_end - y_start
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                           linewidth=3, edgecolor='red', facecolor='none', label='Crop Region')
            ax.add_patch(rect)

            # Add text annotation showing the actual size
            crop_height = y_end - y_start
            crop_width = x_end - x_start
            aspect_ratio = crop_height / crop_width

            info_text = f'Crop Size: {crop_height}px (height) × {crop_width}px (width)\nAspect Ratio: {aspect_ratio:.3f}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label='Chlorophyll concentration (mg/m³)', shrink=0.8)

            # Labels and title
            ax.set_xlabel('X pixel index (Longitude direction)')
            ax.set_ylabel('Y pixel index (Latitude direction)')
            ax.set_title(f'MODIS Full Image with Crop Region\nPeriod: {period_str}')
            ax.legend(loc='upper right')

            # Save
            png_filename = f"MODIS_crop_region_{period_str}.png"
            png_path = self.output_dir / png_filename
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved MODIS crop region visualization to {png_path}")
            logger.info(f"  Crop size: {crop_height}px (세로) × {crop_width}px (가로)")
            logger.info(f"  Aspect ratio: {aspect_ratio:.3f}")

        except Exception as e:
            logger.error(f"Error saving MODIS crop visualization: {e}")
            import traceback
            traceback.print_exc()

    def save_comparison_map_as_png(self,
                                  period_str: str,
                                  ust21_data: np.ndarray,
                                  modis_data: np.ndarray):
        """
        Save comparison map with UST21, MODIS, and difference

        Args:
            period_str: Period string
            ust21_data: UST21 data
            modis_data: MODIS data
        """
        try:
            # Apply land masks (check shape compatibility)
            if self.ust_land_mask is not None and self.ust_land_mask.shape == ust21_data.shape:
                ust21_masked = self.apply_land_mask(ust21_data.copy(), self.ust_land_mask)
                modis_masked = self.apply_land_mask(modis_data.copy(), self.ust_land_mask)
            else:
                ust21_masked = ust21_data.copy()
                modis_masked = modis_data.copy()

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

            # UST21 coordinate bounds (from NetCDF metadata)
            lat_min, lat_max = 23.17, 49.1579
            lon_min, lon_max = 111.604, 150.855
            extent = [lon_min, lon_max, lat_min, lat_max]

            # Calculate aspect ratio to preserve image dimensions
            height, width = ust21_masked.shape
            extent_width = lon_max - lon_min  # degrees longitude
            extent_height = lat_max - lat_min  # degrees latitude
            pixel_aspect = width / height
            geo_aspect = extent_width / extent_height
            aspect_ratio = geo_aspect / pixel_aspect

            # Get common scaling for UST21 and MODIS (0-10 mg/m³)
            valid_ust21 = ust21_masked[~np.isnan(ust21_masked)]
            valid_modis = modis_masked[~np.isnan(modis_masked)]

            if len(valid_ust21) > 0 and len(valid_modis) > 0:
                chl_vmin = 0
                chl_vmax = 10

                # UST21 map (0-10 mg/m³)
                im1 = axes[0].imshow(np.ma.masked_invalid(ust21_masked), cmap='viridis',
                                    vmin=chl_vmin, vmax=chl_vmax, origin='upper', extent=extent, aspect=aspect_ratio)
                axes[0].set_title('UST21 Chlorophyll')
                axes[0].set_xlabel('Longitude (°E)')
                axes[0].set_ylabel('Latitude (°N)')
                plt.colorbar(im1, ax=axes[0], label='mg/m³')

                # MODIS map (0-10 mg/m³)
                im2 = axes[1].imshow(np.ma.masked_invalid(modis_masked), cmap='viridis',
                                    vmin=chl_vmin, vmax=chl_vmax, origin='upper', extent=extent, aspect=aspect_ratio)
                axes[1].set_title('MODIS Chlorophyll')
                axes[1].set_xlabel('Longitude (°E)')
                axes[1].set_ylabel('Latitude (°N)')
                plt.colorbar(im2, ax=axes[1], label='mg/m³')

                # Difference map (-10~10 mg/m³)
                difference = ust21_masked - modis_masked
                im3 = axes[2].imshow(np.ma.masked_invalid(difference), cmap='RdBu_r',
                                    vmin=-10, vmax=10, origin='upper', extent=extent, aspect=aspect_ratio)
                axes[2].set_title('UST21 - MODIS Difference')
                axes[2].set_xlabel('Longitude (°E)')
                axes[2].set_ylabel('Latitude (°N)')
                plt.colorbar(im3, ax=axes[2], label='mg/m³')

                fig.suptitle(f'Chlorophyll Comparison (Period: {period_str})', fontsize=14)

                png_filename = f"UST21_MODIS_comparison_{period_str}.png"
                png_path = self.output_dir / png_filename
                plt.tight_layout()
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved comparison map PNG to {png_path}")

        except Exception as e:
            logger.error(f"Error saving comparison map: {e}")

    def process_8day_period(self, start_date: str):
        """
        Process a single 8-day period

        Args:
            start_date: Start date in YYYYMMDD format

        Returns:
            (success: bool, rmse: float or None, period_str: str)
        """
        logger.info(f"Processing 8-day period starting from {start_date}")

        # Calculate end date (8 days later)
        from datetime import datetime, timedelta
        date_obj = datetime.strptime(start_date, '%Y%m%d')
        end_date_obj = date_obj + timedelta(days=7)
        end_date = end_date_obj.strftime('%Y%m%d')
        period_str = f"{start_date}_{end_date}"

        try:
            # Load UST21 data
            ust21_data = self.load_ust21_8day_average(start_date)
            if ust21_data is None:
                logger.error(f"Failed to load UST21 data for {start_date}")
                return False, None, period_str

            logger.info(f"UST21 data shape: {ust21_data.shape}")

            # Load MODIS data
            modis_data = self.load_modis_8day_average(start_date)
            if modis_data is None:
                logger.error(f"Failed to load MODIS data for {start_date}")
                return False, None, period_str

            logger.info(f"MODIS data shape (original): {modis_data.shape}")

            # Crop and align MODIS to UST21 region
            modis_cropped, crop_info = self.crop_and_align_modis_to_ust21(modis_data, ust21_data)
            if modis_cropped is None:
                logger.error(f"Failed to crop and align MODIS data")
                return False, None, period_str

            logger.info(f"MODIS data shape (final): {modis_cropped.shape}")

            # Debug: Check MODIS data range before interpolation
            valid_modis_crop = modis_cropped[~np.isnan(modis_cropped)]
            if len(valid_modis_crop) > 0:
                logger.info(f"MODIS cropped data range: min={np.min(valid_modis_crop):.4f}, max={np.max(valid_modis_crop):.4f}, mean={np.mean(valid_modis_crop):.4f}")

            modis_interpolated = modis_cropped

            # Debug: Check MODIS data range after interpolation
            valid_modis_interp = modis_interpolated[~np.isnan(modis_interpolated)]
            if len(valid_modis_interp) > 0:
                logger.info(f"MODIS interpolated data range: min={np.min(valid_modis_interp):.4f}, max={np.max(valid_modis_interp):.4f}, mean={np.mean(valid_modis_interp):.4f}")

            # Debug: Check UST21 data range
            valid_ust21 = ust21_data[~np.isnan(ust21_data)]
            if len(valid_ust21) > 0:
                logger.info(f"UST21 data range: min={np.min(valid_ust21):.4f}, max={np.max(valid_ust21):.4f}, mean={np.mean(valid_ust21):.4f}")

            # Calculate difference map
            difference = self.calculate_difference_map(ust21_data, modis_interpolated)
            if difference is None:
                logger.error(f"Failed to calculate difference map")
                return False, None, period_str

            # Save visualizations and get RMSE
            rmse = self.save_difference_map_as_png(period_str, difference)
            self.save_comparison_map_as_png(period_str, ust21_data, modis_interpolated)
            self.save_modis_crop_visualization(period_str, modis_data, crop_info)

            logger.info(f"Successfully processed period {period_str}")
            return True, rmse, period_str

        except Exception as e:
            logger.error(f"Error processing 8-day period for {start_date}: {e}")
            import traceback
            traceback.print_exc()
            return False, None, period_str


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate difference maps between UST21 and MODIS chlorophyll'
    )
    parser.add_argument('--ust21_perf_dir', type=str, required=True,
                       help='UST21 performance directory')
    parser.add_argument('--modis_dir', type=str, required=True,
                       help='MODIS Aqua 8-day directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--ust_land_mask',
                       help='Path to UST21 land mask file')
    parser.add_argument('--dates', type=str, nargs='+',
                       help='Specific start dates to process (YYYYMMDD format)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize processor
    processor = UST21ModisDifferenceMap(
        args.ust21_perf_dir,
        args.modis_dir,
        args.output_dir,
        ust_land_mask_path=args.ust_land_mask
    )

    # Default dates (adjust as needed)
    if args.dates:
        target_dates = args.dates
    else:
        # 8-day periods: 01~08, 02~09, 03~10, ..., 12~12
        target_dates = [
            '20201201', '20201202', '20201203', '20201204', '20201205',
            '20201206', '20201207', '20201208', '20201209', '20201210',
            '20201211', '20201212', '20201213', '20201214', '20201215',
            '20201216', '20201217', '20201218', '20201219', '20201220',
            '20201221', '20201222', '20201223', '20201224'
        ]

    success_count = 0
    failed_count = 0
    rmse_results = []  # Store RMSE results for each period

    for date_str in target_dates:
        success, rmse, period_str = processor.process_8day_period(date_str)
        if success:
            success_count += 1
            if rmse is not None:
                rmse_results.append({
                    'period': period_str,
                    'start_date': date_str,
                    'rmse': rmse
                })
        else:
            failed_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total periods requested: {len(target_dates)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to: {args.output_dir}")

    # Save RMSE results to CSV
    if rmse_results:
        import csv
        csv_path = Path(args.output_dir) / 'rmse_summary.csv'

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Period', 'Start_Date', 'RMSE (mg/m³)'])

            for result in rmse_results:
                writer.writerow([result['period'], result['start_date'], f"{result['rmse']:.6f}"])

            # Calculate and write average RMSE
            avg_rmse = np.mean([r['rmse'] for r in rmse_results])
            writer.writerow([])
            writer.writerow(['Average', '', f"{avg_rmse:.6f}"])

        print(f"\n{'='*60}")
        print(f"RMSE STATISTICS")
        print(f"{'='*60}")
        print(f"Average RMSE: {avg_rmse:.6f} mg/m³")
        print(f"RMSE summary saved to: {csv_path}")

        # Print individual period RMSE values
        print(f"\nPer-period RMSE values:")
        for result in rmse_results:
            print(f"  {result['period']}: {result['rmse']:.6f} mg/m³")


if __name__ == '__main__':
    main()
