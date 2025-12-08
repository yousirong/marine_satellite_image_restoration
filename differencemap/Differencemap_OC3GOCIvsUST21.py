#!/usr/bin/env python3
"""
Daily OC3 chlorophyll data processing and difference map calculation.
This script supports two modes:
- Hourly mode: Averages multiple hourly OC3 results into daily chlorophyll
- Daily mode (default): Uses OC3 results from daily-averaged RRS data (no averaging needed)

For both modes:
1. Loads OC3 chlorophyll data (either averaging hourly or loading daily directly)
2. Calculates difference map between OC3 daily and existing KHOA daily chlorophyll
3. Saves results as PNG visualization files
"""

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import glob
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.ndimage import zoom

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DailyAveraging:
    """
    Process daily-averaged OC3 chlorophyll data and calculate difference maps
    Supports both hourly-averaged and direct daily OC3 results
    """

    def __init__(self,
                 oc3_results_dir: str,
                 khoa_daily_dir: str,
                 output_dir: str,
                 goci_land_mask_path: str = None,
                 ust_land_mask_path: str = None,
                 daily_mode: bool = False):
        """
        Initialize DailyAveraging processor

        Args:
            oc3_results_dir: Directory containing OC3 results
                             - hourly mode: YYYYMMDD_HHMMSS subdirectories
                             - daily mode: YYYYMMDD_daily subdirectories
            khoa_daily_dir: Directory containing KHOA daily chlorophyll files
            output_dir: Output directory for daily averages and difference maps
            goci_land_mask_path: Path to GOCI land mask file
            ust_land_mask_path: Path to UST21 land mask file
            daily_mode: If True, expects daily-averaged OC3 results (no hourly averaging needed)
        """
        self.oc3_results_dir = Path(oc3_results_dir)
        self.khoa_daily_dir = Path(khoa_daily_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.daily_mode = daily_mode

        # Load land masks
        self.goci_land_mask = None
        self.ust_land_mask = None

        if goci_land_mask_path and Path(goci_land_mask_path).exists():
            self.goci_land_mask = np.load(goci_land_mask_path)
            # logger.info(f"Loaded GOCI land mask: {self.goci_land_mask.shape}")

        if ust_land_mask_path and Path(ust_land_mask_path).exists():
            self.ust_land_mask = np.load(ust_land_mask_path)
            # logger.info(f"Loaded UST21 land mask: {self.ust_land_mask.shape}")

    def get_daily_timestamps(self, date_str: str) -> List[str]:
        """
        Get all hourly timestamps for a given date (hourly mode) or daily timestamp (daily mode)

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            List of timestamp directories
            - hourly mode: YYYYMMDD_HHMMSS format
            - daily mode: [YYYYMMDD_daily]
        """
        # date_str이 완전한 형식(YYYYMMDD)인지 확인
        if len(date_str) != 8:
            # logger.warning(f"Invalid date format: {date_str}. Expected YYYYMMDD format.")
            return []

        if self.daily_mode:
            # For daily-averaged data: look for YYYYMMDD_daily directory
            daily_timestamp = f"{date_str}_daily"
            daily_dir = self.oc3_results_dir / daily_timestamp
            if daily_dir.exists() and daily_dir.is_dir():
                # logger.info(f"Found daily result for {date_str}: {daily_timestamp}")
                return [daily_timestamp]
            else:
                # logger.warning(f"Daily directory not found: {daily_dir}")
                return []
        else:
            # List all directories matching the date pattern (YYYYMMDD_*)
            date_pattern = f"{date_str}_*"
            timestamp_dirs = sorted(self.oc3_results_dir.glob(date_pattern))

            timestamps = [d.name for d in timestamp_dirs if d.is_dir()]
            # logger.info(f"Found {len(timestamps)} hourly results for {date_str}: {timestamps}")

            return timestamps

    def clip_chlorophyll_range(self, data: np.ndarray, min_val: float = 0.01, max_val: float = 10.0) -> np.ndarray:
        """
        Filter chlorophyll values to scientifically valid range
        Values outside the range are set to NaN (excluded from analysis)

        Args:
            data: Chlorophyll data array
            min_val: Minimum valid value (default: 0.01 mg/m³)
            max_val: Maximum valid value (default: 10.0 mg/m³)

        Returns:
            Filtered data with out-of-range values set to NaN
        """
        filtered = data.copy()

        # Set values outside the valid range to NaN
        invalid_mask = (filtered < min_val) | (filtered > max_val)
        filtered[invalid_mask] = np.nan

        return filtered.astype(np.float32)

    def load_tile_data(self,
                       timestamp: str,
                       tile_id: str) -> Optional[np.ndarray]:
        """
        Load chlorophyll data for a specific tile and timestamp

        Args:
            timestamp: Timestamp in YYYYMMDD_HHMMSS format (hourly mode) or YYYYMMDD_daily (daily mode)
            tile_id: Tile identifier (e.g., mask_10_y0000_x2304 or img_504_y5429_x4864)

        Returns:
            Numpy array of chlorophyll data or None if file not found
        """
        chl_file = (self.oc3_results_dir / timestamp / timestamp / tile_id /
                    'chlorophyll_oc3.csv')

        if not chl_file.exists():
            # logger.warning(f"File not found: {chl_file}")
            return None

        try:
            data = np.loadtxt(chl_file, delimiter=',', dtype=np.float32)

            # Remove extreme outliers and invalid values
            # Set values outside valid range to 0
            data[data < 0] = 0.0          # Negative values → 0

            # Clip to valid chlorophyll range (0.01 ~ 10.0 mg/m³)
            data = self.clip_chlorophyll_range(data, min_val=0.01, max_val=10.0)

            return data
        except Exception as e:
            logger.error(f"Error loading {chl_file}: {e}")
            return None

    def get_available_tiles(self, timestamp: str) -> List[str]:
        """
        Get list of available tiles for a given timestamp

        Args:
            timestamp: Timestamp in YYYYMMDD_HHMMSS format (hourly mode) or YYYYMMDD_daily (daily mode)

        Returns:
            List of tile IDs
        """
        tile_dir = self.oc3_results_dir / timestamp / timestamp

        if not tile_dir.exists():
            # logger.warning(f"Tile directory not found: {tile_dir}")
            return []

        tiles = [d.name for d in tile_dir.iterdir() if d.is_dir()]
        return sorted(tiles)

    def average_hourly_to_daily(self,
                                date_str: str,
                                tile_id: str) -> Optional[np.ndarray]:
        """
        Average hourly chlorophyll data into daily average for a specific tile

        Args:
            date_str: Date in YYYYMMDD format
            tile_id: Tile identifier

        Returns:
            Daily averaged chlorophyll array or None if insufficient data
        """
        timestamps = self.get_daily_timestamps(date_str)

        if not timestamps:
            # logger.warning(f"No timestamps found for {date_str}")
            return None

        hourly_data = []

        for timestamp in timestamps:
            data = self.load_tile_data(timestamp, tile_id)
            if data is not None:
                hourly_data.append(data)

        if not hourly_data:
            # logger.warning(f"No valid data loaded for tile {tile_id} on {date_str}")
            return None

        # Stack data arrays
        stacked_data = np.stack(hourly_data, axis=0)

        # Calculate daily average, ignoring NaN values
        # Suppress RuntimeWarning for mean of empty slice
        with np.errstate(invalid='ignore'):
            daily_avg = np.nanmean(stacked_data, axis=0)

        # Replace all-nan slices with NaN
        daily_avg = np.where(np.isnan(daily_avg), np.nan, daily_avg)

        # Check if the entire tile is filled with a single value (empty patch filled with median/mean)
        # Exclude NaN and 0 from the check
        valid_values = daily_avg[(~np.isnan(daily_avg)) & (daily_avg != 0)]
        if len(valid_values) > 0:
            unique_vals = np.unique(valid_values)
            if len(unique_vals) == 1:
                # Entire tile has the same non-zero value - likely empty patch
                # Fill with 0
                daily_avg[:] = 0.0
                # logger.warning(f"Empty tile detected (all values = {unique_vals[0]:.6f}), filled with 0: {tile_id}")

        return daily_avg.astype(np.float32)

    def load_khoa_daily_data(self, date_str: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Load KHOA daily chlorophyll data

        Args:
            date_str: Date in YYYYMMDD format (e.g., 20210101)

        Returns:
            Tuple of (chlorophyll_data, metadata) or None if not found
        """
        # Construct filename: KHOA_Chla_L3_Z001_D01_WGS250M_U20210101.nc
        nc_filename = f"KHOA_Chla_L3_Z001_D01_WGS250M_U{date_str}.nc"
        nc_path = self.khoa_daily_dir / nc_filename

        if not nc_path.exists():
            # logger.warning(f"KHOA file not found: {nc_path}")
            return None

        try:
            import netCDF4 as nc
            ds = nc.Dataset(str(nc_path), 'r')

            # Extract chlorophyll data
            if 'merged_daily_Chl' in ds.variables:
                chl_data = ds.variables['merged_daily_Chl'][:].astype(np.float32)

                # Clip to valid chlorophyll range (0.01 ~ 10.0 mg/m³)
                chl_data = self.clip_chlorophyll_range(chl_data, min_val=0.01, max_val=10.0)
            else:
                logger.error(f"Variable 'merged_daily_Chl' not found in {nc_path}")
                ds.close()
                return None

            # Extract metadata
            metadata = {
                'lon': ds.variables['lon'][:].astype(np.float32) if 'lon' in ds.variables else None,
                'lat': ds.variables['lat'][:].astype(np.float32) if 'lat' in ds.variables else None,
                'time': ds.variables['time'][0] if 'time' in ds.variables else None,
                'attributes': {}
            }

            # Safely extract attributes
            try:
                for attr_name in ds.ncattrs():
                    try:
                        metadata['attributes'][attr_name] = ds.getncattr(attr_name)
                    except:
                        pass
            except:
                pass

            ds.close()

            # logger.info(f"Loaded KHOA data from {nc_path} with shape {chl_data.shape}")
            return chl_data, metadata

        except Exception as e:
            logger.error(f"Error loading KHOA file {nc_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def reconstruct_daily_from_tiles(self,
                                     date_str: str) -> Optional[np.ndarray]:
        """
        Reconstruct daily chlorophyll data from all tiles
        - In hourly mode: averages hourly tiles into daily
        - In daily mode: directly loads daily tiles (no averaging needed)

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            Reconstructed 2D daily chlorophyll array or None
        """
        # Get available tiles from first timestamp
        timestamps = self.get_daily_timestamps(date_str)
        if not timestamps:
            logger.error(f"No timestamps found for {date_str}")
            return None

        tiles = self.get_available_tiles(timestamps[0])
        if not tiles:
            logger.error(f"No tiles found for {date_str}")
            return None

        # logger.info(f"Processing {len(tiles)} tiles for {date_str}")

        # Dictionary to store data for each tile
        tile_data = {}

        if self.daily_mode:
            # For daily mode: directly load tiles (no averaging needed)
            for tile_id in tiles:
                tile_array = self.load_tile_data(timestamps[0], tile_id)
                if tile_array is not None:
                    tile_data[tile_id] = tile_array
        else:
            # For hourly mode: average hourly tiles into daily
            for tile_id in tiles:
                daily_tile = self.average_hourly_to_daily(date_str, tile_id)
                if daily_tile is not None:
                    tile_data[tile_id] = daily_tile

        if not tile_data:
            logger.error(f"No valid tile data found for {date_str}")
            return None

        # Parse tile coordinates to reconstruct the full grid
        # Tile format: mask_<idx>_y<y_start>_x<x_start> or img_<idx>_y<y_start>_x<x_start>
        # Each tile is typically 256x256 pixels
        full_data = self.reconstruct_grid_from_tiles(tile_data)

        # Remove vertical stripe artifacts (columns filled with single value)
        if full_data is not None:
            full_data = self.remove_vertical_stripes(full_data)

        return full_data

    def reconstruct_grid_from_tiles(self, tile_data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Reconstruct full grid from tile data

        Args:
            tile_data: Dictionary mapping tile_id to data array

        Returns:
            Reconstructed full grid or None
        """
        try:
            # Extract position information from tile IDs
            positions = {}
            for tile_id, data in tile_data.items():
                # Parse: mask_<idx>_y<y_start>_x<x_start> or img_<idx>_y<y_start>_x<x_start>
                parts = tile_id.split('_')
                if len(parts) >= 4:
                    y_start = int(parts[2][1:])  # Remove 'y' prefix
                    x_start = int(parts[3][1:])  # Remove 'x' prefix
                    positions[tile_id] = (y_start, x_start, data.shape)

            if not positions:
                logger.error("Could not parse tile positions")
                return None

            # Determine grid dimensions
            max_y = max(y + shape[0] for y, x, shape in positions.values())
            max_x = max(x + shape[1] for y, x, shape in positions.values())

            # logger.info(f"Reconstructing grid: {max_y} x {max_x}")

            # Initialize full grid with NaN
            full_grid = np.full((max_y, max_x), np.nan, dtype=np.float32)

            # Fill in tile data
            for tile_id, data in tile_data.items():
                y_start, x_start, _ = positions[tile_id]
                y_end = y_start + data.shape[0]
                x_end = x_start + data.shape[1]
                full_grid[y_start:y_end, x_start:x_end] = data

            # logger.info(f"Successfully reconstructed grid with shape {full_grid.shape}")
            return full_grid

        except Exception as e:
            logger.error(f"Error reconstructing grid from tiles: {e}")
            return None

    def remove_vertical_stripes(self, data: np.ndarray, tile_width: int = 256) -> np.ndarray:
        """
        Remove vertical stripe artifacts caused by empty tiles filled with uniform values
        Also handles near-zero values that appear as artifacts

        Args:
            data: Full grid data
            tile_width: Width of tiles (default 256)

        Returns:
            Data with vertical stripes removed (replaced with 0)
        """
        try:
            cleaned_data = data.copy()
            stripe_count = 0

            # Check each vertical stripe at tile boundaries
            for x_start in range(0, data.shape[1], tile_width):
                x_end = min(x_start + tile_width, data.shape[1])
                column_block = cleaned_data[:, x_start:x_end]

                # Check if this column block is filled with a single non-zero value
                valid_values = column_block[(~np.isnan(column_block)) & (column_block != 0)]

                if len(valid_values) > 100:  # Require at least 100 pixels to detect pattern
                    unique_vals = np.unique(valid_values)
                    # Check for uniform value (artifact) or very small values
                    if len(unique_vals) == 1:
                        val = unique_vals[0]
                        # Remove if it's a uniform value (artifact from empty tile)
                        if val < 0.01 or (np.std(valid_values) < 0.001 and len(unique_vals) == 1):
                            cleaned_data[:, x_start:x_end] = 0
                            stripe_count += 1
                            # logger.warning(f"Removed vertical stripe at x={x_start}:{x_end} (value={val:.6f})")

            if stripe_count > 0:
                # logger.info(f"Removed {stripe_count} vertical stripes from reconstructed data")
                pass

            return cleaned_data

        except Exception as e:
            logger.error(f"Error removing vertical stripes: {e}")
            return data

    def interpolate_to_goci_resolution(self,
                                       ust_data: np.ndarray,
                                       target_shape: tuple = None) -> np.ndarray:
        """
        Interpolate UST21 data to target GOCI resolution

        Args:
            ust_data: UST21 data at UST21 resolution (8000x10500)
            target_shape: Target shape (height, width). If None, uses GOCI standard shape (5685x5566)

        Returns:
            UST21 data interpolated to target resolution
        """
        try:
            # Use target shape if provided, otherwise use standard GOCI shape
            if target_shape is None:
                target_shape = (5685, 5566)

            zoom_y = target_shape[0] / ust_data.shape[0]
            zoom_x = target_shape[1] / ust_data.shape[1]

            # logger.info(f"Interpolating UST21 from {ust_data.shape} to {target_shape}")
            # logger.info(f"Zoom factors: y={zoom_y:.4f}, x={zoom_x:.4f}")

            # Use zoom for interpolation (linear interpolation for better quality downsampling)
            interpolated = zoom(ust_data, (zoom_y, zoom_x), order=1, prefilter=False)

            # logger.info(f"Interpolated shape: {interpolated.shape}")
            return interpolated.astype(np.float32)

        except Exception as e:
            logger.error(f"Error interpolating UST21 data: {e}")
            return None

    def apply_land_masks(self,
                        data: np.ndarray,
                        is_land_mask: np.ndarray = None) -> np.ndarray:
        """
        Apply land mask to data (keep original values, will be overlaid with black)

        Args:
            data: Data array
            is_land_mask: Land mask where 999 = land, 1 = water

        Returns:
            Data array with land pixels unchanged (black overlay will be applied separately)
        """
        if is_land_mask is None:
            return data

        try:
            # Return data unchanged - land pixels will be visualized via black overlay
            # This allows the overlay to work correctly without NaN rendering issues
            masked_data = data.copy()
            # logger.info(f"Land masking will be applied via overlay visualization")
            return masked_data

        except Exception as e:
            logger.error(f"Error in apply_land_masks: {e}")
            return data

    def calculate_difference_map(self,
                                 oc3_daily: np.ndarray,
                                 khoa_daily: np.ndarray):
        """
        Prepare OC3 and KHOA maps in original resolutions without interpolation

        Args:
            oc3_daily: OC3 daily chlorophyll array (GOCI resolution: 5685x5566)
            khoa_daily: KHOA daily chlorophyll array (UST21 resolution: 8000x10500)

        Returns:
            Tuple of (oc3_masked, khoa_masked, oc3_land_mask, khoa_land_mask)
            - oc3_masked: OC3 map with land masked at GOCI resolution
            - khoa_masked: KHOA map with land masked at UST21 resolution
            - oc3_land_mask: Land mask for OC3 at GOCI resolution
            - khoa_land_mask: Land mask for KHOA at UST21 resolution
        """
        try:
            # Keep both datasets in original resolutions - NO INTERPOLATION
            # logger.info(f"OC3 shape: {oc3_daily.shape}, KHOA shape: {khoa_daily.shape}")
            # logger.info("Creating comparison: OC3 at GOCI resolution, KHOA at original resolution")

            # Apply land masks to each dataset independently using appropriate masks
            # OC3 gets GOCI land mask
            if self.goci_land_mask is not None:
                oc3_masked = self.apply_land_masks(oc3_daily.copy(), self.goci_land_mask)
                oc3_land_mask = (self.goci_land_mask == 999)
                # logger.info(f"OC3 land mask: {np.sum(oc3_land_mask)} land pixels at GOCI resolution")
            elif self.ust_land_mask is not None:
                oc3_masked = self.apply_land_masks(oc3_daily.copy(), self.ust_land_mask)
                oc3_land_mask = None
                # logger.info("OC3 using UST21 land mask (not ideal)")
            else:
                oc3_masked = oc3_daily.copy()
                oc3_land_mask = None

            # KHOA gets UST21 land mask
            if self.ust_land_mask is not None:
                khoa_masked = self.apply_land_masks(khoa_daily.copy(), self.ust_land_mask)
                khoa_land_mask = (self.ust_land_mask == 999)
                # logger.info(f"KHOA land mask: {np.sum(khoa_land_mask)} land pixels at UST21 resolution")
            else:
                khoa_masked = khoa_daily.copy()
                khoa_land_mask = None

            # logger.info(f"Prepared OC3 and KHOA maps in original resolutions")
            # Return both masked maps and land mask info for visualization
            return oc3_masked, khoa_masked, oc3_land_mask, khoa_land_mask

        except Exception as e:
            logger.error(f"Error preparing maps for comparison: {e}")
            return None, None, None, None

    def interpolate_land_mask_to_oc3(self,
                                     land_mask: np.ndarray,
                                     target_shape: tuple) -> np.ndarray:
        """
        Interpolate land mask to OC3 resolution
        Uses nearest neighbor interpolation for land/water masks

        Args:
            land_mask: Land mask array (999=land, other=water)
            target_shape: Target shape (OC3 resolution)

        Returns:
            Interpolated land mask at target resolution
        """
        try:
            if land_mask.shape == target_shape:
                return land_mask

            logger.info(f"Interpolating land mask from {land_mask.shape} to {target_shape}")

            # Calculate zoom factors
            zoom_y = target_shape[0] / land_mask.shape[0]
            zoom_x = target_shape[1] / land_mask.shape[1]

            logger.info(f"Land mask zoom factors: y={zoom_y:.4f}, x={zoom_x:.4f}")

            # Use nearest neighbor (order=0) for categorical mask data
            mask_interpolated = zoom(land_mask, (zoom_y, zoom_x), order=0, prefilter=False)

            # Ensure exact shape match
            mask_interpolated = mask_interpolated[:target_shape[0], :target_shape[1]]

            logger.info(f"Land mask interpolated to shape: {mask_interpolated.shape}")
            return mask_interpolated.astype(np.float32)

        except Exception as e:
            logger.error(f"Error interpolating land mask: {e}")
            return None

    def align_khoa_to_oc3_resolution(self,
                                     oc3_data: np.ndarray,
                                     khoa_data: np.ndarray) -> np.ndarray:
        """
        Interpolate KHOA data to OC3 resolution for proper alignment

        Args:
            oc3_data: OC3 chlorophyll data (reference resolution)
            khoa_data: KHOA chlorophyll data (needs to be interpolated)

        Returns:
            KHOA data interpolated to OC3 resolution
        """
        try:
            if oc3_data.shape == khoa_data.shape:
                logger.info(f"OC3 and KHOA already same shape: {oc3_data.shape}")
                return khoa_data

            oc3_shape = oc3_data.shape
            khoa_shape = khoa_data.shape

            logger.info(f"Interpolating KHOA from {khoa_shape} to OC3 shape {oc3_shape}")

            # Calculate zoom factors
            zoom_y = oc3_shape[0] / khoa_shape[0]
            zoom_x = oc3_shape[1] / khoa_shape[1]

            logger.info(f"Zoom factors: y={zoom_y:.4f}, x={zoom_x:.4f}")

            # Use scipy zoom for bilinear interpolation
            khoa_interpolated = zoom(khoa_data, (zoom_y, zoom_x), order=1, prefilter=False)

            # Ensure exact shape match
            khoa_interpolated = khoa_interpolated[:oc3_shape[0], :oc3_shape[1]]

            logger.info(f"KHOA interpolated to shape: {khoa_interpolated.shape}")
            return khoa_interpolated.astype(np.float32)

        except Exception as e:
            logger.error(f"Error aligning KHOA to OC3 resolution: {e}")
            return None

    def calculate_oc3_aligned_difference(self,
                                        date_str: str,
                                        oc3_daily: np.ndarray,
                                        khoa_daily: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate difference map with OC3 as reference resolution
        Interpolates KHOA to OC3 resolution for proper comparison

        Args:
            date_str: Date string for logging
            oc3_daily: OC3 chlorophyll data at GOCI resolution
            khoa_daily: KHOA chlorophyll data at UST21 resolution

        Returns:
            Difference map array (OC3 - KHOA) at OC3 resolution
        """
        try:
            logger.info(f"Starting OC3-aligned difference calculation")
            logger.info(f"OC3 input shape: {oc3_daily.shape}")
            logger.info(f"KHOA input shape: {khoa_daily.shape}")
            logger.info(f"GOCI land mask shape: {self.goci_land_mask.shape if self.goci_land_mask is not None else 'None'}")

            # Step 1: Interpolate KHOA to OC3 resolution
            khoa_aligned = self.align_khoa_to_oc3_resolution(oc3_daily, khoa_daily)
            if khoa_aligned is None:
                logger.error(f"Failed to align KHOA to OC3 resolution")
                return None

            logger.info(f"OC3 shape: {oc3_daily.shape}, KHOA aligned shape: {khoa_aligned.shape}")

            # Step 2: Apply land masks (ensure shapes match)
            oc3_masked = oc3_daily.copy()

            if self.goci_land_mask is not None:
                # Crop GOCI land mask to match OC3 data shape
                goci_mask_cropped = self.goci_land_mask[:oc3_daily.shape[0], :oc3_daily.shape[1]]

                # If still not matching, crop both to match
                if goci_mask_cropped.shape != oc3_masked.shape:
                    logger.warning(f"GOCI mask shape {goci_mask_cropped.shape} != OC3 shape {oc3_masked.shape}, cropping both")
                    min_h = min(goci_mask_cropped.shape[0], oc3_masked.shape[0])
                    min_w = min(goci_mask_cropped.shape[1], oc3_masked.shape[1])
                    goci_mask_cropped = goci_mask_cropped[:min_h, :min_w]
                    oc3_masked = oc3_masked[:min_h, :min_w]
                    khoa_aligned = khoa_aligned[:min_h, :min_w]

                oc3_masked[goci_mask_cropped == 999] = np.nan
                logger.info(f"Applied GOCI land mask to OC3, final shape: {oc3_masked.shape}")
            else:
                logger.warning("No GOCI land mask provided")

            # Step 3: Apply KHOA land mask (interpolate UST21 mask to OC3 resolution)
            khoa_masked = khoa_aligned.copy()

            if self.ust_land_mask is not None:
                # Interpolate UST21 land mask to OC3 resolution for proper alignment
                ust_mask_interp = self.interpolate_land_mask_to_oc3(self.ust_land_mask, khoa_masked.shape)

                if ust_mask_interp is not None:
                    if ust_mask_interp.shape != khoa_masked.shape:
                        logger.warning(f"UST mask interp shape {ust_mask_interp.shape} != KHOA aligned shape {khoa_masked.shape}, cropping both")
                        min_h = min(ust_mask_interp.shape[0], khoa_masked.shape[0])
                        min_w = min(ust_mask_interp.shape[1], khoa_masked.shape[1])
                        ust_mask_interp = ust_mask_interp[:min_h, :min_w]
                        khoa_masked = khoa_masked[:min_h, :min_w]
                        oc3_masked = oc3_masked[:min_h, :min_w]

                    khoa_masked[ust_mask_interp == 999] = np.nan
                    logger.info(f"Applied interpolated UST21 land mask to KHOA, final shape: {khoa_masked.shape}")
                else:
                    logger.warning("Failed to interpolate UST21 land mask")
            else:
                logger.warning("No UST21 land mask provided")

            logger.info(f"After masking - OC3: {oc3_masked.shape}, KHOA: {khoa_masked.shape}")

            # Step 4: Calculate difference (ensure shapes match)
            if oc3_masked.shape != khoa_masked.shape:
                logger.warning(f"Shape mismatch before difference calculation: {oc3_masked.shape} vs {khoa_masked.shape}")
                min_h = min(oc3_masked.shape[0], khoa_masked.shape[0])
                min_w = min(oc3_masked.shape[1], khoa_masked.shape[1])
                oc3_masked = oc3_masked[:min_h, :min_w]
                khoa_masked = khoa_masked[:min_h, :min_w]

            # Clip both datasets to valid chlorophyll range before calculating difference
            oc3_masked = self.clip_chlorophyll_range(oc3_masked, min_val=0.01, max_val=10.0)
            khoa_masked = self.clip_chlorophyll_range(khoa_masked, min_val=0.01, max_val=10.0)

            difference = np.where(
                (~np.isnan(oc3_masked)) & (~np.isnan(khoa_masked)),
                oc3_masked - khoa_masked,
                np.nan
            )

            # Step 5: Clean up near-zero values
            # Set values very close to zero to exactly zero
            threshold = 0.01
            difference[(np.abs(difference) < threshold) & (~np.isnan(difference))] = 0.0

            logger.info(f"Calculated OC3-aligned difference map with shape {difference.shape}")

            return difference.astype(np.float32)

        except Exception as e:
            logger.error(f"Error calculating OC3-aligned difference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_difference_map_as_png(self,
                                   date_str: str,
                                   difference: np.ndarray,
                                   land_mask: np.ndarray = None,
                                   vmin: float = None,
                                   vmax: float = None):
        """
        Save difference map as PNG with proper visualization
        Handles 0 values by displaying them as white
        Uses percentiles to avoid extreme outliers

        Args:
            date_str: Date string for filename
            difference: Difference map array
            land_mask: Boolean array where True = land pixels (will be gray)
            vmin: Minimum value for colormap scaling
            vmax: Maximum value for colormap scaling
        """
        try:
            png_filename = f"OC3_KHOA_difference_{date_str}.png"
            png_path = self.output_dir / png_filename

            # Calculate statistics excluding NaN
            valid_data = difference[~np.isnan(difference)]
            if len(valid_data) == 0:
                logger.warning(f"No valid data in difference map for {date_str}")
                return

            actual_min = np.nanmin(difference)
            actual_max = np.nanmax(difference)
            actual_mean = np.nanmean(difference)
            actual_std = np.nanstd(difference)

            logger.info(f"Difference map statistics:")
            logger.info(f"  Min: {actual_min:.4f}, Max: {actual_max:.4f}")
            logger.info(f"  Mean: {actual_mean:.4f}, Std: {actual_std:.4f}")

            # Set default vmin/vmax if not provided
            if vmin is None or vmax is None:
                # Use percentiles to handle outliers gracefully
                valid_data_nonzero = difference[(~np.isnan(difference)) & (difference != 0.0)]
                if len(valid_data_nonzero) > 0:
                    p5 = np.percentile(valid_data_nonzero, 5)
                    p95 = np.percentile(valid_data_nonzero, 95)
                    abs_max = max(abs(p5), abs(p95))
                else:
                    abs_max = max(abs(actual_min), abs(actual_max))

                if vmin is None:
                    vmin = -abs_max
                if vmax is None:
                    vmax = abs_max

            logger.info(f"PNG scaling: vmin={vmin:.4f}, vmax={vmax:.4f}")

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

            # Create RGB visualization
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap('RdBu_r')

            # Initialize RGB array (white background for all pixels)
            rgb_data = np.ones((difference.shape[0], difference.shape[1], 3), dtype=np.float32)

            # Apply colormap only to non-zero, non-NaN pixels
            valid_mask = (~np.isnan(difference)) & (difference != 0.0)
            if land_mask is not None:
                valid_mask = valid_mask & (~land_mask)

            if np.any(valid_mask):
                rgb_data[valid_mask] = cmap(norm(difference[valid_mask]))[:, :3]

            # Apply land mask as gray (if provided)
            if land_mask is not None:
                rgb_data[land_mask] = [0.5, 0.5, 0.5]  # Gray color for land

            # White remains for 0 values (already initialized)

            # Display the RGB image
            ax.imshow(rgb_data, origin='upper', interpolation='none')

            # Add colorbar using ScalarMappable
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Chlorophyll difference (mg/m³)',
                               shrink=0.8)

            # Set labels and title
            ax.set_xlabel('Longitude pixel')
            ax.set_ylabel('Latitude pixel')
            ax.set_title(f'OC3 - KHOA Daily Chlorophyll Difference Map\n{date_str}\n' +
                        f'Mean: {actual_mean:.4f} mg/m³, Std: {actual_std:.4f} mg/m³\n' +
                        f'(White: 0 difference, Gray: land, Red-Blue: positive-negative difference)')

            # Save figure
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved difference map PNG to {png_path}")

        except Exception as e:
            logger.error(f"Error saving difference map as PNG: {e}")
            import traceback
            traceback.print_exc()

    def save_oc3_aligned_comparison_as_png(self,
                                          date_str: str,
                                          oc3_data: np.ndarray,
                                          khoa_data: np.ndarray,
                                          difference_map: np.ndarray):
        """
        Save comparison maps with OC3 as reference resolution
        Aligns KHOA to OC3 for proper visualization

        Args:
            date_str: Date string for filename
            oc3_data: OC3 chlorophyll data at GOCI resolution (reference)
            khoa_data: KHOA chlorophyll data at original resolution (will be interpolated)
            difference_map: Pre-calculated difference map (OC3 - KHOA aligned)
        """
        try:
            # Interpolate KHOA to OC3 resolution for visualization
            khoa_aligned = self.align_khoa_to_oc3_resolution(oc3_data, khoa_data)
            if khoa_aligned is None:
                logger.error(f"Failed to align KHOA for visualization")
                return

            logger.info(f"Creating comparison: OC3 {oc3_data.shape}, KHOA aligned {khoa_aligned.shape}, Diff {difference_map.shape if difference_map is not None else 'None'}")

            # Ensure all data have the same shape
            oc3_masked = oc3_data.copy()
            khoa_masked = khoa_aligned.copy()

            # Ensure shapes match (crop to smallest)
            if oc3_masked.shape != khoa_masked.shape or (difference_map is not None and oc3_masked.shape != difference_map.shape):
                logger.warning(f"Shape mismatch: OC3 {oc3_masked.shape}, KHOA {khoa_masked.shape}, Diff {difference_map.shape if difference_map is not None else 'None'}")
                min_h = min(oc3_masked.shape[0], khoa_masked.shape[0])
                if difference_map is not None:
                    min_h = min(min_h, difference_map.shape[0])
                min_w = min(oc3_masked.shape[1], khoa_masked.shape[1])
                if difference_map is not None:
                    min_w = min(min_w, difference_map.shape[1])

                oc3_masked = oc3_masked[:min_h, :min_w]
                khoa_masked = khoa_masked[:min_h, :min_w]
                if difference_map is not None:
                    difference_map = difference_map[:min_h, :min_w]
                logger.info(f"All data cropped to: {oc3_masked.shape}")

            # Apply land masks
            oc3_land_mask_bool = None
            if self.goci_land_mask is not None:
                # Crop GOCI land mask to match data shape
                goci_mask_cropped = self.goci_land_mask[:oc3_masked.shape[0], :oc3_masked.shape[1]]
                if goci_mask_cropped.shape == oc3_masked.shape:
                    oc3_masked[goci_mask_cropped == 999] = np.nan
                    oc3_land_mask_bool = (goci_mask_cropped == 999)
                else:
                    logger.warning(f"Could not apply GOCI mask: shape {goci_mask_cropped.shape} != {oc3_masked.shape}")

            # Apply KHOA land mask (interpolate UST21 mask to OC3 resolution)
            khoa_land_mask_bool = None
            if self.ust_land_mask is not None:
                # Interpolate UST21 land mask to OC3 resolution for proper alignment
                ust_mask_interp = self.interpolate_land_mask_to_oc3(self.ust_land_mask, khoa_masked.shape)
                if ust_mask_interp is not None and ust_mask_interp.shape == khoa_masked.shape:
                    khoa_masked[ust_mask_interp == 999] = np.nan
                    khoa_land_mask_bool = (ust_mask_interp == 999)
                    logger.info(f"Applied interpolated UST21 land mask to KHOA")
                else:
                    logger.warning(f"Could not apply interpolated UST mask: shape {ust_mask_interp.shape if ust_mask_interp is not None else 'None'} != {khoa_masked.shape}")

            # GOCI coordinate bounds (from HDF5 metadata)
            # Geographic coordinates for full GOCI image (5685 x 5566 pixels)
            lat_min, lat_max = 7.125, 54.527
            lon_min, lon_max = 87.909, 172.091

            # Calculate extent based on actual data shape to maintain pixel correspondence
            # extent = [left, right, bottom, top] for imshow with origin='upper'
            # Adjust extent to match the actual cropped data shape
            data_height, data_width = oc3_masked.shape
            full_goci_height, full_goci_width = 5685, 5566

            # Scale the geographic extent to match the actual data size
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min

            # Calculate extent for the actual data (may be cropped)
            extent = [
                lon_min,  # left
                lon_min + (lon_range * data_width / full_goci_width),  # right
                lat_min + (lat_range * (full_goci_height - data_height) / full_goci_height),  # bottom (adjusted for cropping from top)
                lat_min + (lat_range * full_goci_height / full_goci_height)  # top
            ]

            # Calculate aspect ratio to preserve image dimensions
            extent_width = extent[1] - extent[0]  # degrees longitude
            extent_height = extent[3] - extent[2]  # degrees latitude
            pixel_aspect = data_width / data_height
            geo_aspect = extent_width / extent_height
            aspect_ratio = geo_aspect / pixel_aspect

            # Create comparison figure with 3 columns
            fig = plt.figure(figsize=(20, 6), dpi=150)
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

            # Column 1: OC3 data
            ax1 = fig.add_subplot(gs[0, 0])
            valid_oc3 = oc3_masked[~np.isnan(oc3_masked)]
            if len(valid_oc3) > 0:
                # For display: set values >=10 to 0, and NaN values to 0
                oc3_display = oc3_masked.copy()
                oc3_display[oc3_display >= 10.0] = 0.0
                oc3_display[np.isnan(oc3_display)] = 0.0  # NaN values shown as 0 (dark purple)

                vmin_oc3 = 0.0
                vmax_oc3 = 10.0
                im1 = ax1.imshow(oc3_display, cmap='viridis',
                                vmin=vmin_oc3, vmax=vmax_oc3, origin='upper', interpolation='nearest',
                                extent=extent, aspect=aspect_ratio)
                ax1.set_title(f'OC3 Daily Chlorophyll\n({oc3_data.shape[0]}×{oc3_data.shape[1]})')
                ax1.set_xlabel('Longitude (°E)', fontsize=11)
                ax1.set_ylabel('Latitude (°N)', fontsize=11)
                cbar1 = plt.colorbar(im1, ax=ax1, label='mg/m³', shrink=0.8)

                # Overlay land mask as light gray
                if oc3_land_mask_bool is not None:
                    land_overlay = np.ma.masked_where(~oc3_land_mask_bool,
                                                     np.ones_like(oc3_land_mask_bool, dtype=float))
                    ax1.imshow(land_overlay, cmap='gray', vmin=0, vmax=1,
                             origin='upper', interpolation='none', extent=extent, aspect=aspect_ratio, alpha=1.0)

            # Column 2: KHOA aligned to OC3
            ax2 = fig.add_subplot(gs[0, 1])
            valid_khoa = khoa_masked[~np.isnan(khoa_masked)]
            if len(valid_khoa) > 0:
                vmin_khoa = np.nanmin(khoa_masked)
                vmax_khoa = np.nanmax(khoa_masked)
                vmax_khoa = max(vmax_khoa, 0.01)
                im2 = ax2.imshow(khoa_masked, cmap='viridis',
                                vmin=vmin_khoa, vmax=vmax_khoa, origin='upper', interpolation='nearest',
                                extent=extent, aspect=aspect_ratio)
                ax2.set_title(f'KHOA Daily (Aligned to OC3)\n({khoa_aligned.shape[0]}×{khoa_aligned.shape[1]})')
                ax2.set_xlabel('Longitude (°E)', fontsize=11)
                ax2.set_ylabel('Latitude (°N)', fontsize=11)
                cbar2 = plt.colorbar(im2, ax=ax2, label='mg/m³', shrink=0.8)

                # Overlay land mask as gray
                if khoa_land_mask_bool is not None:
                    land_overlay = np.ma.masked_where(~khoa_land_mask_bool,
                                                     np.ones_like(khoa_land_mask_bool, dtype=float) * 0.5)
                    ax2.imshow(land_overlay, cmap='gray', vmin=0, vmax=1,
                             origin='upper', interpolation='none', extent=extent, aspect=aspect_ratio, alpha=0.6)

            # Column 3: Difference map
            ax3 = fig.add_subplot(gs[0, 2])
            if difference_map is not None:
                valid_diff = difference_map[~np.isnan(difference_map)]
                if len(valid_diff) > 0:
                    # Use percentiles to avoid extreme outliers
                    p5 = np.nanpercentile(difference_map, 5)
                    p95 = np.nanpercentile(difference_map, 95)
                    abs_max = max(abs(p5), abs(p95))
                    vmin_diff = -abs_max
                    vmax_diff = abs_max

                    logger.info(f"Difference map range: [{p5:.4f}, {p95:.4f}], using vmin/vmax: [{vmin_diff:.4f}, {vmax_diff:.4f}]")

                    # Create RGB visualization
                    from matplotlib.colors import Normalize
                    norm = Normalize(vmin=vmin_diff, vmax=vmax_diff)
                    cmap = plt.get_cmap('RdBu_r')

                    # Initialize RGB array (white background)
                    rgb_data = np.ones((difference_map.shape[0], difference_map.shape[1], 3), dtype=np.float32)

                    # Apply colormap to non-zero, non-NaN pixels
                    valid_mask = (~np.isnan(difference_map)) & (difference_map != 0.0)
                    if khoa_land_mask_bool is not None:
                        valid_mask = valid_mask & (~khoa_land_mask_bool)

                    if np.any(valid_mask):
                        rgb_data[valid_mask] = cmap(norm(difference_map[valid_mask]))[:, :3]

                    # Apply land mask as gray
                    if khoa_land_mask_bool is not None:
                        rgb_data[khoa_land_mask_bool] = [0.5, 0.5, 0.5]

                    # Display RGB image
                    ax3.imshow(rgb_data, origin='upper', interpolation='none', extent=extent, aspect=aspect_ratio)

                    # Colorbar
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar3 = plt.colorbar(sm, ax=ax3, label='Difference (mg/m³)', shrink=0.8)

                    # Title and labels
                    ax3.set_title(f'OC3 - KHOA Difference\n({difference_map.shape[0]}×{difference_map.shape[1]})')
                    ax3.set_xlabel('Longitude (°E)', fontsize=11)
                    ax3.set_ylabel('Latitude (°N)', fontsize=11)

                    # Statistics annotation
                    diff_stats_text = f'Mean: {np.nanmean(difference_map):.4f}\nStd: {np.nanstd(difference_map):.4f}'
                    ax3.text(0.02, 0.98, diff_stats_text, transform=ax3.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=9)
                else:
                    ax3.text(0.5, 0.5, 'No valid\ndifference data', ha='center', va='center', fontsize=12)
                    ax3.axis('off')
            else:
                ax3.text(0.5, 0.5, 'Difference map\nnot available', ha='center', va='center', fontsize=12)
                ax3.axis('off')

            # Figure title
            fig.suptitle(f'Chlorophyll Comparison (OC3-aligned) - {date_str}', fontsize=16, fontweight='bold', y=0.98)

            # Save
            png_filename = f"OC3_KHOA_comparison_{date_str}.png"
            png_path = self.output_dir / png_filename
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            plt.savefig(png_path, dpi=150)  # Removed bbox_inches='tight' to preserve aspect ratio
            plt.close()

            logger.info(f"Saved OC3-aligned comparison map to {png_path}")

        except Exception as e:
            logger.error(f"Error saving OC3-aligned comparison: {e}")
            import traceback
            traceback.print_exc()

    def save_chlorophyll_map_as_png(self,
                                    date_str: str,
                                    oc3_data: np.ndarray,
                                    khoa_data: np.ndarray):
        """
        Legacy function - kept for backward compatibility
        Use save_oc3_aligned_comparison_as_png instead
        """
        pass


    def calculate_rmse(self, oc3_data: np.ndarray, khoa_data: np.ndarray) -> Optional[float]:
        """
        Calculate linear RMSE between OC3 and KHOA data

        Args:
            oc3_data: OC3 chlorophyll data
            khoa_data: KHOA chlorophyll data (must be same shape as oc3_data)

        Returns:
            RMSE value or None if calculation fails
        """
        try:
            # Ensure same shape
            if oc3_data.shape != khoa_data.shape:
                logger.warning(f"Shape mismatch for RMSE: OC3 {oc3_data.shape} vs KHOA {khoa_data.shape}")
                return None

            # Clip both datasets to valid chlorophyll range
            oc3_data = self.clip_chlorophyll_range(oc3_data, min_val=0.01, max_val=10.0)
            khoa_data = self.clip_chlorophyll_range(khoa_data, min_val=0.01, max_val=10.0)

            # Find valid pixels (non-NaN in both)
            valid_mask = ~np.isnan(oc3_data) & ~np.isnan(khoa_data)

            if not np.any(valid_mask):
                logger.warning("No valid pixels for RMSE calculation")
                return None

            # Calculate RMSE on valid pixels
            oc3_valid = oc3_data[valid_mask]
            khoa_valid = khoa_data[valid_mask]

            rmse = np.sqrt(np.mean((oc3_valid - khoa_valid) ** 2))

            logger.info(f"Linear RMSE calculated: {rmse:.4f} mg/m³ (valid pixels: {np.sum(valid_mask)})")
            return rmse

        except Exception as e:
            logger.error(f"Error calculating RMSE: {e}")
            return None

    def calculate_log_rmse(self, oc3_data: np.ndarray, khoa_data: np.ndarray) -> Optional[float]:
        """
        Calculate log-scale RMSE between OC3 and KHOA data
        This is more appropriate for chlorophyll-a comparisons as it reduces
        the influence of high values and is more scientifically standard

        Args:
            oc3_data: OC3 chlorophyll data
            khoa_data: KHOA chlorophyll data (must be same shape as oc3_data)

        Returns:
            Log-scale RMSE value or None if calculation fails
        """
        try:
            # Ensure same shape
            if oc3_data.shape != khoa_data.shape:
                logger.warning(f"Shape mismatch for log RMSE: OC3 {oc3_data.shape} vs KHOA {khoa_data.shape}")
                return None

            # Clip both datasets to valid chlorophyll range
            oc3_data = self.clip_chlorophyll_range(oc3_data, min_val=0.01, max_val=10.0)
            khoa_data = self.clip_chlorophyll_range(khoa_data, min_val=0.01, max_val=10.0)

            # Find valid pixels (non-NaN in both)
            valid_mask = ~np.isnan(oc3_data) & ~np.isnan(khoa_data)

            if not np.any(valid_mask):
                logger.warning("No valid pixels for log RMSE calculation")
                return None

            # Calculate log-scale RMSE on valid pixels
            oc3_valid = oc3_data[valid_mask]
            khoa_valid = khoa_data[valid_mask]

            # Log10 transformation
            log_oc3 = np.log10(oc3_valid)
            log_khoa = np.log10(khoa_valid)

            # Calculate RMSE in log space
            log_rmse = np.sqrt(np.mean((log_oc3 - log_khoa) ** 2))

            logger.info(f"Log-scale RMSE calculated: {log_rmse:.6f} (valid pixels: {np.sum(valid_mask)})")
            return log_rmse

        except Exception as e:
            logger.error(f"Error calculating log RMSE: {e}")
            return None

    def process_date(self, date_str: str) -> bool:
        """
        Process a single date: average hourly to daily and calculate difference
        Uses OC3 as reference resolution, aligns KHOA to OC3

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Processing date {date_str}")

        # Step 1: Reconstruct daily OC3 data from tiles
        oc3_daily = self.reconstruct_daily_from_tiles(date_str)
        if oc3_daily is None:
            logger.error(f"Failed to reconstruct OC3 daily data for {date_str}")
            return False

        logger.info(f"OC3 daily shape: {oc3_daily.shape}")

        # Step 2: Load KHOA daily data
        result = self.load_khoa_daily_data(date_str)
        if result is None:
            logger.error(f"Failed to load KHOA daily data for {date_str}")
            return False

        khoa_daily, metadata = result
        logger.info(f"KHOA daily shape: {khoa_daily.shape}")

        # Step 3: Calculate OC3-aligned difference map
        difference_map = self.calculate_oc3_aligned_difference(date_str, oc3_daily, khoa_daily)
        if difference_map is None:
            logger.error(f"Failed to calculate OC3-aligned difference for {date_str}")
            return False

        logger.info(f"Difference map shape: {difference_map.shape}")

        # Step 4: Calculate RMSE (log-scale only)
        # Align KHOA to OC3 resolution first
        khoa_aligned = self.align_khoa_to_oc3_resolution(oc3_daily, khoa_daily)
        if khoa_aligned is not None:
            # Calculate log-scale RMSE (standard for chlorophyll comparison)
            log_rmse = self.calculate_log_rmse(oc3_daily, khoa_aligned)

            if log_rmse is not None:
                # Save log RMSE to CSV file
                rmse_csv_path = self.output_dir / "rmse_results.csv"

                # Check if file exists to determine if we need to write header
                write_header = not rmse_csv_path.exists()

                with open(rmse_csv_path, 'a') as f:
                    if write_header:
                        f.write("date,rmse\n")
                    f.write(f"{date_str},{log_rmse:.6f}\n")

                logger.info(f"Log-scale RMSE saved to {rmse_csv_path}: {log_rmse:.6f}")

        # Step 5: Save OC3-aligned comparison visualization
        self.save_oc3_aligned_comparison_as_png(date_str, oc3_daily, khoa_daily, difference_map)

        # Step 6: Also save standalone difference map
        self.save_difference_map_as_png(date_str, difference_map)

        logger.info(f"Successfully processed {date_str}")
        return True

    def process_date_range(self, start_date: str, end_date: str):
        """
        Process a range of dates

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format (inclusive)
        """
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')

        current = start
        successful = 0
        failed = 0
        total_days = (end - start).days + 1

        while current <= end:
            date_str = current.strftime('%Y%m%d')
            days_processed = (current - start).days + 1
            # logger.info(f"[{days_processed}/{total_days}] Processing {date_str}...")
            if self.process_date(date_str):
                successful += 1
                # logger.info(f"[{days_processed}/{total_days}] ✓ Success: {date_str}")
            else:
                failed += 1
                # logger.info(f"[{days_processed}/{total_days}] ✗ Failed: {date_str}")
            current += timedelta(days=1)

        # logger.info(f"Processing complete: {successful}/{total_days} successful, {failed}/{total_days} failed")

        # Calculate and save average RMSE
        self.calculate_and_save_average_rmse()

    def calculate_and_save_average_rmse(self):
        """
        Calculate average RMSE from all processed dates and append to CSV
        """
        rmse_csv_path = self.output_dir / "rmse_results.csv"

        if not rmse_csv_path.exists():
            logger.warning("No RMSE results file found")
            return

        try:
            # Read RMSE results
            import pandas as pd
            df = pd.read_csv(rmse_csv_path)

            if 'rmse' not in df.columns or len(df) == 0:
                logger.warning("No RMSE data found in CSV")
                return

            # Calculate average
            avg_rmse = df['rmse'].mean()

            # Append average to CSV
            with open(rmse_csv_path, 'a') as f:
                f.write(f"\nAverage,{avg_rmse:.6f}\n")

            logger.info(f"Average RMSE calculated and saved: {avg_rmse:.6f}")
            print(f"\n{'='*60}")
            print(f"OVERALL AVERAGE RMSE: {avg_rmse:.6f}")
            print(f"{'='*60}")

        except Exception as e:
            logger.error(f"Error calculating average RMSE: {e}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Daily averaging of OC3 chlorophyll and difference map calculation'
    )
    parser.add_argument('--oc3_dir', required=True,
                       help='Directory containing OC3 results (hourly or daily)')
    parser.add_argument('--khoa_dir', required=True,
                       help='Directory containing KHOA daily chlorophyll files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--goci_land_mask',
                       help='Path to GOCI land mask file')
    parser.add_argument('--ust_land_mask',
                       help='Path to UST21 land mask file')
    parser.add_argument('--daily_mode', action='store_true', default=True,
                       help='Process daily-averaged OC3 results (default: True)')
    parser.add_argument('--date',
                       help='Single date to process (YYYYMMDD format)')
    parser.add_argument('--start_date',
                       help='Start date for range processing (YYYYMMDD format)')
    parser.add_argument('--end_date',
                       help='End date for range processing (YYYYMMDD format)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize processor
    processor = DailyAveraging(args.oc3_dir, args.khoa_dir, args.output_dir,
                              goci_land_mask_path=args.goci_land_mask,
                              ust_land_mask_path=args.ust_land_mask,
                              daily_mode=args.daily_mode)

    # Process dates
    if args.start_date and args.end_date:
        processor.process_date_range(args.start_date, args.end_date)
    elif args.date:
        processor.process_date(args.date)
    else:
        logger.error("Must provide either --date or both --start_date and --end_date")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
