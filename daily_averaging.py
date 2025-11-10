#!/usr/bin/env python3
"""
Daily averaging of OC3 chlorophyll data and difference map calculation.
This script:
1. Averages 8 hourly OC3 results into daily chlorophyll
2. Calculates difference map between OC3 daily and existing KHOA daily chlorophyll
3. Saves results as NetCDF files
"""

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.ndimage import zoom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DailyAveraging:
    """
    Process hourly OC3 chlorophyll data to daily averages and calculate difference maps
    """

    def __init__(self,
                 oc3_results_dir: str,
                 khoa_daily_dir: str,
                 output_dir: str,
                 goci_land_mask_path: str = None,
                 ust_land_mask_path: str = None):
        """
        Initialize DailyAveraging processor

        Args:
            oc3_results_dir: Directory containing hourly OC3 results
            khoa_daily_dir: Directory containing KHOA daily chlorophyll files
            output_dir: Output directory for daily averages and difference maps
            goci_land_mask_path: Path to GOCI land mask file
            ust_land_mask_path: Path to UST21 land mask file
        """
        self.oc3_results_dir = Path(oc3_results_dir)
        self.khoa_daily_dir = Path(khoa_daily_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load land masks
        self.goci_land_mask = None
        self.ust_land_mask = None

        if goci_land_mask_path and Path(goci_land_mask_path).exists():
            self.goci_land_mask = np.load(goci_land_mask_path)
            logger.info(f"Loaded GOCI land mask: {self.goci_land_mask.shape}")

        if ust_land_mask_path and Path(ust_land_mask_path).exists():
            self.ust_land_mask = np.load(ust_land_mask_path)
            logger.info(f"Loaded UST21 land mask: {self.ust_land_mask.shape}")

    def get_daily_timestamps(self, date_str: str) -> List[str]:
        """
        Get all hourly timestamps for a given date

        Args:
            date_str: Date in YYYYMMDD format

        Returns:
            List of timestamp directories in YYYYMMDD_HHMMSS format
        """
        # List all directories matching the date
        date_pattern = f"{date_str}_*"
        timestamp_dirs = sorted(self.oc3_results_dir.glob(date_pattern))

        timestamps = [d.name for d in timestamp_dirs if d.is_dir()]
        logger.info(f"Found {len(timestamps)} hourly results for {date_str}: {timestamps}")

        return timestamps

    def load_tile_data(self,
                       timestamp: str,
                       tile_id: str) -> Optional[np.ndarray]:
        """
        Load chlorophyll data for a specific tile and timestamp

        Args:
            timestamp: Timestamp in YYYYMMDD_HHMMSS format
            tile_id: Tile identifier (e.g., mask_10_y0000_x2304)

        Returns:
            Numpy array of chlorophyll data or None if file not found
        """
        chl_file = (self.oc3_results_dir / timestamp / timestamp / tile_id /
                    'chlorophyll_oc3.csv')

        if not chl_file.exists():
            logger.warning(f"File not found: {chl_file}")
            return None

        try:
            data = np.loadtxt(chl_file, delimiter=',', dtype=np.float32)
            return data
        except Exception as e:
            logger.error(f"Error loading {chl_file}: {e}")
            return None

    def get_available_tiles(self, timestamp: str) -> List[str]:
        """
        Get list of available tiles for a given timestamp

        Args:
            timestamp: Timestamp in YYYYMMDD_HHMMSS format

        Returns:
            List of tile IDs
        """
        tile_dir = self.oc3_results_dir / timestamp / timestamp

        if not tile_dir.exists():
            logger.warning(f"Tile directory not found: {tile_dir}")
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
            logger.warning(f"No timestamps found for {date_str}")
            return None

        hourly_data = []

        for timestamp in timestamps:
            data = self.load_tile_data(timestamp, tile_id)
            if data is not None:
                hourly_data.append(data)

        if not hourly_data:
            logger.warning(f"No valid data loaded for tile {tile_id} on {date_str}")
            return None

        # Stack data arrays
        stacked_data = np.stack(hourly_data, axis=0)
        logger.debug(f"Stacked data shape: {stacked_data.shape} for tile {tile_id}")

        # Calculate daily average, ignoring NaN values
        daily_avg = np.nanmean(stacked_data, axis=0)

        # Replace all-nan slices with NaN
        daily_avg = np.where(np.isnan(daily_avg), np.nan, daily_avg)

        logger.info(f"Averaged {len(hourly_data)} hourly datasets for tile {tile_id}")

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
            logger.warning(f"KHOA file not found: {nc_path}")
            return None

        try:
            ds = nc.Dataset(str(nc_path), 'r')

            # Extract chlorophyll data
            if 'merged_daily_Chl' in ds.variables:
                chl_data = ds.variables['merged_daily_Chl'][:].astype(np.float32)
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

            logger.info(f"Loaded KHOA data from {nc_path} with shape {chl_data.shape}")
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

        logger.info(f"Processing {len(tiles)} tiles for {date_str}")

        # Dictionary to store averaged data for each tile
        tile_data = {}

        for tile_id in tiles:
            daily_tile = self.average_hourly_to_daily(date_str, tile_id)
            if daily_tile is not None:
                tile_data[tile_id] = daily_tile
                logger.debug(f"Processed tile {tile_id}: shape {daily_tile.shape}")

        if not tile_data:
            logger.error(f"No valid tile data found for {date_str}")
            return None

        # Parse tile coordinates to reconstruct the full grid
        # Tile format: mask_<idx>_y<y_start>_x<x_start>
        # Each tile is typically 256x256 pixels
        full_data = self.reconstruct_grid_from_tiles(tile_data)

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
                # Parse: mask_<idx>_y<y_start>_x<x_start>
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

            logger.info(f"Reconstructing grid: {max_y} x {max_x}")

            # Initialize full grid with NaN
            full_grid = np.full((max_y, max_x), np.nan, dtype=np.float32)

            # Fill in tile data
            for tile_id, data in tile_data.items():
                y_start, x_start, _ = positions[tile_id]
                y_end = y_start + data.shape[0]
                x_end = x_start + data.shape[1]
                full_grid[y_start:y_end, x_start:x_end] = data

            logger.info(f"Successfully reconstructed grid with shape {full_grid.shape}")
            return full_grid

        except Exception as e:
            logger.error(f"Error reconstructing grid from tiles: {e}")
            return None

    def interpolate_to_ust_resolution(self,
                                      oc3_data: np.ndarray) -> np.ndarray:
        """
        Interpolate OC3 data from GOCI resolution to UST21 resolution

        Args:
            oc3_data: OC3 data at GOCI resolution (5685x5567)

        Returns:
            OC3 data interpolated to UST21 resolution (8000x10500)
        """
        try:
            # Calculate zoom factors
            ust_shape = (8000, 10500)
            zoom_y = ust_shape[0] / oc3_data.shape[0]
            zoom_x = ust_shape[1] / oc3_data.shape[1]

            logger.info(f"Interpolating OC3 from {oc3_data.shape} to {ust_shape}")
            logger.info(f"Zoom factors: y={zoom_y:.4f}, x={zoom_x:.4f}")

            # Use zoom for interpolation (nearest neighbor for masked data)
            interpolated = zoom(oc3_data, (zoom_y, zoom_x), order=1, prefilter=False)

            logger.info(f"Interpolated shape: {interpolated.shape}")
            return interpolated.astype(np.float32)

        except Exception as e:
            logger.error(f"Error interpolating OC3 data: {e}")
            return None

    def apply_land_masks(self,
                        data: np.ndarray,
                        is_land_mask: np.ndarray = None) -> np.ndarray:
        """
        Apply land mask to data (set land pixels to NaN)

        Args:
            data: Data array
            is_land_mask: Land mask where 999 = land, 1 = water

        Returns:
            Masked data array
        """
        if is_land_mask is None:
            return data

        try:
            # Create masked copy
            masked_data = data.copy()

            # Land is marked as 999 in mask
            land_pixels = (is_land_mask == 999)
            masked_data[land_pixels] = np.nan

            logger.info(f"Applied land mask: {np.sum(land_pixels)} land pixels masked")
            return masked_data

        except Exception as e:
            logger.error(f"Error applying land mask: {e}")
            return data

    def calculate_difference_map(self,
                                 oc3_daily: np.ndarray,
                                 khoa_daily: np.ndarray) -> np.ndarray:
        """
        Calculate difference map between OC3 daily and KHOA daily chlorophyll
        with proper resolution matching and land mask application

        Args:
            oc3_daily: OC3 daily chlorophyll array (GOCI resolution)
            khoa_daily: KHOA daily chlorophyll array (UST21 resolution)

        Returns:
            Difference map (OC3 - KHOA) at UST21 resolution
        """
        try:
            # Step 1: Interpolate OC3 to UST21 resolution if needed
            if oc3_daily.shape != khoa_daily.shape:
                logger.info(f"Interpolating OC3 from {oc3_daily.shape} to {khoa_daily.shape}")
                oc3_interpolated = self.interpolate_to_ust_resolution(oc3_daily)
                if oc3_interpolated is None:
                    logger.error("Failed to interpolate OC3 data")
                    return None
            else:
                oc3_interpolated = oc3_daily.copy()

            # Step 2: Apply land masks
            if self.ust_land_mask is not None:
                oc3_interpolated = self.apply_land_masks(oc3_interpolated, self.ust_land_mask)
                khoa_daily = self.apply_land_masks(khoa_daily, self.ust_land_mask)

            # Step 3: Calculate difference, preserving NaN values
            difference = np.where(
                (~np.isnan(oc3_interpolated)) & (~np.isnan(khoa_daily)),
                oc3_interpolated - khoa_daily,
                np.nan
            )

            logger.info(f"Calculated difference map with shape {difference.shape}")
            return difference.astype(np.float32)

        except Exception as e:
            logger.error(f"Error calculating difference map: {e}")
            return None

    def save_difference_map_as_png(self,
                                   date_str: str,
                                   difference: np.ndarray,
                                   vmin: float = None,
                                   vmax: float = None):
        """
        Save difference map as PNG with proper visualization

        Args:
            date_str: Date string for filename
            difference: Difference map array
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
                # Use symmetric range around 0, or ±3 std
                abs_max = max(abs(actual_min), abs(actual_max))
                if vmin is None:
                    vmin = -abs_max
                if vmax is None:
                    vmax = abs_max

            logger.info(f"PNG scaling: vmin={vmin:.4f}, vmax={vmax:.4f}")

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

            # Create masked array for proper handling of NaN (land)
            masked_diff = np.ma.masked_invalid(difference)

            # Use diverging colormap centered at 0
            im = ax.imshow(masked_diff, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                          origin='upper', interpolation='none')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Chlorophyll difference (mg/m³)',
                               shrink=0.8)

            # Set labels and title
            ax.set_xlabel('Longitude index')
            ax.set_ylabel('Latitude index')
            ax.set_title(f'OC3 - KHOA Daily Chlorophyll Difference Map\n{date_str}\n' +
                        f'Mean: {actual_mean:.4f}, Std: {actual_std:.4f}')

            # Save figure
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved difference map PNG to {png_path}")

        except Exception as e:
            logger.error(f"Error saving difference map as PNG: {e}")

    def save_chlorophyll_map_as_png(self,
                                    date_str: str,
                                    oc3_data: np.ndarray,
                                    khoa_data: np.ndarray):
        """
        Save chlorophyll comparison maps as PNG

        Args:
            date_str: Date string for filename
            oc3_data: OC3 chlorophyll data (will be interpolated if needed)
            khoa_data: KHOA chlorophyll data
        """
        try:
            # Interpolate OC3 to same resolution as KHOA if needed
            if oc3_data.shape != khoa_data.shape:
                oc3_data = self.interpolate_to_ust_resolution(oc3_data)

            # Apply land masks
            if self.ust_land_mask is not None:
                oc3_masked = self.apply_land_masks(oc3_data.copy(), self.ust_land_mask)
                khoa_masked = self.apply_land_masks(khoa_data.copy(), self.ust_land_mask)
            else:
                oc3_masked = oc3_data
                khoa_masked = khoa_data

            # Create comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

            # Get common scaling (log scale for chlorophyll)
            valid_oc3 = oc3_masked[~np.isnan(oc3_masked)]
            valid_khoa = khoa_masked[~np.isnan(khoa_masked)]

            if len(valid_oc3) > 0 and len(valid_khoa) > 0:
                vmin = min(np.nanmin(oc3_masked), np.nanmin(khoa_masked))
                vmax = max(np.nanmax(oc3_masked), np.nanmax(khoa_masked))

                # OC3 map
                im1 = axes[0].imshow(np.ma.masked_invalid(oc3_masked), cmap='viridis',
                                    vmin=vmin, vmax=vmax, origin='upper')
                axes[0].set_title('OC3 Daily Chlorophyll')
                axes[0].set_xlabel('Longitude index')
                axes[0].set_ylabel('Latitude index')
                plt.colorbar(im1, ax=axes[0], label='mg/m³')

                # KHOA map
                im2 = axes[1].imshow(np.ma.masked_invalid(khoa_masked), cmap='viridis',
                                    vmin=vmin, vmax=vmax, origin='upper')
                axes[1].set_title('KHOA Daily Chlorophyll')
                axes[1].set_xlabel('Longitude index')
                axes[1].set_ylabel('Latitude index')
                plt.colorbar(im2, ax=axes[1], label='mg/m³')

                # Difference map
                difference = oc3_masked - khoa_masked
                im3 = axes[2].imshow(np.ma.masked_invalid(difference), cmap='RdBu_r',
                                    origin='upper')
                axes[2].set_title('OC3 - KHOA Difference')
                axes[2].set_xlabel('Longitude index')
                axes[2].set_ylabel('Latitude index')
                plt.colorbar(im3, ax=axes[2], label='mg/m³')

                fig.suptitle(f'Chlorophyll Comparison - {date_str}', fontsize=14)

                png_filename = f"OC3_KHOA_comparison_{date_str}.png"
                png_path = self.output_dir / png_filename
                plt.tight_layout()
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved comparison map PNG to {png_path}")
            else:
                logger.warning(f"Insufficient valid data for comparison map on {date_str}")
                plt.close()

        except Exception as e:
            logger.error(f"Error saving chlorophyll maps as PNG: {e}")

    def save_as_netcdf(self,
                       date_str: str,
                       oc3_daily: np.ndarray,
                       khoa_daily: np.ndarray,
                       difference: np.ndarray,
                       metadata: Dict):
        """
        Save daily averages and difference map as NetCDF files

        Args:
            date_str: Date in YYYYMMDD format
            oc3_daily: OC3 daily chlorophyll array
            khoa_daily: KHOA daily chlorophyll array
            difference: Difference map
            metadata: Metadata from KHOA file
        """
        try:
            # Create output filenames
            oc3_filename = f"OC3_daily_Chl_{date_str}.nc"
            diff_filename = f"OC3_KHOA_difference_{date_str}.nc"

            oc3_path = self.output_dir / oc3_filename
            diff_path = self.output_dir / diff_filename

            # Save OC3 daily
            self._save_netcdf_file(oc3_path, oc3_daily, metadata,
                                   "OC3 daily chlorophyll-a concentration", date_str)

            # Save difference map
            self._save_netcdf_file(diff_path, difference, metadata,
                                   "Difference map: OC3 daily - KHOA daily chlorophyll", date_str)

            logger.info(f"Saved NetCDF files for {date_str}")

        except Exception as e:
            logger.error(f"Error saving NetCDF files: {e}")

    def _save_netcdf_file(self,
                          filepath: Path,
                          data: np.ndarray,
                          metadata: Dict,
                          description: str,
                          date_str: str):
        """
        Helper function to save a single NetCDF file

        Args:
            filepath: Output file path
            data: Data array to save
            metadata: Metadata dictionary
            description: Variable description
            date_str: Date string
        """
        ds = nc.Dataset(str(filepath), 'w', format='NETCDF4')

        try:
            # Create dimensions
            ds.createDimension('x', data.shape[1])
            ds.createDimension('y', data.shape[0])
            ds.createDimension('time', 1)

            # Create variables
            time_var = ds.createVariable('time', 'f4', ('time',))
            var = ds.createVariable('chlorophyll_data', 'f4', ('y', 'x'), zlib=True)

            if metadata['lon'] is not None:
                lon_var = ds.createVariable('lon', 'f4', ('y', 'x'))
                lon_var[:] = metadata['lon']
                lon_var.units = 'degree_east'

            if metadata['lat'] is not None:
                lat_var = ds.createVariable('lat', 'f4', ('y', 'x'))
                lat_var[:] = metadata['lat']
                lat_var.units = 'degree_north'

            # Set data
            time_var[:] = metadata['time'] if metadata['time'] is not None else 0
            var[:] = data

            # Set attributes
            var.long_name = description
            var.units = 'mg/m^3'
            var.standard_name = 'mass_concentration_chlorophyll_a_concentration_in_sea_water'

            # Set global attributes
            ds.title = description
            ds.date = date_str
            ds.creation_date = datetime.now().isoformat()
            ds.source = 'OC3 Algorithm'

            logger.info(f"Successfully saved {filepath}")

        finally:
            ds.close()

    def process_date(self, date_str: str) -> bool:
        """
        Process a single date: average hourly to daily and calculate difference

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

        # Step 3: Calculate difference map
        # Note: You may need to align/interpolate if shapes don't match
        difference = self.calculate_difference_map(oc3_daily, khoa_daily)
        if difference is None:
            logger.warning(f"Could not calculate difference map for {date_str}")
            # Continue anyway, save OC3 daily

        # Step 4: Save as NetCDF (use interpolated OC3 to match KHOA shape)
        oc3_interpolated = self.interpolate_to_ust_resolution(oc3_daily) if oc3_daily.shape != khoa_daily.shape else oc3_daily
        self.save_as_netcdf(date_str, oc3_interpolated, khoa_daily,
                           difference if difference is not None else oc3_interpolated,
                           metadata)

        # Step 5: Save PNG visualizations
        if difference is not None:
            self.save_difference_map_as_png(date_str, difference)

        # Save comparison map
        self.save_chlorophyll_map_as_png(date_str, oc3_daily, khoa_daily)

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

        while current <= end:
            date_str = current.strftime('%Y%m%d')
            if self.process_date(date_str):
                successful += 1
            else:
                failed += 1
            current += timedelta(days=1)

        logger.info(f"Processing complete: {successful} successful, {failed} failed")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Daily averaging of OC3 chlorophyll and difference map calculation'
    )
    parser.add_argument('--oc3_dir', required=True,
                       help='Directory containing hourly OC3 results')
    parser.add_argument('--khoa_dir', required=True,
                       help='Directory containing KHOA daily chlorophyll files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--goci_land_mask',
                       help='Path to GOCI land mask file')
    parser.add_argument('--ust_land_mask',
                       help='Path to UST21 land mask file')
    parser.add_argument('--date', required=True,
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
                              ust_land_mask_path=args.ust_land_mask)

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
