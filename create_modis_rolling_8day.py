#!/usr/bin/env python3
"""
MODIS 8-Day Rolling Window Data Processing Script
Converts L2 MODIS data to L3m format with 8-day rolling window averaging
(Days 1-8, 2-9, 3-10, ..., 358-365)
"""

import os
import glob
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import netCDF4 as nc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODIS_L2_DIR = "/home/juneyonglee/Desktop/AY_ust/My_Book/MODIS/MODIS-Aqua/OC"
OUTPUT_DIR = "/home/juneyonglee/Desktop/AY_ust/My_Book/MODIS/MODIS_aqua_8days"
YEARS = range(2012, 2022)  # 2012-2021

# L3m grid parameters (standard MODIS 4km)
LAT_SIZE = 4320
LON_SIZE = 8640
LAT_MIN = -90.0
LAT_MAX = 90.0
LON_MIN = -180.0
LON_MAX = 180.0


class MODIS_L2_to_L3m_Processor:
    """Process MODIS L2 data to L3m format with 8-day rolling average"""

    def __init__(self, l2_dir, output_dir):
        self.l2_dir = l2_dir
        self.output_dir = output_dir
        self.lat_grid = np.linspace(LAT_MAX, LAT_MIN, LAT_SIZE)  # 90 to -90
        self.lon_grid = np.linspace(LON_MIN, LON_MAX, LON_SIZE)  # -180 to 180

        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initialized processor with L3m grid: {LAT_SIZE}x{LON_SIZE}")

    def get_l2_files_for_year(self, year):
        """Get all L2 files for a given year, sorted by date"""
        year_pattern = os.path.join(self.l2_dir, str(year), "*", "*", "AQUA_MODIS.*.L2.OC.nc")
        files = sorted(glob.glob(year_pattern))

        # Extract date from filename and sort
        def extract_date(filepath):
            basename = os.path.basename(filepath)
            date_str = basename.split('.')[1]  # YYYYMMDDTHHMMSS
            return date_str[:8]  # YYYYMMDD

        files_with_dates = [(f, extract_date(f)) for f in files]
        files_with_dates.sort(key=lambda x: x[1])

        logger.info(f"Found {len(files_with_dates)} L2 files for {year}")
        return files_with_dates

    def read_l2_chlor_a(self, filepath):
        """
        Read chlorophyll-a data from L2 NetCDF file
        Returns: chlor_a (2D array), lat (2D), lon (2D)
        """
        try:
            ds = nc.Dataset(filepath, 'r')

            # Read chlorophyll-a from geophysical_data group
            if 'geophysical_data' not in ds.groups:
                ds.close()
                return None, None, None

            geo = ds.groups['geophysical_data']
            if 'chlor_a' not in geo.variables:
                ds.close()
                return None, None, None

            chlor_a = geo.variables['chlor_a'][:]

            # Read navigation data
            if 'navigation_data' not in ds.groups:
                ds.close()
                return None, None, None

            nav = ds.groups['navigation_data']
            lat = nav.variables['latitude'][:] if 'latitude' in nav.variables else None
            lon = nav.variables['longitude'][:] if 'longitude' in nav.variables else None

            ds.close()

            if lat is None or lon is None:
                return None, None, None

            return chlor_a, lat, lon

        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None, None, None

    def regrid_l2_to_l3m(self, chlor_a, lat, lon):
        """
        Regrid L2 data to L3m grid (4320x8640)
        Uses nearest neighbor assignment

        L2 data has shape (2030, 1354) - each pixel has explicit lat/lon
        L3m is global 4320x8640 grid
        """
        if chlor_a is None or lat is None or lon is None:
            return None

        # Initialize L3m grid
        l3m_grid = np.full((LAT_SIZE, LON_SIZE), np.nan, dtype=np.float32)
        count_grid = np.zeros((LAT_SIZE, LON_SIZE), dtype=np.int32)

        # Flatten arrays
        chlor_flat = chlor_a.flatten()
        lat_flat = lat.flatten()
        lon_flat = lon.flatten()

        # Convert geospatial coordinates to grid indices
        lat_idx = ((LAT_MAX - lat_flat) / (LAT_MAX - LAT_MIN) * LAT_SIZE).astype(int)
        lon_idx = ((lon_flat - LON_MIN) / (LON_MAX - LON_MIN) * LON_SIZE).astype(int)

        # Filter valid indices and valid chlorophyll values
        valid_mask = (lat_idx >= 0) & (lat_idx < LAT_SIZE) & \
                     (lon_idx >= 0) & (lon_idx < LON_SIZE) & \
                     (chlor_flat > 0) & (chlor_flat <= 100) & \
                     ~np.isnan(chlor_flat)

        valid_lat_idx = lat_idx[valid_mask]
        valid_lon_idx = lon_idx[valid_mask]
        valid_chlor = chlor_flat[valid_mask]

        # Accumulate values in L3m grid (for averaging)
        np.add.at(l3m_grid, (valid_lat_idx, valid_lon_idx), valid_chlor)
        np.add.at(count_grid, (valid_lat_idx, valid_lon_idx), 1)

        # Average where we have data
        valid_cells = count_grid > 0
        l3m_grid[valid_cells] = l3m_grid[valid_cells] / count_grid[valid_cells]
        l3m_grid[~valid_cells] = np.nan

        return l3m_grid

    def generate_8day_windows(self, year):
        """Generate 8-day rolling windows for a year"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)

        windows = []
        current_date = start_date

        while current_date <= end_date - timedelta(days=7):
            window_end = current_date + timedelta(days=7)
            windows.append((current_date, window_end))
            current_date += timedelta(days=1)

        return windows

    def process_8day_window(self, year, window_start, window_end, l2_files_with_dates):
        """
        Process an 8-day window:
        1. Collect L2 files within the window
        2. Regrid each to L3m
        3. Average the grids
        4. Save as L3m NetCDF
        """
        # Find L2 files in this window
        window_files = []
        for filepath, date_str in l2_files_with_dates:
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if window_start <= file_date <= window_end:
                window_files.append((filepath, file_date))

        if not window_files:
            return False

        logger.info(f"Processing window {window_start.date()} to {window_end.date()} ({len(window_files)} L2 files)")

        # Read and regrid each L2 file
        l3m_grids = []

        for filepath, _ in window_files:
            chlor_a, lat, lon = self.read_l2_chlor_a(filepath)
            if chlor_a is None:
                continue

            l3m_grid = self.regrid_l2_to_l3m(chlor_a, lat, lon)
            if l3m_grid is not None:
                l3m_grids.append(l3m_grid)

        if not l3m_grids:
            logger.warning(f"No valid L2 data for window {window_start.date()}")
            return False

        # Calculate average (ignoring NaN)
        l3m_stack = np.array(l3m_grids)
        l3m_average = np.nanmean(l3m_stack, axis=0).astype(np.float32)

        # Save to NetCDF
        self.save_l3m_netcdf(l3m_average, window_start, window_end, year)

        return True

    def save_l3m_netcdf(self, data, window_start, window_end, year):
        """Save L3m data to NetCDF file"""
        # Create filename
        date_str_start = window_start.strftime("%Y%m%d")
        date_str_end = window_end.strftime("%Y%m%d")
        filename = f"AQUA_MODIS.{date_str_start}_{date_str_end}.L3m.8D.CHL.chlor_a.4km.nc"
        filepath = os.path.join(self.output_dir, filename)

        try:
            # Create dataset
            ds = nc.Dataset(filepath, 'w', format='NETCDF4')

            # Create dimensions
            ds.createDimension('lat', LAT_SIZE)
            ds.createDimension('lon', LON_SIZE)

            # Create coordinate variables
            lat_var = ds.createVariable('lat', 'f4', ('lat',))
            lon_var = ds.createVariable('lon', 'f4', ('lon',))

            lat_var[:] = self.lat_grid
            lon_var[:] = self.lon_grid

            lat_var.units = 'degrees_north'
            lat_var.long_name = 'latitude'
            lat_var.standard_name = 'latitude'

            lon_var.units = 'degrees_east'
            lon_var.long_name = 'longitude'
            lon_var.standard_name = 'longitude'

            # Create data variable
            chlor_var = ds.createVariable('chlor_a', 'f4', ('lat', 'lon'), fill_value=-32767)
            chlor_var[:] = data

            chlor_var.long_name = 'Chlorophyll a concentration'
            chlor_var.units = 'mg m^-3'
            chlor_var.standard_name = 'mass_concentration_of_chlorophyll_a_in_sea_water'
            chlor_var.valid_min = 0.001
            chlor_var.valid_max = 100.0

            # Add global attributes
            ds.setncattr('title', 'MODIS-Aqua Level-3 Mapped 8-Day Chlorophyll Concentration (Rolling Window)')
            ds.setncattr('instrument', 'MODIS')
            ds.setncattr('platform', 'Aqua')
            ds.setncattr('processing_version', 'Rolling 8-Day Window')
            ds.setncattr('temporal_range', '8-day rolling window')
            ds.setncattr('spatialResolution', '4 km')
            ds.setncattr('map_projection', 'Equidistant Cylindrical')
            ds.setncattr('geospatial_lat_min', LAT_MIN)
            ds.setncattr('geospatial_lat_max', LAT_MAX)
            ds.setncattr('geospatial_lon_min', LON_MIN)
            ds.setncattr('geospatial_lon_max', LON_MAX)

            time_coverage_start = window_start.strftime("%Y-%m-%dT00:00:00.000Z")
            time_coverage_end = window_end.strftime("%Y-%m-%dT23:59:59.999Z")
            ds.setncattr('time_coverage_start', time_coverage_start)
            ds.setncattr('time_coverage_end', time_coverage_end)

            ds.close()

            logger.info(f"Saved: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            return False

    def process_year(self, year):
        """Process all 8-day windows for a year"""
        logger.info(f"=== Processing year {year} ===")

        # Get all L2 files for this year
        l2_files = self.get_l2_files_for_year(year)
        if not l2_files:
            logger.warning(f"No L2 files found for {year}")
            return

        # Generate 8-day windows
        windows = self.generate_8day_windows(year)
        logger.info(f"Generated {len(windows)} 8-day windows for {year}")

        # Process each window
        success_count = 0
        for window_start, window_end in windows:
            if self.process_8day_window(year, window_start, window_end, l2_files):
                success_count += 1

        logger.info(f"Successfully processed {success_count}/{len(windows)} windows for {year}")

    def process_all_years(self):
        """Process all years"""
        for year in YEARS:
            self.process_year(year)

        logger.info("=== All years processed ===")


def main():
    """Main entry point"""
    logger.info("Starting MODIS L2 to L3m rolling 8-day processing")
    logger.info(f"Input directory: {MODIS_L2_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    processor = MODIS_L2_to_L3m_Processor(MODIS_L2_DIR, OUTPUT_DIR)
    processor.process_all_years()

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
