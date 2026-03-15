#!/usr/bin/env python3
"""
MODIS 8-Day Rolling Window Data Processing
Converts L2 MODIS data to L3m format with 8-day rolling window averaging
(Days 1-8, 2-9, 3-10, ..., 358-365)
"""

import os
import glob
from datetime import datetime, timedelta
import logging
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


def read_l2_file(filepath):
    """
    Read L2 MODIS file and extract chlorophyll, lat, lon
    Returns: chlor_a, lat, lon (or None if error)
    """
    try:
        ds = nc.Dataset(filepath, 'r')

        # Read geophysical data
        if 'geophysical_data' not in ds.groups:
            ds.close()
            return None, None, None

        geo = ds.groups['geophysical_data']
        if 'chlor_a' not in geo.variables:
            ds.close()
            return None, None, None

        # Read chlorophyll-a (masked array)
        chlor_a = geo.variables['chlor_a'][:]

        # Convert masked array to float with NaN for masked values
        if np.ma.is_masked(chlor_a):
            chlor_a = chlor_a.filled(np.nan).astype(np.float32)
        else:
            chlor_a = chlor_a.astype(np.float32)

        # Read navigation data
        if 'navigation_data' not in ds.groups:
            ds.close()
            return None, None, None

        nav = ds.groups['navigation_data']
        lat = nav.variables['latitude'][:]
        lon = nav.variables['longitude'][:]

        # Convert masked arrays if needed
        if np.ma.is_masked(lat):
            lat = lat.filled(np.nan)
        if np.ma.is_masked(lon):
            lon = lon.filled(np.nan)

        ds.close()

        return chlor_a.astype(np.float32), lat.astype(np.float32), lon.astype(np.float32)

    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None, None, None


def regrid_l2_to_l3m(chlor_a, lat, lon):
    """
    Regrid L2 data to L3m grid using averaging
    L2 has shape (2030, 1354), L3m is (4320, 8640)
    """
    if chlor_a is None:
        return None

    # Initialize accumulators
    grid_data = np.zeros((LAT_SIZE, LON_SIZE), dtype=np.float64)
    grid_count = np.zeros((LAT_SIZE, LON_SIZE), dtype=np.uint16)

    # Flatten for processing
    chlor_flat = chlor_a.ravel()
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()

    # Calculate grid indices for each data point
    lat_idx = ((LAT_MAX - lat_flat) / (LAT_MAX - LAT_MIN) * LAT_SIZE).astype(np.int32)
    lon_idx = ((lon_flat - LON_MIN) / (LON_MAX - LON_MIN) * LON_SIZE).astype(np.int32)

    # Create validity mask
    valid = (lat_idx >= 0) & (lat_idx < LAT_SIZE) & \
            (lon_idx >= 0) & (lon_idx < LON_SIZE) & \
            (chlor_flat > 0) & (chlor_flat <= 100) & \
            np.isfinite(chlor_flat)

    if not np.any(valid):
        return None

    # Extract valid data
    lat_idx_valid = lat_idx[valid]
    lon_idx_valid = lon_idx[valid]
    chlor_valid = chlor_flat[valid]

    # Accumulate values in grid using 2D indexing
    np.add.at(grid_data, (lat_idx_valid, lon_idx_valid), chlor_valid)
    np.add.at(grid_count, (lat_idx_valid, lon_idx_valid), 1)

    # Calculate average
    grid_result = np.full((LAT_SIZE, LON_SIZE), np.nan, dtype=np.float32)
    mask = grid_count > 0
    grid_result[mask] = (grid_data[mask] / grid_count[mask]).astype(np.float32)

    return grid_result


def get_l2_files_for_date_range(year, start_date, end_date):
    """Get all L2 files within date range"""
    files = []
    pattern = os.path.join(MODIS_L2_DIR, str(year), "*", "*", "AQUA_MODIS.*.L2.OC.nc")

    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        try:
            date_str = basename.split('.')[1]  # YYYYMMDDTHHMMSS
            file_date = datetime.strptime(date_str[:8], "%Y%m%d")

            if start_date <= file_date <= end_date:
                files.append(filepath)
        except:
            continue

    return sorted(files)


def save_l3m_file(data, start_date, end_date):
    """Save data as L3m NetCDF file"""
    date_str_start = start_date.strftime("%Y%m%d")
    date_str_end = end_date.strftime("%Y%m%d")
    filename = f"AQUA_MODIS.{date_str_start}_{date_str_end}.L3m.8D.CHL.chlor_a.4km.nc"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        # Create dataset
        ds = nc.Dataset(filepath, 'w', format='NETCDF4', clobber=True)

        # Create dimensions
        ds.createDimension('lat', LAT_SIZE)
        ds.createDimension('lon', LON_SIZE)

        # Create coordinate variables
        lat_var = ds.createVariable('lat', 'f4', ('lat',), zlib=True)
        lon_var = ds.createVariable('lon', 'f4', ('lon',), zlib=True)

        lat_var[:] = np.linspace(LAT_MAX, LAT_MIN, LAT_SIZE)
        lon_var[:] = np.linspace(LON_MIN, LON_MAX, LON_SIZE)

        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'

        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'

        # Create data variable (fill_value must be set at creation time)
        chlor_var = ds.createVariable('chlor_a', 'f4', ('lat', 'lon'),
                                      zlib=True, complevel=4, fill_value=-32767.0)
        chlor_var[:] = data

        chlor_var.long_name = 'Chlorophyll a concentration'
        chlor_var.units = 'mg m^-3'
        chlor_var.valid_min = 0.001
        chlor_var.valid_max = 100.0

        # Global attributes
        ds.title = 'MODIS-Aqua Level-3 Mapped 8-Day Chlorophyll Concentration (Rolling Window)'
        ds.instrument = 'MODIS'
        ds.platform = 'Aqua'
        ds.processing_version = 'Rolling Window 8-Day'
        ds.temporal_range = '8-day rolling window'
        ds.spatialResolution = '4 km'
        ds.map_projection = 'Equidistant Cylindrical'
        ds.geospatial_lat_min = LAT_MIN
        ds.geospatial_lat_max = LAT_MAX
        ds.geospatial_lon_min = LON_MIN
        ds.geospatial_lon_max = LON_MAX
        ds.time_coverage_start = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
        ds.time_coverage_end = end_date.strftime("%Y-%m-%dT23:59:59.999Z")

        ds.close()
        logger.info(f"Saved: {filename}")
        return True

    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False


def process_year(year):
    """Process all 8-day rolling windows for a year"""
    logger.info(f"=== Processing year {year} ===")

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    # Generate 8-day rolling windows
    windows = []
    current = start_date
    while current + timedelta(days=7) <= end_date:
        window_end = current + timedelta(days=7)
        windows.append((current, window_end))
        current += timedelta(days=1)

    logger.info(f"Generated {len(windows)} 8-day windows")

    success_count = 0
    for window_start, window_end in windows:
        # Get L2 files for this window
        l2_files = get_l2_files_for_date_range(year, window_start, window_end)

        if not l2_files:
            continue

        logger.info(f"Window {window_start.date()}-{window_end.date()}: {len(l2_files)} L2 files")

        # Process each L2 file and accumulate
        grids = []
        for l2_file in l2_files:
            chlor_a, lat, lon = read_l2_file(l2_file)
            if chlor_a is None:
                continue

            grid = regrid_l2_to_l3m(chlor_a, lat, lon)
            if grid is not None:
                grids.append(grid)

        if grids:
            # Average all grids
            grids_array = np.array(grids)
            avg_grid = np.nanmean(grids_array, axis=0).astype(np.float32)

            # Save
            if save_l3m_file(avg_grid, window_start, window_end):
                success_count += 1

    logger.info(f"Year {year}: {success_count}/{len(windows)} windows processed")


def main():
    """Main entry point"""
    logger.info("Starting MODIS L2 to L3m rolling 8-day processing")
    logger.info(f"Input: {MODIS_L2_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for year in YEARS:
        process_year(year)

    logger.info("=== Processing complete ===")


if __name__ == "__main__":
    main()
