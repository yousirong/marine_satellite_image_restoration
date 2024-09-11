# nc 파일 열어서 탐색하는 파일 
import netCDF4 as nc

# NetCDF 파일 경로
# path = '/media/juneyonglee/My Book/VIIRS/VIIRS-SNPP/OC/2013/01/01/SNPP_VIIRS.20130101T035400.L2.OC.nc'
# path = '/media/juneyonglee/My Book/MODIS/MODIS-Aqua/OC/2012/01/01/AQUA_MODIS.20120101T031000.L2.OC.nc'
# path = '/media/juneyonglee/My Book/UST21/Daily/2012/01/UST21_L3_Merged-Chla-1D_20120104.nc'
path = '/home/juneyonglee/Documents/COMS_GOCI_L2A_GA_20110401001641.RRS.he5'
def print_group_info(group, indent=0):
    indent_str = "  " * indent
    print(f"{indent_str}Group: {group.path}")
    for name, variable in group.variables.items():
        print(f"{indent_str}  Variable: {name} {variable.shape} {variable.dtype}")
    for name, dim in group.dimensions.items():
        print(f"{indent_str}  Dimension: {name} {len(dim)}")
    for name, subgroup in group.groups.items():
        print_group_info(subgroup, indent + 1)

# NetCDF 파일 열기
f = nc.Dataset(path, 'r')

# 파일 내부 구조 탐색
print(f"Variables: {f.variables.keys()}")
print(f"Dimensions: {f.dimensions.keys()}")
print(f"Groups: {f.groups.keys()}")

# 파일 내부 구조 탐색
print_group_info(f)

f.close()

'''
(unir) juneyonglee@juneyonglee-RTX3090-2:~/Desktop/AY_ust$ python preprocessing/nc_file_config.py
Variables: dict_keys([])
Dimensions: dict_keys(['number_of_lines', 'pixels_per_line', 'bands_per_pixel', 'number_of_reflectance_location_values', 'pixel_control_points', 'number_of_bands', 'number_of_reflective_bands'])
Groups: dict_keys(['sensor_band_parameters', 'scan_line_attributes', 'geophysical_data', 'navigation_data', 'processing_control'])
Group: /
  Dimension: number_of_lines 2030
  Dimension: pixels_per_line 1354
  Dimension: bands_per_pixel 16
  Dimension: number_of_reflectance_location_values 10
  Dimension: pixel_control_points 1354
  Dimension: number_of_bands 24
  Dimension: number_of_reflective_bands 16
  Group: /sensor_band_parameters
    Variable: wavelength (24,) int32
    Variable: vcal_gain (16,) float32
    Variable: vcal_offset (16,) float32
    Variable: F0 (16,) float32
    Variable: aw (16,) float32
    Variable: bbw (16,) float32
    Variable: k_oz (16,) float32
    Variable: k_no2 (16,) float32
    Variable: Tau_r (16,) float32
  Group: /scan_line_attributes
    Variable: year (2030,) int32
    Variable: day (2030,) int32
    Variable: msec (2030,) int32
    Variable: detnum (2030,) int8
    Variable: mside (2030,) int8
    Variable: slon (2030,) float32
    Variable: clon (2030,) float32
    Variable: elon (2030,) float32
    Variable: slat (2030,) float32
    Variable: clat (2030,) float32
    Variable: elat (2030,) float32
    Variable: csol_z (2030,) float32
  Group: /geophysical_data
    Variable: aot_869 (2030, 1354) int16
    Variable: angstrom (2030, 1354) int16
    Variable: Rrs_412 (2030, 1354) int16
    Variable: Rrs_443 (2030, 1354) int16
    Variable: Rrs_469 (2030, 1354) int16
    Variable: Rrs_488 (2030, 1354) int16
    Variable: Rrs_531 (2030, 1354) int16
    Variable: Rrs_547 (2030, 1354) int16
    Variable: Rrs_555 (2030, 1354) int16
    Variable: Rrs_645 (2030, 1354) int16
    Variable: Rrs_667 (2030, 1354) int16
    Variable: Rrs_678 (2030, 1354) int16
    Variable: chlor_a (2030, 1354) float32
    Variable: Kd_490 (2030, 1354) int16
    Variable: pic (2030, 1354) int16
    Variable: poc (2030, 1354) int16
    Variable: ipar (2030, 1354) int16
    Variable: nflh (2030, 1354) int16
    Variable: par (2030, 1354) int16
    Variable: l2_flags (2030, 1354) int32
  Group: /navigation_data
    Variable: longitude (2030, 1354) float32
    Variable: latitude (2030, 1354) float32
    Variable: cntl_pt_cols (1354,) int32
    Variable: cntl_pt_rows (2030,) int32
    Variable: tilt (2030,) float32
  Group: /processing_control
    Group: /processing_control/input_parameters
    Group: /processing_control/flag_percentages

    


    GOCI 관련 
 (unir) juneyonglee@juneyonglee-RTX3090-2:~/Desktop/AY_ust$ python preprocessing/nc_file_config.py
Variables: dict_keys([])
Dimensions: dict_keys([])
Groups: dict_keys(['HDFEOS', 'HDFEOS INFORMATION'])
Group: /
  Group: /HDFEOS
    Group: /HDFEOS/ADDITIONAL
      Group: /HDFEOS/ADDITIONAL/FILE_ATTRIBUTES
    Group: /HDFEOS/GRIDS
      Group: /HDFEOS/GRIDS/Image Data
        Group: /HDFEOS/GRIDS/Image Data/Data Fields
          Variable: Band 1 RRS Image Pixel Values (5685, 5567) float32
          Variable: Band 2 RRS Image Pixel Values (5685, 5567) float32
          Variable: Band 3 RRS Image Pixel Values (5685, 5567) float32
          Variable: Band 4 RRS Image Pixel Values (5685, 5567) float32
          Variable: Band 5 RRS Image Pixel Values (5685, 5567) float32
          Variable: Band 6 RRS Image Pixel Values (5685, 5567) float32
          Variable: Band 7 RRS Image Pixel Values (5685, 5567) float32
          Variable: Band 8 RRS Image Pixel Values (5685, 5567) float32
          Dimension: phony_dim_0 5685
          Dimension: phony_dim_1 5567
    Group: /HDFEOS/POINTS
      Group: /HDFEOS/POINTS/Ephemeris
        Group: /HDFEOS/POINTS/Ephemeris/Data
        Group: /HDFEOS/POINTS/Ephemeris/Linkage
      Group: /HDFEOS/POINTS/Event
        Group: /HDFEOS/POINTS/Event/Data
        Group: /HDFEOS/POINTS/Event/Linkage
      Group: /HDFEOS/POINTS/File Descripter Metadata
        Group: /HDFEOS/POINTS/File Descripter Metadata/Data
        Group: /HDFEOS/POINTS/File Descripter Metadata/Linkage
      Group: /HDFEOS/POINTS/Map Projection
        Group: /HDFEOS/POINTS/Map Projection/Data
        Group: /HDFEOS/POINTS/Map Projection/Linkage
      Group: /HDFEOS/POINTS/Navigation for GOCI
        Group: /HDFEOS/POINTS/Navigation for GOCI/Data
          Variable: Navigation for GOCI (128,) {'names': ['Band number', 'Slot number', 'Relative time', 'Spacecraft attitude', 'XO', 'YO', 'XS', 'YS', 'XPO', 'YPO', 'XPS', 'YPS', 'Number of valid A parameters', 'A parameters value', 'Number of valid B parameters', 'B parameters value', 'Number of valid C parameters', 'C parameters value', 'Number of valid D parameters', 'D parameters value', 'Number of valid A prime parameters', 'A prime parameters value', 'Number of valid B prime parameters', 'B prime parameters value', 'Number of valid C prime parameters', 'C prime parameters value', 'Number of valid D prime parameters', 'D prime parameters value'], 'formats': ['<i4', '<i4', '<f4', ('<f4', (3,)), '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4', '<i4', ('<f4', (16,)), '<i4', ('<f4', (16,)), '<i4', ('<f4', (16,)), '<i4', ('<f4', (16,)), '<i4', ('<f4', (16,)), '<i4', ('<f4', (16,)), '<i4', ('<f4', (16,)), '<i4', ('<f4', (16,))], 'offsets': [0, 4, 8, 12, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 124, 128, 192, 196, 260, 264, 328, 332, 396, 400, 464, 468, 532, 536], 'itemsize': 600, 'aligned': True}
          Dimension: phony_dim_2 128
        Group: /HDFEOS/POINTS/Navigation for GOCI/Linkage
      Group: /HDFEOS/POINTS/Radiometric Calibration for GOCI
        Group: /HDFEOS/POINTS/Radiometric Calibration for GOCI/Data
        Group: /HDFEOS/POINTS/Radiometric Calibration for GOCI/Linkage
      Group: /HDFEOS/POINTS/Scene Header
        Group: /HDFEOS/POINTS/Scene Header/Data
        Group: /HDFEOS/POINTS/Scene Header/Linkage
      Group: /HDFEOS/POINTS/Validataion (trailer)
        Group: /HDFEOS/POINTS/Validataion (trailer)/Data
        Group: /HDFEOS/POINTS/Validataion (trailer)/Linkage
  Group: /HDFEOS INFORMATION
    Variable: StructMetadata.0 () <class 'str'>
'''