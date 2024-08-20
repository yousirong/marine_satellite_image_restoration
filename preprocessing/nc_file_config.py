import netCDF4 as nc

# NetCDF 파일 경로
# path = '/media/juneyonglee/My Book/VIIRS/VIIRS-SNPP/OC/2013/01/01/SNPP_VIIRS.20130101T035400.L2.OC.nc'
path = '/media/juneyonglee/My Book/MODIS/MODIS-Aqua/OC/2012/01/01/AQUA_MODIS.20120101T031000.L2.OC.nc'
# path = '/media/juneyonglee/My Book/UST21/Daily/2012/01/UST21_L3_Merged-Chla-1D_20120104.nc'

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
