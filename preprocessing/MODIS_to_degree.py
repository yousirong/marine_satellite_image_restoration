import netCDF4 as nc
import numpy as np
import os
import math
from global_land_mask import globe

lat_min = 35.1  # latitude, wedo
lat_max = 36.2
long_min = 129.1  # longitude, kyungdo
long_max = 130.25

# GOCI-II
x_min = 2702
x_max = 2958
y_min = 2397
y_max = 2653

def landorocean(lat, lon):
    is_on_land = globe.is_land(lat, lon)
    # True:land, False:Ocean
    return is_on_land

# other satellite data
img_path = '/media/juneyonglee/My Book/VIIRS/VIIRS-SNPP/OC'
save_path = '/media/juneyonglee/My Book/Preprocessed/VIIRS'

satellite_type = "VIIRS"  # set satellite_type(GOCI, GOCI-II, ...)

data_type = 'RRS'  # set data_type(RRS, CHL, ...)

year_list = os.listdir(img_path)
mask = np.load("preprocessing/is_land_on_GOCI.npy")  # get GOCI Land mask
ocean_idx = np.load("preprocessing/ocean_idx_arr.npy", allow_pickle=True)

for y in year_list:
    if y in ["2021"]:
        path = os.path.join(img_path, y)
        print(str(y))
        
        for root, dirs, files in os.walk(path):
            for file in files:
                f_path = os.path.join(root, file)
                
                if satellite_type == "VIIRS" and data_type == 'RRS':
                    try:
                        with nc.Dataset(f_path, 'r') as f:
                            if 'geophysical_data' not in f.groups:
                                print(f"'geophysical_data' group not found in {f_path}")
                                continue
                            
                            geophysical_data = f.groups['geophysical_data']
                            rrs_vars = [v for v in geophysical_data.variables if v.startswith('Rrs_')]
                            
                            for var_name in rrs_vars:
                                try:
                                    a = geophysical_data.variables[var_name][:]
                                    np_a = np.array(a)
                                    np_a = np.nan_to_num(np_a)
                                    
                                    # 데이터 범위 확인
                                    print(f"Processing variable {var_name} in file {file}")
                                    print(f"Data min: {np.min(np_a)}, max: {np.max(np_a)}, mean: {np.mean(np_a)}")

                                    row, col = np_a.shape
                                    
                                    idx = 0
                                    for k in range(0, row, 256):
                                        for r in range(0, col, 256):
                                            if idx in ocean_idx:
                                                new_arr = np_a[k:k+256, r:r+256]
                                                if new_arr.shape != (256, 256):
                                                    continue
                                                row_col = '_r' + str(k) + '_c' + str(r)
                                                
                                                count_idx = new_arr[np.where(new_arr == 0)]
                                                count = count_idx.size
                                                pct = count / (256 * 256) * 100
                                                print(f"Loss rate: {pct}% for segment {row_col} in variable {var_name}")
                                                if pct > 80:
                                                    continue
                                                np.savetxt(save_path + '/' + file[:-7] + row_col, new_arr, delimiter=",")
                                            idx = idx + 1
                                except Exception as e:
                                    print(f"Error processing variable {var_name} in file {file}: {e}")
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
