# GOCI
# img_path = '/media/pmilab/My Book1/GOCI/L2_Chl-a'
# save_path = '/home/pmilab/Documents/GOCI/Chl-a'
# save_degree_path = '/home/pmilab/Documents/GOCI/Rrs_degree'

# other satellite data
img_path = '/media/juneyonglee/My Book/VIIRS/VIIRS-SNPP/OC'
save_path = '/media/juneyonglee/My Book/Preprocessed/VIIRS'

# GOCI-II
# img_path = '/media/pmilab/My Book/GOCI-II/Chl/5/1'
# save_path = '/home/pmilab/Documents/GOCI-II/Chl'
# img_path = '/media/pmilab/My Book1/GOCI-II/Rrs(AC)'

# test route
# img_path = '/home/pmilab/Documents/GOCI_make_image/test'
# save_path = '/home/pmilab/Documents/GOCI_make_image/save'


import netCDF4 as nc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import cv2
import csv
from global_land_mask import globe
import glob
import math

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

satellite_type = "GOCI-II"  # set satellite_type(GOCI, GOCI-II, ...)

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
                
                if satellite_type == "GOCI":
                    if data_type == 'CHL':
                        try:
                            print(file)
                            f = h5.File(f_path, 'r+')
                            a = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['CHL Image Pixel Values']
                            np_a = np.array(a)
                            np_a = np.where(np_a == -999.0, 0, np_a)
                            
                            dst = np_a
                            idx = 0
                            row, col = np_a.shape

                            for k in range(0, row, 256):
                                for r in range(0, col, 256):
                                    if idx in ocean_idx:
                                        new_arr = np_a[k:k+256, r+r+256]
                                        if new_arr.shape != (256, 256):
                                            continue
                                        row_col = '_r' + str(k) + '_c' + str(r)
                                        count_idx = new_arr[np.where(new_arr == 0)]
                                        count = count_idx.size
                                        pct = count / (256 * 256) * 10  # get a loss rate
                                        pct = math.floor(pct) * 10
                                        print(pct)
                                        if pct == 100:
                                            continue
                                        
                                        np.savetxt(save_path + '/' + y + '/' + str(pct) + '/' + file[:-7] + row_col, new_arr, delimiter=",")  # save a .txt file
                                    idx = idx + 1
                            f.close()
                        except Exception as e:
                            print(f"Error processing file {file}: {e}")
                    elif data_type == 'RRS':
                        try:
                            f = h5.File(f_path, 'r+')
                            print(f)
                            for band in range(1, 2):  # set band range
                                try:
                                    a = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Band ' + str(band) + ' RRS Image Pixel Values']
                                    np_a = np.array(a)
                                    row, col = np_a.shape
                                    np_a = np.where(np_a == -999.0, 0, np_a)

                                    np_a = np_a * 255  # for setting model input

                                    idx = 0

                                    for k in range(0, row, 256):
                                        for r in range(0, col, 256):
                                            if idx in ocean_idx:
                                                new_arr = np_a[k:k+256, r+r+256]
                                                if new_arr.shape != (256, 256):
                                                    continue
                                                row_col = '_r' + str(k) + '_c' + str(r)
                                                
                                                count_idx = new_arr[np.where(new_arr == 0)]
                                                count = count_idx.size
                                                pct = count / (256 * 256) * 10
                                                pct = math.floor(pct) * 10
                                                print(pct)
                                                if pct == 100:
                                                    continue
                                                cv2.imwrite(save_path + '/' + y + '/' + str(band) + '/' + str(pct) + '/' + file[:-7] + row_col + '.tiff', new_arr)  # save a .tiff file
                                            
                                            idx = idx + 1
                                except Exception as e:
                                    print(f"Error processing band {band} in file {file}: {e}")
                            f.close()
                        except Exception as e:
                            print(f"Error opening file {file}: {e}")

                elif satellite_type == "GOCI-II":
                    if data_type == 'CHL':
                        try:
                            csv_f = open(save_path + '/' + satellite_type + '_' + data_type + 'data_info.csv', 'w', newline='')
                            wr = csv.writer(csv_f)
                            wr.writerow(['index', 'min', 'max', 'mean', 'var', 'std'])
                            print(file)
                            f = nc.Dataset(f_path, 'r')
                            a = f['geophysical_data']['Chl']
                            
                            np_a = np.array(a)
                            np_a = np.where(np_a == -999.0, 0, np_a)

                            a_min = np.nanmin(np_a)
                            a_max = np.nanmax(np_a)
                            a_var = np.nanvar(np_a)
                            a_mean = np.nanmean(np_a)
                            a_std = np.nanstd(np_a)
                            wr.writerow([file, a_min, a_max, a_mean, a_var, a_std])
                            np_a = np_a * 255

                            f.close()
                            csv_f.close()
                        except Exception as e:
                            print(f"Error processing file {file}: {e}")
                    elif data_type == 'RRS':
                        try:
                            csv_f = open(info_save_path + '/' + satellite_type + '_' + data_type + 'data_info.csv', 'w', newline='')
                            wr = csv.writer(csv_f)
                            wr.writerow(['index', 'min', 'max', 'mean', 'var', 'std'])

                            global_data_info = {}
                            band_lst = [380, 412, 443, 490, 510, 555, 620, 660, 680, 709, 745, 865]
                            for i in band_lst:
                                global_data_info[i] = {}
                                global_data_info[i]["min"] = []
                                global_data_info[i]["max"] = []
                                global_data_info[i]["var"] = []
                                global_data_info[i]["mean"] = []
                                global_data_info[i]["std"] = []

                            for month in os.listdir(f_path):
                                day_list = os.listdir(os.path.join(f_path, month))
                                for day in day_list:
                                    file_list = os.listdir(os.path.join(f_path, month, day))
                                    for i in file_list:
                                        if "S007" in i:  # only using slot number 7, because slot 7 contains Korea peninsula
                                            try:
                                                f = nc.Dataset(os.path.join(f_path, month, day, i), 'r')
                                                print(i)
                                                for band in [380, 412, 443, 490, 510, 555, 620, 660, 680, 709, 745, 865]:    
                                                    print(band)
                                                    a = f['geophysical_data']['Rrs']['Rrs_' + str(band)]  # rrs data per band
                                                    np_a = np.array(a)
                                                    np_a = np.where(np_a == -999.0, 0, np_a)
                                                    np_a = np_a[x_min:x_max+1, y_min:y_max+1]

                                                    a_min = np.nanmin(np_a)
                                                    a_max = np.nanmax(np_a)
                                                    a_var = np.nanvar(np_a)
                                                    a_mean = np.nanmean(np_a)
                                                    a_std = np.nanstd(np_a)

                                                    global_data_info[band]["min"].append(a_min)
                                                    global_data_info[band]["max"].append(a_min)
                                                    global_data_info[band]["var"].append(a_min)
                                                    global_data_info[band]["mean"].append(a_min)
                                                    global_data_info[band]["std"].append(a_min)

                                                f.close()
                                            except Exception as e:
                                                print(f"Error processing file {i} in day {day}: {e}")
                            for i in global_data_info:
                                wr.writerow([i, min(global_data_info[i]["min"]), max(global_data_info[i]["max"])])
                            csv_f.close()
                        except Exception as e:
                            print(f"Error processing files in {f_path}: {e}")

                else:  # other satellite type(MODIS, VIIRS, ...)
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