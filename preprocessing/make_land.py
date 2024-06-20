import h5py as h5
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

lat_path = "./LAT_LON/COMS_GOCI_L2P_GA_20110524031644.LAT.he5"
lon_path = "./LAT_LON/COMS_GOCI_L2P_GA_20110524031644.LON.he5"

f_lat = h5.File(lat_path,'r+')
f_lon = h5.File(lon_path,'r+')

lat_ = f_lat['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Latitude Image Pixel Values']
lat_ = list(lat_)
np_lat = np.array(lat_)

lon_ = f_lon['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Longitude Image Pixel Values']
lon_ = list(lon_)
np_lon = np.array(lon_)

x, y = np_lon.shape
print(x, y)

new_array = np.zeros((x,y))
for i in range(x):
    for j in range(y):
        lat = np_lat[i, j]
        lon = np_lon[i, j]
        is_on_land = globe.is_land(lat,lon)
        print(is_on_land)
        if is_on_land : 
            new_array[i, j] = -1
        else:
            new_array[i, j] = 0

np.save("./is_land_on_GOCI.npy", new_array)