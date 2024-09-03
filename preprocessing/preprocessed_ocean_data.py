# GOCI data preprocessing

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


lat_min = 35.1#latitude, wedo
lat_max = 36.2
long_min = 129.1#longtitude, kyungdo
long_max = 130.25



####FOR GOCI-II####
#x_min=935
#x_max=1287
#y_min=2361
#y_max=2771
###################

######south#######
'''
x_min = 2346
x_max = 2602
y_min = 3053
y_max = 3309
'''
###################

######west#######
'''
x_min = 1761
x_max = 2017
y_min = 2309
y_max = 2565
'''
###################

######east#######

x_min = 2702
x_max = 2958
y_min = 2397
y_max = 2653

###################
def landorocean(lat,lon):
    is_on_land = globe.is_land(lat,lon)
    #True:land, False:Ocean
    return is_on_land
    
#GOCI
# img_path = '/media/pmilab/My Book1/GOCI/L2_Chl-a'
# save_path = '/home/pmilab/Documents/GOCI/Chl-a'
# save_degree_path = '/home/pmilab/Documents/GOCI/Rrs_degree'

#other sattelite data
# img_path = '/home/pmilab/Documents/validation/out/chl/VIIRS'
# save_path = '/home/pmilab/Documents/VIIRS/2021_chl'

#GOCI-II
#img_path = '/media/pmilab/My Book/GOCI-II/Chl/5/1'
#save_path = '/home/pmilab/Documents/GOCI-II/Chl'
#img_path = '/media/pmilab/My Book1/GOCI-II/Rrs(AC)'

#test route
#img_path = '/home/pmilab/Documents/GOCI_make_image/test'
#save_path = '/home/pmilab/Documents/GOCI_make_image/save'

#CUR_working
img_path = '/media/pmilab/My Book1/GOCI/L2_Chl-a'
save_path = '/home/pmilab/Documents/New_prep/GOCI/Chl-a'

if not os.path.isdir(save_path):
    os.makedirs(save_path)


satellite_type = "GOCI" #set satellite_type(GOCI, GOCI-II, ...) 

data_type = 'CHL' #set data_type(RRS, CHL, ...)

if "GOCI-II" in img_path:
    satellite_type = "GOCI-II"

if 'Chl' in img_path:
    data_type = 'CHL'
elif 'Rrs' in img_path:
    data_type = 'RRS'
elif 'SSC' in img_path:
    data_type = 'SSC'

#img_list = os.listdir(img_path)
year_list = os.listdir(img_path)
mask = np.load("./is_land_on_GOCI.npy") #get GOCI Land mask 
ocean_idx = np.load("./ocean_idx_arr.npy")
from tqdm import tqdm
for y in year_list:
    if y in ["2021"]:
        path = img_path+'/'+y
        #target = path+"/*.he5"
        #folder_list = glob.glob(target)
        #folder_list = [f for f in os.listdir(path) if f.endswith('.he5')]
        print(str(y))
       
        #folder_list = os.listdir(path)
        img_list = os.listdir(path)

        if satellite_type=="GOCI":
            if data_type=='CHL':
                for i in tqdm(img_list):
                    if not "he5" in i:
                        continue
                    i = os.path.basename(i)
                    #print(i)
                    f = h5.File(path+'/'+i,'r+')
                    a = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['CHL Image Pixel Values']
                    a = list(a)
                    np_a = np.array(a)
                    np_a = np.where(np_a==-999.0, 0, np_a)
                   
                    dst = np_a
                    #print(dst.shape)
                    idx = 0 
                    row, col = np_a.shape

                    for k in range(0, row, 256):
                        for r in range(0, col, 256):
                            if idx in ocean_idx:
                                new_arr = np_a[k:k+256, r:r+256]
                                if new_arr.shape!=(256,256):
                                    continue
                                row_col = '_r' + str(k) + '_c' + str(r)
                                count_idx = new_arr[np.where(new_arr==0)]
                                count = count_idx.size
                                pct = count/(256*256) *10 #get a loss rate 
                                pct = math.floor(pct)*10
                                # print(pct)
                                if pct == 100 : 
                                    continue
                                if not os.path.isdir(os.path.join(save_path,y,str(pct))):
                                    os.makedirs(os.path.join(save_path,y,str(pct)))
                                np.savetxt(save_path+'/'+ y+'/' +str(pct)+'/'+ i[:-7] + row_col, new_arr, delimiter=",") # save a .txt file
                            idx = idx+1
               
                    f.close()
            elif data_type=='RRS':
                for i in folder_list:
                    f_path = path+"/"+i
                    try : 
                        f = h5.File(f_path,'r+')
                        print(f)
                    except:
                        continue
                    for band in range(1,2):#set band range     
                        try:
                            a = f['HDFEOS']['GRIDS']['Image Data']['Data Fields']['Band '+str(band)+ ' RRS Image Pixel Values']
                        except :
                            continue 
                        np_a = np.array(a)
                        row, col = np_a.shape 
                        np_a = np.where(np_a==-999.0, 0, np_a)

                        #if np.nanmax(np_a) > 5 : 
                        #    continue
                        np_a = np_a*255 #for setting model input 

                        idx = 0 

                        for k in range(0, row, 256):
                            for r in range(0, col, 256):
                                if idx in ocean_idx:
                                    new_arr = np_a[k:k+256, r:r+256]
                                    if new_arr.shape!=(256,256):
                                        continue
                                    row_col = '_r' + str(k) + '_c' + str(r)
                                    
                                    count_idx = new_arr[np.where(new_arr==0)]
                                    count = count_idx.size
                                    pct = count/(256*256) *10
                                    pct = math.floor(pct)*10
                                    print(pct)
                                    if pct == 100 : 
                                        continue
                                    cv2.imwrite(save_path + '/'+ y+'/'+ str(band)+ '/' +str(pct)+'/'+ i[:-7] + row_col +'.tiff', new_arr )#save a .tiff file
                                   
                                idx = idx+1

        elif satellite_type=="GOCI-II":
            if data_type=='CHL':
                csv_f = open(save_path+'/'+satellite_type+'_'+data_type+'data_info.csv','w',newline='')
                wr = csv.writer(csv_f)
                wr.writerow(['index','min','max','mean','var','std'])
                for i in img_list:
                    print(i)
                    #f = h5.File(img_path+'/'+i,'r+')
                    f = nc.Dataset(img_path+'/'+i,'r')
                    a = f['geophysical_data']['Chl']
                   
                    np_a = np.array(a)
                    np_a = np.where(np_a==-999.0, 0, np_a)

                    a_min = np.nanmin(np_a)
                    a_max = np.nanmax(np_a)
                    a_var = np.nanvar(np_a)
                    a_mean = np.nanmean(np_a)
                    a_std = np.nanstd(np_a)
                    wr.writerow([i,a_min,a_max,a_mean,a_var,a_std])
                    np_a = np_a*255

                    #cv2.imwrite(save_path+'/'+i[:-3]+'.png',np_a)
                    #plt.imsave(save_path+'/'+i[:-3]+'plt.png',np_a)
                    f.close()
                csv_f.close()
            elif data_type=='RRS':
                csv_f = open(info_save_path+'/'+satellite_type+'_'+data_type+'data_info.csv','w',newline='')
                wr = csv.writer(csv_f)
                wr.writerow(['index','min','max','mean','var','std'])

                global_data_info = {}
                band_lst = [380,412,443,490,510,555,620,660,680,709,745,865]
                for i in band_lst:
                    global_data_info[i]={}
                    global_data_info[i]["min"] = []
                    global_data_info[i]["max"] = []
                    global_data_info[i]["var"] = []
                    global_data_info[i]["mean"] = []
                    global_data_info[i]["std"] = []

                for month in img_list:
                    day_list = os.listdir(img_path+"/"+month)
                    for day in day_list:
                        file_list = os.listdir(img_path+"/"+month+"/"+day)
                        for i in file_list:
                            if "S007" in i:#only using slot number 7, because slot 7 contains Korea peninsula
                                f = nc.Dataset(img_path+"/"+month+"/"+day+'/'+i,'r')
                                print(i)
                                for band in [380,412,443,490,510,555,620,660,680,709,745,865]:    
                                    print(band)
                                    a = f['geophysical_data']['Rrs']['Rrs_'+str(band)]#rrs data per band
                                    #lat = f['navigation_data']['latitude']
                                    #lon = f['navigation_data']['longitude']

                                    #np_lat = np.array(lat)
                                    #np_lon = np.array(lon)
                                    np_a = np.array(a)
                                    np_a = np.where(np_a==-999.0, 0, np_a)
                                    np_a = np_a[x_min:x_max+1,y_min:y_max+1]

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


                                    #wr.writerow([i[:-3]+str(band),a_min,a_max,a_mean,a_var,a_std])#individual image info
                                    ##global image info


                                    np_a = np.array(a)*255
                                    #cv2.imwrite(save_path+'/'+i[:-3]+str(band)+'.png',np_a)
                                    #plt.imsave(save_path+'/'+i[:-3]+'plt.png',np_a)
                                f.close()
                            else:
                                continue

                for i in global_data_info:
                    wr.writerow([i,min(global_data_info[i]["min"]),max(global_data_info[i]["max"])])    
               
                csv_f.close()
            elif data_type=='SSC':
                csv_f = open(save_path+'/'+satellite_type+'_'+data_type+'data_info.csv','w',newline='')
                wr = csv.writer(csv_f)
                wr.writerow(['index','min','max','mean','var','std'])
                for i in img_list:
                    f = nc.Dataset(img_path+'/'+i,'r')
                    for band in ['SSC_direction','SSC_speed','SSC_u','SSC_v']:    
                        a = f['geophysical_data'][band]
                        np_a = np.array(a)
                        np_a = np.where(np_a==-999.0, 0, np_a)

                        a_min = np.nanmin(np_a)
                        a_max = np.nanmax(np_a)
                        a_var = np.nanvar(np_a)
                        a_mean = np.nanmean(np_a)
                        a_std = np.nanstd(np_a)
                        wr.writerow([i[:-3]+str(band),a_min,a_max,a_mean,a_var,a_std])

                        np_a = np.array(a)*255
                        cv2.imwrite(save_path+'/'+i[:-3]+str(band)+'.png',np_a)
                        #plt.imsave(save_path+'/'+i[:-3]+'plt.png',np_a)
                    f.close()
                csv_f.close()
        else: # other satellite type(MODIS, VIIRS, ...)
            if data_type=='RRS':
                
                for i in img_list:
                    f_path = path+"/"+i

                    np_a = np.load(f_path)
                   
                    row, col = np_a.shape 
                    np_a = np.nan_to_num(np_a)
                    #np_a = np.where(np_a==-999.0, 0, np_a)
    
                    idx = 0 
                    for k in range(0, row, 256):
                        for r in range(0, col, 256):
                            if idx in ocean_idx:
                                new_arr = np_a[k:k+256, r:r+256]
                                if new_arr.shape!=(256,256):
                                    continue
                                row_col = '_r' + str(k) + '_c' + str(r)
                                
                                count_idx = new_arr[np.where(new_arr==0)]
                                count = count_idx.size
                                pct = count/(256*256) *10
                                pct = math.floor(pct)*10
                                print(pct)
                                print(new_arr)
                                if pct > 80 : 
                                    continue
                                #cv2.imwrite(save_path + '/'+ y+'/'+ str(band)+ '/'+ i[:-8] + row_col +'.tiff', new_arr )
                                #cv2.imwrite(save_path + '/'+ y+'/'+ i[:-7] + row_col +'.tiff', new_arr)
                                print(i[:-7],row_col)
                                #cv2.imwrite(t_path + '/'+str(pct)+'/'+ i[:-7] + row_col +'.png', new_arr)
                                np.savetxt(save_path+'/'+ i[:-7] + row_col, new_arr, delimiter=",")
                            idx = idx+1
                    

