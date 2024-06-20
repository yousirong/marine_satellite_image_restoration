from PIL import Image
import cv2
import os
import math
import numpy as np
import glob

#rrs_path_1 = './interpolation/rrs_test/2/50'
#rrs_path_2 = './interpolation/rrs_test/3/50'
#rrs_path_3 ='./interpolation/rrs_test/4/50'
#rrs_mask ='/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/GOCI_RRS/Rrs_test/2021/Rrs_mask/3/50'
#save_path = 'interpolation/chl/50/'

#current
rrs_path_1 = './basic_interpolation/2021/rrs_test/2/50'
rrs_path_2 = './basic_interpolation/2021/rrs_test/3/50'
rrs_path_3 ='./basic_interpolation/2021/rrs_test/4/50'
rrs_mask ='/home/pmilab/Desktop/RFR/data/GOCI_RRS/Rrs_test/2021/Rrs_mask/3/50'
save_path = './basic_interpolation/ay_chl/50/'

if not os.path.isdir(save_path):
    os.makedirs(save_path)
else:
    print("Folder already exists")

mask_files_list = glob.glob(os.path.join(rrs_mask, '*'), recursive=True)
rrs1_files_list = glob.glob(os.path.join(rrs_path_1, '*'), recursive=True)
rrs2_files_list = glob.glob(os.path.join(rrs_path_2, '*'), recursive=True)
rrs3_files_list = glob.glob(os.path.join(rrs_path_3, '*'), recursive=True)

#image_path = "COMS_GOCI_L2A_GA_20200210021642._r4096_c1792"
data_type = ['443', '490', '555']
from tqdm import trange
for i in trange(len(rrs1_files_list)):

    img = []
    f_name = rrs2_files_list[i].split('/')
    f_name = f_name[-1]
    #f_name = f_name[:-5]
    #print(rrs1_files_list[i]) 
    #print(rrs2_files_list[i]) 
    #print(rrs3_files_list[i]) 
    #print("@@@@")
    rrs1 = np.loadtxt(rrs1_files_list[i], delimiter=',',dtype='float32')/255
    rrs2 = np.loadtxt(rrs2_files_list[i], delimiter=',',dtype='float32')/255
    rrs3 = np.loadtxt(rrs3_files_list[i], delimiter=',',dtype='float32')/255
    img.append(rrs1)
    img.append(rrs2)
    img.append(rrs3)
    
    #mask = np.loadtxt(mask_files_list[i], delimiter=',',dtype='float32')
    mask = cv2.imread(mask_files_list[i], cv2.IMREAD_GRAYSCALE)
    '''
    for i, typ in enumerate(data_type):
        img_name = image_path + '_' + str(typ) + '.tiff'
        assert os.path.exists(img_name), f"{img_name} 파일이 존재하지 않습니다."
        img.append(cv2.imread(img_name, cv2.IMREAD_UNCHANGED))
    '''
    R_rs = np.stack(img, axis=0)
    _, height, width = R_rs.shape

    a = [0.2515, -2.3798, 1.5823, -0.6372, -0.5692]

    Chl_oc3 = np.empty((height,width))

    for h in range(height):
        for w in range(width):
            if R_rs[2, h, w] <= 0 or R_rs[0, h, w] <= 0 or R_rs[1, h, w] <= 0:
                Chl_oc3[h,w] = 0
            else:
                term = np.sum(a[i] * (np.log10(np.max(R_rs[:2,h,w])/R_rs[2,h,w]))**i for i in range(1, 5))
                Chl_oc3[h,w] = 10 ** (a[0] + term)
                #print(Chl_oc3[h,w])
    np.savetxt(save_path+f_name, Chl_oc3)
        
#print(img.shape)
