import os
import glob
from PIL import Image 
import numpy as np
import math
import cv2
import torch
import decimal

info_band = [380,412,443,490,510,555,620,660,680,709,745,865]
info_min = [-0.040952116, -0.03115786, -0.03290336, -0.030595144, -0.029857721, -0.028025944, -0.026336318, -0.024608022, -0.019731289, -0.013473362, -0.0006118686, -0.0003371486]
info_max = [0.15080376, 0.1442169, 0.11957388,0.1284895,0.11004408,0.07235966,0.069359526,0.044848032,0.048895203,0.03874212,0.1749536, 0.029107913]


#ground-truth image path
#gt_path = "data/GOCI_RRS/Rrs_test/gt/1/"
gt_path = "results/GOCI_RRS_degree/1/10/gt/"
#restored image path
restored_path = "results/GOCI_RRS_degree/1/10/recon/"

#mask image path
#mask_path = "data/GOCI_RRS/Rrs_test/Rrs_mask/1/10/"
mask_path = "results/GOCI_RRS_degree/1/10/mask/"

gt_list = os.listdir(gt_path)
restored_list = os.listdir(restored_path)
mask_list = os.listdir(mask_path)

gt_list = sorted(gt_list)

restored_list = sorted(restored_list)

mask_list = sorted(mask_list)

mse_list = []
rmse_list = []
mape_list=[]
mae_list = []
cloud_count_list = []

temp_rmspe =0
temp_mae = 0
temp_rmse = 0
cloud_count = 0
outlier = 0
tmp = 0
print("len(gt_path)", len(gt_list))
min_restored = []
#for every restored image
for i in range(len(gt_list)):
    #gt_np =cv2.imread(gt_path+gt_list[i], cv2.IMREAD_UNCHANGED)
    gt_np = np.loadtxt(gt_path+gt_list[i], delimiter=',',dtype='float32')
    gt_np = gt_np / 255.
    #print(gt_np)
    
    #restored_np = cv2.imread(restored_path+restored_list[i],cv2.IMREAD_UNCHANGED)
    restored_np = np.loadtxt(restored_path+restored_list[i], delimiter=',',dtype='float32')
    restored_np = restored_np / 255.
    min_restored.append(np.min(abs(restored_np)))
    #print(restored_np)
    #restored_np = np.sum(restored_np, -1) / 3 

    
    #mask = cv2.imread(mask_path+mask_list[i])
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.loadtxt(mask_path+mask_list[i], delimiter=',',dtype='float32')
    
    #print(mask.shape)
    #print(mask)
    '''
    gt = Image.open(gt_path+gt_list[i]).convert("L")
    restored = Image.open(restored_path+gt_list[i]).convert("L")
    mask = Image.open(mask_path+gt_list[i])
    gt_np = np.asarray(gt)
    restored_np = np.asarray(restored)
    mask_np = np.asarray(mask)

    #For real value
    #gt_np = info_min[0]+gt_np*(info_max[0]-info_min[0])#for real value
    #restored_np = info_min[0]+restored_np*(info_max[0]-info_min[0])#for real value

    #For 0~1 value
    gt_np = gt_np/254.0#0~1 normalized
    restored_np = restored_np/254.0#0~1 normalized

    
    #mask_np = info_min[0]+mask_np*(info_max[0]-info_min[0])
'''
    w = gt_np.shape[0]
    h = gt_np.shape[0]

    #temp_mse = 0
    #temp_rmse = 0
    #temp_mae = 0
    #cloud_count = 0
    for i in range(w):
        for j in range(h):
            #only restoration area performance
            if mask[i,j]==0:
                loss = (gt_np[i,j]-restored_np[i,j])/gt_np[i,j]

                if gt_np[i,j]<0 or restored_np[i,j] < 0:
                    outlier = outlier+1
                    #rint("loss : ", loss)
                    #print("gt : ", gt_np[i,j])
                    #print("retored_np : ", restored_np[i,j])
                    continue
                if math.isinf(loss):
                    continue
                #temp_mse = temp_mse + (gt_np[i,j]-restored_np[i,j])**2
                
                #if loss>10:
                #    print(loss)
                #    print(restored_np[i,j])
                #    print(gt_np[i,j])
                #    tmp = tmp+ 1
                #    print("===================")
                temp_rmse = temp_rmse + (gt_np[i,j]-restored_np[i,j])**2
                temp_rmspe = temp_rmspe + loss**2
                temp_mape = temp_mape + abs(loss)
                cloud_count = cloud_count+1
                #print(loss**2)
            
            #temp_mse = temp_mse + (gt_np[i,j]-restored_np[i,j])**2
            #temp_rmse = temp_rmse + (gt_np[i,j]-restored_np[i,j])**2
            #temp_mae = temp_mae + abs(gt_np[i,j]-restored_np[i,j])
            #cloud_count = cloud_count+1
    
    #print("MSE:",temp_mse/cloud_count)
print("MAPE:", temp_mape/cloud_count)
print("RMSE:",math.sqrt(temp_rmse/cloud_count))
print("RMSPE:",math.sqrt(temp_rmspe/cloud_count))
print("cloud count: ", cloud_count)



    #mse_list.append(temp_mse/cloud_count)
    #rmse_list.append(math.sqrt(temp_rmse/cloud_count))
    #mae_list.append(temp_mae/cloud_count)
    #cloud_count_list.append(cloud_count)

#print("MSE")
#print(sum(mse_list)/len(mse_list))

#print("RMSE")
#print(sum(rmse_list)/len(rmse_list))

#print("MAPE")
#print((sum(mae_list)/len(mae_list))*100)




'''
file_name = './performance/3/30/mse.txt'
with open(file_name, 'w+') as file:
    file.write('\n'.join(mse_list))
file_name = './performance/3/30/rmse.txt'
with open(file_name, 'w+') as file:
    file.write('\n'.join(rmse_list))
file_name = './performance/3/30/mae.txt'
with open(file_name, 'w+') as file:
    file.write('\n'.join(mae_list))
file_name = './performance/3/30/cloud.txt'
with open(file_name, 'w+') as file:
    file.write('\n'.join(cloud_count_list))
'''

                






#MSE
#0.000169050546701005
#RMSE
#0.010008637449858658
#MAE
#0.0047770616987512764


#MSE
#0.0008690927320306929
#RMSE
#0.025579065219706522
#MAE
#0.0164974467045689



