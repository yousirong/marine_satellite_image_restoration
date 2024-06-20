# cv2.imread(img_path+"/"+img, cv2.IMREAD_UNCHANGED)

import os
import glob
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score as r2_
def plot_parity(true, pred, rmse_, mae_, kind="scatter", 
                xlabel="true", ylabel="predict", title="Loss 10-20%", 
                hist2d_kws=None, scatter_kws=None, kde_kws=None,
                equal=True, metrics=True, metrics_position="lower right",
                figsize=(8, 8), ax=None, filename='./performance/chl/viirs/'):
    
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # data range
    val_min = min(true.min(), pred.min())
    val_max = max(true.max(), pred.max())

    # data plot
    if "scatter" in kind:
        if not scatter_kws:
            scatter_kws={'color':'green', 'alpha':0.5}
        ax.scatter(true, pred, s = 25, **scatter_kws)
    elif "hist2d" in kind:
        if not hist2d_kws:
            hist2d_kws={'cmap':'Greens', 'vmin':1}
        ax.hist2d(true, pred, **hist2d_kws)
    elif "kde" in kind:
        if not kde_kws:
            kde_kws={'cmap':'viridis', 'levels':5}
        sns.kdeplot(x=true, y=pred, **kde_kws, ax=ax)

    # x, y bounds
    xbounds = ax.get_xbound()
    ybounds = ax.get_ybound()
    max_bounds = [min(xbounds[0], ybounds[0]), max(xbounds[1], ybounds[1])]
    ax.set_xlim(max_bounds)
    ax.set_ylim(max_bounds)

    # x, y ticks, ticklabels
    y = ax.get_yticks()
    ticks = [round(y[i],3) for i in range (len(y)) if not y[i]<0 and i%2==1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=15)

    # grid
    ax.grid(True)

    # 기준선
    ax.plot(max_bounds, max_bounds, c="k", alpha=0.3)

    # x, y label
    font_label = {"color":"gray", "fontsize":20}
    ax.set_xlabel(xlabel, fontdict=font_label, labelpad=8)
    ax.set_ylabel(ylabel, fontdict=font_label, labelpad=8)

    # title
    font_title = {"color": "gray", "fontsize":20, "fontweight":"bold"}
    ax.set_title(title, fontdict=font_title, pad=16)

    # metrics
    if metrics:
        #rmse = mean_squared_error(true, pred, squared=False)
        #mae = mean_absolute_error(true, pred)
        #r2 = r2_score(true, pred)
        rmse = rmse_
        mae = mae_
        r2 = r2_(true, pred)
        #r2 = r2_
        font_metrics = {'color':'k', 'fontsize':14}

        if metrics_position == "lower right":
            text_pos_x = 0.98
            text_pos_y = 0.3
            ha = "right"
        elif metrics_position == "upper left":
            text_pos_x = 0.1
            text_pos_y = 0.9
            ha = "left"
        else:
            text_pos_x, text_pos_y = 0.1 #text_position
            ha = "left"

        ax.text(text_pos_x, text_pos_y, f"RMSE = {rmse:.8f}", 
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y-0.1, f"MAE = {mae:.8f}", 
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y-0.2, f"R2 = {r2:.3f}", 
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)

    # 파일로 저장
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(filename+'10.png')
    if not ax:
        fig.tight_layout()
        if filename:
            fig.savefig(filename+'band443_40.png')
    else: print("fail@@")
    return ax

goci_path  = '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/results/modis_rrs_degree/10/recon'
#goci_path = './interpolation/rrs_test/10/'
#goci_path = '/home/pmilab/Documents/validation_data/recon/GOCI_RRS_degree/2020/2/10/img/'
#modis_path = '/home/pmilab/Documents/validation_data/modis/2021/'
#modis_path = '/home/pmilab/Documents/GOCI/Chl-a/2021/0/'
modis_path = '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/results/modis_rrs_degree/10/gt/'
#mask_path = '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/data/GOCI_RRS/Rrs_test/2021/Rrs_mask/3/10/'
mask_path = '/media/pmilab/3dbe7506-c248-4dac-a1f3-866a0bc3ecf8/home/pmimoon/Documents/RFR/results/modis_rrs_degree/10/mask/'

save_path = "./performance/rrs/modis"
time_range = 2

modis_files_list = glob.glob(os.path.join(modis_path, '*'), recursive=True)
goci_files_list = glob.glob(os.path.join(goci_path, '*'), recursive=True)
mask_files_list = glob.glob(os.path.join(mask_path, '*'), recursive=True)
print("len(modis_files_list):", len(modis_files_list))
print("len(goci_files_list):", len(goci_files_list))
print("len(mask_files_list):", len(mask_files_list))
temp_rmspe =0
temp_mae = 0
temp_mape = 0
temp_rmse = 0
cloud_count = 0
outlier = 0
temp_mse = 0 
tmp = 0
cnt = 0

plt_gt = []
plt_res = []
for i in range(len(goci_files_list)):

    goci_parts = goci_files_list[i].split('/')

    goci_file_name = goci_parts[-1] 
    goci_file_name_parts = goci_file_name.split('_')

    goci_date = goci_file_name_parts[4][:8]
    goci_time = goci_file_name_parts[4][8:10]

    goci_r_value = goci_file_name_parts[-2][1:] 
    goci_c_value = goci_file_name_parts[-1][1:]

    # print("날짜:", goci_date)
    # print("시간:", goci_time)
    # print("r 값:", goci_r_value)
    # print("c 값:", goci_c_value)     


    match_list =[]

    for j in range(len(modis_files_list)):

        modis_parts = modis_files_list[j].split('/')


        modis_file_name = modis_parts[-1] 
        modis_file_name_parts = modis_file_name.split('_')

        modis_date = modis_file_name_parts[1][6:14]
        modis_time = modis_file_name_parts[1][15:17]

        modis_r_value = modis_file_name_parts[-2][1:] 
        modis_c_value = modis_file_name_parts[-1].split('.')[0][1:]

        #print("날짜:", modis_date)
        #print("시간:", modis_time)
        #print("r 값:", modis_r_value)
        #print("c 값:", modis_c_value)

        if modis_date == goci_date and modis_r_value == goci_r_value and modis_c_value == goci_c_value:
            #print(goci_files_list[i], modis_files_list[j])
            modis_timeint = int(modis_time)
            goci_timeint = int(goci_time)
            
            if  (abs(goci_timeint - modis_timeint)) <= time_range :
                #print(modis_timeint, goci_timeint)
                match_list.append(modis_files_list[j])
    if len(match_list) > 1:
        #print(sorted(match_list))
        selected_goci = goci_files_list[i]
        selected_modis = sorted(match_list)[-1]
        print("!!!!")
        continue
    elif len(match_list) == 1:
        cnt = cnt+1
        selected_goci = goci_files_list[i]
        selected_mask = mask_files_list[i]
        selected_modis = match_list[0]
        print(selected_mask,selected_goci, selected_modis)        

        restored_np = np.loadtxt(selected_goci, delimiter=' ',dtype='float32')
        #restored_np = restored_np / 255.
        
        #mask = cv2.imread(selected_mask, cv2.IMREAD_UNCHANGED)
        mask = np.loadtxt(selected_mask, delimiter=',',dtype='float32')
        gt_np = np.loadtxt(selected_modis, delimiter=',', dtype='float32')
     
        W = gt_np.shape[0]
        H = gt_np.shape[0]
        #print(w,h)
        for w in range(W):
            for h in range(H):
                #only restoration area performance
                if mask[w,h]==0 and gt_np[w,h]!=0 and restored_np[w,h]!=0:
                    loss = (gt_np[w,h]-restored_np[w,h])/gt_np[w,h]
                    
                    if gt_np[w,h]<0 or restored_np[w,h] < 0 or restored_np[w,h] > 10 or gt_np[w,h]>10:
                            outlier = outlier+1
                            continue
                    if math.isinf(loss) or math.isnan(gt_np[w,h]) or math.isnan(restored_np[w,h]): 
                            continue
                    
                    #if restored_np[i,j] < 0.0125 :
                    plt_gt.append(gt_np[w,h])
                    plt_res.append(restored_np[w,h])
                    #if loss>10:
                    #    print(loss)
                    #    print(restored_np[i,j])
                    #    print(gt_np[i,j])
                    #    tmp = tmp+ 1
                    #    print("===================")
                    
                    temp_mae = temp_mae + abs(gt_np[w,h]-restored_np[w,h])
                    #temp_mse = temp_mse + (gt_np[i,j]-restored_np[i,j])**2
                    temp_rmse = temp_rmse + (gt_np[w,h]-restored_np[w,h])**2
                    #temp_rmspe = temp_rmspe + loss**2
                    #temp_mape = temp_mape + abs(loss)
                    cloud_count = cloud_count+1
#plotting(save_path,plt_gt, plt_res)
plt_gt = np.array(plt_gt)
plt_res = np.array(plt_res)
plot_parity(plt_gt, plt_res, math.sqrt(temp_rmse/cloud_count), temp_mae/cloud_count)
print("MAE:", temp_mae/cloud_count)
print("MAPE:", temp_mape/cloud_count)
print("MSE", temp_mse/cloud_count)
print("RMSE:",math.sqrt(temp_rmse/cloud_count))
print("RMSPE:",math.sqrt(temp_rmspe/cloud_count))
#print("tmp : ", tmp)
print("cloud count: ", cloud_count)
print("count : ", cnt)         
    
    #################
    # Enter content #
    # val1 : selected_goci
    # val2 : selected_modis
    #################

     
