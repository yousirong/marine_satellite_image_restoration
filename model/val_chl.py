import os
import glob
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from tqdm import trange, tqdm
import random
from sklearn.metrics import r2_score as r2_


def plot_parity(filename, loss_rate,true, pred, rmse_, mape_, kind="scatter", 
                xlabel="true", ylabel="predict", title="Loss 50-60%", 
                hist2d_kws=None, scatter_kws=None, kde_kws=None,
                equal=True, metrics=True, metrics_position="lower right",
                figsize=(8, 8), ax=None, save_file = True,):
    
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # data range
    val_min = min(true.min(), pred.min())
    val_max = max(true.max(), pred.max())

    # data plot
    if "scatter" in kind:
        if not scatter_kws:
            scatter_kws={'color':'green', 'alpha':0.5}
        ax.scatter(true, pred, s = 1, **scatter_kws)
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
        mape = mape_
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
        ax.text(text_pos_x, text_pos_y-0.1, f"MAE = {mape:.8f}", 
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y-0.2, f"R2 = {r2:.3f}", 
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)

    # 파일로 저장
    fig = ax.figure
    fig.tight_layout()
    if save_file:
        fig.savefig(filename+f'/{loss_rate}.png')
    else:
        print("check save file path, saving failed@@")
    plt.show()
    return ax


def validate(loss_rate, data_path, save_path):

    ####path
    recon_path = os.path.join(data_path, 'recon')    #goci_path ==> recon
    gt_path = os.path.join(data_path, 'gt')           #modis_path => gt
    mask_path = os.path.join(data_path, 'mask')   #mask
    assert os.path.isdir(recon_path) and os.path.isdir(gt_path) and os.path.isdir(mask_path), "Please check dataset path is valid"
    

    ####files list
    gt_files_list = sorted(list(glob.glob(os.path.join(gt_path, '*'), recursive=True)))
    recon_files_list = sorted(list(glob.glob(os.path.join(recon_path, '*'), recursive=True)))
    mask_files_list = sorted(list(glob.glob(os.path.join(mask_path, '*'), recursive=True)))
    print("len(gt_files_list):", len(gt_files_list))
    print("len(recon_files_list):", len(recon_files_list))
    print("len(mask_files_list):", len(mask_files_list))

    ####variables
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

    # [0,1] 픽셀만 성능 검증
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i in trange(len(recon_files_list)):
        # for i in trange(10):
            recon_parts = recon_files_list[i].split('/')
            recon_file_name = recon_parts[-1] 
            mask_parts = mask_files_list[i].split('/')
            mask_file_name= mask_parts[-1]
            gt_parts = gt_files_list[i].split('/')
            gt_file_name = gt_parts[-1] 

            cnt = cnt+1

            selected_recon = recon_files_list[i]
            selected_mask = mask_files_list[i]
            selected_gt = gt_files_list[i]
            restored_np = np.loadtxt(selected_recon, delimiter=',',dtype='float32')
            restored_np = restored_np/255.0
            mask = np.loadtxt(selected_mask, delimiter=',',dtype='float32')
            gt_np = np.loadtxt(selected_gt, delimiter=',',dtype='float32')
            gt_np = gt_np/255.0

            W = gt_np.shape[0]
            H = gt_np.shape[0]

            for w in range(W):
                for h in range(H):
                    #only restoration area performance
                    if mask[w,h]==0 and gt_np[w,h]!=0 and restored_np[w,h]!=0:
                        
                        if gt_np[w,h]<0 or restored_np[w,h] < 0 or restored_np[w,h] > 0.078 or gt_np[w,h]>0.078:
    #                     if gt_np[w,h]<0 or restored_np[w,h] < 0 or restored_np[w,h] > 10 or gt_np[w,h]>10:
                            outlier = outlier+1
                            continue
                        else:
                            try:
                                loss = (gt_np[w,h]-restored_np[w,h])/gt_np[w,h]
                                plt_gt.append(gt_np[w, h])
                                plt_res.append(restored_np[w, h])

                            except Warning:
                                print(recon_file_name)
                                print(gt_np[w,h])
                                print(restored_np[w,h])
                                outlier = outlier+1
                                continue

                        if math.isinf(loss) or math.isnan(gt_np[w, h]) or math.isnan(restored_np[w, h]): 
                            continue

                        temp_mape = temp_mape + abs(gt_np[w, h]-restored_np[w, h])
                        #temp_mse = temp_mse + (gt_np[i,j]-restored_np[i,j])**2
                        temp_rmse = temp_rmse + (gt_np[w, h]-restored_np[w, h])**2
                        #temp_rmspe = temp_rmspe + loss**2
                        #temp_mape = temp_mape + abs(loss)
                        cloud_count = cloud_count+1



    plt_gt = np.array(plt_gt)
    plt_res = np.array(plt_res)


    plot_parity(filename = save_path,
                loss_rate = loss_rate,
                true=plt_gt,
                pred=plt_res,
                rmse_=math.sqrt(temp_rmse/cloud_count),
                mape_= temp_mape/cloud_count,
                title=f"Loss {loss_rate}-{int(loss_rate)+9}%",           
    )
