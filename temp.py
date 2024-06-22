import argparse
import os
from model import RFRNetModel
from dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.io import load_yaml, build_save_folder
import torch
# from val_chl import validate

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


def plot_parity(filename, loss_rate,true, pred, rmse_, mae_, kind="scatter", 
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
    if save_file:
        fig.savefig(filename+f'/{loss_rate}.png')
    plt.show()
    return ax


def validate(loss_rate, data_path):

    ####current
    goci_path = f'/home/ubuntu/문서/AY_RFR/model/results/chl_only_range2/{step}/{loss_rate}/degree/recon'    #복원
    modis_path = f'/home/ubuntu/문서/AY_RFR/model/results/chl_only_range2/{step}/{loss_rate}/degree/gt'           #gt
    mask_path = f'/home/ubuntu/문서/AY_RFR/model/results/chl_only_range2/{step}/{loss_rate}/degree/mask'    #mask
    save_path = f'/home/ubuntu/문서/AY_RFR/model/performance/chl_only_range2/goci_to_goci/{step}/{loss_rate}'
    #####

    time_range = 2
    if not os.path.isdir(save_path):
        os.makedirs(save_path)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default="", type=str, help="config file path")
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true')
    args = parser.parse_args()

    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    args = build_save_folder(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"#args.gpu_ids
    
    model = RFRNetModel()

    if args.test:
        model.initialize_model(args.model_path, False, None, args.gpu_ids)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = False, training=False))
        model.test(dataloader, args.result_save_path)
    elif args.val:
        validate(args.loss_rate, args.val_root)
        
    else:
        model.initialize_model(args.model_path, True, args.model_save_path, args.gpu_ids)
        model.cuda()
        dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = False), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        model.train(dataloader, args.model_save_path, args.save_capacity, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()
