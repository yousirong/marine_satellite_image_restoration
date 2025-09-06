#!/usr/bin/env python3
import os
import glob
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from tqdm import trange
from sklearn.metrics import r2_score as r2_
from matplotlib import cm
from matplotlib.colors import Normalize
import re
import argparse

# ust21 land-sea mask 경로
land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_parity(filename, loss_rate, true, pred, rmse_, mape_, kind="scatter",
                xlabel="true (mg/m$^3$)", ylabel="predict (mg/m$^3$)", title="Loss 50-60%",
                hist2d_kws=None, scatter_kws=None, kde_kws=None,
                equal=True, metrics=True, metrics_position="lower right",
                figsize=(8, 8), ax=None, save_file=True):

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Data range for plotting [0.01, 10]
    val_min = 0.01
    val_max = 10

    if kind == "scatter":
        if not scatter_kws:
            scatter_kws = {'s': 1, 'alpha': 0.01}
        ax.scatter(true, pred, **scatter_kws)
    elif kind == "hist2d":
        if not hist2d_kws:
            hist2d_kws = {'cmap': 'Greens', 'vmin': 1}
        ax.hist2d(true, pred, **hist2d_kws)
    elif kind == "kde":
        if not kde_kws:
            kde_kws = {'cmap': 'viridis', 'levels': 5}
        sns.kdeplot(x=true, y=pred, **kde_kws, ax=ax)

    ax.set_xlim([val_min, val_max])
    ax.set_ylim([val_min, val_max])
    ticks = np.arange(0, 11, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=15)
    ax.grid(True)
    ax.plot([val_min, val_max], [val_min, val_max], c="k", alpha=0.3)
    font_label = {"color": "gray", "fontsize": 20}
    ax.set_xlabel(xlabel, fontdict=font_label, labelpad=8)
    ax.set_ylabel(ylabel, fontdict=font_label, labelpad=8)
    font_title = {"color": "gray", "fontsize": 20, "fontweight": "bold"}
    ax.set_title(title, fontdict=font_title, pad=16)

    if metrics:
        font_metrics = {'color': 'k', 'fontsize': 14}
        if metrics_position == "lower right":
            text_pos_x, text_pos_y, ha = 0.98, 0.3, "right"
        elif metrics_position == "upper left":
            text_pos_x, text_pos_y, ha = 0.1, 0.9, "left"
        else:
            text_pos_x, text_pos_y, ha = 0.1, 0.9, "left"
        ax.text(text_pos_x, text_pos_y, f"RMSE = {rmse_:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.1, f"MAE = {mape_:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.2, f"R2 = {r2_(true, pred):.3f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)

    fig.tight_layout()
    if save_file:
        fig.savefig(filename + f'/{loss_rate}.png')
    else:
        print("Check save file path, saving failed.")
    plt.show()
    return ax

def normalize_data(data, vmin=0, vmax=20):
    data_normalized = data.copy()
    land_mask = (data_normalized == 255)
    data_normalized[land_mask] = np.nan
    data_normalized = np.clip(data_normalized, vmin, vmax)
    data_normalized[land_mask] = np.nan
    return data_normalized

# The following function is now disabled because we only want to generate a parity plot.
def save_colormap_image_with_land_mask(data, land_sea_mask_path, row, col, save_path, vmin=0.01, vmax=10, land_color=[0,0,0], recon_file_name=None):
    pass

def validate(loss_rate, data_path, save_path, land_sea_mask_path, reliability_threshold=0.9):
    recon_path = os.path.join(data_path, 'recon')
    gt_path = os.path.join(data_path, 'gt')
    mask_path = os.path.join(data_path, 'mask')
    assert os.path.isdir(recon_path) and os.path.isdir(gt_path) and os.path.isdir(mask_path), \
        "Please check dataset path is valid"

    # We do not save color-map images in this script.
    color_image_path = os.path.join(save_path, f'color_{loss_rate}')
    if not os.path.exists(color_image_path):
        os.makedirs(color_image_path)

    recon_files_list = sorted(glob.glob(os.path.join(recon_path, '*.csv')), key=natural_sort_key)
    gt_files_list = sorted(glob.glob(os.path.join(gt_path, '*.csv')), key=natural_sort_key)
    mask_files_list = sorted(glob.glob(os.path.join(mask_path, '*.csv')), key=natural_sort_key)

    if len(recon_files_list) == 0 or len(gt_files_list) == 0 or len(mask_files_list) == 0:
        print("No image files found in the specified paths.")
        return

    print("len(gt_files_list):", len(gt_files_list))
    print("len(recon_files_list):", len(recon_files_list))
    print("len(mask_files_list):", len(mask_files_list))

    temp_rmse = 0
    temp_mape = 0
    cloud_count = 0
    all_gt = []
    all_pred = []

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i in trange(len(recon_files_list)):
            recon_file_name = os.path.basename(recon_files_list[i])
            mask_file_name = os.path.basename(mask_files_list[i])
            gt_file_name = os.path.basename(gt_files_list[i])

            restored_np = np.loadtxt(recon_files_list[i], delimiter=',', dtype='float32')
            mask = np.loadtxt(mask_files_list[i], delimiter=',', dtype='float32')
            gt_np = np.loadtxt(gt_files_list[i], delimiter=',', dtype='float32')

            # Treat 255 as land => np.nan
            restored_np = np.where(restored_np == 255, np.nan, restored_np)
            gt_np = np.where(gt_np == 255, np.nan, gt_np)
            mask = np.where(mask == 255, np.nan, mask)

            match = re.search(r'r(\d+)_c(\d+)', recon_file_name)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
            else:
                print(f"Filename format does not match for {recon_file_name}")
                continue

            # Commented out color-map saving
            save_colormap_image_with_land_mask(
                data=restored_np,
                land_sea_mask_path=land_sea_mask_path,
                row=row,
                col=col,
                save_path=os.path.join(color_image_path, recon_file_name),
                vmin=0.01,
                vmax=10,
                land_color=[0,0,0],
                recon_file_name=recon_file_name
            )

            valid_indices = (~np.isnan(gt_np)) & (~np.isnan(restored_np)) & (~np.isnan(mask)) & (gt_np > 0)
            print("Valid indices count:", np.sum(valid_indices))
            gt_valid = gt_np[valid_indices]
            pred_valid = restored_np[valid_indices]
            filter_range = (gt_valid > 0) & (gt_valid <= 10) & (pred_valid > 0) & (pred_valid <= 10)
            print("After filtering to [0,10]:", np.sum(filter_range))
            gt_valid = gt_valid[filter_range]
            pred_valid = pred_valid[filter_range]

            all_gt.extend(gt_valid.tolist())
            all_pred.extend(pred_valid.tolist())

            temp_mape += np.nansum(np.abs(gt_valid - pred_valid))
            temp_rmse += np.nansum((gt_valid - pred_valid)**2)
            cloud_count += gt_valid.size

    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)

    if cloud_count == 0:
        print("No valid data found for plotting.")
        return

    rmse_val = math.sqrt(temp_rmse / cloud_count)
    mape_val = temp_mape / cloud_count

    plot_parity(
        filename=save_path,
        loss_rate=loss_rate,
        true=all_gt,
        pred=all_pred,
        rmse_=rmse_val,
        mape_=mape_val,
        title=f"Loss {loss_rate}-{int(loss_rate)+9}%",
        save_file=True
    )
