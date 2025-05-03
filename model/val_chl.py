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

# ust21
# land_sea_mask_path ='/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy'
# goci
land_sea_mask_path ='/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'
# preprocessing/is_land_on_GOCI_modified_1_999.npy

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_parity(filename, loss_rate, true, pred, rmse_, mape_, vmin, vmax, kind="scatter",
                xlabel="true (mg/m$^3$)", ylabel="predict (mg/m$^3$)", title="Loss 50-60%",
                hist2d_kws=None, scatter_kws=None, kde_kws=None,
                equal=True, metrics=True, metrics_position="lower right",
                figsize=(8, 8), ax=None, save_file=True):

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # data plot
    if "scatter" in kind:
        if not scatter_kws:
            scatter_kws = {'color': 'green', 'alpha': 0.5}
        ax.scatter(true, pred, s=1, **scatter_kws)
    elif "hist2d" in kind:
        if not hist2d_kws:
            hist2d_kws = {'cmap': 'Greens', 'vmin': 1}
        ax.hist2d(true, pred, **hist2d_kws)
    elif "kde" in kind:
        if not kde_kws:
            kde_kws = {'cmap': 'viridis', 'levels': 5}
        sns.kdeplot(x=true, y=pred, **kde_kws, ax=ax)

    # x, y bounds (use vmin and vmax dynamically calculated)
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])

    # x, y ticks, ticklabels
    ticks = np.linspace(vmin, vmax, 5)  # Set dynamic ticks based on vmin and vmax
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=15)

    # grid
    ax.grid(True)

    # 기준선
    ax.plot([vmin, vmax], [vmin, vmax], c="k", alpha=0.3)

    # x, y label
    font_label = {"color": "gray", "fontsize": 20}
    ax.set_xlabel(xlabel, fontdict=font_label, labelpad=8)
    ax.set_ylabel(ylabel, fontdict=font_label, labelpad=8)

    # title
    font_title = {"color": "gray", "fontsize": 20, "fontweight": "bold"}
    ax.set_title(title, fontdict=font_title, pad=16)

    # metrics
    if metrics:
        rmse = rmse_
        mape = mape_
        r2 = r2_(true, pred)
        font_metrics = {'color': 'k', 'fontsize': 14}

        if metrics_position == "lower right":
            text_pos_x = 0.98
            text_pos_y = 0.3
            ha = "right"
        elif metrics_position == "upper left":
            text_pos_x = 0.1
            text_pos_y = 0.9
            ha = "left"
        else:
            text_pos_x, text_pos_y = 0.1
            ha = "left"

        ax.text(text_pos_x, text_pos_y, f"RMSE = {rmse:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.1, f"MAE = {mape:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.2, f"R2 = {r2:.3f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)

    # 파일로 저장
    fig = ax.figure
    fig.tight_layout()
    if save_file:
        fig.savefig(filename + f'/{loss_rate}.png')
    else:
        print("check save file path, saving failed@@")
    plt.show()
    return ax


def save_land_mask_image(land_mask_cropped, save_path):
    """
    Save the cropped land-sea mask as an image, where land is black (255) and sea is white (0).
    """
    # Create a binary mask image (land as black, sea as white)
    mask_img = land_mask_cropped

    # # Ensure the save path has the correct extension and remove .csv from the filename if present
    # save_path_with_extension = save_path if save_path.lower().endswith('.png') else save_path + '_mask.png'
    # save_path_with_extension = save_path_with_extension.replace('.csv', '')  # Remove .csv from filename

    # # Save the mask image
    # plt.imsave(save_path_with_extension, mask_img, cmap='gray')

    # # Display the mask image
    # plt.imshow(mask_img, cmap='gray')
    # plt.title(f'Land-Sea Mask')
    # plt.xticks([])
    # plt.yticks([])

    # # Save the mask image with a proper title
    # plt.savefig(save_path_with_extension.replace('.png', '_mask_bar.png'), dpi=300, bbox_inches='tight')
    # plt.close()



def normalize_data_dynamic(data):
    data_normalized = data.copy()
    land_mask = (data_normalized == 255)
    data_normalized[land_mask] = np.nan
    vmin = np.nanmin(data_normalized)
    vmax = np.nanmax(data_normalized)
    if vmax > 20:
        vmin, vmax = 0, 20
    data_normalized = (data_normalized - vmin) / (vmax - vmin)
    data_normalized[land_mask] = np.nan
    return data_normalized, vmin, vmax

def save_colormap_image_with_land_mask(data, land_sea_mask_path, row, col, save_path, land_color=[0, 0, 0], recon_file_name=None):
    date_str = None
    if recon_file_name:
        match = re.search(r'(\d{8})', recon_file_name)  # Extract the date part
        if match:
            date_str = match.group(1)

    land_mask_full = np.load(land_sea_mask_path)
    land_mask_cropped = land_mask_full[row:row + 256, col:col + 256]
    data_normalized, vmin, vmax = normalize_data_dynamic(data)
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.ScalarMappable(norm=norm, cmap='jet')
    colored_img = colormap.to_rgba(data_normalized)[:, :, :3]
    land_mask = (land_mask_cropped == 1)
    colored_img[land_mask] = land_color

    if not save_path.lower().endswith('.png'):
        save_path_with_extension = save_path.replace('.csv', '')  # Remove .csv from filename
        if date_str:
            # Use date_str only once in the file name if present
            save_path_with_extension += f'_nak_r{row}_c{col}.png'
        else:
            save_path_with_extension += '.png'
    else:
        save_path_with_extension = save_path

    plt.imsave(save_path_with_extension, colored_img)
    cmap = plt.get_cmap("jet")
    cmap.set_bad('white', 1.0)
    plt.imshow(data_normalized, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Chlorophyll-a concentration (mg/m³)', ticks=np.linspace(vmin, vmax, num=5))
    plt.title(f'Restored Chlorophyll-a Concentration with Land Mask')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path_with_extension.replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()



def validate(loss_rate, data_path, save_path, land_sea_mask_path):
    recon_path = os.path.join(data_path, 'recon')
    gt_path = os.path.join(data_path, 'gt')
    mask_path = os.path.join(data_path, 'mask')
    assert os.path.isdir(recon_path) and os.path.isdir(gt_path) and os.path.isdir(mask_path), "Please check dataset path is valid"

    color_image_path = os.path.join(save_path, f'color_{loss_rate}')
    if not os.path.exists(color_image_path):
        os.makedirs(color_image_path)

    recon_files_list = sorted(glob.glob(os.path.join(recon_path, '*.csv')), key=lambda s: natural_sort_key(s))
    gt_files_list = sorted(glob.glob(os.path.join(gt_path, '*.csv')), key=lambda s: natural_sort_key(s))
    mask_files_list = sorted(glob.glob(os.path.join(mask_path, '*.csv')), key=lambda s: natural_sort_key(s))

    if len(recon_files_list) == 0 or len(gt_files_list) == 0 or len(mask_files_list) == 0:
        print("No image files found in the specified paths.")
        return

    temp_rmse = 0
    temp_mape = 0
    cloud_count = 0
    plt_gt = []
    plt_res = []

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i in trange(len(recon_files_list)):
            recon_file_name = os.path.basename(recon_files_list[i])
            mask_file_name = os.path.basename(mask_files_list[i])
            gt_file_name = os.path.basename(gt_files_list[i])

            restored_np = np.loadtxt(recon_files_list[i], delimiter=',', dtype='float32')
            mask = np.loadtxt(mask_files_list[i], delimiter=',', dtype='float32')
            gt_np = np.loadtxt(gt_files_list[i], delimiter=',', dtype='float32')

            # 육지(255)는 NaN으로 처리
            restored_np = np.where(restored_np == 255, np.nan, restored_np)
            gt_np = np.where(gt_np == 255, np.nan, gt_np)
            mask = np.where(mask == 255, np.nan, mask)  # 마스크도 NaN 처리

            # 추출한 row와 col 좌표를 파일 이름에서 파싱 (예: recon_file_name에 'rX_cY' 형태로 포함된 경우)
            match = re.search(r'r(\d+)_c(\d+)', recon_file_name)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
            else:
                print(f"Filename format does not match the expected row-col pattern for {recon_file_name}")
                continue

            # 저장된 컬러맵 이미지를 육지 마스크와 함께 저장
            save_colormap_image_with_land_mask(restored_np, land_sea_mask_path, row, col,
                                               os.path.join(color_image_path, recon_file_name))

            # RMSE 및 MAPE 계산
            W, H = gt_np.shape
            for w in range(W):
                for h in range(H):
                    if np.isnan(mask[w, h]) or np.isnan(gt_np[w, h]) or np.isnan(restored_np[w, h]):
                        continue
                    elif gt_np[w, h] == 0:
                        continue  # 0으로 나누는 경우 방지
                    else:
                        loss = (gt_np[w, h] - restored_np[w, h]) / gt_np[w, h]
                        plt_gt.append(gt_np[w, h])
                        plt_res.append(restored_np[w, h])

                        temp_mape += abs(gt_np[w, h] - restored_np[w, h])
                        temp_rmse += (gt_np[w, h] - restored_np[w, h]) ** 2
                        cloud_count += 1

    plt_gt = np.array(plt_gt)
    plt_res = np.array(plt_res)

    if cloud_count == 0:
        print("No valid data found for plotting.")
        return

    # Normalize data using dynamic min-max normalization for validation
    plt_gt_normalized, _, _ = normalize_data_dynamic(plt_gt)
    plt_res_normalized, _, _ = normalize_data_dynamic(plt_res)

    # vmin = min(np.min(plt_gt_normalized), np.min(plt_res_normalized))
    # vmax = max(np.max(plt_gt_normalized), np.max(plt_res_normalized))
    vmin=0
    vmax=20

    plot_parity(filename=save_path,
                loss_rate=loss_rate,
                true=plt_gt_normalized,
                pred=plt_res_normalized,
                rmse_=np.sqrt(temp_rmse / cloud_count) if cloud_count > 0 else float('nan'),
                mape_=temp_mape / cloud_count if cloud_count > 0 else float('nan'),
                vmin=vmin,
                vmax=vmax,
                title=f"Loss {loss_rate}-{int(loss_rate)+9}%"
    )
