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
import random

# Pillow (16비트 TIFF 저장용)
from PIL import Image

# ust21
land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy'
# goci
# land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_parity(filename, loss_rate, true, pred, rmse_, mae_,
                kind="scatter",  # scatter로 기본 설정 변경
                xlabel="true (mg/m$^3$)", ylabel="predict (mg/m$^3$)",
                title="Loss 50-60%",
                hist2d_kws=None, scatter_kws=None, kde_kws=None,
                equal=True, metrics=True, metrics_position="lower right",
                figsize=(8, 8), ax=None, save_file=True):

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # Data range, constrained between 0.01 and 10
    val_min = 0.01
    val_max = 10

    # Data plot
    if "scatter" in kind:
        if not scatter_kws:
            scatter_kws = {'s': 1, 'alpha': 0.01}
        ax.scatter(true, pred, **scatter_kws)
    elif "hist2d" in kind:
        if not hist2d_kws:
            hist2d_kws = {'bins': 300, 'cmap': 'Greens', 'vmin': 1}
        ax.hist2d(true, pred, **hist2d_kws)
    elif "kde" in kind:
        if not kde_kws:
            kde_kws = {'cmap': 'viridis', 'levels': 5}
        sns.kdeplot(x=true, y=pred, **kde_kws, ax=ax)

    # x, y bounds
    ax.set_xlim([val_min, val_max])
    ax.set_ylim([val_min, val_max])

    ticks = np.arange(0, 11, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=15)

    # Grid
    ax.grid(True)

    # Diagonal reference line
    ax.plot([val_min, val_max], [val_min, val_max], c="k", alpha=0.3)

    # x, y labels
    font_label = {"color": "gray", "fontsize": 20}
    ax.set_xlabel(xlabel, fontdict=font_label, labelpad=8)
    ax.set_ylabel(ylabel, fontdict=font_label, labelpad=8)

    # Title
    font_title = {"color": "gray", "fontsize": 20, "fontweight": "bold"}
    ax.set_title(title, fontdict=font_title, pad=16)

    # Metrics
    if metrics:
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
            text_pos_x, text_pos_y = 0.1, 0.9
            ha = "left"

        ax.text(text_pos_x, text_pos_y, f"RMSE = {rmse_:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.1, f"MAE = {mae_:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.2, f"R2 = {r2:.3f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)

    # Save to file
    fig = ax.figure
    fig.tight_layout()
    if save_file:
        os.makedirs(filename, exist_ok=True)
        fig.savefig(os.path.join(filename, f'{loss_rate}.png'))
    else:
        print("Check save file path, saving failed.")
    plt.show()
    return ax

def convert_raw_to_color(data, vmin=0.01, vmax=10, cmap_name='jet'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap_name)
    colored_img = colormap(norm(data))[:, :, :3]
    return colored_img

def save_16bit_grayscale_tiff(data, save_path):
    data_clipped = np.clip(data, 0, 10)
    data_16 = (data_clipped / 10.0 * 65535.0).astype(np.uint16)
    im = Image.fromarray(data_16, mode='I;16')
    if not (save_path.lower().endswith('.tif') or save_path.lower().endswith('.tiff')):
        save_path += '.tif'
    im.save(save_path, format='TIFF')

def save_colormap_image_with_land_mask(data, land_sea_mask_path, row, col,
                                       save_path, global_min, global_max,
                                       land_color=[0, 0, 0],
                                       recon_file_name=None):

    date_str = None
    if recon_file_name:
        match = re.search(r'(\d{8})', recon_file_name)
        if match:
            date_str = match.group(1)

    land_mask_full = np.load(land_sea_mask_path)
    land_mask_cropped = land_mask_full[row:row + 256, col:col + 256]

    data_clipped = data.copy()
    data_clipped = np.where(data_clipped == 255, np.nan, data_clipped)

    # 이미 고정된 global_min, global_max 사용
    scaled_data = 0.01 + (data_clipped - global_min) * (10 - 0.01) / (global_max - global_min)
    colored_img = convert_raw_to_color(scaled_data, vmin=0.01, vmax=10, cmap_name='jet')

    land_mask = (land_mask_cropped == 1)
    colored_img[land_mask] = land_color

    if not save_path.lower().endswith('.png'):
        save_path_with_extension = save_path.replace('.csv', '')
        if date_str:
            save_path_with_extension += f'_{date_str}.png'
        else:
            save_path_with_extension += '.png'
    else:
        save_path_with_extension = save_path

    # (A) 컬러 PNG(8비트) 저장
    plt.imsave(save_path_with_extension, colored_img)

    # (B) 16비트 TIFF 저장 예시 (주석 해제 시 사용 가능)
    # save_16bit_grayscale_tiff(scaled_data, save_path_with_extension.replace('.png','_16bit.tif'))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(colored_img)
    if recon_file_name:
        ax.set_title(f"Restored Chl-a\n{recon_file_name}\n(r{row}_c{col})", fontsize=16)
    else:
        ax.set_title("Restored Chlorophyll-a Concentration with Land Mask", fontsize=16)
    ax.axis('off')
    sm = cm.ScalarMappable(norm=Normalize(vmin=0.01, vmax=10), cmap='jet')
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Chlorophyll-a concentration (mg/m³)',
                 ticks=np.linspace(0.01, 10, num=5))
    fig.tight_layout()
    fig.savefig(save_path_with_extension.replace('.png', '_bar.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def validate(loss_rate, data_path, save_path, land_sea_mask_path, sample_size=None):
    recon_path = os.path.join(data_path, 'recon')
    gt_path = os.path.join(data_path, 'gt')
    mask_path = os.path.join(data_path, 'mask')
    assert os.path.isdir(recon_path) and os.path.isdir(gt_path) and os.path.isdir(mask_path), \
        "Please check dataset path is valid"

    all_recon_files = sorted(glob.glob(os.path.join(recon_path, '*.csv')), key=natural_sort_key)
    all_gt_files = sorted(glob.glob(os.path.join(gt_path, '*.csv')), key=natural_sort_key)
    all_mask_files = sorted(glob.glob(os.path.join(mask_path, '*.csv')), key=natural_sort_key)

    if len(all_recon_files) == 0 or len(all_gt_files) == 0 or len(all_mask_files) == 0:
        print("No image files found in the specified paths.")
        return

    print("Total gt files:", len(all_gt_files))
    print("Total recon files:", len(all_recon_files))
    print("Total mask files:", len(all_mask_files))

    # -------------------------
    # (직접 지정) 전역 스케일링
    # -------------------------
    global_min = -0.6930203437805176
    global_max = 11.1470947265625
    print(f"[Global Scaling] global_min={global_min}, global_max={global_max}")

    # -------------------------
    # 2) 성능 평가 (전역 스케일링 적용, RMSE/MAE 계산 및 결과 시각화)
    # -------------------------
    if sample_size is not None and sample_size < len(all_recon_files):
        random.seed(42)
        sample_indices = sorted(random.sample(range(len(all_recon_files)), sample_size))
        recon_files_list = [all_recon_files[i] for i in sample_indices]
        gt_files_list = [all_gt_files[i] for i in sample_indices]
        mask_files_list = [all_mask_files[i] for i in sample_indices]
        print(f"Randomly sampled {sample_size} files for evaluation.")
    else:
        recon_files_list = all_recon_files
        gt_files_list = all_gt_files
        mask_files_list = all_mask_files

    color_image_path = os.path.join(save_path, f'color_{loss_rate}')
    os.makedirs(color_image_path, exist_ok=True)

    temp_rmse = 0.0
    temp_mae = 0.0
    cloud_count = 0
    plt_gt = []
    plt_res = []

    for i in trange(len(recon_files_list), desc="Processing files", unit="file",
                    bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {percentage:3.0f}%"):
        recon_file_name = os.path.basename(recon_files_list[i])
        mask_file_name = os.path.basename(mask_files_list[i])
        gt_file_name = os.path.basename(gt_files_list[i])

        restored_np = np.loadtxt(recon_files_list[i], delimiter=',', dtype='float32')
        mask = np.loadtxt(mask_files_list[i], delimiter=',', dtype='float32')
        gt_np = np.loadtxt(gt_files_list[i], delimiter=',', dtype='float32')

        restored_np = np.where(restored_np == 255, np.nan, restored_np)
        gt_np = np.where(gt_np == 255, np.nan, gt_np)
        mask = np.where(mask == 255, np.nan, mask)

        # (직접 지정한) 전역 스케일링 적용
        restored_np = 0.01 + (restored_np - global_min) * (10 - 0.01) / (global_max - global_min)
        gt_np = 0.01 + (gt_np - global_min) * (10 - 0.01) / (global_max - global_min)

        match = re.search(r'r(\d+)_c(\d+)', recon_file_name)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
        else:
            print(f"Filename format does not match the expected row-col pattern: {recon_file_name}")
            continue

        save_colormap_image_with_land_mask(
            restored_np,
            land_sea_mask_path,
            row, col,
            os.path.join(color_image_path, recon_file_name),
            global_min=global_min,
            global_max=global_max,
            recon_file_name=recon_file_name
        )

        valid_mask = (~np.isnan(mask)) & (~np.isnan(gt_np)) & (~np.isnan(restored_np)) & (gt_np != 0)
        if np.any(valid_mask):
            diff = gt_np[valid_mask] - restored_np[valid_mask]
            temp_mae += np.sum(np.abs(diff))
            temp_rmse += np.sum(diff ** 2)
            count_valid = np.sum(valid_mask)
            cloud_count += count_valid

            plt_gt.extend(gt_np[valid_mask].tolist())
            plt_res.extend(restored_np[valid_mask].tolist())

    plt_gt = np.array(plt_gt)
    plt_res = np.array(plt_res)

    if cloud_count == 0:
        print("No valid data found for plotting.")
    else:
        rmse_val = math.sqrt(temp_rmse / cloud_count)
        mae_val = temp_mae / cloud_count

        # 최종 파라티 플롯 생성 (scatter 형태)
        plot_parity(
            filename=save_path,
            loss_rate=loss_rate,
            true=plt_gt,
            pred=plt_res,
            rmse_=rmse_val,
            mae_=mae_val,
            title=f"Loss {loss_rate}-{int(loss_rate)+9}%",
            kind="scatter",  # scatter 시각화로 변경
            scatter_kws={'s': 1, 'alpha': 0.01}
        )
