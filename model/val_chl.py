import os
import glob
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from tqdm import trange
from sklearn.metrics import r2_score as r2_
from matplotlib import cm
from matplotlib.colors import Normalize
import re

def natural_sort_key(s):
    # 파일 이름에서 숫자 부분만 추출하여 정렬 키로 사용
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def plot_parity(filename, loss_rate, true, pred, rmse_, mape_, kind="scatter", 
                xlabel="true (mg/m$^3$)", ylabel="predict (mg/m$^3$)", title="Loss 50-60%", 
                hist2d_kws=None, scatter_kws=None, kde_kws=None,
                equal=True, metrics=True, metrics_position="lower right",
                figsize=(8, 8), ax=None, save_file=True):
    
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    # data range
    val_min = min(true.min(), pred.min())
    val_max = max(true.max(), pred.max())

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

    # x, y bounds
    max_bounds = [val_min, val_max]
    ax.set_xlim(max_bounds)
    ax.set_ylim(max_bounds)

    # x, y ticks, ticklabels
    y = ax.get_yticks()
    ticks = [round(y[i], 3) for i in range(len(y)) if not y[i] < 0 and i % 2 == 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=15)

    # grid
    ax.grid(True)

    # 기준선
    ax.plot(max_bounds, max_bounds, c="k", alpha=0.3)

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
            text_pos_x, text_pos_y = 0.1  # text_position
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

def save_colormap_image(data, save_path):
    # Normalize the data to be between 0 and 20
    # norm = Normalize(vmin=0, vmax=20)
    norm = Normalize()

    # Use the viridis colormap to map the grayscale image to a color image
    colormap = cm.ScalarMappable(norm=norm, cmap='viridis')
    colored_img = colormap.to_rgba(data)[:, :, :3]  # Use only RGB values (omit A value)

    # Ensure the save path has the correct extension
    save_path_with_extension = save_path if save_path.lower().endswith('.png') else save_path + '.png'
    plt.imsave(save_path_with_extension, colored_img)

    # Display the image with the updated color bar
    # plt.imshow(data, cmap='viridis', norm=Normalize(vmin=0, vmax=20))
    plt.imshow(data, cmap='viridis', norm=Normalize())
    plt.colorbar(label='Chlorophyll-a concentration (mg/m³)')
    plt.title(f'Restored Chlorophyll-a Concentration')

    # Remove the axis ticks and labels
    plt.xticks([])
    plt.yticks([])
    
    # Save the image with the color bar
    plt.savefig(save_path_with_extension.replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()


def validate(loss_rate, data_path, save_path):

    ####path
    recon_path = os.path.join(data_path, 'recon')
    gt_path = os.path.join(data_path, 'gt')
    mask_path = os.path.join(data_path, 'mask')
    assert os.path.isdir(recon_path) and os.path.isdir(gt_path) and os.path.isdir(mask_path), "Please check dataset path is valid"
    
    # color_images 저장 경로 설정
    color_image_path = os.path.join(save_path, f'color_{loss_rate}')
    if not os.path.exists(color_image_path):
        os.makedirs(color_image_path)

    ####files list
    recon_files_list = sorted(glob.glob(os.path.join(recon_path, '*')), key=natural_sort_key)
    gt_files_list = sorted(glob.glob(os.path.join(gt_path, '*')), key=natural_sort_key)
    mask_files_list = sorted(glob.glob(os.path.join(mask_path, '*')), key=natural_sort_key)
    
    # 이미지 파일이 없을 경우 처리
    if len(recon_files_list) == 0 or len(gt_files_list) == 0 or len(mask_files_list) == 0:
        print("No image files found in the specified paths.")
        return
    
    print("len(gt_files_list):", len(gt_files_list))
    print("len(recon_files_list):", len(recon_files_list))
    print("len(mask_files_list):", len(mask_files_list))

    ####variables
    temp_rmse = 0
    temp_mape = 0
    cloud_count = 0
    outlier = 0
    plt_gt = []
    plt_res = []

    # [0,1] 픽셀만 성능 검증
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i in trange(len(recon_files_list)):
            recon_file_name = os.path.basename(recon_files_list[i])
            mask_file_name = os.path.basename(mask_files_list[i])
            gt_file_name = os.path.basename(gt_files_list[i])

            restored_np = np.loadtxt(recon_files_list[i], delimiter=',', dtype='float32')
            #정규화 하려면 restored_np /=255.0 gt_np /= 255.0
            mask = np.loadtxt(mask_files_list[i], delimiter=',', dtype='float32')
            gt_np = np.loadtxt(gt_files_list[i], delimiter=',', dtype='float32')
            W, H = gt_np.shape

            for w in range(W):
                for h in range(H):
                    if mask[w, h] == 0 and gt_np[w, h] != 0 and restored_np[w, h] != 0:
                        if gt_np[w, h] < 0 or restored_np[w, h] < 0 or restored_np[w, h] > 255 or gt_np[w, h] > 255:
                            outlier += 1
                            continue
                        else:
                            try:
                                loss = (gt_np[w, h] - restored_np[w, h]) / gt_np[w, h]
                                plt_gt.append(gt_np[w, h])
                                plt_res.append(restored_np[w, h])

                            except Warning:
                                print(recon_file_name)
                                print(gt_np[w, h])
                                print(restored_np[w, h])
                                outlier += 1
                                continue

                        if math.isinf(loss) or math.isnan(gt_np[w, h]) or math.isnan(restored_np[w, h]): 
                            continue

                        temp_mape += abs(gt_np[w, h] - restored_np[w, h])
                        temp_rmse += (gt_np[w, h] - restored_np[w, h]) ** 2
                        cloud_count += 1

            # 컬러 이미지로 변환하여 저장 (color_images 폴더 내에 저장)
            save_colormap_image(restored_np, os.path.join(color_image_path, recon_file_name))

    plt_gt = np.array(plt_gt)
    plt_res = np.array(plt_res)

    # 값이 없을 경우 대비 처리
    if cloud_count == 0:
        print("No valid data found for plotting.")
        return

    plot_parity(filename=save_path,
                loss_rate=loss_rate,
                true=plt_gt,
                pred=plt_res,
                rmse_=math.sqrt(temp_rmse / cloud_count) if cloud_count > 0 else float('nan'),
                mape_=temp_mape / cloud_count if cloud_count > 0 else float('nan'),
                title=f"Loss {loss_rate}-{int(loss_rate)+9}%"
    )