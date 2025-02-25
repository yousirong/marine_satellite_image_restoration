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
land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.npy'
# goci
# land_sea_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_parity(filename, loss_rate, true, pred, rmse_, mae_, kind="scatter",
                xlabel="true (mg/m$^3$)", ylabel="predict (mg/m$^3$)", title="Loss 50-60%",
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
            hist2d_kws = {'cmap': 'Greens', 'vmin': 1}
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
        rmse = rmse_
        mae = mae_
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

        ax.text(text_pos_x, text_pos_y, f"RMSE = {rmse:.8f}",
                transform=ax.transAxes, fontdict=font_metrics, ha=ha)
        ax.text(text_pos_x, text_pos_y - 0.1, f"MAE = {mae:.8f}",
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


# def normalize_data(data, vmin=0.01, vmax=10):
#     """
#     Normalize data between vmin and vmax, but exclude 255 values (land) from normalization.
#     """
#     # Create a copy to avoid modifying the original data
#     data_normalized = data.copy()

#     # Mask for land values (255)
#     land_mask = (data_normalized == 255)

#     # Set land values to NaN so they won't affect normalization
#     data_normalized[land_mask] = np.nan

#     # Clip the data to the range [vmin, vmax] and normalize
#     data_normalized = np.clip(data_normalized, vmin, vmax)

#     # Replace NaN values with the original 255 (land)
#     data_normalized[land_mask] = np.nan

#     return data_normalized

# def normalize_data(data):
#     data_normalized = data.copy()
#     land_mask = (data_normalized == 255)
#     data_normalized[land_mask] = np.nan
#     vmin = np.nanmin(data_normalized)
#     vmax = np.nanmax(data_normalized)
#     if vmax > 10:
#         vmin, vmax = 0.01, 10
#     data_normalized = (data_normalized - vmin) / (vmax - vmin)
#     data_normalized[land_mask] = np.nan
#     return data_normalized, vmin, vmax

def save_land_mask_image(land_mask_cropped, save_path):
    """
    Save the cropped land-sea mask as an image, where land is black (255) and sea is white (0).
    """
    # Create a binary mask image (land as black, sea as white)
    mask_img = land_mask_cropped
    # 생략

def convert_raw_to_color(data, vmin=0.01, vmax=10, cmap_name='jet'):
    """
    주어진 2D raw pixel data (값의 범위: vmin ~ vmax)를 cmap (기본 'jet')을 사용하여
    컬러 이미지 (RGB, 0~1 범위)로 변환합니다.
    """
    norm = Normalize(vmin=vmin, vmax=vmax)
    # plt.get_cmap 사용하여 deprecation warning 해결
    colormap = plt.get_cmap(cmap_name)
    colored_img = colormap(norm(data))[:, :, :3]
    return colored_img

def save_colormap_image_with_land_mask(data, land_sea_mask_path, row, col, save_path, vmin=0.01, vmax=10, land_color=[0, 0, 0], recon_file_name=None):
    """
    CSV로 불러온 복원 데이터(data)를 선형 스케일 조정한 후 jet colormap을 적용하고,
    육지 영역(land mask에 따라 1인 부분)은 지정된 색상(기본 검정)으로 표현하여 이미지를 저장합니다.
    파일 이름(recon_file_name)에서 일부 정보를 제목에 포함시켜 raw col 값이 보이도록 합니다.
    """
    # 파일이름에서 날짜 등 정보 추출 (예: 20201201)
    date_str = None
    if recon_file_name:
        match = re.search(r'(\d{8})', recon_file_name)
        if match:
            date_str = match.group(1)

    # Land mask 파일 로드
    land_mask_full = np.load(land_sea_mask_path)
    # (GOCI 데이터인 경우 필요시 아래와 같이 반전:
    #  land_mask_full = np.where(land_mask_full == 0, 1, 0))

    # 256x256 crop으로 자르기
    land_mask_cropped = land_mask_full[row:row + 256, col:col + 256]

    # 원본 CSV 데이터 복사 및 255인 부분은 NaN 처리 (육지 영역)
    data_clipped = data.copy()
    data_clipped = np.where(data_clipped == 255, np.nan, data_clipped)

    # 선형 스케일 조정: 파일별 최소/최대값을 이용해 [0.01, 10] 범위로 매핑
    dmin, dmax = np.nanmin(data_clipped), np.nanmax(data_clipped)
    data_scaled = 0.01 + (data_clipped - dmin) * (10 - 0.01) / (dmax - dmin)

    # jet colormap 적용하여 컬러 이미지 생성 (범위: 0.01 ~ 10)
    colored_img = convert_raw_to_color(data_scaled, vmin=0.01, vmax=10, cmap_name='jet')

    # 육지 영역: cropped mask에서 1인 부분은 지정된 land_color(기본 검정)으로 설정
    land_mask = (land_mask_cropped == 1)
    colored_img[land_mask] = land_color

    # 저장 경로 처리
    if not save_path.lower().endswith('.png'):
        save_path_with_extension = save_path.replace('.csv', '')
        if date_str:
            save_path_with_extension += f'_{date_str}.png'
        else:
            save_path_with_extension += '.png'
    else:
        save_path_with_extension = save_path

    # 컬러 이미지 저장
    plt.imsave(save_path_with_extension, colored_img)

    # Axes 객체를 이용해 이미지와 colorbar 출력
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(colored_img)
    # 파일이름 정보를 제목에 포함 (row, col 값도 함께 표시)
    if recon_file_name:
        ax.set_title(f"Restored Chl-a\n{recon_file_name}\n(r{row}_c{col})", fontsize=16)
    else:
        ax.set_title("Restored Chlorophyll-a Concentration with Land Mask", fontsize=16)
    ax.axis('off')
    sm = cm.ScalarMappable(norm=Normalize(vmin=0.01, vmax=10), cmap='jet')
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Chlorophyll-a concentration (mg/m³)', ticks=np.linspace(0.01, 10, num=5))
    fig.tight_layout()
    fig.savefig(save_path_with_extension.replace('.png', '_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def validate(loss_rate, data_path, save_path, land_sea_mask_path, reliability_threshold=0.9):
    recon_path = os.path.join(data_path, 'recon')
    gt_path = os.path.join(data_path, 'gt')
    mask_path = os.path.join(data_path, 'mask')
    assert os.path.isdir(recon_path) and os.path.isdir(gt_path) and os.path.isdir(mask_path), "Please check dataset path is valid"

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
    temp_mae = 0
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

            # 255인 부분은 NaN 처리
            restored_np = np.where(restored_np == 255, np.nan, restored_np)
            gt_np = np.where(gt_np == 255, np.nan, gt_np)
            mask = np.where(mask == 255, np.nan, mask)

            # 선형 스케일 조정: 각 데이터의 최소/최대를 사용해 [0.01, 10] 범위로 매핑
            rmin, rmax = np.nanmin(restored_np), np.nanmax(restored_np)
            if rmax - rmin == 0:
                restored_np = np.full_like(restored_np, 0.01)
            else:
                restored_np = 0.01 + (restored_np - rmin) * (10 - 0.01) / (rmax - rmin)

            gmin, gmax = np.nanmin(gt_np), np.nanmax(gt_np)
            if gmax - gmin == 0:
                gt_np = np.full_like(gt_np, 0.01)
            else:
                gt_np = 0.01 + (gt_np - gmin) * (10 - 0.01) / (gmax - gmin)

            # 파일 이름에서 row, col 값 추출
            match = re.search(r'r(\d+)_c(\d+)', recon_file_name)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
            else:
                print(f"Filename format does not match the expected row-col pattern for {recon_file_name}")
                continue

            # CSV 데이터 기반 컬러 변환 이미지를 육지 마스크와 함께 저장
            save_colormap_image_with_land_mask(restored_np, land_sea_mask_path, row, col,
                                               os.path.join(color_image_path, recon_file_name))

            # RMSE 및 MAE 계산
            W, H = gt_np.shape
            for w in range(W):
                for h in range(H):
                    if np.isnan(mask[w, h]) or np.isnan(gt_np[w, h]) or np.isnan(restored_np[w, h]):
                        continue
                    elif gt_np[w, h] == 0:
                        continue

                    relative_error = abs(gt_np[w, h] - restored_np[w, h]) / gt_np[w, h]
                    reliability = 1 - relative_error
                    if reliability < reliability_threshold:
                        continue

                    plt_gt.append(gt_np[w, h])
                    plt_res.append(restored_np[w, h])
                    temp_mae += abs(gt_np[w, h] - restored_np[w, h])
                    temp_rmse += (gt_np[w, h] - restored_np[w, h]) ** 2
                    cloud_count += 1

    plt_gt = np.array(plt_gt)
    plt_res = np.array(plt_res)

    if cloud_count == 0:
        print("No valid data found for plotting.")
        return

    # 정규화하지 않고, 원본 [0.01, 10] 스케일 데이터를 그대로 사용
    plot_parity(filename=save_path,
                loss_rate=loss_rate,
                true=plt_gt,
                pred=plt_res,
                rmse_=math.sqrt(temp_rmse / cloud_count) if cloud_count > 0 else float('nan'),
                mae_=temp_mae / cloud_count if cloud_count > 0 else float('nan'),
                title=f"Loss {loss_rate}-{int(loss_rate)+9}%"
    )
