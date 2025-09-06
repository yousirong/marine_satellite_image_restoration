import os
import glob
import re
import numpy as np
import tifffile as tiff
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ================== 설정 ==================
# 처리할 날짜 리스트
DATE_LIST = ['20210101', '20210108', '20210115', '20210122', '20210129'] 

# 기본 경로 설정
BASE_RESULTS_DIR = '/home/juneyonglee/myhdd/GOCI_RRS/results/band2/2021'
BASE_PERFORMANCE_DIR = '/home/juneyonglee/myhdd/GOCI_RRS/performance/band2/2021'

# 고정 설정
LAND_MASK_NPY = '/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy'
PATCH_SIZE = 256
CMAP_NAME = 'jet'
# ========================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]

def convert_raw_to_color(data, vmin, vmax, cmap_name='jet'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap_name)
    return colormap(norm(data))[:, :, :3]  # RGB

def load_full_image_from_patches(patch_dir):
    land_mask = np.load(LAND_MASK_NPY)
    H, W = land_mask.shape
    full_img = np.zeros((H, W), dtype=np.float32)
    csv_files = sorted(glob.glob(os.path.join(patch_dir, '*.csv')), key=natural_sort_key)

    if not csv_files:
        print(f"[WARN] No CSV found in {patch_dir}. Returning empty image.")
        return None

    for fpath in csv_files:
        try:
            arr = np.loadtxt(fpath, delimiter=',', dtype=np.float32)
            if arr.shape != (PATCH_SIZE, PATCH_SIZE):
                print(f"[SKIP] Invalid shape {arr.shape} in {fpath}")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to load {fpath}: {e}")
            continue

        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if not m:
            continue
        y0, x0 = map(int, m.groups())
        full_img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = arr

    return full_img

def save_image_with_details(full_img, out_dir, file_prefix, label_text, cmap_name='jet'):
    land_mask = np.load(LAND_MASK_NPY)  # 999: land
    land_bool = (land_mask == 999)

    os.makedirs(out_dir, exist_ok=True)
    out_tiff = os.path.join(out_dir, f'{file_prefix}.tiff')
    out_png  = os.path.join(out_dir, f'{file_prefix}.png')
    out_bar  = os.path.join(out_dir, f'{file_prefix}_bar.png')

    tiff.imwrite(out_tiff, full_img)
    print(f"[{file_prefix.upper()}] TIFF saved → {out_tiff}")

    if 'mask' in file_prefix:
        vmin, vmax = 0, 1
        print(f"[{file_prefix.upper()}] Using fixed color range [0, 1] for mask.")
    else:
        valid_data = full_img[(full_img != -999) & (full_img != 0)]
        if valid_data.size == 0:
            print(f"[{file_prefix.upper()}] WARN: No valid non-zero data to determine color range. Using default [0, 1].")
            vmin, vmax = 0, 1
        else:
            vmin = np.percentile(valid_data, 1)
            vmax = np.percentile(valid_data, 99)

    clipped = np.clip(full_img, vmin, vmax)
    colored = convert_raw_to_color(clipped, vmin=vmin, vmax=vmax, cmap_name=cmap_name)
    colored[land_bool] = [0.0, 0.0, 0.0]

    plt.imsave(out_png, colored, origin='upper')
    print(f"[{file_prefix.upper()}] Saved colored PNG: {out_png}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(colored, origin='upper')
    ax.axis('off')

    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_name)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(label_text, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_bar, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[{file_prefix.upper()}] Saved colored PNG with colorbar: {out_bar}")

def process_and_average(image_list):
    if not image_list:
        return None
    stacked_images = np.stack(image_list, axis=0)
    masked_stacked_images = np.ma.masked_where(stacked_images == -999, stacked_images)
    averaged_img = np.ma.mean(masked_stacked_images, axis=0).data
    averaged_img[averaged_img == masked_stacked_images.fill_value] = -999
    averaged_img[averaged_img == -999] = 0 # Convert no data to 0 for visualization
    return averaged_img

def process_date(base_date_path, out_root_dir):
    """Main processing logic for a single date."""
    all_recon_images = []
    all_gt_images = []
    all_mask_images = []

    if not os.path.isdir(base_date_path):
        print(f"\n[ERROR] Base date path not found: {base_date_path}. Skipping.")
        return

    time_subdirs = [d for d in os.listdir(base_date_path) if os.path.isdir(os.path.join(base_date_path, d))]

    for time_subdir in sorted(time_subdirs):
        degree_path = os.path.join(base_date_path, time_subdir, 'degree')
        print(f"\n[Main] Processing time subdir: {time_subdir}")

        for data_type, image_list in [('recon', all_recon_images), ('gt', all_gt_images), ('mask', all_mask_images)]:
            patch_dir = os.path.join(degree_path, data_type)
            if os.path.isdir(patch_dir):
                print(f"  - Loading {data_type} data from: {patch_dir}")
                full_img = load_full_image_from_patches(patch_dir)
                if full_img is not None:
                    image_list.append(full_img)
            else:
                print(f"  - '{data_type}' directory not found in {degree_path}. Skipping.")

    # Averaging and Saving
    if all_recon_images:
        print("\n[Main] Averaging and saving RECON image...")
        averaged_recon = process_and_average(all_recon_images)
        if averaged_recon is not None:
            save_image_with_details(averaged_recon, out_root_dir, 'recon_avg', 'Recon Chlorophyll-a (mg/m³)')
    else:
        print("\n[Main] No RECON images to process.")

    if all_gt_images:
        print("\n[Main] Averaging and saving GT image...")
        averaged_gt = process_and_average(all_gt_images)
        if averaged_gt is not None:
            save_image_with_details(averaged_gt, out_root_dir, 'gt_avg', 'GT Chlorophyll-a (mg/m³)')
    else:
        print("\n[Main] No GT images to process.")

    if all_mask_images:
        print("\n[Main] Averaging and saving MASK image...")
        averaged_mask = process_and_average(all_mask_images)
        if averaged_mask is not None:
            save_image_with_details(averaged_mask, out_root_dir, 'mask_avg', 'Mask', cmap_name='gray')
    else:
        print("\n[Main] No MASK images to process.")

# 메인 루프
if __name__ == '__main__':
    for date_str in DATE_LIST:
        print(f"======================================================")
        print(f"========== PROCESSING DATE: {date_str} ==========")
        print(f"======================================================")
        
        base_path = os.path.join(BASE_RESULTS_DIR, date_str)
        out_path = os.path.join(BASE_PERFORMANCE_DIR, date_str)
        
        process_date(base_path, out_path)

    print("\n[Main] All dates processed.")