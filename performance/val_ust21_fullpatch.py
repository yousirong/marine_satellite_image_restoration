import os
import glob
import re
import numpy as np
import tifffile as tiff
from scipy import io

# 추가: 컬러맵 + 컬러바 변환용
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

def convert_raw_to_color(data, vmin=0.01, vmax=10, cmap_name='jet'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap_name)
    return colormap(norm(data))[:, :, :3]  # RGB

def reconstruct_full_image_from_patches(data_path,
                                        land_mask_mat_path,
                                        patch_size=256,
                                        save_tiff_path=None,
                                        save_png_path=None):
    # 1) 전체 해상도 & 육지 마스크 로드
    land_mask_raw = io.loadmat(land_mask_mat_path)['Land']  # Land==1 이면 육지
    H, W = land_mask_raw.shape
    land_bool = (land_mask_raw == 1)

    # 2) 빈 배열 준비
    full_img = np.zeros((H, W), dtype=np.float32)

    # 3) patch CSV 리스트
    recon_dir = os.path.join(data_path, 'recon')
    files = sorted(glob.glob(os.path.join(recon_dir, '*.csv')), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No CSV patches in {recon_dir!r}")

    # 4) 패치 합치기
    for fpath in files:
        arr = np.loadtxt(fpath, delimiter=',', dtype=np.float32)
        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if not m:
            continue
        row, col = map(int, m.groups())
        full_img[row:row+patch_size, col:col+patch_size] = arr

    # 5) TIFF 저장
    if save_tiff_path:
        os.makedirs(os.path.dirname(save_tiff_path), exist_ok=True)
        tiff.imwrite(save_tiff_path, full_img.astype(np.float32))
        print(f"Saved full TIFF: {save_tiff_path}")

# … (위 생략) …

    # 6) PNG 저장 (컬러맵만)
    if save_png_path:
        os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
        clipped = np.clip(full_img, 0.01, 10)

        # (B) jet 컬러맵 적용 → RGB 이미지
        colored = convert_raw_to_color(clipped, vmin=0.01, vmax=10, cmap_name='jet')
        colored[land_bool] = [0.0, 0.0, 0.0]
        plt.imsave(save_png_path, colored, origin='upper')
        print(f"Saved colored PNG: {save_png_path}")

        # 7) PNG + 컬러바 저장 (origin='upper', jet)
        bar_path = save_png_path.replace('.png', '_bar.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        # 이미지 표시 (이미 RGB이므로 cmap은 따로 지정하지 않음)
        ax.imshow(colored, origin='upper')
        ax.axis('off')

        # jet 컬러바를 위한 ScalarMappable 생성
        sm = cm.ScalarMappable(
            norm=Normalize(vmin=0.01, vmax=10),
            cmap='jet'
        )
        sm.set_array([])

        # 컬러바 그리기
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label('Chlorophyll-a (mg/m³)', fontsize=12)

        fig.tight_layout()
        fig.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved colored PNG with colorbar: {bar_path}")

    return full_img

if __name__ == '__main__':
    date_result = '/media/juneyonglee/My Book/results/2020/20201229/degree'
    land_mask   = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'
    out_dir     = '/media/juneyonglee/My Book/performance/2020/20201229'

    # Output paths for recon
    out_tiff    = os.path.join(out_dir, 'full_recon.tiff')
    out_png     = os.path.join(out_dir, 'full_recon.png')

    # --- Recon Image Reconstruction ---
    print("--- Processing Recon Image ---")
    reconstruct_full_image_from_patches(
        data_path=date_result,
        land_mask_mat_path=land_mask,
        patch_size=256,
        save_tiff_path=out_tiff,
        save_png_path=out_png
    )

    # --- GT and Mask Image Reconstruction ---
    # Re-use the function's logic for GT and Mask
    land_mask_raw = io.loadmat(land_mask)['Land']
    H, W = land_mask_raw.shape

    for data_type in ['gt', 'mask']:
        print(f"--- Processing {data_type.upper()} Image ---")
        data_dir = os.path.join(date_result, data_type)
        if not os.path.isdir(data_dir):
            print(f"Directory not found: {data_dir}. Skipping.")
            continue

        full_img = np.zeros((H, W), dtype=np.float32)
        files = sorted(glob.glob(os.path.join(data_dir, '*.csv')), key=natural_sort_key)
        if not files:
            print(f"No CSV patches in {data_dir}. Skipping.")
            continue

        for fpath in files:
            arr = np.loadtxt(fpath, delimiter=',', dtype=np.float32)
            m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
            if not m:
                continue
            row, col = map(int, m.groups())
            full_img[row:row+256, col:col+256] = arr

        # Save as PNG
        output_png_path = os.path.join(out_dir, f'{data_type}.png')
        cmap_choice = 'jet' if data_type == 'gt' else 'gray'
        plt.imsave(output_png_path, full_img, cmap=cmap_choice, origin='upper')
        print(f"Saved {data_type.upper()} PNG: {output_png_path}")
