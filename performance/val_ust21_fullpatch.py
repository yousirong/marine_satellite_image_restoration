import os
import glob
import re
import numpy as np
from scipy import io

# 추가: 컬러맵 + 컬러바 변환용
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

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
            scatter_kws = {'s': 1, 'alpha': 0.1}
        ax.scatter(true, pred, **scatter_kws)
    elif "hist2d" in kind:
        if not hist2d_kws:
            hist2d_kws = {'bins': 300, 'cmap': 'Greens', 'vmin': 1}
        ax.hist2d(true, pred, **hist2d_kws)
    elif "kde" in kind:
        import seaborn as sns
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

    # Metrics (실제 계산된 R² 값 사용)
    if metrics:
        from sklearn.metrics import r2_score as r2_
        r2 = r2_(true, pred)  # Actual R² calculated from the data
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
    # plt.show()
    return ax

def convert_raw_to_color(data, vmin=0.01, vmax=10, cmap_name='jet'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap_name)
    return colormap(norm(data))[:, :, :3]  # RGB

def load_full_image_from_patches(patch_dir, patch_size=256):
    """
    패치 디렉토리에서 전체 이미지 복원 (val_goci_fullpatch.py와 유사)
    마스크의 경우 PNG 파일을 우선적으로 사용하고, 없으면 CSV 사용
    """
    # 마스크인지 확인
    is_mask = 'mask' in patch_dir.lower()

    if is_mask:
        # 마스크의 경우 PNG 파일을 먼저 찾아보기
        png_files = sorted(glob.glob(os.path.join(patch_dir, '*.png')), key=natural_sort_key)

        # 만약 PNG가 없으면 상위 디렉토리의 mask 폴더에서 찾기
        if not png_files:
            parent_dir = os.path.dirname(os.path.dirname(patch_dir))  # degree의 상위
            alt_mask_dir = os.path.join(parent_dir, 'mask')
            if os.path.exists(alt_mask_dir):
                png_files = sorted(glob.glob(os.path.join(alt_mask_dir, '*.png')), key=natural_sort_key)
                print(f"[INFO] Using PNG files from alternative mask directory: {alt_mask_dir}")

        if png_files:
            print(f"[INFO] Loading mask from PNG files: {len(png_files)} files found")
            return load_full_image_from_png_patches(png_files, patch_size)
        else:
            print(f"[INFO] No PNG files found, falling back to CSV files")

    # CSV 파일 사용 (기존 방식)
    files = sorted(glob.glob(os.path.join(patch_dir, '*.csv')), key=natural_sort_key)
    if not files:
        print(f"[WARN] No CSV found in {patch_dir}. Returning empty image.")
        return None

    # 첫 번째 파일로 전체 크기 추정
    sample_file = files[0]
    m = re.search(r'y(\d+)_x(\d+)', os.path.basename(sample_file))
    if not m:
        return None

    # 전체 크기 계산 (모든 패치 좌표를 확인)
    max_row, max_col = 0, 0
    for fpath in files:
        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if m:
            row, col = map(int, m.groups())
            max_row = max(max_row, row + patch_size)
            max_col = max(max_col, col + patch_size)

    full_img = np.zeros((max_row, max_col), dtype=np.float32)

    for fpath in files:
        try:
            arr = np.loadtxt(fpath, delimiter=',', dtype=np.float32)
            if arr.shape != (patch_size, patch_size):
                print(f"[SKIP] Invalid shape {arr.shape} in {fpath}")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to load {fpath}: {e}")
            continue

        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if not m:
            continue
        row, col = map(int, m.groups())
        full_img[row:row+patch_size, col:col+patch_size] = arr

    return full_img

def load_full_image_from_png_patches(png_files, patch_size=256):
    """
    PNG 패치들로부터 전체 이미지 복원
    """
    from PIL import Image

    if not png_files:
        return None

    # 첫 번째 파일로 전체 크기 추정
    sample_file = png_files[0]
    m = re.search(r'y(\d+)_x(\d+)', os.path.basename(sample_file))
    if not m:
        return None

    # 전체 크기 계산
    max_row, max_col = 0, 0
    for fpath in png_files:
        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if m:
            row, col = map(int, m.groups())
            max_row = max(max_row, row + patch_size)
            max_col = max(max_col, col + patch_size)

    full_img = np.zeros((max_row, max_col), dtype=np.float32)

    for fpath in png_files:
        try:
            # PNG 이미지 로드
            img = Image.open(fpath)
            img_array = np.array(img)

            # 그레이스케일로 변환
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array

            # 0-255 범위를 0-1 범위로 정규화
            normalized = gray / 255.0

            if normalized.shape != (patch_size, patch_size):
                print(f"[SKIP] Invalid PNG shape {normalized.shape} in {fpath}")
                continue

        except Exception as e:
            print(f"[ERROR] Failed to load PNG {fpath}: {e}")
            continue

        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(fpath))
        if not m:
            continue
        row, col = map(int, m.groups())
        full_img[row:row+patch_size, col:col+patch_size] = normalized

    return full_img

def save_image_with_details(full_img, out_dir, file_prefix, label_text, land_mask_mat_path, cmap_name='jet'):
    """
    val_goci_fullpatch.py 스타일의 이미지 저장 함수
    """
    # 육지 마스크 로드
    land_mask_raw = io.loadmat(land_mask_mat_path)['Land']
    land_bool = (land_mask_raw == 1)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f'{file_prefix}.png')
    out_bar = os.path.join(out_dir, f'{file_prefix}_bar.png')

    if 'mask' in file_prefix:
        # 마스크의 경우 3가지 구분: 육지(회색), 해양 마스크(검은색), 유효 해양(흰색)
        unique_vals = np.unique(full_img)
        print(f"[{file_prefix.upper()}] Mask unique values: {unique_vals}")

        # 3가지 카테고리로 컬러 이미지 생성 (초기값을 흰색으로 설정)
        colored_mask = np.ones((full_img.shape[0], full_img.shape[1], 3), dtype=np.float32)

        # 육지 영역 확인 (land_bool이 True인 영역)
        land_pixels = np.sum(land_bool)
        ocean_pixels = np.sum(~land_bool)
        print(f"[{file_prefix.upper()}] Land pixels: {land_pixels}, Ocean pixels: {ocean_pixels}")

        # 해양 영역에서 마스크 상태 확인
        ocean_mask_pixels = (~land_bool) & (full_img == 0)  # 해양 영역의 마스크(0값)
        ocean_valid_pixels = (~land_bool) & (full_img == 1)  # 해양 영역의 유효값(1값)

        print(f"[{file_prefix.upper()}] Ocean masked pixels: {np.sum(ocean_mask_pixels)}")
        print(f"[{file_prefix.upper()}] Ocean valid pixels: {np.sum(ocean_valid_pixels)}")

        # 마스크 데이터의 실제 값 분포 확인
        ocean_area = full_img[~land_bool]
        if ocean_area.size > 0:
            ocean_unique = np.unique(ocean_area)
            print(f"[{file_prefix.upper()}] Ocean area unique values: {ocean_unique}")
            for val in ocean_unique:
                count = np.sum(ocean_area == val)
                percentage = (count / ocean_area.size) * 100
                print(f"  Value {val}: {count} pixels ({percentage:.1f}%)")

        # 색상 할당 (뚜렷한 구분을 위해):
        # 먼저 모든 해양 영역을 유효 해양(흰색)으로 설정
        colored_mask[~land_bool] = [1.0, 1.0, 1.0]  # 유효 해양 = 흰색

        # 육지 = 회색 (0.5, 0.5, 0.5)
        colored_mask[land_bool] = [0.5, 0.5, 0.5]

        # 해양 마스크 = 빨간색 (1, 0, 0) - 더 뚜렷하게 보이도록
        colored_mask[ocean_mask_pixels] = [1.0, 0.0, 0.0]

        # PNG 저장
        plt.imsave(out_png, colored_mask, origin='upper')
        print(f"[{file_prefix.upper()}] Saved colored mask PNG: {out_png}")

        # 컬러바 포함 버전 (범례 설명과 함께)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(colored_mask, origin='upper')
        ax.axis('off')
        ax.set_title('Ocean Mask Visualization', fontsize=16)

        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.5, 0.5, 0.5], label='Land (Gray)'),
            Patch(facecolor=[1.0, 0.0, 0.0], label='Ocean Mask (Red)'),
            Patch(facecolor=[1.0, 1.0, 1.0], label='Valid Ocean (White)', edgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        fig.tight_layout()
        fig.savefig(out_bar, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[{file_prefix.upper()}] Saved colored mask PNG with legend: {out_bar}")

    elif 'gt' in file_prefix:
        # GT의 경우 chlorophyll 범위로 처리
        clipped = np.clip(full_img, 0.01, 10)
        colored = convert_raw_to_color(clipped, vmin=0.01, vmax=10, cmap_name=cmap_name)

        # 육지 마스크 크기 조정
        if colored.shape[:2] != land_bool.shape:
            min_h = min(colored.shape[0], land_bool.shape[0])
            min_w = min(colored.shape[1], land_bool.shape[1])
            colored[:min_h, :min_w][land_bool[:min_h, :min_w]] = [0.0, 0.0, 0.0]
        else:
            colored[land_bool] = [0.0, 0.0, 0.0]

        plt.imsave(out_png, colored, origin='upper')
        print(f"[{file_prefix.upper()}] Saved GT PNG: {out_png}")

        # 컬러바 포함 버전
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(colored, origin='upper')
        ax.axis('off')
        ax.set_title(f'GT Chlorophyll-a', fontsize=16)

        sm = cm.ScalarMappable(norm=Normalize(vmin=0.01, vmax=10), cmap=cmap_name)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(label_text, fontsize=12)

        fig.tight_layout()
        fig.savefig(out_bar, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[{file_prefix.upper()}] Saved GT PNG with colorbar: {out_bar}")

def process_all_data_types(date_result, land_mask_mat_path, out_dir):
    """
    recon, gt, mask 모든 데이터 타입을 처리하는 통합 함수
    """
    print(f"\n=== Processing All Data Types ===\nInput: {date_result}\nOutput: {out_dir}")

    for data_type in ['recon', 'gt', 'mask']:
        print(f"\n--- Processing {data_type.upper()} Image ---")
        data_dir = os.path.join(date_result, data_type)

        if not os.path.isdir(data_dir):
            print(f"Directory not found: {data_dir}. Skipping.")
            continue

        # 패치들로부터 전체 이미지 로드
        full_img = load_full_image_from_patches(data_dir, patch_size=256)
        if full_img is None:
            print(f"No valid patches found in {data_dir}. Skipping.")
            continue

        print(f"Loaded {data_type} image: {full_img.shape}")

        # 데이터 타입별 저장
        if data_type == 'recon':
            # Recon: PNG with colorbar only
            png_path = os.path.join(out_dir, 'full_recon.png')

            os.makedirs(out_dir, exist_ok=True)

            # PNG with colorbar using existing function logic
            land_mask_raw = io.loadmat(land_mask_mat_path)['Land']
            land_bool = (land_mask_raw == 1)

            clipped = np.clip(full_img, 0.01, 10)
            colored = convert_raw_to_color(clipped, vmin=0.01, vmax=10, cmap_name='jet')

            # 크기 조정
            if colored.shape[:2] != land_bool.shape:
                min_h = min(colored.shape[0], land_bool.shape[0])
                min_w = min(colored.shape[1], land_bool.shape[1])
                colored[:min_h, :min_w][land_bool[:min_h, :min_w]] = [0.0, 0.0, 0.0]
            else:
                colored[land_bool] = [0.0, 0.0, 0.0]

            plt.imsave(png_path, colored, origin='upper')
            print(f"Saved PNG: {png_path}")

            # Colorbar version
            bar_path = png_path.replace('.png', '_bar.png')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(colored, origin='upper')
            ax.axis('off')
            ax.set_title('Reconstructed Chlorophyll-a', fontsize=16)

            sm = cm.ScalarMappable(norm=Normalize(vmin=0.01, vmax=10), cmap='jet')
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label('Chlorophyll-a (mg/m³)', fontsize=12)

            fig.tight_layout()
            fig.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved PNG with colorbar: {bar_path}")

        elif data_type == 'gt':
            save_image_with_details(
                full_img, out_dir, 'gt',
                'GT Chlorophyll-a (mg/m³)',
                land_mask_mat_path, 'jet'
            )

        elif data_type == 'mask':
            save_image_with_details(
                full_img, out_dir, 'mask',
                'Mask',
                land_mask_mat_path, 'gray'
            )

def filter_top_percent_data(gt_data, recon_data, top_percent=0.95):
    """
    상위 X% 성능 데이터만 필터링하는 함수 (절댓값 오차 기준)
    """
    if len(gt_data) == 0:
        return np.array([]), np.array([])

    # 절댓값 오차 계산
    abs_errors = np.abs(np.array(recon_data) - np.array(gt_data))

    # 상위 X% (오차가 작은 순서로)
    num_top_percent = int(len(abs_errors) * top_percent)
    top_indices = np.argsort(abs_errors)[:num_top_percent]

    filtered_gt = np.array(gt_data)[top_indices]
    filtered_recon = np.array(recon_data)[top_indices]

    print(f"  Filtered top {top_percent*100:.0f}%: {len(filtered_gt):,} / {len(gt_data):,} points")
    print(f"  Error range: {abs_errors[top_indices].min():.6f} - {abs_errors[top_indices].max():.6f}")

    return filtered_gt, filtered_recon

def filter_top_95_percent_data(gt_data, recon_data):
    """
    상위 95% 성능 데이터만 필터링하는 함수 (절댓값 오차 기준)
    """
    return filter_top_percent_data(gt_data, recon_data, 0.95)

def filter_top_99_percent_data(gt_data, recon_data):
    """
    상위 99% 성능 데이터만 필터링하는 함수 (절댓값 오차 기준)
    """
    return filter_top_percent_data(gt_data, recon_data, 0.99)

def validate_with_scatter_plots(date_result_path, out_dir, land_mask_path, sample_ratio=0.2):
    """
    GT와 Recon 데이터를 비교하여 scatter plot을 생성하는 검증 함수 (UST21용, 전체 + 시간패턴별)
    """
    print(f"\n=== Creating Scatter Plot Validation ===")
    print(f"Input path: {date_result_path}")
    print(f"Output path: {out_dir}")

    if not os.path.exists(date_result_path):
        print(f"❌ Input path does not exist: {date_result_path}")
        return

    # GT, Recon 디렉토리 확인
    gt_dir = os.path.join(date_result_path, 'gt')
    recon_dir = os.path.join(date_result_path, 'recon')

    if not (os.path.exists(gt_dir) and os.path.exists(recon_dir)):
        print(f"❌ Missing gt or recon directory")
        return

    # 육지 마스크 로드
    try:
        from scipy import io
        land_mask_raw = io.loadmat(land_mask_path)['Land']
        land_bool = (land_mask_raw == 1)  # 1=육지
    except Exception as e:
        print(f"❌ Error loading land mask: {e}")
        return

    # CSV 파일 리스트 가져오기
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.csv')), key=natural_sort_key)
    recon_files = sorted(glob.glob(os.path.join(recon_dir, '*.csv')), key=natural_sort_key)

    if len(gt_files) == 0 or len(recon_files) == 0:
        print(f"❌ No CSV files found: GT={len(gt_files)}, Recon={len(recon_files)}")
        return

    if len(gt_files) != len(recon_files):
        print(f"⚠️ File count mismatch: GT={len(gt_files)}, Recon={len(recon_files)}")
        min_files = min(len(gt_files), len(recon_files))
        gt_files = gt_files[:min_files]
        recon_files = recon_files[:min_files]

    print(f"Found {len(gt_files)} file pairs")

    # 샘플링
    sample_size = max(1, int(len(gt_files) * sample_ratio))
    sample_indices = np.random.choice(len(gt_files), size=sample_size, replace=False)

    print(f"Sampling {sample_size}/{len(gt_files)} files")

    all_gt_data = []
    all_recon_data = []
    time_data = {}  # 시간 패턴별 데이터 저장

    for idx in sample_indices:
        gt_file = gt_files[idx]
        recon_file = recon_files[idx]

        try:
            # 파일에서 좌표 및 시간 패턴 추출
            filename = os.path.basename(gt_file)

            # 좌표 추출
            coord_match = re.search(r'y(\d+)_x(\d+)', filename)
            if not coord_match:
                continue
            row, col = int(coord_match.group(1)), int(coord_match.group(2))

            # 시간 패턴 추출 (예: img_1_, img_2_ 등의 패턴)
            time_pattern = "unknown"
            time_match = re.search(r'img_(\d+)_', filename)
            if time_match:
                time_pattern = f"img_{time_match.group(1)}"
            else:
                # 다른 시간 패턴들도 시도
                date_match = re.search(r'(\d{8})', filename)
                if date_match:
                    time_pattern = f"date_{date_match.group(1)}"

            # 데이터 로드
            gt_data = np.loadtxt(gt_file, delimiter=',', dtype=np.float32)
            recon_data = np.loadtxt(recon_file, delimiter=',', dtype=np.float32)

            if gt_data.shape != recon_data.shape or gt_data.shape != (256, 256):
                continue

            # 육지 마스크 적용
            try:
                patch_land_mask = land_bool[row:row+256, col:col+256]
                ocean_mask = ~patch_land_mask  # 해양 영역 (육지가 아닌 영역)

                # 해양 영역의 비율 확인
                ocean_ratio = np.sum(ocean_mask) / (256 * 256)
                if ocean_ratio < 0.3:  # 해양 영역이 30% 미만이면 건너뛰기
                    continue

            except IndexError:
                continue

            # 유효한 해양 데이터만 추출
            gt_ocean = gt_data[ocean_mask]
            recon_ocean = recon_data[ocean_mask]

            # 특수값 제거 및 유효 범위 확인 (클로로필 농도: 0.01-10 mg/m³)
            valid_mask = (gt_ocean != 255) & (recon_ocean != 255) & \
                       (~np.isnan(gt_ocean)) & (~np.isnan(recon_ocean)) & \
                       (~np.isinf(gt_ocean)) & (~np.isinf(recon_ocean)) & \
                       (gt_ocean >= 0.01) & (gt_ocean <= 10) & \
                       (recon_ocean >= 0.01) & (recon_ocean <= 10)

            if np.sum(valid_mask) > 10:  # 최소 10개 이상의 유효한 픽셀
                valid_gt = gt_ocean[valid_mask].tolist()
                valid_recon = recon_ocean[valid_mask].tolist()

                # 전체 데이터에 추가
                all_gt_data.extend(valid_gt)
                all_recon_data.extend(valid_recon)

                # 시간 패턴별 데이터에 추가
                if time_pattern not in time_data:
                    time_data[time_pattern] = {'gt': [], 'recon': []}
                time_data[time_pattern]['gt'].extend(valid_gt)
                time_data[time_pattern]['recon'].extend(valid_recon)

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

    if len(all_gt_data) == 0:
        print("❌ No valid data collected for scatter plot")
        return

    print(f"✅ Collected {len(all_gt_data):,} valid data points from ocean areas")
    print(f"Found {len(time_data)} time patterns: {list(time_data.keys())}")

    # 원본 데이터를 그대로 사용
    gt_array = np.array(all_gt_data)
    recon_array = np.array(all_recon_data)

    # 시간 패턴별 데이터도 원본 그대로 사용
    filtered_time_data = {}
    for time_pattern, data in time_data.items():
        if len(data['gt']) > 0:
            filtered_time_data[time_pattern] = {
                'gt': np.array(data['gt']),
                'recon': np.array(data['recon'])
            }

    time_data = filtered_time_data

    # 전체 지표 계산 (원본 데이터로 실제 계산)
    diff = recon_array - gt_array
    rmse = np.sqrt(np.mean(diff ** 2))  # Actual RMSE from all data
    mae = np.mean(np.abs(diff))         # Actual MAE from all data

    # 전체 데이터 범위 계산
    vmin = min(np.min(gt_array), np.min(recon_array))
    vmax = max(np.max(gt_array), np.max(recon_array))

    print(f"Overall data range: GT=[{np.min(gt_array):.3f}, {np.max(gt_array):.3f}], Recon=[{np.min(recon_array):.3f}, {np.max(recon_array):.3f}]")
    print(f"Overall RMSE: {rmse:.6f}")
    print(f"Overall MAE: {mae:.6f}")

    # 출력 디렉토리 생성
    os.makedirs(out_dir, exist_ok=True)

    # 1. 전체 데이터 scatter plot 생성 (원본 데이터 그대로 사용)
    try:
        plot_parity(
            filename=out_dir,
            loss_rate="chlorophyll_validation_overall",
            true=gt_array,  # 원본 데이터 그대로
            pred=recon_array,  # 원본 데이터 그대로
            rmse_=rmse,  # 원본 데이터의 실제 RMSE
            mae_=mae,    # 원본 데이터의 실제 MAE
            xlabel="Ground Truth Chl-a (mg/m³)",
            ylabel="Reconstructed Chl-a (mg/m³)",
            title="UST21 Chlorophyll-a Validation",
            kind="scatter",
            scatter_kws={'s': 2, 'alpha': 0.5}
        )
        print(f"✅ Overall scatter plot saved to: {out_dir}")

    except Exception as e:
        print(f"❌ Error creating overall scatter plot: {e}")

    # 2. 시간 패턴별 scatter plot 생성
    print(f"\n=== Creating Time-pattern based Scatter Plots ===")
    for time_pattern, data in time_data.items():
        try:
            pattern_gt = data['gt']
            pattern_recon = data['recon']

            if len(pattern_gt) < 100:  # 최소 100개 이상의 데이터 포인트 필요
                print(f"⚠️ Skipping {time_pattern}: insufficient data ({len(pattern_gt)} points)")
                continue

            # 시간 패턴별 지표 계산 (원본 데이터로 실제 계산)
            pattern_diff = pattern_recon - pattern_gt
            pattern_rmse = np.sqrt(np.mean(pattern_diff ** 2))  # Actual RMSE from all data
            pattern_mae = np.mean(np.abs(pattern_diff))         # Actual MAE from all data

            plot_parity(
                filename=out_dir,
                loss_rate=f"chlorophyll_validation_{time_pattern}",
                true=pattern_gt,     # 원본 데이터 그대로
                pred=pattern_recon,  # 원본 데이터 그대로
                rmse_=pattern_rmse,     # 원본 데이터의 실제 RMSE
                mae_=pattern_mae,       # 원본 데이터의 실제 MAE
                xlabel="Ground Truth Chl-a (mg/m³)",
                ylabel="Reconstructed Chl-a (mg/m³)",
                title=f"UST21 Chlorophyll-a Validation ({time_pattern})",
                kind="scatter",
                scatter_kws={'s': 2, 'alpha': 0.5}
            )
            print(f"✅ Time pattern {time_pattern} scatter plot saved ({len(pattern_gt):,} points)")

        except Exception as e:
            print(f"❌ Error creating scatter plot for time pattern {time_pattern}: {e}")

    print(f"✅ Scatter plot validation completed for {len(time_data)} time patterns")

    # Calculate R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(gt_array, recon_array)

    # Return metrics
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def process_multiple_dates(base_results_dir, base_performance_dir, land_mask_path, target_dates):
    """
    여러 날짜를 일괄 처리하는 함수

    Args:
        base_results_dir: 결과 기본 경로 (예: '/home/juneyonglee/Desktop/AY_ust/My_Book/results/2020')
        base_performance_dir: 성능 저장 기본 경로 (예: '/home/juneyonglee/Desktop/AY_ust/My_Book/performance/2020')
        land_mask_path: 육지 마스크 파일 경로
        target_dates: 처리할 날짜 리스트 (예: ['20201201', '20201202'])
    """
    print(f"=== Processing Multiple Dates ===")
    print(f"Base results dir: {base_results_dir}")
    print(f"Base performance dir: {base_performance_dir}")
    print(f"Target dates: {target_dates}")

    success_count = 0
    failed_dates = []
    all_metrics = []  # 모든 날짜의 메트릭 저장

    for date_str in target_dates:
        print(f"\n{'='*60}")
        print(f"Processing Date: {date_str}")
        print(f"{'='*60}")

        try:
            # 입력 및 출력 경로 구성
            date_result_path = os.path.join(base_results_dir, date_str, 'degree')
            date_output_path = os.path.join(base_performance_dir, date_str)

            print(f"Input path: {date_result_path}")
            print(f"Output path: {date_output_path}")

            # 경로 존재 확인
            if not os.path.exists(date_result_path):
                print(f"❌ Input path does not exist: {date_result_path}")
                failed_dates.append((date_str, "Input path not found"))
                continue

            # 필요한 하위 디렉토리 확인
            required_dirs = ['recon', 'gt', 'mask']
            missing_dirs = []
            for req_dir in required_dirs:
                dir_path = os.path.join(date_result_path, req_dir)
                if not os.path.exists(dir_path):
                    missing_dirs.append(req_dir)

            if missing_dirs:
                print(f"⚠️  Missing directories: {missing_dirs}")
                print(f"Available directories: {os.listdir(date_result_path) if os.path.exists(date_result_path) else 'None'}")

            # 처리 실행
            process_all_data_types(date_result_path, land_mask_path, date_output_path)

            # Scatter plot 검증 추가
            scatter_output_path = os.path.join(date_output_path, 'scatter_plots')
            date_metrics = validate_with_scatter_plots(date_result_path, scatter_output_path, land_mask_path, sample_ratio=0.2)

            # 메트릭 저장
            if date_metrics is not None:
                all_metrics.append({
                    'date': date_str,
                    'rmse': date_metrics['rmse'],
                    'mae': date_metrics['mae'],
                    'r2': date_metrics['r2']
                })

            success_count += 1
            print(f"✅ Successfully processed {date_str}")

        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")
            failed_dates.append((date_str, str(e)))
            continue

    # 최종 결과 요약
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total dates requested: {len(target_dates)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_dates)}")

    if failed_dates:
        print(f"\nFailed dates:")
        for date_str, reason in failed_dates:
            print(f"  - {date_str}: {reason}")

    if success_count > 0:
        print(f"\nResults saved to: {base_performance_dir}")

    # 전체 평균 메트릭 계산 및 저장
    if len(all_metrics) > 0:
        print(f"\n{'='*60}")
        print(f"OVERALL AVERAGE METRICS ACROSS ALL DATES")
        print(f"{'='*60}")

        avg_rmse = np.mean([m['rmse'] for m in all_metrics])
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_r2 = np.mean([m['r2'] for m in all_metrics])

        print(f"Average RMSE: {avg_rmse:.8f}")
        print(f"Average MAE: {avg_mae:.8f}")
        print(f"Average R²: {avg_r2:.6f}")

        # 메트릭을 CSV 파일로 저장
        metrics_file = os.path.join(base_performance_dir, 'overall_metrics_summary.csv')
        os.makedirs(base_performance_dir, exist_ok=True)

        with open(metrics_file, 'w') as f:
            f.write("Date,RMSE,MAE,R2\n")
            for m in all_metrics:
                f.write(f"{m['date']},{m['rmse']:.8f},{m['mae']:.8f},{m['r2']:.6f}\n")
            f.write(f"\nAverage,{avg_rmse:.8f},{avg_mae:.8f},{avg_r2:.6f}\n")

        print(f"✅ Overall metrics saved to: {metrics_file}")

def generate_date_range(start_date: str, end_date: str):
    """
    시작 날짜부터 종료 날짜까지의 모든 날짜를 생성하는 함수

    Args:
        start_date: 시작 날짜 (YYYYMMDD 형식)
        end_date: 종료 날짜 (YYYYMMDD 형식)

    Returns:
        생성된 날짜 리스트
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    return dates

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process UST21 validation for multiple dates')
    parser.add_argument('--base_results_dir', type=str,
                       default='/home/juneyonglee/Desktop/AY_ust/myhdd/UST21/results/2020',
                       help='Base results directory')
    parser.add_argument('--base_performance_dir', type=str,
                       default='/home/juneyonglee/Desktop/AY_ust/myhdd/UST21/performance/2020',
                       help='Base performance output directory')
    parser.add_argument('--land_mask_path', type=str,
                       default='/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat',
                       help='Land mask file path')
    parser.add_argument('--dates', type=str, nargs='+',
                       help='Individual dates to process (YYYYMMDD format)')
    parser.add_argument('--start_date', type=str,
                       help='Start date for range processing (YYYYMMDD format)')
    parser.add_argument('--end_date', type=str,
                       help='End date for range processing (YYYYMMDD format)')

    args = parser.parse_args()

    # 날짜 리스트 결정
    if args.start_date and args.end_date:
        # 날짜 범위로 처리
        print(f"Generating dates from {args.start_date} to {args.end_date}...")
        target_dates = generate_date_range(args.start_date, args.end_date)
        print(f"Generated {len(target_dates)} dates")
    elif args.dates:
        # 특정 날짜들로 처리
        target_dates = args.dates
        print(f"Processing specific dates: {target_dates}")
    else:
        # 기본값 사용
        target_dates = ['20201201', '20201208','20201215','20201222','20201229','20201231']
        print(f"Using default dates: {target_dates}")

    # 여러 날짜 일괄 처리
    process_multiple_dates(args.base_results_dir, args.base_performance_dir,
                          args.land_mask_path, target_dates)