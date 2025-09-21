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

def convert_raw_to_color(data, vmin=0.01, vmax=10, cmap_name='jet'):
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap_name)
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

if __name__ == '__main__':
    # 설정 변수들
    base_results_dir = '/home/juneyonglee/Desktop/AY_ust/My_Book/results/2020'
    base_performance_dir = '/home/juneyonglee/Desktop/AY_ust/My_Book/performance/2020'
    land_mask_path = '/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat'

    # 처리할 날짜 리스트 (원하는 날짜들을 여기에 추가)
    target_dates = ['20201201', '20201208','20201215','20201222','20201229']

    # 여러 날짜 일괄 처리
    process_multiple_dates(base_results_dir, base_performance_dir, land_mask_path, target_dates)

    # 단일 날짜 처리 예시 (주석 처리됨)
    # single_date_result = '/home/juneyonglee/Desktop/AY_ust/My_Book/results/2020/20201229/degree'
    # single_date_output = '/home/juneyonglee/Desktop/AY_ust/My_Book/performance/2020/20201229'
    # process_all_data_types(single_date_result, land_mask_path, single_date_output)