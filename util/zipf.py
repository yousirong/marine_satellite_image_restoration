import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm

# 전처리된 데이터 경로
preprocessed_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_new'

# 저장 경로 설정 (원본 파일을 덮어쓰지 않으려면 별도의 경로를 설정)
save_path = '/media/juneyonglee/My Book/Preprocessed/GOCI/L2_Rrs_new_nan_to_zero'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# 파일 크기를 체크하는 함수 (파일이 비어있는지 확인)
def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0

# 디렉토리 내 모든 .tiff 파일 처리
for root, dirs, files in os.walk(preprocessed_path):
    for file in tqdm(files):
        if file.endswith('.tiff'):
            file_path = os.path.join(root, file)

            # 파일이 비어있는지 확인
            if is_file_empty(file_path):
                print(f"Skipping empty file: {file_path}")
                continue

            # tiff 파일 읽기 시도
            try:
                img = tiff.imread(file_path)
            except Exception as e:
                print(f"Failed to open {file_path}: {e}")
                continue

            # NaN을 0으로 변경
            img = np.nan_to_num(img, nan=0)

            # 저장할 디렉토리 생성 (원본 디렉토리 구조 유지)
            relative_dir = os.path.relpath(root, preprocessed_path)
            save_dir = os.path.join(save_path, relative_dir)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            # 파일 저장 (같은 이름으로 저장)
            save_file_path = os.path.join(save_dir, file)
            try:
                tiff.imwrite(save_file_path, img.astype(np.uint16))  # uint8로 저장
                print(f"Processed and saved: {save_file_path}")
            except Exception as e:
                print(f"Failed to save {save_file_path}: {e}")

print("All files processed and saved with NaN replaced by 0.")
