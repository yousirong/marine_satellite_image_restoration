#!/usr/bin/env python3
import os
import cv2
import numpy as np

def is_invalid_data(arr):
    """모든 값이 NaN/Inf인 경우 True"""
    return not np.isfinite(arr).any()

def is_invalid_mask(mask):
    """1) 전부 0 or 2) 0 픽셀이 전체의 1% 미만이면 True"""
    if mask.shape != (256, 256):
        return False  # 크기 불일치시 판단하지 않음

    total_pixels = 256 * 256
    threshold = int(total_pixels * 0.01)
    zero_count = np.count_nonzero(mask == 0)

    if zero_count == total_pixels:
        return True  # 전부 0
    if zero_count < threshold:
        return True  # 거의 255

    return False

def clean_dataset_recursive(data_root, mask_root):
    deleted = 0
    valid_exts = ('.png', '.tif', '.tiff', '.npy', '.jpg')

    for dirpath, _, filenames in os.walk(data_root):
        for fname in filenames:
            if not fname.lower().endswith(valid_exts):
                continue

            data_path = os.path.join(dirpath, fname)
            # data_root를 기준으로 한 상대경로
            rel_path = os.path.relpath(data_path, data_root)
            mask_path = os.path.join(mask_root, rel_path)

            # mask 파일이 없으면 건너뛰기
            if not os.path.exists(mask_path):
                print(f"[SKIP] mask not found: {rel_path}")
                continue

            # 로드
            try:
                if fname.lower().endswith(".npy"):
                    data = np.load(data_path)
                    mask = np.load(mask_path)
                else:
                    data = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if data is None or mask is None:
                    raise ValueError("read returned None")
            except Exception as e:
                print(f"[ERROR] Load failed ({rel_path}): {e}")
                continue

            # 삭제 조건 검사
            if is_invalid_data(data) or is_invalid_data(mask) or is_invalid_mask(mask):
                try:
                    os.remove(data_path)
                    os.remove(mask_path)
                    print(f"[DELETE] {rel_path}")
                    deleted += 1
                except Exception as e:
                    print(f"[ERROR] Delete failed ({rel_path}): {e}")

    print(f"총 삭제된 파일 쌍: {deleted}")

if __name__ == "__main__":
    data_root = "/media/juneyonglee/My Book/Preprocessed/GOCI_RRS/band4/train"
    mask_root = "/media/juneyonglee/My Book/Preprocessed/GOCI_RRS/mask/band4/train"
    clean_dataset_recursive(data_root, mask_root)
