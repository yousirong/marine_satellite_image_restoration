# import os
# import numpy as np
# import tifffile as tiff
# from concurrent.futures import ProcessPoolExecutor, as_completed

# def check_and_delete(fpath, invalid_value, threshold):
#     """
#     단일 파일을 읽어서 invalid_value 또는 NaN 비율이 threshold 초과 시 삭제.
#     반환: (fpath, deleted(bool), ratio)
#     """
#     try:
#         img = tiff.imread(fpath).astype(np.float32)
#     except Exception as e:
#         print(f"[ERROR] {fpath} 읽기 실패: {e}")
#         return fpath, False, None

#     total = img.size
#     count_invalid = np.count_nonzero(img == invalid_value)
#     count_nan     = np.count_nonzero(np.isnan(img))
#     ratio = (count_invalid + count_nan) / total

#     if ratio > threshold:
#         try:
#             os.remove(fpath)
#             return fpath, True, ratio
#         except Exception as e:
#             print(f"[ERROR] {fpath} 삭제 실패: {e}")
#             return fpath, False, ratio

#     return fpath, False, ratio

# def clean_patches(root_dir, invalid_value=-999, threshold=0.6, num_workers=32):
#     """
#     root_dir 아래 모든 .tif/.tiff 파일을 멀티프로세싱으로 검사·삭제합니다.
#     """
#     # 1) 삭제 대상 파일 목록 수집
#     tiff_files = []
#     for dirpath, _, filenames in os.walk(root_dir):
#         for fname in filenames:
#             if fname.lower().endswith(('.tif', '.tiff')):
#                 tiff_files.append(os.path.join(dirpath, fname))

#     # 2) ProcessPoolExecutor로 병렬 처리
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         future_to_path = {
#             executor.submit(check_and_delete, fpath, invalid_value, threshold): fpath
#             for fpath in tiff_files
#         }
#         for future in as_completed(future_to_path):
#             fpath = future_to_path[future]
#             try:
#                 _, deleted, ratio = future.result()
#                 if deleted:
#                     print(f"[DELETED] {fpath} ({ratio:.2%} invalid+NaN)")
#             except Exception as e:
#                 print(f"[ERROR] {fpath} 처리 중 예외 발생: {e}")

# if __name__ == "__main__":
#     root = "/media/juneyonglee/My Book1/Preprocessed/GOCI_RRS"
#     clean_patches(root_dir=root, invalid_value=-999, threshold=0.6)
import os
import cv2
import numpy as np

def clean_small_masks(root_dir, threshold=0.1, patch_size=(256, 256)):
    """
    root_dir 하위의 모든 .png 파일을 검사해서,
    검은색(0) 픽셀이 전체 픽셀의 threshold 미만이면 파일을 삭제합니다.

    Args:
        root_dir (str): 마스크들이 저장된 최상위 디렉터리
        threshold (float): 최소 검은색 비율 (예: 0.1 == 10%)
        patch_size (tuple): 기대하는 패치 크기 (가로, 세로)
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.png'):
                continue

            fpath = os.path.join(dirpath, fname)
            mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] 읽기 실패: {fpath}")
                continue

            # 크기 확인 (원한다면 이 줄을 제외하고 모든 크기에 대해 실행할 수 있습니다)
            if mask.shape != patch_size:
                print(f"[SKIP] 크기 불일치 {mask.shape}: {fpath}")
                continue

            total = mask.size
            black = int((mask == 0).sum())
            ratio = black / total

            if ratio < threshold:
                try:
                    os.remove(fpath)
                    print(f"[DELETED] {fpath} — 검은색 비율 {ratio:.2%}")
                except Exception as e:
                    print(f"[ERROR] 삭제 실패 {fpath}: {e}")

if __name__ == "__main__":
    root_directory = "/media/juneyonglee/My Book1/Preprocessed/GOCI_RRS/mask"
    clean_small_masks(root_directory, threshold=0.1, patch_size=(256,256))
