import os
import cv2
import numpy as np

def delete_black_masks(root_dir):
    """
    root_dir 아래 모든 서브폴더를 순회하면서,
    256x256 크기의 png 이미지를 검사해,
    1) 완전히 0(검정)인 경우
    2) 0 픽셀이 전체의 1% 미만인 경우(즉 거의 전부 255인 경우)
    를 찾아 삭제한다.
    """

    total_pixels = 256 * 256  # 65536
    threshold = int(total_pixels * 0.01)  # 655.36 -> 655

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith('.png'):
                file_path = os.path.join(root, filename)

                # 이미지 로드 (흑백)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                if img.shape != (256, 256):
                    # 크기가 다른 경우는 여기서는 스킵(또는 필요시 삭제)
                    continue

                # 0(검정) 픽셀 카운트
                zero_count = np.count_nonzero(img == 0)

                # 1) 전부 0인가?
                if zero_count == total_pixels:
                    os.remove(file_path)
                    print(f"[DELETE] All-zero mask: {file_path}")
                    continue  # 다음 파일로

                # 2) 0 픽셀이 1% 미만인가?
                if zero_count < threshold:
                    os.remove(file_path)
                    print(f"[DELETE] Zero-pixel < 1%: {file_path}")
                    continue

if __name__ == "__main__":
    root_directory = "/media/juneyonglee/My Book/Preprocessed/UST21/mask"
    delete_black_masks(root_directory)
