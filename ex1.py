import os
import shutil

# 검색할 디렉토리와 저장할 디렉토리
search_dir = "/media/juneyonglee/My Book/results/50"
save_dir = "/home/juneyonglee/Documents/5years/50"
target_keyword = "89_20201202_r3785_c3615"

# 저장할 디렉토리가 없으면 생성
os.makedirs(save_dir, exist_ok=True)

# 디렉토리 순회하며 파일 검색 및 복사 (degree 폴더 제외)
for root, dirs, files in os.walk(search_dir):
    # 'degree' 폴더 제외
    dirs[:] = [d for d in dirs if d != "degree"]

    for file in files:
        if target_keyword in file:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(save_dir, file)
            shutil.copy2(src_path, dst_path)
            print(f"복사 완료: {src_path} -> {dst_path}")
