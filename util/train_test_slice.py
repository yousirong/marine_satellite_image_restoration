import os
import random
import shutil
from tqdm import tqdm

# 경로 설정
save_base = '/media/juneyonglee/My Book/Preprocessed/UST/Chl-a_8day'
train_base = os.path.join(save_base, 'train')
test_base = os.path.join(save_base, 'test')

# 이동할 비율 설정 (20%)
test_ratio = 0.2

# 각 loss 폴더별로 파일을 확인하고, 20%를 test 폴더로 이동하는 함수
def move_files_to_test(train_base, test_base, test_ratio):
    loss_folders = [f for f in os.listdir(train_base) if os.path.isdir(os.path.join(train_base, f))]

    # 각 loss 폴더에 대해서
    for loss_folder in tqdm(loss_folders, desc="Processing folders"):
        train_folder = os.path.join(train_base, loss_folder)
        test_folder = os.path.join(test_base, loss_folder)
        
        # test 폴더가 없으면 생성
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        # train 폴더 내의 모든 파일을 확인
        train_files = [f for f in os.listdir(train_folder) if f.endswith('.tiff')]
        
        # 파일을 랜덤하게 섞은 후, 20% 선택
        num_files_to_move = int(len(train_files) * test_ratio)
        selected_files = random.sample(train_files, num_files_to_move)

        # 선택된 파일을 test 폴더로 이동
        for file_name in selected_files:
            src_path = os.path.join(train_folder, file_name)
            dest_path = os.path.join(test_folder, file_name)
            shutil.move(src_path, dest_path)
            print(f"Moved {file_name} to test folder.")

# 파일 이동 실행
move_files_to_test(train_base, test_base, test_ratio)
