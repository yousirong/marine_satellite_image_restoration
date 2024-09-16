# 8day ust21 실제 좌표값으로 변경
import os
import re

# 낙동강과 새만금의 기본 좌표
x_nak, y_nak = (3377, 3664)
x_sae, y_sae = (3751, 4629)

# 파일 경로 설정
save_base = '/media/juneyonglee/My Book/Preprocessed/UST/Chl-a_8day/train'

# 낙동강과 새만금 패치 파일 패턴 (r{row}_c{col} 형식)
file_pattern = re.compile(r'_r(\d+)_c(\d+)\.tiff$')

# 디렉토리 내 파일 탐색
for root, dirs, files in os.walk(save_base):
    for file in files:
        if file.endswith(".tiff"):
            # 파일명에서 r{row}_c{col} 추출
            match = file_pattern.search(file)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                
                # 낙동강 파일 이름인지 새만금 파일 이름인지 확인
                if 'nak' in file:
                    actual_row = x_nak + row
                    actual_col = y_nak + col
                    new_filename = file.replace(f'_r{row}_c{col}', f'_r{actual_row}_c{actual_col}')
                elif 'sae' in file:
                    actual_row = x_sae + row
                    actual_col = y_sae + col
                    new_filename = file.replace(f'_r{row}_c{col}', f'_r{actual_row}_c{actual_col}')
                else:
                    continue  # 'nak' 또는 'sae'가 없으면 건너뜀

                # 기존 파일 경로
                old_file_path = os.path.join(root, file)
                # 새로운 파일 경로
                new_file_path = os.path.join(root, new_filename)
                
                # 파일 이름 변경
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {file} -> {new_filename}")
