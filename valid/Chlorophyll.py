import os
import numpy as np
import glob
from tqdm import trange

# 현재 사용 중인 경로 설정
rrs_path_1 = 'model/results/GOCI_RRS_band2_perfect/550000/10/'
rrs_path_2 = 'model/results/GOCI_RRS_band3_perfect/550000/10'  # 2번째 밴드 경로
rrs_path_3 = 'model/results/GOCI_RRS_band4_perfect/550000/10'  # 3번째 밴드 경로
save_path = 'model/results/GOCI_chl/550000/10'

# 결과 저장 폴더 생성
if not os.path.isdir(save_path):
    os.makedirs(save_path)
else:
    print("Folder already exists")

# RRS 파일 리스트 가져오기
rrs1_files_list = glob.glob(os.path.join(rrs_path_1, '*'), recursive=True) if os.path.exists(rrs_path_1) else []
rrs2_files_list = glob.glob(os.path.join(rrs_path_2, '*'), recursive=True) if os.path.exists(rrs_path_2) else []
rrs3_files_list = glob.glob(os.path.join(rrs_path_3, '*'), recursive=True) if os.path.exists(rrs_path_3) else []

# 클로로필 계산을 위한 계수
a = [0.2515, -2.3798, 1.5823, -0.6372, -0.5692]

# 각 경로 존재 여부 출력
if not rrs1_files_list:
    print(f"{rrs_path_1} 경로가 존재하지 않거나 파일이 없습니다. 이 경로를 건너뜁니다.")
if not rrs2_files_list:
    print(f"{rrs_path_2} 경로가 존재하지 않거나 파일이 없습니다. 이 경로를 건너뜁니다.")
if not rrs3_files_list:
    print(f"{rrs_path_3} 경로가 존재하지 않거나 파일이 없습니다. 이 경로를 건너뜁니다.")

# 모든 경로가 비어있으면 종료
if not rrs1_files_list or not rrs2_files_list or not rrs3_files_list:
    print("처리할 파일이 없습니다. 프로그램을 종료합니다.")
else:
    # 파일 순회하며 처리
    for i in trange(len(rrs1_files_list)):
        if i >= len(rrs2_files_list) or i >= len(rrs3_files_list):
            print(f"인덱스 {i}에 해당하는 RRS2 또는 RRS3 파일이 없습니다. 건너뜁니다.")
            continue

        img = []
        f_name = os.path.basename(rrs1_files_list[i])

        # RRS 데이터를 읽어서 0-1 사이 값으로 변환 (각 밴드 처리)
        rrs1 = np.loadtxt(rrs1_files_list[i], delimiter=',', dtype='float32') / 255
        rrs2 = np.loadtxt(rrs2_files_list[i], delimiter=',', dtype='float32') / 255
        rrs3 = np.loadtxt(rrs3_files_list[i], delimiter=',', dtype='float32') / 255

        # 세 개의 밴드를 img에 추가
        img.append(rrs1)
        img.append(rrs2)
        img.append(rrs3)

        # R_rs 3차원 배열로 쌓기
        R_rs = np.stack(img, axis=0)  # (3, height, width)
        _, height, width = R_rs.shape

        # 클로로필 값을 저장할 배열 생성
        Chl_oc3 = np.empty((height, width))

        # 클로로필 값을 계산하여 저장
        for h in range(height):
            for w in range(width):
                # 육지 영역을 마스크로 제외한 후 클로로필 계산
                if R_rs[2, h, w] <= 0 or R_rs[0, h, w] <= 0 or R_rs[1, h, w] <= 0:
                    Chl_oc3[h, w] = 0
                else:
                    term = np.sum(a[i] * (np.log10(np.max(R_rs[:2, h, w]) / R_rs[2, h, w])) ** i for i in range(1, 5))
                    Chl_oc3[h, w] = 10 ** (a[0] + term)

        # 계산된 클로로필 값을 콤마로 구분하여 저장
        np.savetxt(os.path.join(save_path, f_name), Chl_oc3, delimiter=',')
        print(f"Chlorophyll map saved at {os.path.join(save_path, f_name)}")
