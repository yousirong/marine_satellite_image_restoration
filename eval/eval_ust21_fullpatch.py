import os
import sys

import re
import numpy as np
import tifffile as tiff
from scipy import io
import torch
from torch.utils.data import Dataset, DataLoader
from model.model import RFRNetModel

class UST21TileDataset(Dataset):
    """
    UST21 daily tiles 폴더에서 *.tiff 파일을 읽어
    (patch_tensor, mask_tensor, filename) 을 반환.
    육지 영역은 mask=0 으로 처리합니다.
    """
    def __init__(self, base_dir, land_mask_mat_path, patch_size=256, transform=None):
        # 1) 전역 Land/Sea 마스크 로드 (MAT 파일: Land=1이면 육지)
        land_mask_raw = io.loadmat(land_mask_mat_path)['Land']
        # 바다=1, 육지=0 으로 뒤집기
        self.sea_mask_global = np.where(land_mask_raw == 0, 1, 0).astype(np.uint8)

        self.patch_size = patch_size
        self.transform = transform

        # ** 날짜 폴더(base_dir) 내 tiff만 순서대로 수집 **
        # ex) base_dir = '/…/UST21_tiles_daily/2020/20201201'
        files = [f for f in os.listdir(base_dir) if f.endswith('.tiff')]
        files.sort()
        self.files = [os.path.join(base_dir, f) for f in files]
        print(f"[Dataset] found {len(self.files)} tiles in {base_dir!r}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = tiff.imread(path).astype(np.float32)

        # 파일명에서 y0, x0 추출
        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(path))
        if not m:
            raise ValueError(f"파일명에서 좌표를 추출할 수 없습니다: {path}")
        y0, x0 = int(m.group(1)), int(m.group(2))

        sea_patch = self.sea_mask_global[
            y0 : y0 + self.patch_size,
            x0 : x0 + self.patch_size
        ].astype(np.float32)

        data_mask = (arr > 0).astype(np.float32)
        mask = data_mask * sea_patch

        # torch tensor 변환: shape=(1,H,W)
        patch_tensor = torch.from_numpy(arr).unsqueeze(0)
        mask_tensor  = torch.from_numpy(mask).unsqueeze(0)

        # → partial conv in_channels=3 에 맞춰 3채널로 복제
        patch_tensor = patch_tensor.repeat(3, 1, 1)
        mask_tensor  = mask_tensor.repeat(3, 1, 1)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)

        return patch_tensor, mask_tensor, path


if __name__ == '__main__':
    # — 모델 초기화 (변경 없음) —
    model = RFRNetModel()
    model.initialize_model(
        path='/home/juneyonglee/MyData/5th_years/ust21_chl_8day/g_1000000.pth',
        train=False, gpu_ids=[0,1]
    )
    model.cuda()

    # — 연도 폴더 지정: 안에 날짜별 서브폴더가 있습니다 —
    year_folder  = '/media/juneyonglee/My Book/Preprocessed/UST21_tiles_daily/2020'
    base_result  = '/media/juneyonglee/My Book/results'
    year         = os.path.basename(year_folder)  # '2020'

    # 각 날짜 폴더(예: '20201201')를 순회
    for date in sorted(os.listdir(year_folder)):
        date_folder = os.path.join(year_folder, date)
        if not os.path.isdir(date_folder):
            continue

        print(f"\n[Main] Processing date: {date}")
        # 이 date_folder 안엔 *.tiff 패치들이 있어야 합니다
        dataset = UST21TileDataset(
            base_dir=date_folder,
            land_mask_mat_path='/home/juneyonglee/Desktop/AY_ust/preprocessing/Land_mask/Land_mask.mat',
            patch_size=256
        )
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        # 결과 저장 경로: /…/results/2020/20201201
        result_save_path = os.path.join(base_result, year, date)
        print(f"[Main] saving results under: {result_save_path}")

        model.test(
            test_loader=loader,
            result_save_path=result_save_path
        )

    print("\n[Main] All dates processed.")

