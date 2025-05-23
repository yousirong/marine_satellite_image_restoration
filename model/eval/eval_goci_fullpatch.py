#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GOCI fullpatch evaluation script for RFRNetModel
"""
import sys, os

# ─── 프로젝트 루트를 sys.path에 추가 ───
# 스크립트 경로: AY_ust/model/eval/eval_goci_fullpatch.py
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 이후 임포트
import re
import numpy as np
import tifffile as tiff
from scipy import io
import torch
from torch.utils.data import Dataset, DataLoader

from model.model import RFRNetModel


class GOCITileDataset(Dataset):
    """
    GOCI_tiles_daily/... 의 모든 *.tiff 파일을 읽어
    (patch_tensor, mask_tensor, filename)을 반환.
    육지(원 mask==999) 영역 제외, 해양(mask==1)에서 arr==0인 부분만 검은색(0), 나머지는 흰색(1)
    """
    def __init__(self, base_dir, land_sea_mask_npy_path, patch_size=256, transform=None):
        original_mask = np.load(land_sea_mask_npy_path)
        # 1=ocean, 999=land → sea_mask_global: 1=ocean, 0=land
        self.sea_mask_global = (original_mask == 1).astype(np.uint8)
        self.patch_size = patch_size
        self.transform = transform

        self.files = []
        for root, _, fnames in os.walk(base_dir):
            for f in fnames:
                if f.lower().endswith('.tiff'):
                    self.files.append(os.path.join(root, f))
        self.files.sort()
        print(f"[Dataset] found {len(self.files)} tiles under {base_dir!r}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr  = tiff.imread(path).astype(np.float32)

        m = re.search(r'y(\d+)_x(\d+)', os.path.basename(path))
        if not m:
            raise ValueError(f"Cannot extract coords from filename: {path}")
        y0, x0 = map(int, m.groups())

        # 1) sea_patch: 1=ocean, 0=land
        sea_patch = self.sea_mask_global[
            y0 : y0 + self.patch_size,
            x0 : x0 + self.patch_size
        ]

        # 2) hole map: ocean & arr==0 → 1, else 0
        hole = ((sea_patch == 1) & (arr == 0)).astype(np.float32)

        # 3) invert → mask: hole=0 (black), others=1 (white)
        mask = 1.0 - hole

        # to torch, shape=(3,H,W)
        patch_tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3,1,1)
        mask_tensor  = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1)

        if self.transform:
            patch_tensor = self.transform(patch_tensor)
        return patch_tensor, mask_tensor, path


if __name__ == '__main__':
    model = RFRNetModel()
    model.initialize_model(
        path='/home/juneyonglee/MyData/5th_years/GOCI_RRS_band2_1day/g_1000000.pth',
        train=False, gpu_ids=[0,1]
    )
    model.cuda()

    year_folder   = '/media/juneyonglee/My Book/Preprocessed/GOCI_tiles_daily/band2/2021'
    base_result   = '/media/juneyonglee/My Book/results/band2'
    year          = os.path.basename(year_folder)

    for date in sorted(os.listdir(year_folder)):
        date_folder = os.path.join(year_folder, date)
        if not os.path.isdir(date_folder): continue

        for time in sorted(os.listdir(date_folder)):
            time_folder = os.path.join(date_folder, time)
            if not os.path.isdir(time_folder): continue
            if time[:2] not in {'00','01','02','03','04','05','06','07'}:
                continue

            print(f"\n[Main] Processing: {date} {time}")
            dataset = GOCITileDataset(
                base_dir=time_folder,
                land_sea_mask_npy_path='/home/juneyonglee/Desktop/AY_ust/preprocessing/is_land_on_GOCI_modified_1_999.npy',
                patch_size=256
            )
            loader = DataLoader(
                dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
            )

            result_save_path = os.path.join(base_result, year, date, time)
            os.makedirs(result_save_path, exist_ok=True)
            print(f"[Main] saving results under: {result_save_path}")

            model.test(
                test_loader=loader,
                result_save_path=result_save_path
            )

    print("\n[Main] All GOCI dates processed.")