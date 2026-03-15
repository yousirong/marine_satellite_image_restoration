# -*- coding: utf-8 -*-
"""
UST21 Full Patch Dataset
========================

UST21 (Chlorophyll-a) 데이터를 위한 데이터셋 클래스
"""

import os
import re
import numpy as np
import tifffile as tiff
import cv2
from typing import Dict, Any, Optional

from testbed.core.base_dataset import BaseFullPatchDataset
from testbed.utils.mask_utils import (
    load_ust21_mask,
    extract_patch_coords,
    get_mask_patch,
    expand_to_3channels,
    calculate_ocean_stats
)


class UST21FullPatchDataset(BaseFullPatchDataset):
    """
    UST21 Full Patch 데이터셋

    UST21 Chlorophyll-a 데이터를 처리합니다.

    특징:
    - 파일명 패턴: y{:04d}_x{:04d}.tiff
    - Land-Sea 마스크: .mat 형식 (Land=1=육지)
    - 유효 범위: 0.01 ~ 10 mg/m³
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 데이터셋 설정
                - image_dir: 이미지 디렉터리
                - land_mask_path: Land-Sea 마스크 경로 (.mat)
                - patch_size: 패치 크기 (기본: 256)
                - valid_range: 유효 범위 (기본: [0.01, 10])
                - min_ocean_ratio: 최소 해양 비율 (기본: 0.3)
        """
        # 설정값 추출
        self.valid_range = config.get('valid_range', [0.01, 10])
        self.min_ocean_ratio = config.get('min_ocean_ratio', 0.3)

        super().__init__(config)

    def _load_land_sea_mask(self):
        """UST21 Land-Sea 마스크 로드"""
        if not self.land_mask_path or not os.path.exists(self.land_mask_path):
            print(f"[Warning] Land mask not found: {self.land_mask_path}")
            return

        self.land_sea_mask = load_ust21_mask(self.land_mask_path)
        print(f"[UST21] Land-sea mask shape: {self.land_sea_mask.shape}")

    def load_image(self, index: int) -> np.ndarray:
        """
        이미지 로드

        Args:
            index: 파일 인덱스

        Returns:
            np.ndarray: 로드된 이미지 (H, W)
        """
        file_path = self.files[index]
        image = tiff.imread(file_path).astype(np.float32)
        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        UST21 데이터 전처리

        1. 채널 처리 (3채널로 확장)
        2. 크기 조정
        3. 유효 범위 외 값 처리
        4. NaN 처리

        Args:
            image: 입력 이미지

        Returns:
            np.ndarray: 전처리된 이미지 (3, H, W)
        """
        # 채널 처리
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

        # 크기 조정
        if image.shape[0] != self.patch_size or image.shape[1] != self.patch_size:
            image = cv2.resize(image, (self.patch_size, self.patch_size))

        # 유효 범위 외 값 처리
        # 음수 또는 0 이하 값 → 0
        image[image <= 0] = 0
        # 상한 초과 값 → 0
        image[image > self.valid_range[1]] = 0

        # NaN 처리
        image[np.isnan(image)] = 0

        # (H, W, C) → (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return image

    def create_mask(self, image: np.ndarray, index: int) -> np.ndarray:
        """
        결측치 마스크 생성

        해양 영역에서 유효한 데이터가 있는 부분만 1로 설정합니다.

        Args:
            image: 전처리된 이미지 (C, H, W)
            index: 파일 인덱스

        Returns:
            np.ndarray: 마스크 (3, H, W), 1=유효, 0=결측
        """
        filename = os.path.basename(self.files[index])

        # 좌표 추출
        y0, x0 = extract_patch_coords(filename, pattern='y_x')
        if y0 is None or x0 is None:
            print(f"[Warning] Failed to extract coords from {filename}")
            return np.ones((3, self.patch_size, self.patch_size), dtype=np.float32)

        # Land-Sea 마스크 패치 추출
        try:
            sea_mask_patch = get_mask_patch(
                self.land_sea_mask, y0, x0, self.patch_size
            )
        except ValueError as e:
            print(f"[Warning] {e}")
            return np.ones((3, self.patch_size, self.patch_size), dtype=np.float32)

        # 데이터 유효 마스크 (첫 번째 채널 기준)
        img_2d = image[0]
        data_valid = (img_2d > 0).astype(np.float32)

        # 최종 마스크: 해양 영역 × 데이터 유효
        # - 해양(sea_mask=1) & 데이터 있음(data_valid=1) → 1
        # - 해양(sea_mask=1) & 데이터 없음(data_valid=0) → 0 (복원 필요)
        # - 육지(sea_mask=0) → 1 (무시)
        final_mask = np.ones_like(img_2d, dtype=np.float32)
        ocean_holes = (sea_mask_patch == 1) & (data_valid == 0)
        final_mask[ocean_holes] = 0

        # 3채널로 확장
        return expand_to_3channels(final_mask).astype(np.float32)

    def validate_sample(self, image: np.ndarray, mask: np.ndarray, index: int) -> bool:
        """
        샘플 유효성 검사

        Args:
            image: 이미지 (C, H, W)
            mask: 마스크 (C, H, W)
            index: 파일 인덱스

        Returns:
            bool: 유효한 샘플이면 True
        """
        filename = os.path.basename(self.files[index])

        # 좌표 추출
        y0, x0 = extract_patch_coords(filename, pattern='y_x')
        if y0 is None:
            return False

        # 해양 마스크 추출
        try:
            sea_mask_patch = get_mask_patch(
                self.land_sea_mask, y0, x0, self.patch_size
            )
        except ValueError:
            return False

        # 해양 비율 계산
        total_pixels = self.patch_size * self.patch_size
        ocean_pixels = (sea_mask_patch == 1).sum()
        ocean_ratio = ocean_pixels / total_pixels

        # 최소 해양 비율 검사
        if ocean_ratio < self.min_ocean_ratio:
            return False

        return True
