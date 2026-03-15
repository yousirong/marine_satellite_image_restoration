# -*- coding: utf-8 -*-
"""
GOCI Full Patch Dataset
=======================

GOCI (Remote Sensing Reflectance) 데이터를 위한 데이터셋 클래스
"""

import os
import re
import numpy as np
import tifffile as tiff
import cv2
from typing import Dict, Any, Optional

from testbed.core.base_dataset import BaseFullPatchDataset
from testbed.utils.mask_utils import (
    load_goci_mask,
    extract_patch_coords,
    get_mask_patch,
    expand_to_3channels,
    calculate_ocean_stats
)


class GOCIFullPatchDataset(BaseFullPatchDataset):
    """
    GOCI Full Patch 데이터셋

    GOCI Rrs (Remote Sensing Reflectance) 데이터를 처리합니다.

    특징:
    - 파일명 패턴: y{:04d}_x{:04d}.tiff
    - Land-Sea 마스크: .npy 형식 (999=바다)
    - 결측값: -999, 0, 1000
    - 해양 영역 검증: min 30%, hole ratio 1-95%
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 데이터셋 설정
                - image_dir: 이미지 디렉터리
                - land_mask_path: Land-Sea 마스크 경로 (.npy)
                - patch_size: 패치 크기 (기본: 256)
                - missing_values: 결측값 목록 (기본: [-999, 0, 1000])
                - valid_range: 유효 범위 (기본: [-1000, 1000])
                - min_ocean_ratio: 최소 해양 비율 (기본: 0.3)
                - min_hole_ratio: 최소 결측 비율 (기본: 0.01)
                - max_hole_ratio: 최대 결측 비율 (기본: 0.95)
        """
        # 설정값 추출
        self.missing_values = config.get('missing_values', [-999, 0, 1000])
        self.valid_range = config.get('valid_range', [-1000, 1000])
        self.min_ocean_ratio = config.get('min_ocean_ratio', 0.3)
        self.min_hole_ratio = config.get('min_hole_ratio', 0.01)
        self.max_hole_ratio = config.get('max_hole_ratio', 0.95)

        super().__init__(config)

    def _load_land_sea_mask(self):
        """GOCI Land-Sea 마스크 로드"""
        if not self.land_mask_path or not os.path.exists(self.land_mask_path):
            print(f"[Warning] Land mask not found: {self.land_mask_path}")
            return

        self.land_sea_mask = load_goci_mask(self.land_mask_path)
        print(f"[GOCI] Land-sea mask shape: {self.land_sea_mask.shape}")

    def load_image(self, index: int) -> np.ndarray:
        """
        이미지 로드

        Args:
            index: 파일 인덱스

        Returns:
            np.ndarray: 로드된 이미지 (H, W) 또는 (H, W, C)
        """
        file_path = self.files[index]
        image = tiff.imread(file_path).astype(np.float32)
        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        GOCI 데이터 전처리

        1. 채널 처리 (3채널로 확장)
        2. 크기 조정
        3. 극값 처리
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

        # 극값 처리
        extreme_mask = (image < self.valid_range[0]) | (image > self.valid_range[1])
        image[extreme_mask] = self.missing_values[0]

        # NaN 처리 - 결측값으로 대체
        image[np.isnan(image)] = self.missing_values[0]

        # (H, W, C) → (C, H, W)
        image = np.transpose(image, (2, 0, 1))

        return image

    def create_mask(self, image: np.ndarray, index: int) -> np.ndarray:
        """
        결측치 마스크 생성

        해양 영역에서만 결측치를 마스킹합니다.
        육지 영역은 항상 유효(1)로 처리됩니다.

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
            # 좌표 추출 실패 시 전체 유효 마스크 반환
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

        # 결측 영역 탐지 (첫 번째 채널 기준)
        img_2d = image[0]
        hole_mask = np.zeros_like(img_2d, dtype=np.uint8)
        for val in self.missing_values:
            hole_mask |= (img_2d == val).astype(np.uint8)
        hole_mask |= np.isnan(img_2d).astype(np.uint8)

        # 최종 마스크 생성
        # - 기본값: 1 (유효)
        # - 해양 영역의 결측치: 0 (복원 필요)
        final_mask = np.ones_like(img_2d, dtype=np.float32)
        ocean_holes = (sea_mask_patch == 1) & (hole_mask == 1)
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

        # 통계 계산
        stats = calculate_ocean_stats(
            image[0], sea_mask_patch, self.missing_values
        )

        # 조건 검사
        # 1. 최소 해양 비율
        if stats['ocean_ratio'] < self.min_ocean_ratio:
            return False

        # 2. 결측 비율 범위
        if stats['hole_ratio'] < self.min_hole_ratio:
            return False
        if stats['hole_ratio'] > self.max_hole_ratio:
            return False

        return True
