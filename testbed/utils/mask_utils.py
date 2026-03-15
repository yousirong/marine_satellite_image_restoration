# -*- coding: utf-8 -*-
"""
Mask Utilities
==============

GOCI와 UST21의 Land-Sea 마스크 로드 및 변환 유틸리티
"""

import numpy as np
import re
from scipy import io
from typing import Tuple, Optional


def load_goci_mask(mask_path: str) -> np.ndarray:
    """
    GOCI Land-Sea 마스크 로드

    원본 형식: .npy 파일
    - 1 = 육지
    - 999 = 바다

    변환 후:
    - 0 = 육지
    - 1 = 바다

    Args:
        mask_path: 마스크 파일 경로 (.npy)

    Returns:
        np.ndarray: 변환된 마스크 (0=육지, 1=바다)
    """
    lm = np.load(mask_path)
    print(f"[GOCI Mask] Original shape: {lm.shape}, unique: {np.unique(lm)}")

    # 999 → 1 (바다), 그 외 → 0 (육지)
    sea_mask = np.where(lm == 999, 1, 0).astype(np.uint8)
    print(f"[GOCI Mask] Converted unique: {np.unique(sea_mask)}")

    return sea_mask


def load_ust21_mask(mask_path: str) -> np.ndarray:
    """
    UST21 Land-Sea 마스크 로드

    원본 형식: .mat 파일 (MATLAB)
    - Land 변수: 1 = 육지, 0 = 바다

    변환 후:
    - 0 = 육지
    - 1 = 바다

    Args:
        mask_path: 마스크 파일 경로 (.mat)

    Returns:
        np.ndarray: 변환된 마스크 (0=육지, 1=바다)
    """
    mat_data = io.loadmat(mask_path)
    land_mask_raw = mat_data['Land']
    print(f"[UST21 Mask] Original shape: {land_mask_raw.shape}, unique: {np.unique(land_mask_raw)}")

    # 0 (바다) → 1, 1 (육지) → 0
    sea_mask = np.where(land_mask_raw == 0, 1, 0).astype(np.uint8)
    print(f"[UST21 Mask] Converted unique: {np.unique(sea_mask)}")

    return sea_mask


def extract_patch_coords(filename: str, pattern: str = 'y_x') -> Tuple[Optional[int], Optional[int]]:
    """
    파일명에서 패치 좌표 추출

    Args:
        filename: 파일명 (예: y0256_x0512.tiff)
        pattern: 좌표 패턴
            - 'y_x': y{num}_x{num} 형식
            - 'r_c': r{num}_c{num} 형식

    Returns:
        Tuple[int, int]: (row/y, col/x) 좌표, 추출 실패 시 (None, None)
    """
    if pattern == 'y_x':
        match = re.search(r'y(\d+)_x(\d+)', filename)
    elif pattern == 'r_c':
        match = re.search(r'r(\d+)_c(\d+)', filename)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def get_mask_patch(global_mask: np.ndarray, y0: int, x0: int,
                   patch_size: int = 256) -> np.ndarray:
    """
    전역 마스크에서 패치 추출

    Args:
        global_mask: 전역 마스크 (H, W)
        y0: 시작 y 좌표
        x0: 시작 x 좌표
        patch_size: 패치 크기

    Returns:
        np.ndarray: 마스크 패치 (patch_size, patch_size)
    """
    H, W = global_mask.shape

    # 경계 검사
    if y0 + patch_size > H or x0 + patch_size > W:
        raise ValueError(
            f"Patch out of bounds: y0={y0}, x0={x0}, "
            f"patch_size={patch_size}, mask_shape={global_mask.shape}"
        )

    return global_mask[y0:y0+patch_size, x0:x0+patch_size].copy()


def expand_to_3channels(mask: np.ndarray) -> np.ndarray:
    """
    2D 마스크를 3채널로 확장

    Args:
        mask: 2D 마스크 (H, W)

    Returns:
        np.ndarray: 3채널 마스크 (3, H, W)
    """
    if mask.ndim == 2:
        return np.repeat(mask[np.newaxis, :, :], 3, axis=0)
    elif mask.ndim == 3 and mask.shape[0] == 3:
        return mask
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")


def create_ocean_hole_mask(image: np.ndarray, sea_mask: np.ndarray,
                           missing_values: list = None) -> np.ndarray:
    """
    해양 영역의 결측치 마스크 생성

    Args:
        image: 입력 이미지 (H, W) 또는 (C, H, W)
        sea_mask: 해양 마스크 (H, W), 1=바다, 0=육지
        missing_values: 결측값 목록

    Returns:
        np.ndarray: 결측 마스크, 1=유효, 0=결측(복원 필요)
    """
    missing_values = missing_values or [-999, 0]

    # 이미지가 3채널인 경우 첫 번째 채널 사용
    if image.ndim == 3:
        img_2d = image[0] if image.shape[0] in [1, 3] else image[:, :, 0]
    else:
        img_2d = image

    # 결측 영역 탐지
    hole_mask = np.zeros_like(img_2d, dtype=np.uint8)
    for val in missing_values:
        hole_mask |= (img_2d == val).astype(np.uint8)
    hole_mask |= np.isnan(img_2d).astype(np.uint8)

    # 최종 마스크: 기본값 1(유효), 해양 영역의 결측치만 0
    final_mask = np.ones_like(hole_mask, dtype=np.uint8)
    ocean_holes = (sea_mask == 1) & (hole_mask == 1)
    final_mask[ocean_holes] = 0

    return final_mask


def calculate_ocean_stats(image: np.ndarray, sea_mask: np.ndarray,
                          missing_values: list = None) -> dict:
    """
    해양 영역 통계 계산

    Args:
        image: 입력 이미지
        sea_mask: 해양 마스크 (1=바다)
        missing_values: 결측값 목록

    Returns:
        dict: 통계 정보
    """
    missing_values = missing_values or [-999, 0]

    if image.ndim == 3:
        img_2d = image[0] if image.shape[0] in [1, 3] else image[:, :, 0]
    else:
        img_2d = image

    total_pixels = img_2d.size
    ocean_pixels = (sea_mask == 1).sum()
    land_pixels = total_pixels - ocean_pixels

    # 결측 픽셀 수
    hole_mask = np.zeros_like(img_2d, dtype=bool)
    for val in missing_values:
        hole_mask |= (img_2d == val)
    hole_mask |= np.isnan(img_2d)

    ocean_holes = (sea_mask == 1) & hole_mask
    ocean_valid = (sea_mask == 1) & ~hole_mask

    return {
        'total_pixels': total_pixels,
        'ocean_pixels': int(ocean_pixels),
        'land_pixels': int(land_pixels),
        'ocean_ratio': float(ocean_pixels / total_pixels) if total_pixels > 0 else 0,
        'ocean_hole_pixels': int(ocean_holes.sum()),
        'ocean_valid_pixels': int(ocean_valid.sum()),
        'hole_ratio': float(ocean_holes.sum() / ocean_pixels) if ocean_pixels > 0 else 0
    }
