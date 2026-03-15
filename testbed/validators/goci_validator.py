# -*- coding: utf-8 -*-
"""
GOCI Validator
==============

GOCI 복원 결과 검증기
"""

import os
import glob
import numpy as np
from typing import Dict, Any, Tuple, List

from testbed.core.evaluator import BaseEvaluator


class GOCIValidator(BaseEvaluator):
    """
    GOCI 결과 검증기

    GOCI Rrs 복원 결과를 검증하고 성능 지표를 계산합니다.

    특징:
    - CSV 파일 기반 결과 로드
    - 육지 영역 (255 값) 필터링
    - 결측값 처리
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 검증 설정
                - metrics: 계산할 지표 목록
                - top_percent: 상위 X% 필터링
                - land_value: 육지 표시 값 (기본: 255)
        """
        super().__init__(config)
        self.land_value = config.get('land_value', 255)
        self.missing_values = config.get('missing_values', [-999, 0, 1000])

    def load_results(self, result_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        GOCI 결과 로드

        degree/gt/, degree/recon/ 폴더에서 CSV 파일을 로드합니다.

        Args:
            result_path: 결과 저장 경로

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - gt: Ground Truth 값들
                - pred: 예측 값들
                - mask: 마스크 값들
        """
        gt_dir = os.path.join(result_path, 'degree', 'gt')
        recon_dir = os.path.join(result_path, 'degree', 'recon')
        mask_dir = os.path.join(result_path, 'degree', 'mask')

        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.csv')))
        recon_files = sorted(glob.glob(os.path.join(recon_dir, '*.csv')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.csv')))

        print(f"[GOCI Validator] Found {len(gt_files)} GT files, {len(recon_files)} recon files")

        if len(gt_files) == 0 or len(recon_files) == 0:
            return np.array([]), np.array([]), np.array([])

        all_gt = []
        all_pred = []
        all_mask = []

        for i, (gt_file, recon_file) in enumerate(zip(gt_files, recon_files)):
            try:
                gt_data = np.loadtxt(gt_file, delimiter=',')
                pred_data = np.loadtxt(recon_file, delimiter=',')

                # 마스크 로드 (있는 경우)
                if i < len(mask_files):
                    mask_data = np.loadtxt(mask_files[i], delimiter=',')
                else:
                    mask_data = np.ones_like(gt_data)

                all_gt.append(gt_data.flatten())
                all_pred.append(pred_data.flatten())
                all_mask.append(mask_data.flatten())

            except Exception as e:
                print(f"[Warning] Failed to load {gt_file}: {e}")
                continue

        if len(all_gt) == 0:
            return np.array([]), np.array([]), np.array([])

        gt = np.concatenate(all_gt)
        pred = np.concatenate(all_pred)
        mask = np.concatenate(all_mask)

        return gt, pred, mask

    def filter_land(self, gt: np.ndarray, pred: np.ndarray,
                    mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        육지 및 결측 영역 필터링

        Args:
            gt: Ground Truth 값들
            pred: 예측 값들
            mask: 마스크

        Returns:
            Tuple[np.ndarray, np.ndarray]: 해양 영역만 남긴 (gt, pred)
        """
        # 유효 조건:
        # 1. 육지 아님 (gt != 255)
        # 2. NaN 아님
        # 3. 결측값 아님
        valid = (gt != self.land_value) & ~np.isnan(gt) & ~np.isnan(pred)

        # 결측값 제외
        for val in self.missing_values:
            valid &= (gt != val)

        # 마스크 적용 (있는 경우)
        if mask is not None:
            valid &= (mask > 0)

        return gt[valid], pred[valid]

    def get_ocean_hole_mask(self, result_path: str) -> List[np.ndarray]:
        """
        해양 결측 영역 마스크 로드

        Args:
            result_path: 결과 저장 경로

        Returns:
            List[np.ndarray]: 마스크 배열 리스트
        """
        mask_dir = os.path.join(result_path, 'degree', 'mask')
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.csv')))

        masks = []
        for mask_file in mask_files:
            try:
                mask_data = np.loadtxt(mask_file, delimiter=',')
                masks.append(mask_data)
            except Exception as e:
                print(f"[Warning] Failed to load mask {mask_file}: {e}")

        return masks
