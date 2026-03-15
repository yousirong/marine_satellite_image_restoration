# -*- coding: utf-8 -*-
"""
Performance Metrics
===================

복원 결과 평가를 위한 성능 지표 계산 함수들
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import r2_score


def calculate_rmse(gt: np.ndarray, pred: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
    """
    Root Mean Squared Error (RMSE) 계산

    Args:
        gt: Ground Truth 값
        pred: 예측 값
        mask: 유효 마스크 (선택적)

    Returns:
        float: RMSE 값
    """
    if mask is not None:
        valid = mask.astype(bool)
        gt = gt[valid]
        pred = pred[valid]

    diff = pred - gt
    return float(np.sqrt(np.mean(diff ** 2)))


def calculate_mae(gt: np.ndarray, pred: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> float:
    """
    Mean Absolute Error (MAE) 계산

    Args:
        gt: Ground Truth 값
        pred: 예측 값
        mask: 유효 마스크 (선택적)

    Returns:
        float: MAE 값
    """
    if mask is not None:
        valid = mask.astype(bool)
        gt = gt[valid]
        pred = pred[valid]

    return float(np.mean(np.abs(pred - gt)))


def calculate_r2(gt: np.ndarray, pred: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> float:
    """
    R² (결정계수) 계산

    Args:
        gt: Ground Truth 값
        pred: 예측 값
        mask: 유효 마스크 (선택적)

    Returns:
        float: R² 값
    """
    if mask is not None:
        valid = mask.astype(bool)
        gt = gt[valid]
        pred = pred[valid]

    if len(gt) < 2:
        return 0.0

    return float(r2_score(gt, pred))


def calculate_mape(gt: np.ndarray, pred: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (MAPE) 계산

    Args:
        gt: Ground Truth 값
        pred: 예측 값
        mask: 유효 마스크 (선택적)
        epsilon: 0으로 나누기 방지 값

    Returns:
        float: MAPE 값 (%)
    """
    if mask is not None:
        valid = mask.astype(bool)
        gt = gt[valid]
        pred = pred[valid]

    # 0에 가까운 값 제외
    nonzero = np.abs(gt) > epsilon
    gt = gt[nonzero]
    pred = pred[nonzero]

    if len(gt) == 0:
        return 0.0

    return float(np.mean(np.abs((gt - pred) / gt)) * 100)


def calculate_bias(gt: np.ndarray, pred: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
    """
    평균 편향(Bias) 계산

    Args:
        gt: Ground Truth 값
        pred: 예측 값
        mask: 유효 마스크 (선택적)

    Returns:
        float: Bias 값
    """
    if mask is not None:
        valid = mask.astype(bool)
        gt = gt[valid]
        pred = pred[valid]

    return float(np.mean(pred - gt))


def calculate_all_metrics(gt: np.ndarray, pred: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          metrics: list = None) -> Dict[str, float]:
    """
    모든 성능 지표 계산

    Args:
        gt: Ground Truth 값
        pred: 예측 값
        mask: 유효 마스크 (선택적)
        metrics: 계산할 지표 목록 (기본: ['rmse', 'mae', 'r2'])

    Returns:
        Dict[str, float]: 지표명 → 값 딕셔너리
    """
    metrics = metrics or ['rmse', 'mae', 'r2']
    results = {}

    if mask is not None:
        valid = mask.astype(bool)
        gt_valid = gt[valid]
        pred_valid = pred[valid]
    else:
        gt_valid = gt.flatten()
        pred_valid = pred.flatten()

    # NaN 제거
    valid_idx = ~(np.isnan(gt_valid) | np.isnan(pred_valid))
    gt_valid = gt_valid[valid_idx]
    pred_valid = pred_valid[valid_idx]

    if len(gt_valid) == 0:
        return {m: 0.0 for m in metrics}

    if 'rmse' in metrics:
        results['rmse'] = calculate_rmse(gt_valid, pred_valid)
    if 'mae' in metrics:
        results['mae'] = calculate_mae(gt_valid, pred_valid)
    if 'r2' in metrics:
        results['r2'] = calculate_r2(gt_valid, pred_valid)
    if 'mape' in metrics:
        results['mape'] = calculate_mape(gt_valid, pred_valid)
    if 'bias' in metrics:
        results['bias'] = calculate_bias(gt_valid, pred_valid)

    # 샘플 수 추가
    results['n_samples'] = len(gt_valid)

    return results


def filter_top_percent(gt: np.ndarray, pred: np.ndarray,
                       percent: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    상위 X% 성능 데이터 필터링

    절대 오차가 작은 상위 percent 비율의 데이터만 반환합니다.

    Args:
        gt: Ground Truth 값
        pred: 예측 값
        percent: 상위 비율 (0~1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: 필터링된 (gt, pred)
    """
    abs_errors = np.abs(pred - gt)
    num_top = int(len(abs_errors) * percent)
    top_indices = np.argsort(abs_errors)[:num_top]

    return gt[top_indices], pred[top_indices]


def calculate_statistics(values: np.ndarray) -> Dict[str, float]:
    """
    기본 통계량 계산

    Args:
        values: 값 배열

    Returns:
        Dict[str, float]: 통계량 딕셔너리
    """
    valid = values[~np.isnan(values)]

    if len(valid) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'count': 0
        }

    return {
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'median': float(np.median(valid)),
        'count': int(len(valid))
    }
