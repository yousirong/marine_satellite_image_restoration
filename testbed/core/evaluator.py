# -*- coding: utf-8 -*-
"""
Evaluator Base Classes
======================

복원 결과 평가를 위한 베이스 클래스 및 팩토리
"""

import os
import glob
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from testbed.utils.metrics import (
    calculate_all_metrics,
    filter_top_percent,
    calculate_statistics
)


class BaseEvaluator(ABC):
    """
    평가기 베이스 클래스

    복원 결과를 로드하고 성능 지표를 계산합니다.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 평가 설정
                - metrics: 계산할 지표 목록
                - top_percent: 상위 X% 필터링 (선택적)
                - colormap: 시각화 컬러맵
        """
        self.config = config
        self.metrics = config.get('metrics', ['rmse', 'mae', 'r2'])
        self.top_percent = config.get('top_percent', 0.95)
        self.colormap = config.get('colormap', 'jet')

    @abstractmethod
    def load_results(self, result_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        결과 로드

        Args:
            result_path: 결과 저장 경로

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - gt: Ground Truth 값들
                - pred: 예측 값들
                - mask: 유효 마스크 (선택적)
        """
        pass

    @abstractmethod
    def filter_land(self, gt: np.ndarray, pred: np.ndarray,
                    mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        육지 영역 필터링

        Args:
            gt: Ground Truth 값들
            pred: 예측 값들
            mask: 마스크

        Returns:
            Tuple[np.ndarray, np.ndarray]: 필터링된 (gt, pred)
        """
        pass

    def evaluate(self, result_path: str) -> Dict[str, Any]:
        """
        평가 실행

        Args:
            result_path: 결과 저장 경로

        Returns:
            Dict[str, Any]: 평가 결과
        """
        # 결과 로드
        gt, pred, mask = self.load_results(result_path)

        if len(gt) == 0:
            return {'error': 'No results found'}

        # 육지 필터링
        gt_ocean, pred_ocean = self.filter_land(gt, pred, mask)

        if len(gt_ocean) == 0:
            return {'error': 'No ocean data found'}

        # 전체 지표 계산
        all_metrics = calculate_all_metrics(gt_ocean, pred_ocean, metrics=self.metrics)

        # 상위 X% 필터링 후 지표 계산
        if self.top_percent < 1.0:
            gt_top, pred_top = filter_top_percent(gt_ocean, pred_ocean, self.top_percent)
            top_metrics = calculate_all_metrics(gt_top, pred_top, metrics=self.metrics)
            all_metrics['top_metrics'] = top_metrics
            all_metrics['top_percent'] = self.top_percent

        # 통계 추가
        all_metrics['gt_stats'] = calculate_statistics(gt_ocean)
        all_metrics['pred_stats'] = calculate_statistics(pred_ocean)

        return all_metrics

    def print_results(self, results: Dict[str, Any], title: str = "Evaluation Results"):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")

        if 'error' in results:
            print(f"Error: {results['error']}")
            return

        print(f"Samples: {results.get('n_samples', 'N/A')}")
        print(f"\n[All Data]")
        for metric in self.metrics:
            if metric in results:
                print(f"  {metric.upper()}: {results[metric]:.6f}")

        if 'top_metrics' in results:
            print(f"\n[Top {results['top_percent']*100:.0f}%]")
            for metric in self.metrics:
                if metric in results['top_metrics']:
                    print(f"  {metric.upper()}: {results['top_metrics'][metric]:.6f}")

        print(f"{'='*60}")


class EvaluatorFactory:
    """
    평가기 팩토리

    모델 타입에 따라 적절한 평가기 클래스를 생성합니다.
    """

    @staticmethod
    def create(model_type: str, config: Dict[str, Any]) -> BaseEvaluator:
        """
        평가기 생성

        Args:
            model_type: 'goci' 또는 'ust21'
            config: 평가 설정

        Returns:
            BaseEvaluator: 생성된 평가기
        """
        if model_type.lower() == 'goci':
            from testbed.validators.goci_validator import GOCIValidator
            return GOCIValidator(config)
        elif model_type.lower() == 'ust21':
            from testbed.validators.ust21_validator import UST21Validator
            return UST21Validator(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def list_available() -> List[str]:
        """사용 가능한 평가기 타입 목록"""
        return ['goci', 'ust21']
