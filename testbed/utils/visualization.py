# -*- coding: utf-8 -*-
"""
Visualization Utilities
=======================

복원 결과 시각화를 위한 도구들
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Dict, Any, Optional, Tuple


class UnifiedVisualizer:
    """
    통합 시각화 클래스

    GOCI와 UST21 결과를 시각화합니다.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 시각화 설정
                - colormap: 컬러맵 (기본: 'jet')
                - figsize: 그림 크기
                - dpi: 해상도
        """
        config = config or {}
        self.cmap = config.get('colormap', 'jet')
        self.figsize = config.get('figsize', (8, 8))
        self.dpi = config.get('dpi', 300)

    def plot_scatter(self, gt: np.ndarray, pred: np.ndarray,
                     metrics: Dict[str, float],
                     output_path: str,
                     title: str = "Validation",
                     xlabel: str = "Ground Truth",
                     ylabel: str = "Prediction"):
        """
        Scatter plot (Parity plot) 생성

        Args:
            gt: Ground Truth 값
            pred: 예측 값
            metrics: 성능 지표 딕셔너리
            output_path: 저장 경로
            title: 그래프 제목
            xlabel: X축 라벨
            ylabel: Y축 라벨
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 값 범위
        vmin = min(np.min(gt), np.min(pred))
        vmax = max(np.max(gt), np.max(pred))

        # Scatter plot
        ax.scatter(gt, pred, s=1, alpha=0.3, c='blue', rasterized=True)

        # 1:1 line
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5, label='1:1 line')

        # 설정
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_aspect('equal')

        # Metrics 표시
        text_lines = []
        if 'rmse' in metrics:
            text_lines.append(f"RMSE: {metrics['rmse']:.6f}")
        if 'mae' in metrics:
            text_lines.append(f"MAE: {metrics['mae']:.6f}")
        if 'r2' in metrics:
            text_lines.append(f"R²: {metrics['r2']:.4f}")
        if 'n_samples' in metrics:
            text_lines.append(f"N: {metrics['n_samples']:,}")

        text = '\n'.join(text_lines)
        ax.text(0.95, 0.05, text, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot saved: {output_path}")

    def plot_image_comparison(self, gt_img: np.ndarray, pred_img: np.ndarray,
                              mask_img: np.ndarray, output_path: str,
                              vmin: float = None, vmax: float = None,
                              title: str = "Comparison"):
        """
        GT vs Prediction 이미지 비교 플롯

        Args:
            gt_img: Ground Truth 이미지 (H, W)
            pred_img: 예측 이미지 (H, W)
            mask_img: 마스크 이미지 (H, W)
            output_path: 저장 경로
            vmin: 최소값
            vmax: 최대값
            title: 제목
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 값 범위 계산
        if vmin is None:
            vmin = min(np.nanmin(gt_img), np.nanmin(pred_img))
        if vmax is None:
            vmax = max(np.nanmax(gt_img), np.nanmax(pred_img))

        # GT
        im0 = axes[0].imshow(gt_img, cmap=self.cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Mask
        im1 = axes[1].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Mask')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Prediction
        im2 = axes[2].imshow(pred_img, cmap=self.cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Difference
        diff = pred_img - gt_img
        diff_max = np.nanmax(np.abs(diff))
        im3 = axes[3].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[3].set_title('Difference (Pred - GT)')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=14)

        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Comparison plot saved: {output_path}")

    def plot_histogram(self, gt: np.ndarray, pred: np.ndarray,
                       output_path: str, bins: int = 100,
                       title: str = "Value Distribution"):
        """
        GT와 Prediction의 히스토그램 비교

        Args:
            gt: Ground Truth 값
            pred: 예측 값
            output_path: 저장 경로
            bins: 빈 개수
            title: 제목
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 범위 계산
        vmin = min(np.min(gt), np.min(pred))
        vmax = max(np.max(gt), np.max(pred))

        # 히스토그램
        ax.hist(gt, bins=bins, range=(vmin, vmax), alpha=0.5,
                label=f'GT (mean={np.mean(gt):.4f})', color='blue')
        ax.hist(pred, bins=bins, range=(vmin, vmax), alpha=0.5,
                label=f'Pred (mean={np.mean(pred):.4f})', color='red')

        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()

        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Histogram saved: {output_path}")

    def plot_error_distribution(self, gt: np.ndarray, pred: np.ndarray,
                                output_path: str, bins: int = 100,
                                title: str = "Error Distribution"):
        """
        오차 분포 히스토그램

        Args:
            gt: Ground Truth 값
            pred: 예측 값
            output_path: 저장 경로
            bins: 빈 개수
            title: 제목
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        error = pred - gt
        abs_error = np.abs(error)

        # 오차 히스토그램
        axes[0].hist(error, bins=bins, alpha=0.7, color='blue')
        axes[0].axvline(x=0, color='red', linestyle='--', label='Zero')
        axes[0].axvline(x=np.mean(error), color='green', linestyle='-',
                        label=f'Mean={np.mean(error):.4f}')
        axes[0].set_xlabel('Error (Pred - GT)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Error Distribution', fontsize=14)
        axes[0].legend()

        # 절대 오차 히스토그램
        axes[1].hist(abs_error, bins=bins, alpha=0.7, color='orange')
        axes[1].axvline(x=np.mean(abs_error), color='green', linestyle='-',
                        label=f'MAE={np.mean(abs_error):.4f}')
        axes[1].set_xlabel('Absolute Error', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Absolute Error Distribution', fontsize=14)
        axes[1].legend()

        plt.suptitle(title, fontsize=16)

        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Error distribution saved: {output_path}")

    def create_summary_report(self, results: Dict[str, Any],
                              output_path: str, model_type: str):
        """
        텍스트 형식의 요약 리포트 생성

        Args:
            results: 평가 결과 딕셔너리
            output_path: 저장 경로
            model_type: 모델 타입 ('goci' 또는 'ust21')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"  {model_type.upper()} RESTORATION EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")

            f.write("[Performance Metrics]\n")
            f.write("-"*40 + "\n")
            if 'rmse' in results:
                f.write(f"  RMSE: {results['rmse']:.6f}\n")
            if 'mae' in results:
                f.write(f"  MAE:  {results['mae']:.6f}\n")
            if 'r2' in results:
                f.write(f"  R²:   {results['r2']:.4f}\n")
            if 'n_samples' in results:
                f.write(f"  N:    {results['n_samples']:,}\n")

            if 'top_metrics' in results:
                f.write(f"\n[Top {results.get('top_percent', 0.95)*100:.0f}% Metrics]\n")
                f.write("-"*40 + "\n")
                top = results['top_metrics']
                if 'rmse' in top:
                    f.write(f"  RMSE: {top['rmse']:.6f}\n")
                if 'mae' in top:
                    f.write(f"  MAE:  {top['mae']:.6f}\n")
                if 'r2' in top:
                    f.write(f"  R²:   {top['r2']:.4f}\n")

            if 'gt_stats' in results:
                f.write(f"\n[Ground Truth Statistics]\n")
                f.write("-"*40 + "\n")
                stats = results['gt_stats']
                f.write(f"  Mean:   {stats.get('mean', 0):.6f}\n")
                f.write(f"  Std:    {stats.get('std', 0):.6f}\n")
                f.write(f"  Min:    {stats.get('min', 0):.6f}\n")
                f.write(f"  Max:    {stats.get('max', 0):.6f}\n")

            if 'pred_stats' in results:
                f.write(f"\n[Prediction Statistics]\n")
                f.write("-"*40 + "\n")
                stats = results['pred_stats']
                f.write(f"  Mean:   {stats.get('mean', 0):.6f}\n")
                f.write(f"  Std:    {stats.get('std', 0):.6f}\n")
                f.write(f"  Min:    {stats.get('min', 0):.6f}\n")
                f.write(f"  Max:    {stats.get('max', 0):.6f}\n")

            f.write("\n" + "="*60 + "\n")

        print(f"Summary report saved: {output_path}")
