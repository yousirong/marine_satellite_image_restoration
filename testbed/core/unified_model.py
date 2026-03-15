# -*- coding: utf-8 -*-
"""
Unified Model Wrapper
=====================

기존 RFRNetModel을 감싸서 GOCI/UST21 공통 인터페이스를 제공합니다.
"""

import sys
import os
import torch
import torch.nn as nn

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.model import RFRNetModel


class UnifiedModel:
    """
    GOCI/UST21 통합 모델 래퍼

    기존 RFRNetModel을 감싸서 통일된 인터페이스를 제공합니다.

    Args:
        model_type: 모델 유형 ('goci' 또는 'ust21')
        checkpoint_path: 체크포인트 파일 경로
        gpu_ids: 사용할 GPU ID 리스트

    Example:
        >>> model = UnifiedModel('goci', '/path/to/checkpoint.pth', [0, 1])
        >>> model.initialize()
        >>> model.infer(dataloader, '/path/to/results')
    """

    def __init__(self, model_type: str, checkpoint_path: str, gpu_ids: list = None):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.gpu_ids = gpu_ids or [0]
        self.model = None
        self.initialized = False

    def initialize(self):
        """모델 초기화 및 체크포인트 로드"""
        print(f"\n{'='*50}")
        print(f"Initializing {self.model_type.upper()} Model")
        print(f"{'='*50}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"GPU IDs: {self.gpu_ids}")

        # RFRNetModel 인스턴스 생성
        self.model = RFRNetModel()

        # 모델 초기화 (체크포인트 로드)
        self.model.initialize_model(
            path=self.checkpoint_path,
            train=False,
            gpu_ids=self.gpu_ids
        )

        # CUDA로 이동
        self.model.cuda()

        # 모델 건강 상태 체크
        if hasattr(self.model, 'check_model_health'):
            self.model.check_model_health()

        self.initialized = True
        print(f"Model initialized successfully!")

    def infer(self, dataloader, result_save_path: str):
        """
        추론 실행

        Args:
            dataloader: PyTorch DataLoader
            result_save_path: 결과 저장 경로
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # 결과 저장 디렉터리 생성
        os.makedirs(result_save_path, exist_ok=True)

        print(f"\nRunning inference...")
        print(f"Result path: {result_save_path}")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Total samples: {len(dataloader.dataset)}")

        # 추론 실행
        self.model.test(
            test_loader=dataloader,
            result_save_path=result_save_path
        )

        print(f"Inference completed!")

    def get_model(self) -> RFRNetModel:
        """내부 RFRNetModel 반환"""
        return self.model

    def __repr__(self):
        return f"UnifiedModel(type={self.model_type}, initialized={self.initialized})"
