# -*- coding: utf-8 -*-
"""
Base Dataset Classes
====================

Full Patch 데이터셋의 베이스 클래스 및 팩토리
"""

import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseFullPatchDataset(Dataset, ABC):
    """
    Full Patch 데이터셋 베이스 클래스

    GOCI와 UST21 데이터셋의 공통 인터페이스를 정의합니다.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 데이터셋 설정
                - image_dir: 이미지 디렉터리
                - land_mask_path: Land-Sea 마스크 경로
                - patch_size: 패치 크기 (기본: 256)
        """
        super().__init__()
        self.config = config
        self.image_dir = config.get('image_dir', '')
        self.land_mask_path = config.get('land_mask_path', '')
        self.patch_size = config.get('patch_size', 256)

        # 파일 목록 및 마스크
        self.files: List[str] = []
        self.land_sea_mask: Optional[np.ndarray] = None

        # 초기화
        self._load_files()
        self._load_land_sea_mask()

    def _load_files(self):
        """이미지 파일 목록 로드"""
        if not os.path.isdir(self.image_dir):
            print(f"[Warning] Image directory not found: {self.image_dir}")
            return

        for root, _, files in os.walk(self.image_dir):
            for f in files:
                if f.lower().endswith(('.tiff', '.tif', '.png')):
                    self.files.append(os.path.join(root, f))

        self.files.sort()
        print(f"[Dataset] Loaded {len(self.files)} files from {self.image_dir}")

    @abstractmethod
    def _load_land_sea_mask(self):
        """Land-Sea 마스크 로드 (서브클래스에서 구현)"""
        pass

    @abstractmethod
    def load_image(self, index: int) -> np.ndarray:
        """이미지 로드 (서브클래스에서 구현)"""
        pass

    @abstractmethod
    def create_mask(self, image: np.ndarray, index: int) -> np.ndarray:
        """결측치 마스크 생성 (서브클래스에서 구현)"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """데이터 전처리 (서브클래스에서 구현)"""
        pass

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        데이터 아이템 반환

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]:
                - image: 전처리된 이미지 (C, H, W)
                - mask: 결측 마스크 (C, H, W), 1=유효, 0=결측
                - filename: 파일명
        """
        # 이미지 로드 및 전처리
        image = self.load_image(index)
        image = self.preprocess(image)

        # 마스크 생성
        mask = self.create_mask(image, index)

        # 파일명
        filename = os.path.basename(self.files[index])

        # Tensor 변환
        image_tensor = self._to_tensor(image)
        mask_tensor = self._to_tensor(mask)

        return image_tensor, mask_tensor, filename

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """NumPy 배열을 Tensor로 변환"""
        if arr.dtype == object:
            arr = arr.astype(np.float32)

        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)

        return torch.from_numpy(arr.astype(np.float32))

    def get_file_path(self, index: int) -> str:
        """인덱스로 파일 경로 반환"""
        return self.files[index]


class DatasetFactory:
    """
    데이터셋 팩토리

    모델 타입에 따라 적절한 데이터셋 클래스를 생성합니다.
    """

    @staticmethod
    def create(model_type: str, config: Dict[str, Any]) -> BaseFullPatchDataset:
        """
        데이터셋 생성

        Args:
            model_type: 'goci' 또는 'ust21'
            config: 데이터셋 설정

        Returns:
            BaseFullPatchDataset: 생성된 데이터셋
        """
        if model_type.lower() == 'goci':
            from testbed.datasets.goci_dataset import GOCIFullPatchDataset
            return GOCIFullPatchDataset(config)
        elif model_type.lower() == 'ust21':
            from testbed.datasets.ust21_dataset import UST21FullPatchDataset
            return UST21FullPatchDataset(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def list_available() -> List[str]:
        """사용 가능한 데이터셋 타입 목록"""
        return ['goci', 'ust21']
