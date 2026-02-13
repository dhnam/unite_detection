from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Sequence, TypedDict

import transformers
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from unite_detection.schemas import DatasetConfig


class Sample(TypedDict):
    path: str
    chunk_idx: int
    label: int
    total_frames: int


class DeepFakeBaseDataset(Dataset, ABC):
    def __init__(
        self, paths: Sequence[Path | str], config: DatasetConfig | None = None
    ):
        self.config = config if config else DatasetConfig()
        self.samples: list[Sample] = []

        print(f"Processing {len(paths)} paths...")
        self._prepare_samples(paths)
        print(
            f"Loaded {len(self.samples)} samples from {len(paths)} files/directories."
        )

        self.preprocessor = None
        if config.encoder.use_auto_processor:
            self.preprocessor = transformers.AutoProcessor.from_pretrained(
                config.encoder.model, use_fast=True
            )

    @abstractmethod
    def _get_label(self, path: str) -> int | None:
        raise NotImplementedError

    def _prepare_samples(self, paths: Sequence[Path | str]):
        """상속받는 클래스에서 각자의 방식으로 samples 리스트를 채움"""
        for path in paths:
            path_str = str(path)
            label = self._get_label(path_str)
            if label is None:
                continue

            frame_cnt = self._get_frame_count(path_str)
            if frame_cnt <= 0:
                continue

            num_chunks = self._calculate_num_chunks(frame_cnt)
            for i in range(num_chunks):
                self.samples.append(
                    {
                        "path": path_str,
                        "chunk_idx": i,
                        "label": label,
                        "total_frames": frame_cnt,
                    }
                )

    @abstractmethod
    def _get_frame_count(self, path: str) -> int:
        """자식 클래스에서 구현: 전체 프레임 수 반환"""
        raise NotImplementedError

    @abstractmethod
    def _calculate_num_chunks(self, frame_cnt: int) -> int:
        """자식 클래스에서 구현: 프레임 수에 따른 청크 개수 계산"""
        raise NotImplementedError

    def get_label_counter(self) -> Counter[int]:
        return Counter([x["label"] for x in self.samples])

    def __len__(self):
        return len(self.samples)

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> tuple[Float[Tensor, "channel batch h w"], Int[Tensor, "batch"]]:  # ty:ignore[invalid-method-override]
        """자식 클래스에서 구현: 실제 데이터 로드"""
        raise NotImplementedError
