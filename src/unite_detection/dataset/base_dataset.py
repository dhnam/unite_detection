from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset

from unite_detection.processor import BaseProcessor
from unite_detection.schemas import DatasetConfig, FileMeta


class DeepFakeBaseDataset(Dataset, ABC):
    def __init__(
        self,
        paths: Sequence[Path | str],
        config: DatasetConfig | None = None,
    ):
        self.config = config or DatasetConfig()
        self._samples: list[FileMeta] | None = None
        self._processor: BaseProcessor | None = None
        self._paths = paths


    @property
    def processor(self) -> BaseProcessor:
        """Lazy Initialization"""
        if self._processor is None:
            self._processor = self._create_processor()
        return self._processor

    @property
    def samples(self) -> list[FileMeta]:
        """Lazy Initialization"""
        if self._samples is None:
            print(f"Processing {len(self._paths)} paths...")
            self._samples = []
            self._prepare_samples(self._paths)
            print(
                f"Loaded {len(self.samples)} samples "
                f"from {len(self._paths)} files/directories.",
            )
        return self._samples


    @abstractmethod
    def _create_processor(self) -> BaseProcessor:
        raise NotImplementedError

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

            frame_cnt = self.processor.get_frame_count(path_str)
            if frame_cnt <= 0:
                continue

            num_chunks = self.processor.calculate_num_chunks(frame_cnt)
            for i in range(num_chunks):
                self._samples.append(
                    {
                        "path": path_str,
                        "chunk_idx": i,
                        "label": label,
                        "total_frames": frame_cnt,
                    },
                )

    def get_label_counter(self) -> Counter[int]:
        return Counter([x["label"] for x in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[Float[Tensor, "channel batch h w"], Int[Tensor, "batch"]]:  # ty:ignore[invalid-method-override]

        return self.processor.getitem(self.samples[idx])
