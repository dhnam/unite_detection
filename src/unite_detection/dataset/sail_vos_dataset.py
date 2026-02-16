from collections.abc import Sequence
from pathlib import Path
from typing import overload

from unite_detection.processor import ImageProcessor
from unite_detection.schemas import DatasetConfig

from .base_dataset import DeepFakeBaseDataset


class SailVosDataset(DeepFakeBaseDataset):
    def __init__(
        self,
        paths: Sequence[Path | str],
        config: DatasetConfig | None = None,
        ext: str = ".png",
    ):
        super().__init__(paths, config)

        self.ext = ext

    @overload
    def _create_processor(self):
        return ImageProcessor(self.config, self.idx_to_filename)

    def idx_to_filename(self, idx: int) -> str:
        return f"{idx:06d}{self.ext}"

    def _get_label(self, path: str) -> int | None:
        return 1
