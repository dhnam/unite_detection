from pathlib import Path
from typing import Sequence

from unite_detection.dataset import DeepFakeBaseDataset, ImageProcessor
from unite_detection.schemas import DatasetConfig


class SailVosDataset(DeepFakeBaseDataset):
    def __init__(
        self,
        paths: Sequence[Path | str],
        config: DatasetConfig | None = None,
        ext: str = ".png",
    ):
        self.ext = ext
        if config is None:
            config = DatasetConfig()
        self.processor = ImageProcessor(self.idx_to_filename, config.size)

        super().__init__(paths, config)

    def idx_to_filename(self, idx: int) -> str:
        return f"{idx:06d}{self.ext}"

    def _get_label(self, path: str) -> int | None:
        return 1

    def _get_frame_count(self, path: str) -> int:
        return self.processor.get_frame_count(path)

    def _calculate_num_chunks(self, frame_cnt: int) -> int:
        return self.processor.calculate_num_chunks(self, frame_cnt)

    def __getitem__(self, idx):
        return self.processor.getitem(self, idx)
