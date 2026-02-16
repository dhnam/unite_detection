from pathlib import Path
from typing import override

from unite_detection.processor import VideoProcessor

from .base_dataset import DeepFakeBaseDataset


class CustomVideoDataset(DeepFakeBaseDataset):
    @override
    def _get_label(self, path: Path) -> int:
        return 0  # dummy value

    @override
    def _create_processor(self):
        return VideoProcessor(self.config)
