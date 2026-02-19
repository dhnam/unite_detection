from pathlib import Path
from typing import override

from unite_detection.processor import VideoProcessor

from .base_dataset import DeepFakeBaseDataset


class FFDataset(DeepFakeBaseDataset):
    @override
    def _get_label(self, path: Path) -> int:
        path_parts = path.parts
        if "original" in path_parts:
            return 0
        else:
            return 1

    @override
    def _create_processor(self):
        return VideoProcessor(self.config)
