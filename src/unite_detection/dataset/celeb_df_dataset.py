from abc import ABC
from pathlib import Path
from typing import override

from unite_detection.processor import ImageProcessor, VideoProcessor

from .base_dataset import DeepFakeBaseDataset


class CelebDFBaseDataset(DeepFakeBaseDataset, ABC):
    """
    Celeb-DF 데이터셋의 공통 기능을 담은 부모 클래스
    """

    @override
    def _get_label(self, path: Path) -> int | None:
        """폴더명 기반 레이블 결정"""
        rel_path = path.parts
        if "YouTube-real" in rel_path or "Celeb-real" in rel_path:
            return 0
        elif "Celeb-synthesis" in rel_path:
            return 1
        return None


class CelebDFVideoDataset(CelebDFBaseDataset):
    """
    비디오 파일(.mp4 등)에서 직접 프레임을 추출하는 데이터셋
    """

    @override
    def _create_processor(self):
        return VideoProcessor(self.config)


class CelebDFImageDataset(CelebDFBaseDataset):
    """
    이미지 파일이 저장된 폴더에서 프레임을 읽어오는 데이터셋
    """

    @override
    def _create_processor(self):
        return ImageProcessor(self.config, self.idx_to_filename)

    @staticmethod
    def idx_to_filename(idx: int) -> str:
        return f"frame_{idx + 1:06d}.jpg"
