import math
from abc import ABC
from pathlib import Path
from typing import Sequence, cast

import cv2
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from transformers import BatchFeature

from unite_detection.dataset import DeepFakeBaseDataset, ImageProcessor
from unite_detection.schemas import DatasetConfig


class CelebDFBaseDataset(DeepFakeBaseDataset, ABC):
    """
    Celeb-DF 데이터셋의 공통 기능을 담은 부모 클래스
    """

    def _get_label(self, path: str) -> int | None:
        """폴더명 기반 레이블 결정"""
        rel_path = path.replace("\\", "/")
        if "YouTube-real" in rel_path or "Celeb-real" in rel_path:
            return 0
        elif "Celeb-synthesis" in rel_path:
            return 1
        return None


class CelebDFVideoDataset(CelebDFBaseDataset):
    """
    비디오 파일(.mp4 등)에서 직접 프레임을 추출하는 데이터셋
    """

    def _get_frame_count(self, path: str) -> int:
        cap = cv2.VideoCapture(path)
        cnt = 0
        if cap.isOpened():
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return cnt

    def _calculate_num_chunks(self, frame_cnt: int) -> int:
        # Stride=2를 고려한 유효 프레임 기반 계산
        effective_frames = math.ceil(frame_cnt / 2)
        return math.ceil(effective_frames / self.config.num_frames)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        video_path, chunk_idx, label = meta["path"], meta["chunk_idx"], meta["label"]

        try:
            decoder = VideoDecoder(video_path, device="cpu")
            total_frames = decoder.metadata.num_frames or 100000

            # Stride 2 적용하여 인덱스 계산
            start_frame = chunk_idx * self.config.num_frames * 2
            indices = [
                min(start_frame + (i * 2), total_frames - 1)
                for i in range(self.config.num_frames)
            ]

            frames_batch = decoder.get_frames_at(indices=indices)
            pixel_value_raw: Int[Tensor, "batch channel h w"] = frames_batch.data

            pixel_value: Float[Tensor, "batch channel h w"]
            if self.config.transform:
                # Video transform이 필요한 경우 여기서 처리 (Batch 단위 지원 필요)
                pixel_value = cast(
                    Float[Tensor, "batch channel h w"],
                    self.config.transform(pixel_value_raw),
                )
            else:
                pixel_value = pixel_value_raw.to(torch.float32)

            frames_tensor: Float[Tensor, "batch channel h w"]
            if self.config.encoder.use_auto_processer:
                assert self.preprocessor is not None
                processed = cast(
                    BatchFeature,
                    self.preprocessor(images=pixel_value, return_tensors="pt"),
                )
                frames_tensor = cast(
                    Float[Tensor, "batch channel h w"], processed.pixel_values
                )
            else:
                frames_tensor = pixel_value

            # (N, C, H, W) -> (C, N, H, W)
            frames_tensor_out: Float[Tensor, "channel batch h w"] = (
                frames_tensor.permute(1, 0, 2, 3)
            )

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            frames_tensor_out: Float[Tensor, "channel batch h w"]
            frames_tensor_out = torch.zeros(
                (3, self.config.num_frames, self.config.size[1], self.config.size[0]),
                dtype=torch.float32,
            )

        return frames_tensor_out.contiguous(), torch.tensor(label)


class CelebDFImageDataset(CelebDFBaseDataset):
    """
    이미지 파일이 저장된 폴더에서 프레임을 읽어오는 데이터셋
    """

    def __init__(
        self, paths: Sequence[Path | str], config: DatasetConfig | None = None
    ):
        super().__init__(paths, config)
        self.processor = ImageProcessor(
            CelebDFImageDataset.idx_to_filename, self.config.size
        )

    @staticmethod
    def idx_to_filename(idx: int) -> str:
        return f"frame_{idx + 1:06d}.jpg"

    def _get_frame_count(self, path: str) -> int:
        return self.processor.get_frame_count(path)

    def _calculate_num_chunks(self, frame_cnt: int) -> int:
        return self.processor.calculate_num_chunks(self, frame_cnt)

    def __getitem__(self, idx):
        return self.processor.getitem(self, idx)
