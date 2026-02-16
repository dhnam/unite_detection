import math
from typing import override

import cv2
from jaxtyping import Int
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from unite_detection.schemas import FileMeta

from .base_processor import BaseProcessor


class VideoProcessor(BaseProcessor):
    @override
    def get_frame_count(self, path: str) -> int:
        cap = cv2.VideoCapture(path)
        cnt = 0
        if cap.isOpened():
            cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return cnt

    @override
    def calculate_num_chunks(self, frame_cnt: int) -> int:
        # Stride=2를 고려한 유효 프레임 기반 계산
        effective_frames = math.ceil(frame_cnt / 2)
        return math.ceil(effective_frames / self.config.arch.num_frames)

    @override
    def _load_frames(self, meta: FileMeta) -> Int[Tensor, "batch channel h w"]:
        video_path, chunk_idx = meta["path"], meta["chunk_idx"]

        decoder = VideoDecoder(video_path, device=self.config.video_decode_device)
        total_frames = decoder.metadata.num_frames or 100000

        # Stride 2 적용하여 인덱스 계산
        start_frame = chunk_idx * self.config.arch.num_frames * 2
        indices = [
            min(start_frame + (i * 2), total_frames - 1)
            for i in range(self.config.arch.num_frames)
        ]

        frames_batch = decoder.get_frames_at(indices=indices)
        pixel_value_raw: Int[Tensor, "batch channel h w"] = frames_batch.data
        return pixel_value_raw
