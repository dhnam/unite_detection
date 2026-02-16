import math
import os
from collections.abc import Callable
from typing import override

import torch
from jaxtyping import Int
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image, read_file

from unite_detection.schemas import DatasetConfig, FileMeta

from .base_processor import BaseProcessor


class ImageProcessor(BaseProcessor):
    def __init__(self, config: DatasetConfig, naming_fn: Callable[[int], str]):
        super().__init__(config)
        self.naming_fn = naming_fn

    @override
    def get_frame_count(self, path: str) -> int:
        try:
            return len(
                [f for f in os.listdir(path) if f.endswith((".jpg", ".png", ".bmp"))],
            )
        except Exception:
            return 0

    @override
    def calculate_num_chunks(self, frame_cnt: int) -> int:
        return math.ceil(frame_cnt / self.config.arch.num_frames)

    @override
    def _load_frames(self, meta: FileMeta) -> Int[Tensor, "batch channel h w"]:
        folder_path, chunk_idx, total_frames = (
            meta["path"],
            meta["chunk_idx"],
            meta["total_frames"],
        )
        frames_list: list[Int[Tensor, "channel h w"]] = []
        start_frame_idx = chunk_idx * self.config.arch.num_frames

        for i in range(self.config.arch.num_frames):
            current_idx = min(start_frame_idx + i, total_frames - 1)
            file_name = self.naming_fn(current_idx)
            img_path = os.path.join(folder_path, file_name)

            try:
                img_tensor_raw: Int[Tensor, "channel h w"] = decode_image(
                    read_file(img_path),
                    mode=ImageReadMode.RGB,
                )
                frames_list.append(img_tensor_raw)

            except Exception:
                if len(frames_list) > 0:
                    frames_list.append(frames_list[-1])
                else:
                    frames_list.append(
                        torch.zeros(
                            (
                                3,
                                self.config.arch.img_size,
                                self.config.arch.img_size,
                            ),
                            dtype=torch.uint8,
                        ),
                    )

        frames_tensor: Int[Tensor, "batch channel h w"] = torch.stack(
            frames_list,
            dim=0,
        )
        return frames_tensor
