import math
import os
from typing import Callable, cast

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image, read_file
from torchvision.transforms import v2
from transformers import BatchFeature

from unite_detection.dataset import DeepFakeBaseDataset


class ImageProcessor:
    def __init__(self, naming_fn: Callable[[int], str], size: tuple[int, int]):
        self.naming_fn = naming_fn
        self.size = size

    def get_frame_count(self, path: str) -> int:
        try:
            return len(
                [f for f in os.listdir(path) if f.endswith((".jpg", ".png", ".bmp"))]
            )
        except Exception:
            return 0

    def calculate_num_chunks(self, dataset: DeepFakeBaseDataset, frame_cnt: int) -> int:
        return math.ceil(frame_cnt / dataset.config.arch.num_frames)

    def getitem(
        self, dataset: DeepFakeBaseDataset, idx: int
    ) -> tuple[Float[Tensor, "channel batch h w"], Int[Tensor, "batch"]]:
        meta = dataset.samples[idx]
        folder_path, chunk_idx, label, total_frames = (
            meta["path"],
            meta["chunk_idx"],
            meta["label"],
            meta["total_frames"],
        )

        frames_list: list[Float[Tensor, "channel h w"]] = []
        start_frame_idx = chunk_idx * dataset.config.arch.num_frames

        for i in range(dataset.config.arch.num_frames):
            current_idx = min(start_frame_idx + i, total_frames - 1)
            file_name = self.naming_fn(current_idx)
            img_path = os.path.join(folder_path, file_name)

            try:
                img_tensor_raw: Int[Tensor, "channel h w"] = decode_image(
                    read_file(img_path), mode=ImageReadMode.RGB
                )
                img_tensor_resized = v2.Resize(self.size)(img_tensor_raw)
                img_tensor: Float[Tensor, "channel h w"]
                if dataset.config.transform:
                    img_tensor = dataset.config.transform(img_tensor_resized)
                else:
                    img_tensor = img_tensor_resized.to(torch.float32)
                frames_list.append(img_tensor)
            except Exception:
                frames_list.append(
                    torch.zeros(
                        (3, dataset.config.size[1], dataset.config.size[0]),
                        dtype=torch.float32,
                    )
                )

        if frames_list:
            frames_tensor: Float[Tensor, "batch channel h w"] = torch.stack(
                frames_list, dim=0
            )
            if dataset.config.encoder.use_auto_processor:
                assert dataset.preprocessor is not None
                processed = cast(
                    BatchFeature,
                    dataset.preprocessor(images=frames_tensor, return_tensors="pt"),
                )
                frames_tensor = cast(
                    Float[Tensor, "batch channel h w"], processed.pixel_values
                )
            frames_tensor_out: Float[Tensor, "channel batch h w"] = (
                frames_tensor.permute(1, 0, 2, 3)
            )
        else:
            frames_tensor_out = torch.zeros(
                (
                    3,
                    dataset.config.arch.num_frames,
                    dataset.config.size[1],
                    dataset.config.size[0],
                ),
                dtype=torch.float32,
            )

        return frames_tensor_out.contiguous(), torch.tensor(label)
