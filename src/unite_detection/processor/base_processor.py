from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import torch
import transformers
from jaxtyping import Float, Int
from torch import Tensor
from torchvision.transforms import v2

from unite_detection.schemas import DatasetConfig, FileMeta

if TYPE_CHECKING:
    from transformers import BatchFeature


class BaseProcessor(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        if self.config.encoder.use_auto_processor:
            self.auto_processor = transformers.AutoProcessor.from_pretrained(
                self.config.encoder.model,
                use_fast=True,
            )

    @abstractmethod
    def get_frame_count(self, path: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def calculate_num_chunks(self, frame_cnt: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _load_frames(self, meta: FileMeta) -> Int[Tensor, "batch channel h w"]:
        raise NotImplementedError

    def getitem(
        self, meta: FileMeta
    ) -> tuple[Float[Tensor, "channel batch h w"], Int[Tensor, "batch"]]:  # ty:ignore[invalid-method-override]
        raw_frames_tensor = self._load_frames(meta)
        frames_tensor_resized = cast(
            'Int[Tensor, "batch channel h w"]',
            v2.Resize(self.config.arch.img_size)(raw_frames_tensor),
        )

        transformed_tensor: Float[Tensor, "batch channel h w"]
        if self.config.transform:
            transformed_tensor = self.config.transform(frames_tensor_resized)
        else:
            transformed_tensor = frames_tensor_resized.to(torch.float32)

        processed_tensor: Float[Tensor, "batch channel h w"]
        if self.config.encoder.use_auto_processor:
            assert self.auto_processor is not None
            processed = cast(
                "BatchFeature",
                self.auto_processor(images=transformed_tensor, return_tensors="pt"),
            )
            processed_tensor = cast(
                'Float[Tensor, "batch channel h w"]',
                processed.pixel_values,
            )
        else:
            processed_tensor = transformed_tensor

        # (N, C, H, W) -> (C, N, H, W)
        processed_tensor_out: Float[Tensor, "channel batch h w"] = (
            processed_tensor.permute(1, 0, 2, 3)
        )

        return processed_tensor_out.contiguous(), Tensor(
            meta["label"], dtype=torch.long
        )
