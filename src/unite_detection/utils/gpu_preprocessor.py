from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torchvision.transforms import v2
from transformers import Siglip2ImageProcessor, Siglip2Processor


class GPUSigLIPProcessor:
    def __init__(self, processor: Siglip2Processor):
        config = cast(Siglip2ImageProcessor, processor.image_processor)  # type: ignore

        # 1. 리사이즈 설정: Bilinear + Antialias=True가 핵심
        # Fast 프로세서가 텐서를 처리할 때 사용하는 로직과 일치시킵니다.
        self.resize = v2.Resize(
            size=(config.size["height"], config.size["width"]),  # ty:ignore[unresolved-attribute]
            interpolation=v2.InterpolationMode.BILINEAR,  # resample=2
            antialias=True,  # 오차를 줄이는 가장 중요한 설정
        )

        # 2. 정규화 설정
        # (x - 0.5) / 0.5 연산
        self.mean: Int[Tensor, "1 3 1 1"] = torch.tensor(config.image_mean).view(
            1, 3, 1, 1
        )
        self.std: Int[Tensor, "1 3 1 1"] = torch.tensor(config.image_std).view(
            1, 3, 1, 1
        )
        self.rescale_factor = config.rescale_factor

    def __call__(
        self, video_tensor: Float[Tensor, "batch 3 frame H W"]
    ) -> Float[Tensor, "batch_frame 3 h w"]:
        """
        video_tensor: (batch, 3, frame, H, W), float32, GPU
        """
        b, c, t, h, w = video_tensor.shape
        device = video_tensor.device

        # 차원 변경 (B*T, C, H, W)
        x: Float[Tensor, "batch frame 3 H W"] = video_tensor.permute(0, 2, 1, 3, 4)
        x: Float[Tensor, "batch_frame 3 H W"] = x.flatten(0, 1)

        # [Step 1] Resize (uint8 상태에서 수행하거나 float32에서 수행)
        # torchvision v2는 uint8 입력을 받아 내부적으로 고정밀 연산을 수행합니다.
        x: Float[Tensor, "batch_frame 3 h w"] = self.resize(x)

        # [Step 2] Rescale (0~255 -> 0~1)
        x: Float[Tensor, "batch_frame 3 h w"] = x * self.rescale_factor

        # [Step 3] Normalize (x - 0.5) / 0.5
        # mean, std를 캐싱하여 속도 최적화
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        x = (x - self.mean) / self.std

        # [Step 4] 최종 모델 입력형태인 float16으로 반환
        return x.to(torch.float16)
