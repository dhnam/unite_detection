from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Protocol, Sequence, runtime_checkable

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch import Tensor


class EncoderConfig(BaseModel):
    model: str = "google/siglip2-base-patch16-384"
    use_auto_processor: bool = True


class ArchSchema(BaseModel):
    """
    Architecture Schema
    """

    num_cls: int = 2
    num_heads: int = 12
    num_frames: int = 32
    img_size: int = 384


class UNITEConfig(BaseModel):
    dropout: float = 0.1
    use_bfloat: bool = True
    arch: ArchSchema = Field(default_factory=ArchSchema, exclude=True)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig, exclude=True)


class UNITEOutput(NamedTuple):
    res: Float[Tensor, "batch cls"]
    ad_param: Float[Tensor, "batch head frame"] | None = None
    embed: Float[Tensor, "batch embed"] | None = None


class ADLossConfig(BaseModel):
    delta_within: tuple[float, float] = (0.01, -2.0)
    delta_between: float = 0.5
    eta: float = 0.05
    arch: ArchSchema = Field(default_factory=ArchSchema, exclude=True)


class OptimizerConfig(BaseModel):
    lr: float = 1e-4
    decay_steps: int = 1000


class LossConfig(BaseModel):
    lambda_1: float = 0.5
    lambda_2: float = 0.5


class UNITEClassifierConfig(BaseModel):
    arch: ArchSchema = Field(default_factory=ArchSchema)
    unite_model: UNITEConfig = Field(default_factory=UNITEConfig)
    ad_loss: ADLossConfig = Field(default_factory=ADLossConfig)
    optim: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)

    def model_post_init(self, __context):
        self.unite_model.arch = self.arch
        self.ad_loss.arch = self.arch


class DatasetConfig(BaseModel):
    video_decode_device: str = "cpu"
    transform: Callable | None = Field(None, exclude=True)
    arch: ArchSchema = Field(default_factory=ArchSchema)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)


class DataLoaderConfig(BaseModel):
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int | None = 1
    pin_memory: bool = True
    persistent_workers: bool = True


class SamplerConfig(BaseModel):
    real_weight: float = 0.4
    fake_weight: float = 0.35
    gta_weight: float = 0.25
    seed: int | None = None
    run_sample: int = 20000


class DataModuleConfig(BaseModel):
    celeb_df_preprocess_path: Path = Path("./celeb_preprocessed")
    from_img: bool = True
    val_split_ratio: float = 0.1
    use_gta_v: bool = True
    gta_v_preprocess_path: Path = Path("./gta_v_preprocessed")
    gta_v_zip_path: Path = Path("./mini-ref-sailvos.zip")
    gta_v_down_path: Path = Path("./gta_v")
    gta_v_gdrive_id: str = "1-0Vu4X-pqb4Da226g1OALHytAlMRCC84"
    do_preprocess: bool = True
    loader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)


class VisualizationData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: Tensor
    labels: Tensor
    embeds: Tensor
    ps: Tensor | None = None
    cs: Tensor | None = None

    @computed_field
    @property
    def preds(self) -> Tensor:
        return torch.argmax(self.logits, dim=1)

    @classmethod
    def from_step_output(cls, output: list[VisualizationData]) -> VisualizationData:
        if not output:
            return cls(
                logits=torch.empty(0),
                labels=torch.empty(0),
                embeds=torch.empty(0),
            )

        return cls(
            logits=torch.cat([x.logits for x in output], dim=0),
            labels=torch.cat([x.labels for x in output], dim=0),
            embeds=torch.cat([x.embeds for x in output], dim=0),
            ps=torch.cat([x.ps for x in output if x.ps is not None], dim=0),
            cs=torch.cat([x.cs for x in output if x.cs is not None], dim=0),
        )


class AugmentationConfig(BaseModel):
    horizontal_flip: bool = True
    rotation: bool = False
    rotation_range: int = 10
    gaussian_blur: bool = True
    gaussian_kernel: tuple[int, int] = (3, 7)
    color_jitter: bool = False
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    jpeg: bool = True
    jpeg_quality_range: tuple[int, int] = (60, 100)

@runtime_checkable
class Visualizable(Protocol):
    num_cls: int
    class_names: Sequence[str]
    num_heads: int
    val_output: VisualizationData | None
    test_output: VisualizationData | None


class PlotContext(BaseModel):
    class_names: Sequence[str]
    stage: str
    current_epoch: int
    num_heads: int
