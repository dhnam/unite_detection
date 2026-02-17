from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import torch
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from torch import Tensor

if TYPE_CHECKING:
    from jaxtyping import Float


class EncoderConfig(BaseModel):
    model: str = "google/siglip2-base-patch16-384"
    use_auto_processor: bool = True


class FileMeta(TypedDict):
    path: Path
    chunk_idx: int
    label: int
    total_frames: int


class ArchSchema(BaseModel):
    """
    Architecture Schema
    """

    num_cls: int = 2
    num_heads: int = 12
    num_frames: int = 16
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
    warmup_steps: int = 100


class LossConfig(BaseModel):
    lambda_1: float = 0.5
    lambda_2: float = 0.5


class UNITEClassifierConfig(BaseModel):
    arch: ArchSchema = Field(default_factory=ArchSchema)
    unite_model: UNITEConfig = Field(default_factory=UNITEConfig)
    ad_loss: ADLossConfig = Field(default_factory=ADLossConfig)
    optim: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)

    @model_validator(mode="after")
    def sync_internal(self):
        self.unite_model.arch = self.arch
        self.ad_loss.arch = self.arch
        return self


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
    from_img: bool = False
    val_split_ratio: float = 0.1
    use_gta_v: bool = True
    gta_v_preprocess_path: Path = Path("./gta_v_preprocessed")
    gta_v_zip_path: Path = Path("./mini-ref-sailvos.zip")
    gta_v_down_path: Path = Path("./gta_v")
    gta_v_gdrive_id: str = "1-0Vu4X-pqb4Da226g1OALHytAlMRCC84"
    do_preprocess: bool = False
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
            ps=torch.cat([x.ps for x in output], dim=0)
            if output.ps is not None
            else None,
            cs=torch.cat([x.cs for x in output], dim=0)
            if output.cs is not None
            else None,
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


class TrainConfig(BaseModel):
    project_name: str = "UNITE_deepfake_classification"
    use_ckpt: bool = True
    ckpt_path: Path = Path("./checkpoints/")
    ckpt_monitor: Literal[
        "MulticlassAveragePrecision",
        "MulticlassAccuracy",
        "AUROC",
    ] = "MulticlassAveragePrecision"
    compile: bool = True
    wandb_watch: bool = True
    wandb_log_model: bool | Literal["all"] = False
    max_epoch: int = 20
    acc_grad: int = 2
    arch: ArchSchema = Field(default_factory=ArchSchema)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    datamodule: DataModuleConfig = Field(default_factory=DataModuleConfig)
    lit_unite: UNITEClassifierConfig = Field(default_factory=UNITEClassifierConfig)
    augment: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @model_validator(mode="after")
    def sync_internal(self, __context):
        self.datamodule.dataset.arch = self.arch
        self.datamodule.dataset.encoder = self.encoder

        self.lit_unite.arch = self.arch
        self.lit_unite.unite_model.encoder = self.encoder
        return self


@runtime_checkable
class Visualizable(Protocol):
    @property
    def num_cls(self) -> int: ...

    @property
    def class_names(self) -> Sequence[str]: ...

    @property
    def num_heads(self) -> int: ...

    @property
    def val_output(self) -> VisualizationData | None: ...

    @property
    def test_output(self) -> VisualizationData | None: ...


class PlotContext(BaseModel):
    class_names: Sequence[str]
    stage: str
    current_epoch: int
    num_heads: int
