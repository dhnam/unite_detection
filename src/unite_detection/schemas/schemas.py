from typing import NamedTuple, Callable

from jaxtyping import Float
from pydantic import BaseModel, Field, model_validator
from torch import Tensor


class UNITEConfig(BaseModel):
    num_channel: int = 3
    num_cls: int = 2
    num_heads: int = 12
    num_frames: int = 32
    dropout: float = 0.1
    encoder_model: str = "google/siglip2-base-patch16-384"
    use_bfloat: bool = True
    cpu_preprocess: bool = True


class UNITEOutput(NamedTuple):
    res: Float[Tensor, "batch cls"]
    ad_param: Float[Tensor, "batch head frame"] | None = None
    embed: Float[Tensor, "batch embed"] | None = None


class ADLossConfig(BaseModel):
    num_cls: int = 2
    num_heads: int = 12
    num_frames: int = 32
    delta_within: tuple[float, float] = (0.01, -2.0)
    delta_between: float = 0.5
    eta: float = 0.05


class OptimizerConfig(BaseModel):
    lr: float = 1e-4
    decay_steps: int = 1000


class LossConfig(BaseModel):
    lambda_1: float = 0.5
    lambda_2: float = 0.5


class UNITEClassifierConfig(BaseModel):
    num_cls: int = 2
    num_heads: int = 12
    num_frames: int = 32
    unite_model: UNITEConfig = Field(default_factory=UNITEConfig)
    ad_loss: ADLossConfig = Field(default_factory=ADLossConfig)
    optim: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)

    @model_validator(mode="after")
    def ensure_param(self):
        self.unite_model.num_cls = self.num_cls
        self.unite_model.num_heads = self.num_heads
        self.unite_model.num_frames = self.num_frames

        self.ad_loss.num_cls = self.num_cls
        self.ad_loss.num_heads = self.num_heads
        self.ad_loss.num_frames = self.num_frames

        return self


class DatasetConfig(BaseModel):
    num_frames: int = 32
    size: tuple[int, int] = (384, 384)
    device: str = 'cpu'
    encoder_model: str = "google/siglip2-base-patch16-384"
    cpu_preprocess: bool = True
    transform: Callable | None = None
