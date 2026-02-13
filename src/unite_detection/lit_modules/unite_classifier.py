from typing import Literal, NotRequired, TypedDict, cast, override


import lightning.pytorch as L
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    Precision,
    Recall,
)
from unite_detection.schemas import VisualizationData, Visualizable

from unite_detection.losses import ADLoss
from unite_detection.models import UNITE
from unite_detection.schemas import UNITEClassifierConfig, UNITEOutput


class LitUNITEClassifier(L.LightningModule, Visualizable):
    def __init__(self, config: UNITEClassifierConfig | None = None):
        super().__init__()
        self.config = config or UNITEClassifierConfig()
        self.save_hyperparameters(
            self.config.model_dump(
                exclude={
                    "unite_model": {"arch"},
                    "ad_loss": {"arch"},
                }
            )
        )

        self.model: UNITE = UNITE(self.config.unite_model)
        self.ce_loss: nn.Module = nn.CrossEntropyLoss()
        self.ad_loss: ADLoss = ADLoss(self.config.ad_loss)

        class MetricType(TypedDict):
            task: Literal["multiclass", "binary"]
            num_classes: NotRequired[int]
            average: NotRequired[Literal["macro"]]

        metric_param: MetricType = {
            "task": "multiclass",
            "num_classes": self.config.arch.num_cls,
            "average": "macro",
        }

        metrics = MetricCollection(
            [
                Accuracy(**metric_param),
                AveragePrecision(**metric_param),
                Precision(**metric_param),
                Recall(**metric_param),
                AUROC(**metric_param),
            ]
        )

        self.val_metrics: MetricCollection = metrics.clone(prefix="val/")
        self.test_metrics: MetricCollection = metrics.clone(prefix="test/")

        self.class_names: list[str] = (
            ["Real", "Fake"]
            if self.config.arch.num_cls == 2
            else [f"Class {i}" for i in range(self.config.arch.num_cls)]
        )
        self.val_output: VisualizationData | None = None
        self.test_output: VisualizationData | None = None
        self.num_heads:int = self.config.arch.num_heads
        
        self._val_buffer: list[VisualizationData] = []
        self._test_buffer: list[VisualizationData] = []

    @override
    def forward(self, x):  # pyright: ignore[reportAny, reportUnknownParameterType, reportMissingParameterType]
        return self.model(x)  # pyright: ignore[reportAny]

    @override
    def on_save_checkpoint(self, checkpoint):  # pyright: ignore[reportMissingParameterType]
        """체크포인트 저장 시 Frozen 파라미터(Backbone)를 제외합니다."""
        state_dict = checkpoint["state_dict"]  # pyright: ignore[reportAny]

        # 저장하지 않을 키 필터링 (vis_encoder 관련 키들 제거)
        # 키 이름은 모델 구조에 따라 'model.vis_encoder.'로 시작합니다.
        keys_to_remove = [k for k in state_dict.keys() if "model.vis_encoder" in k]  # pyright: ignore[reportAny]

        for k in keys_to_remove:  # pyright: ignore[reportAny]
            del state_dict[k]

    @override
    def on_load_checkpoint(self, checkpoint):  # pyright: ignore[reportMissingParameterType]
        """로드 시 체크포인트에 없는 백본 가중치를 현재 모델에서 복사해서 채워줌"""
        state_dict = checkpoint["state_dict"]  # pyright: ignore[reportAny]
        model_state_dict = self.state_dict()

        # 현재 모델(self.model.vis_encoder)은 이미 __init__에서
        # from_pretrained로 로드된 상태입니다.
        # 체크포인트에 없는 키(백본 가중치)들을 모델의 현재 가중치로 채워줍니다.
        for key in model_state_dict:
            if key not in state_dict:
                state_dict[key] = model_state_dict[key]

    @override
    def training_step(
        self,
        batch: tuple[Float[Tensor, "batch channel frame h w"], Float[Tensor, "batch"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        x, y = batch
        logit, P, _ = cast(UNITEOutput, self.model(x, return_ad_param=True))
        assert P is not None
        loss_ad, within, between, head_dist_mean = cast(
            tuple[
                Float[Tensor, ""],
                Float[Tensor, ""],
                Float[Tensor, ""],
                Float[Tensor, ""],
            ],
            self.ad_loss(P, y, log_detail=True),
        )
        loss_ce = cast(Float[Tensor, ""], self.ce_loss(logit, y))
        loss = loss_ce * self.config.loss.lambda_1 + loss_ad * self.config.loss.lambda_2
        c_magnitude = cast(
            Float[Tensor, ""],
            torch.norm(self.ad_loss.C, p=2, dim=-1).mean(),  # pyright: ignore[reportUnknownMemberType]
        )
        self.log_dict(
            {
                "train/loss_ad": loss_ad,
                "train/loss_ad/loss_within": within,
                "train/loss_ad/loss_between": between,
                "train/head_dist_mean": head_dist_mean,
                "train/loss_ce": loss_ce,
                "train/loss": loss,
                "train/C_magnitude": c_magnitude,
            },
            logger=True,
        )

        return loss

    @override
    def on_validation_epoch_start(self):
        self._val_buffer = []

    @override
    def validation_step(
        self,
        batch: tuple[Float[Tensor, "batch channel frame h w"], Float[Tensor, "batch"]],
        batch_idx: int,
    ):
        x, y = batch
        logit, P, embed = cast(
            UNITEOutput, self.model(x, return_ad_param=True, return_embed=True)
        )
        assert P is not None and embed is not None
        loss_ad = cast(Float[Tensor, ""], self.ad_loss(P, y))
        loss_ce = cast(Float[Tensor, ""], self.ce_loss(logit, y))
        loss = loss_ce * self.config.loss.lambda_1 + loss_ad * self.config.loss.lambda_2
        self.log("val/loss_ad", loss_ad, logger=True)
        self.log("val/loss_ce", loss_ce, logger=True)
        self.log("val/loss", loss, prog_bar=True, logger=True)

        self.val_metrics.update(logit, y)
        self.log_dict(self.val_metrics, logger=True)

        self._val_buffer.append(VisualizationData(
            logits=logit.detach().cpu(),
            labels=y.detach().cpu(),
            embeds=embed.detach().cpu(),
            ps=F.normalize(P, p=2, dim=2).detach().cpu(),
            cs=self.ad_loss.C.detach().cpu(),
        ))

    @override
    def on_validation_epoch_end(self):
        # 모든 배치 데이터 병합
        self.val_output = VisualizationData.from_step_output(self._val_buffer)
        self._val_buffer.clear()

    @override
    def on_test_epoch_start(self):
        self._test_buffer = []

    @override
    def test_step(
        self,
        batch: tuple[Float[Tensor, "batch channel frame h w"], Float[Tensor, "batch"]],
        batch_idx: int,
    ):
        x, y = batch
        logit, _, embed = cast(UNITEOutput, self.model(x, return_embed=True))
        assert embed is not None

        self._test_buffer.append(VisualizationData(
            logits=logit.detach().cpu(),
            labels=y.detach().cpu(),
            embeds=embed.detach().cpu(),
        ))

        self.test_metrics.update(logit, y)
        self.log_dict(self.test_metrics, logger=True)


    @override
    def on_test_epoch_end(self):
        self.test_output = VisualizationData.from_step_output(self._test_buffer)
        self._test_buffer.clear()

    @override
    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.config.optim.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.config.optim.decay_steps, gamma=0.5
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
