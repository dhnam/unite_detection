from typing import Literal, NotRequired, TypedDict, cast, override


import lightning.pytorch as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Float
from sklearn.manifold import TSNE  # pyright: ignore[reportMissingTypeStubs]
from torch import Tensor, nn
from torchmetrics import ConfusionMatrix, MetricCollection
from torchmetrics.classification import (
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    Precision,
    Recall,
)
from torchmetrics.metric import Metric

from unite_detection.losses import ADLoss
from unite_detection.models import UNITE
from unite_detection.schemas import UNITEClassifierConfig, UNITEOutput


class LitUNITEClassifier(L.LightningModule):
    def __init__(self, config: UNITEClassifierConfig | None = None):
        super().__init__()
        if config is None:
            config = UNITEClassifierConfig()
        self.save_hyperparameters(
            config.model_dump(
                exclude={
                    "unite_model": {"num_cls", "num_heads", "num_frames"},
                    "ad_loss": {"num_cls", "num_heads", "num_frames"},
                }
            )
        )

        self.config: UNITEClassifierConfig = config
        self.model: UNITE = UNITE(config.unite_model)
        self.ce_loss: nn.Module = nn.CrossEntropyLoss()
        self.ad_loss: ADLoss = ADLoss(config.ad_loss)

        MetricType = TypedDict(
            "MetricType",
            {
                "task": Literal["multiclass"] | Literal["binary"],
                "num_classes": NotRequired[int],
                "average": NotRequired[Literal["macro"]],
            },
        )

        metric_param: MetricType = {
            "task": "multiclass",
            "num_classes": config.num_cls,
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
        self.val_roc: Metric = ROC(**metric_param)
        self.test_metrics: MetricCollection = metrics.clone(prefix="test/")
        self.test_roc: Metric = ROC(**metric_param)

        self.confmat: Metric = ConfusionMatrix(
            task="multiclass", num_classes=config.num_cls
        )

        self.class_names: list[str] = (
            ["Real", "Fake"]
            if config.num_cls == 2
            else [f"Class {i}" for i in range(config.num_cls)]
        )
        self.val_preds: list[Tensor] = []
        self.val_labels: list[Tensor] = []
        self.val_embeds: list[Tensor] = []
        self.val_ps: list[Tensor] = []
        self.test_preds: list[Tensor] = []
        self.test_labels: list[Tensor] = []
        self.test_embeds: list[Tensor] = []

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
        self.val_preds = []
        self.val_labels = []
        self.val_embeds = []
        self.val_ps = []

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

        # 데이터 수집 (CPU로 이동하여 메모리 절약)
        preds = torch.argmax(logit, dim=1)
        self.val_roc.update(logit, y)  # ty:ignore[invalid-argument-type]

        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        self.val_embeds.append(embed.detach().cpu())  # [Batch, 768]
        self.val_ps.append(F.normalize(P, p=2, dim=2).detach().cpu())

    @override
    def on_validation_epoch_end(self):
        # 모든 배치 데이터 병합
        all_preds = torch.cat(self.val_preds)
        all_labels = torch.cat(self.val_labels)
        all_embeds = torch.cat(self.val_embeds)
        all_ps = torch.cat(self.val_ps)

        # 1. Confusion Matrix 시각화 및 로그
        self._log_confusion_matrix(all_preds, all_labels, stage="val")

        # 2. t-SNE 시각화 및 로그
        self._log_tsne(all_embeds, all_labels, predicts=all_preds, stage="val")

        fig, ax = self.val_roc.plot()  # pyright: ignore[reportAny]
        if self.logger:
            self.logger.experiment.log({"val/ROC_curve": wandb.Image(fig)})  # ty:ignore[unresolved-attribute]
        self.val_roc.reset()
        plt.close(fig)  # pyright: ignore[reportAny]

        # 3. C heatmap
        for cls in range(self.config.num_cls):
            cls_C_norm = F.normalize(self.ad_loss.C[cls], p=2, dim=-1)
            sim_matrix = torch.mm(cls_C_norm, cls_C_norm.t())
            heatmap = go.Heatmap(
                x=[f"Head_{i}" for i in range(self.config.num_heads)],
                y=[f"Head_{i}" for i in range(self.config.num_heads)],
                z=sim_matrix.cpu().detach().numpy(),
            )
            fig = go.Figure([heatmap])

            self.logger.experiment.log(  # ty:ignore[unresolved-attribute]
                {f"val/center_sim_mat_{self.class_names[cls]}": wandb.Plotly(fig)}
            )

        # 4. t-SNE of P and C
        # 데이터 준비
        tsne_embeds: list[np.ndarray] = []
        tsne_labels: list[str] = []  # Real/Fake 구분용
        tsne_heads: list[int] = []  # Head ID 구분용 (색상)
        tsne_types: list[
            Literal["Center", "Sample"]
        ] = []  # Center vs Sample 구분용 (크기)

        # [Centers 추출]
        for c_idx in range(self.config.num_cls):
            for h_idx in range(self.config.num_heads):
                c_vec = self.ad_loss.C[c_idx, h_idx].detach().cpu().numpy()
                tsne_embeds.append(c_vec)
                tsne_labels.append(self.class_names[c_idx])  # "Real" or "Fake"
                tsne_heads.append(h_idx)
                tsne_types.append("Center")

        # [Samples (P) 추출 및 샘플링]
        num_total_samples = all_ps.shape[0]
        sample_size = min(
            num_total_samples, 400
        )  # 브라우저 부하 방지를 위해 적절히 조절
        indices = torch.randperm(num_total_samples)[:sample_size]

        for i in indices:
            label_name = self.class_names[int(all_labels[i].item())]
            for h_idx in range(self.config.num_heads):
                p_vec = all_ps[i, h_idx].detach().cpu().numpy()
                tsne_embeds.append(p_vec)
                tsne_labels.append(label_name)
                tsne_heads.append(h_idx)
                tsne_types.append("Sample")

        # t-SNE 실행
        tsne_embeds_arr = np.array(tsne_embeds)
        tsne_2d = cast(
            np.ndarray,
            TSNE(n_components=2, random_state=42).fit_transform(tsne_embeds_arr),
        )

        # 시각화 시작
        fig, ax = plt.subplots(figsize=(10, 8))

        # 설정: Head별 색상, Class별 마커
        cmap = sns.color_palette("husl", self.config.num_heads)
        marker_map = {
            self.class_names[0]: "o",
            self.class_names[1]: "X",
        }  # 0: Real(Circle), 1: Fake(X)

        for i in range(len(tsne_2d)):
            h_idx = tsne_heads[i]
            label = tsne_labels[i]
            m_type = tsne_types[i]

            color = cmap[h_idx]
            marker = marker_map.get(label, "d")
            size = 150 if m_type == "Center" else 50
            alpha = 1.0 if m_type == "Center" else 0.4
            edge = "black" if m_type == "Center" else "none"
            zorder = 3 if m_type == "Center" else 2

            _ = ax.scatter(
                tsne_2d[i, 0],
                tsne_2d[i, 1],
                c=[color],
                marker=marker,
                s=size,
                alpha=alpha,
                edgecolors=edge,
                linewidths=1.5,
                zorder=zorder,
            )

        # 범례 설정 (Custom Legend)
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Real",
                markerfacecolor="gray",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                label="Fake",
                markerfacecolor="gray",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Center",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=12,
            ),
        ]
        # Head별 색상 범례 추가
        for h in range(self.config.num_heads):
            legend_elements.append(
                Line2D([0], [0], color=cmap[h], lw=3, label=f"Head {h}")
            )

        _ = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
        _ = ax.set_title(f"t-SNE of P and C (Epoch {self.current_epoch})")
        plt.tight_layout()

        # WandB 로그
        if self.logger:
            self.logger.experiment.log({"val/P_C_tsne_plot": wandb.Image(fig)})  # ty:ignore[unresolved-attribute]

        plt.close(fig)

        # 메모리 해제 및 초기화
        self.val_preds.clear()
        self.val_labels.clear()
        self.val_embeds.clear()
        self.val_ps.clear()  # 추가됨
        if hasattr(self, "val_roc"):
            self.val_roc.reset()

    @override
    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_labels = []
        self.test_embeds = []

    @override
    def test_step(
        self,
        batch: tuple[Float[Tensor, "batch channel frame h w"], Float[Tensor, "batch"]],
        batch_idx: int,
    ):
        x, y = batch
        logit, _, embed = cast(UNITEOutput, self.model(x, return_embed=True))
        assert embed is not None
        preds = torch.argmax(logit, dim=1)

        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(y.detach().cpu())
        self.test_embeds.append(embed.detach().cpu())

        self.test_metrics.update(logit, y)
        self.log_dict(self.test_metrics, logger=True)

        self.test_roc.update(logit, y)  # ty:ignore[invalid-argument-type]

    @override
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)
        all_embeds = torch.cat(self.test_embeds)

        # 1. Confusion Matrix
        self._log_confusion_matrix(all_preds, all_labels, stage="test")

        # 2. t-SNE
        self._log_tsne(all_embeds, all_labels, predicts=all_preds, stage="test")

        fig, _ = self.test_roc.plot()
        if self.logger:
            self.logger.experiment.log({"test/ROC_curve": wandb.Image(fig)})  # ty:ignore[unresolved-attribute]

        plt.close(fig)
        self.test_roc.reset()

        self.test_preds.clear()
        self.test_labels.clear()
        self.test_embeds.clear()

    def _log_confusion_matrix(
        self,
        preds: Float[Tensor, "total"],
        labels: Float[Tensor, "total"],
        stage: str = "val",
    ):
        """Confusion Matrix를 그리고 WandB에 로그합니다."""
        conf_matrix = cast(
            Tensor, self.confmat(preds.to(self.device), labels.to(self.device))
        )

        fig = plt.figure(figsize=(8, 6))
        _ = sns.heatmap(
            conf_matrix.cpu().numpy(),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        _ = plt.xlabel("Predicted")
        _ = plt.ylabel("True")
        _ = plt.title(f"{stage.capitalize()} Confusion Matrix")

        # WandB 로그
        if self.logger:
            self.logger.experiment.log({f"{stage}/confusion_matrix": wandb.Image(fig)})  # ty:ignore[unresolved-attribute]

        plt.close(fig)

    def _log_tsne(
        self,
        embeds: Tensor,
        labels: Tensor,
        predicts: Tensor | None = None,
        stage: str = "val",
        max_samples: int = 2000,
    ):
        """t-SNE를 계산하고 WandB에 로그합니다. 데이터가 많으면 샘플링합니다."""

        # 데이터가 너무 많으면 t-SNE 속도가 매우 느려지므로 샘플링
        num_samples = embeds.shape[0]
        if num_samples > max_samples:
            indices = torch.randperm(num_samples)[:max_samples]
            embeds = embeds[indices]
            labels = labels[indices]
            if predicts is not None:
                predicts = predicts[indices]

        # 텐서를 numpy로 변환
        X = embeds.numpy()
        y = labels.numpy()

        # t-SNE 계산
        # perplexity는 데이터 수보다 작아야 함 (보통 30~50)
        n_components = 2
        perplexity = min(30, num_samples - 1)
        if perplexity < 5:
            return  # 데이터가 너무 적으면 패스

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42,
        )
        X_embedded = cast(np.ndarray, tsne.fit_transform(X))

        # DataFrame 생성 (시각화 용이성 위해)
        df_tsne = pd.DataFrame(X_embedded, columns=["x", "y"])
        df_tsne["label"] = [self.class_names[i] for i in y]

        style_col = None
        if predicts is not None:
            y_pred = predicts
            df_tsne["prediction"] = [self.class_names[i] for i in y_pred]
            style_col = "prediction"

        # Plotting
        fig = plt.figure(figsize=(15, 12))
        sns.scatterplot(
            data=df_tsne,
            x="x",
            y="y",
            hue="label",
            style=style_col,
            palette=["blue", "red"],  # Real: Blue, Fake: Red
            markers={"Real": "o", "Fake": "X"},
            alpha=0.7,
            s=100,
        )
        plt.title(f"{stage.capitalize()} t-SNE Visualization")
        plt.grid(True, linestyle="--", alpha=0.3)

        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout()

        # WandB 로그
        if self.logger:
            self.logger.experiment.log({f"{stage}/t_sne": wandb.Image(fig)})  # ty:ignore[unresolved-attribute]
        plt.close(fig)

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
