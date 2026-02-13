from typing import override

import matplotlib.pyplot as plt
import pytorch_lightning as L
import torch
import wandb

from unite_detection.schemas import PlotContext, Visualizable, VisualizationData
from unite_detection.utils.plots import (
    plot_conf_mat,
    plot_encoder_tsne,
    plot_roc,
    plot_tsne,
)


class VisualizationCallback(L.Callback):
    def _visualize_all(
        self, pl_module: L.LightningModule, data: VisualizationData, stage: str
    ):
        assert isinstance(pl_module, Visualizable)
        ctx = PlotContext(
            class_names=pl_module.class_names,
            stage=stage,
            current_epoch=pl_module.current_epoch,
            num_heads=pl_module.num_heads,
        )
        log_dict: dict[str, wandb.Image] = {}

        fig_roc = plot_roc(data, ctx)
        log_dict[f"{stage}/ROC_curve"] = wandb.Image(fig_roc)
        plt.close(fig_roc)

        fig_cm = plot_conf_mat(data, ctx)
        log_dict[f"{stage}/confusion_matrix"] = wandb.Image(fig_cm)
        plt.close(fig_cm)

        fig_tsne = plot_tsne(data, ctx, max_samples=20000)
        log_dict[f"{stage}/t_sne"] = wandb.Image(fig_tsne)
        plt.close(fig_tsne)

        if not (
            data.ps is None
            or data.cs is None
            or data.ps.size(0) == 0
            or data.cs.size(0) == 0
        ):
            fig_enc_tsne = plot_encoder_tsne(data, ctx, max_samples=400)
            log_dict[f"{stage}/P_C_tsne_plot"] = wandb.Image(fig_enc_tsne)
            plt.close(fig_enc_tsne)

        if pl_module.logger:
            pl_module.logger.experiment.log(log_dict)

    @override
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if trainer.sanity_checking:
            return
        if not isinstance(pl_module, Visualizable):
            return
        data = pl_module.val_output
        if data is None:
            return

        self._visualize_all(pl_module, data, "val")

    @override
    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not isinstance(pl_module, Visualizable):
            return
        data = pl_module.test_output
        if data is None:
            return

        self._visualize_all(pl_module, data, "test")
