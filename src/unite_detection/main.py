from pathlib import Path
from typing import Annotated, Literal, override

import kagglehub
import lightning.pytorch as L
import torch
import typer
import wandb
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel, Field
from rich import print
from torchvision.transforms import v2

from unite_detection.lit_modules import (
    DFDataModule,
    LitUNITEClassifier,
    VisualizationCallback,
)
from unite_detection.schemas import (
    ArchSchema,
    AugmentationConfig,
    DataModuleConfig,
    EncoderConfig,
    UNITEClassifierConfig,
)


class TrainConfig(BaseModel):
    project_name: str = "UNITE_deepfaek_classification"
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

    @override
    def model_post_init(self, __context):
        self.datamodule.dataset.arch = self.arch
        self.datamodule.dataset.encoder = self.encoder

        self.lit_unite.arch = self.arch
        self.lit_unite.unite_model.encoder = self.encoder


def represent_path(dumper: yaml.SafeDumper, data: Path):
    return dumper.represent_str(str(data))


yaml.add_multi_representer(Path, represent_path, Dumper=yaml.SafeDumper)

app = typer.Typer(no_args_is_help=True)


@app.command(short_help="Initialize config yaml file. Will override current file.")
def init_yaml(path: Path = Path("./example.yaml")):
    default_config = TrainConfig()
    model_dict = default_config.model_dump(
        exclude={"datamodule": {"dataset": {"arch", "encoder"}}, "lit_unite": {"arch"}},
    )
    with open(path, "w") as f:
        yaml.safe_dump(model_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Default config file generated - check {path}.")


@app.command(short_help="Train UNITE model using given yaml file.")
def train(
    config_path: Path = Path("./example.yaml"),
    resume: bool = False,
    run_name: str | None = None,
    fast_dev_run: Annotated[bool, typer.Option("--fast-dev-run")] = False,
):

    print("Logging into wandb and kaggle...")
    wandb.login()
    kagglehub.login()

    model_dict: dict
    with open(config_path) as f:
        model_dict = yaml.safe_load(f)
    config = TrainConfig.model_validate(model_dict)

    transform_list: list[v2.Transform | None] = [
        v2.ToDtype(torch.uint8),
        v2.RandomHorizontalFlip(p=0.5) if config.augment.horizontal_flip else None,
        v2.RandomApply(
            [
                v2.RandomRotation(
                    (-1 * config.augment.rotation_range, config.augment.rotation_range),
                ),
            ],
        )
        if config.augment.rotation
        else None,  # Quite slow?
        v2.RandomApply([v2.GaussianBlur(kernel_size=config.augment.gaussian_kernel)])
        if config.augment.gaussian_blur
        else None,  # Managable...
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=config.augment.color_jitter_brightness,
                    contrast=config.augment.color_jitter_contrast,
                ),
            ],
        )
        if config.augment.color_jitter
        else None,  # Looks slow...
        v2.RandomApply([v2.JPEG(config.augment.jpeg_quality_range)])
        if config.augment.jpeg
        else None,
        v2.ToDtype(torch.float32),
    ]
    transform_list = [x for x in transform_list if x is not None]
    transform = v2.Compose(transform_list)

    config.datamodule.dataset.transform = transform
    datamodule = DFDataModule(config.datamodule)
    lit_classifier = LitUNITEClassifier(config.lit_unite)
    if config.compile and not fast_dev_run:
        lit_classifier = torch.compile(lit_classifier)

    if not resume:
        wandb_logger = WandbLogger(
            project=config.project_name,
            name=run_name,
            log_model=config.wandb_log_model,
        )
    else:
        run_id: str = typer.prompt("Input run ID to resume")
        wandb_logger = WandbLogger(
            project=config.project_name,
            name=run_name,
            log_model=config.wandb_log_model,
            id=run_id,
            resume="must",
        )

    callbacks: list[Callback] = [VisualizationCallback(), LearningRateMonitor()]
    if config.use_ckpt and not fast_dev_run:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.ckpt_path,
                monitor=f"val/{config.ckpt_monitor}",
                mode="max",
                save_last=True,
            ),
        )

    trainer = L.Trainer(
        precision="bf16-mixed" if config.lit_unite.unite_model.use_bfloat else 16,
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.acc_grad,
        fast_dev_run=fast_dev_run,
    )

    if config.wandb_watch and not fast_dev_run:
        wandb_logger.watch(lit_classifier)

    if not resume:
        trainer.fit(lit_classifier, datamodule=datamodule)
    else:
        ckpt: str = typer.prompt("Input checkpoint path")
        trainer.fit(lit_classifier, datamodule=datamodule, ckpt_path=ckpt)

    wandb.finish()


if __name__ == "__main__":
    app()
