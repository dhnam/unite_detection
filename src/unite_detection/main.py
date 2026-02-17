from pathlib import Path, PosixPath
from typing import Annotated

import kagglehub
import lightning.pytorch as L
import torch
import typer
import wandb
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from rich import print
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from unite_detection.dataset import CustomVideoDataset
from unite_detection.lit_modules import (
    DFDataModule,
    LitUNITEClassifier,
    VisualizationCallback,
)
from unite_detection.schemas import TrainConfig


def represent_path(dumper: yaml.SafeDumper, data: Path):
    return dumper.represent_str(str(data))


yaml.add_multi_representer(Path, represent_path, Dumper=yaml.SafeDumper)

app = typer.Typer(no_args_is_help=True)


@app.command(short_help="Initialize config yaml file. Will override current file.")
def init_yaml(
    path: Annotated[
        Path, typer.Option(exists=False, file_okay=True, dir_okay=False)
    ] = Path("./example.yaml"),
):
    default_config = TrainConfig()
    model_dict = default_config.model_dump(
        exclude={"datamodule": {"dataset": {"arch", "encoder"}}, "lit_unite": {"arch"}},
    )
    with open(path, "w") as f:
        yaml.safe_dump(model_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Default config file generated - check {path}.")


@app.command(short_help="Train UNITE model using given yaml file.")
def train(
    config_path: Annotated[
        Path, typer.Option(exists=True, file_okay=True, dir_okay=False)
    ] = Path("./example.yaml"),
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

    if config.lit_unite.unite_model.use_bfloat:
        torch.set_float32_matmul_precision("high")

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
    transform = v2.Compose(transform_list)  # ty:ignore[invalid-argument-type]

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

    callbacks: list[Callback] = [VisualizationCallback(), LearningRateMonitor()]  # ty:ignore[invalid-assignment]
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
        max_epochs=config.max_epoch,
        precision="bf16-mixed" if config.lit_unite.unite_model.use_bfloat else 16,
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.acc_grad,
        fast_dev_run=fast_dev_run,
    )

    if config.wandb_watch and not fast_dev_run:
        wandb_logger.watch(lit_classifier)  # ty:ignore[invalid-argument-type]

    if not resume:
        trainer.fit(lit_classifier, datamodule=datamodule)  # ty:ignore[invalid-argument-type]
    else:
        ckpt: str = typer.prompt("Input checkpoint path")
        trainer.fit(lit_classifier, datamodule=datamodule, ckpt_path=ckpt)  # ty:ignore[invalid-argument-type]

    wandb.finish()


@app.command(short_help="Test the model using test dataset")
def test(
    ckpt_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)
    ],
    config_path: Annotated[
        Path, typer.Option(exists=True, file_okay=True, dir_okay=False)
    ] = Path("./example.yaml"),
    run_id: Annotated[
        str | None,
        typer.Option(
            prompt="Input run id to log (or leave it blank)", show_default=False
        ),
    ] = "",
    fast_dev_run: Annotated[bool, typer.Option("--fast-dev-run")] = False,
):
    run_id = run_id if run_id != "" else None

    print("Logging into wandb and kaggle...")
    wandb.login()
    kagglehub.login()

    model_dict: dict
    with open(config_path) as f:
        model_dict = yaml.safe_load(f)
    config = TrainConfig.model_validate(model_dict)

    if config.lit_unite.unite_model.use_bfloat:
        torch.set_float32_matmul_precision("high")

    datamodule = DFDataModule(config.datamodule)
    torch.serialization.add_safe_globals([PosixPath])
    lit_classifier = LitUNITEClassifier.load_from_checkpoint(ckpt_path)
    wandb_logger: WandbLogger
    if run_id:
        wandb_logger = WandbLogger(
            project=config.project_name,
            log_model=config.wandb_log_model,
            id=run_id,
            resume="must",
        )
    else:
        wandb_logger = WandbLogger(
            project=config.project_name,
            log_model=config.wandb_log_model,
        )

    callbacks: list[Callback] = [VisualizationCallback()]  # ty:ignore[invalid-assignment]

    trainer = L.Trainer(
        precision="bf16-mixed" if config.lit_unite.unite_model.use_bfloat else 16,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
    )

    trainer.test(lit_classifier, datamodule=datamodule)
    wandb.finish()


@app.command(short_help="Predict with given video directory.")
def predict(
    ckpt_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)
    ],
    input_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)
    ],
    config_path: Annotated[
        Path, typer.Option(exists=True, file_okay=True, dir_okay=False)
    ] = Path("./example.yaml"),
    fast_dev_run: Annotated[bool, typer.Option("--fast-dev-run")] = False,
):
    model_dict: dict
    with open(config_path) as f:
        model_dict = yaml.safe_load(f)
    config = TrainConfig.model_validate(model_dict)

    predict_files = [file for file in input_dir.glob("*") if file.is_file()]
    predict_dataset = CustomVideoDataset(predict_files, config.datamodule.dataset)
    loader = DataLoader(
        predict_dataset,
        config.datamodule.loader.batch_size,
    )
    torch.serialization.add_safe_globals([PosixPath])
    lit_classifier = LitUNITEClassifier.load_from_checkpoint(ckpt_path)
    trainer = L.Trainer(
        precision="bf16-mixed" if config.lit_unite.unite_model.use_bfloat else 16,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
    )

    print(trainer.predict(lit_classifier, loader))


@app.command(short_help="Export to ONNX file")
def export(
    ckpt_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)
    ],
    onnx_path: Annotated[
        Path, typer.Argument(exists=False, file_okay=True, dir_okay=False)
    ],
):
    torch.serialization.add_safe_globals([PosixPath])
    lit_classifier = LitUNITEClassifier.load_from_checkpoint(ckpt_path)
    lit_classifier.to_onnx(onnx_path, export_params=True)


if __name__ == "__main__":
    app()
