from pathlib import Path
from typing import Literal

# from lightning.pytorch.profilers import AdvancedProfiler
import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import v2

from unite_detection.lit_modules import (
    DFDataModule,
    LitUNITEClassifier,
    VisualizationCallback,
)
from unite_detection.schemas import (
    ADLossConfig,
    ArchSchema,
    DataLoaderConfig,
    DataModuleConfig,
    DatasetConfig,
    EncoderConfig,
    LossConfig,
    OptimizerConfig,
    UNITEClassifierConfig,
    UNITEConfig,
)

BATCH_SIZE = 13
ACC_GRAD = 2
DECAY_STEPS = (1000 * 32) // (BATCH_SIZE * ACC_GRAD)
LENGTH = 16
SIZE_SIDE: Literal[224, 384] = 384
SIZE = (SIZE_SIDE, SIZE_SIDE)
ENCODER = f"google/siglip2-base-patch16-{SIZE_SIDE}"

# To test: (384, 384), 16 vs (224, 224), 32

transform = v2.Compose(
    [
        v2.ToDtype(torch.uint8),
        v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomApply([v2.RandomRotation((-10, 10))]), # Quite slow?
        v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 7))]),  # Managable...
        # v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2)]), # Looks slow...
        v2.RandomApply([v2.JPEG((60, 100))]),
        v2.ToDtype(torch.float32),
    ]
)

arch = ArchSchema(num_frames=LENGTH, img_size=(SIZE_SIDE, SIZE_SIDE))

encoder = EncoderConfig(
    model=ENCODER,
    use_auto_processor=True,
)

datamodule_config = DataModuleConfig(
    celeb_df_preprocess_path=Path("/content/preprocessed"),
    from_img=False,
    use_gta_v=True,
    gta_v_preprocess_path=Path("/content/preprocessed_gta"),
    do_preprocess=False,
    loader=DataLoaderConfig(
        batch_size=BATCH_SIZE,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    ),
    dataset=DatasetConfig(
        video_decode_device="cpu",
        transform=transform,
        arch=arch,
        encoder=encoder,
    ),
)

datamodule = DFDataModule(datamodule_config)

unite_classifier_config = UNITEClassifierConfig(
    arch=arch,
    unite_model=UNITEConfig(
        dropout=0.1,
        use_bfloat=True,
        encoder=encoder,
    ),
    ad_loss=ADLossConfig(
        delta_within=(0.01, -2),
        delta_between=0.5,
    ),
    optim=OptimizerConfig(
        lr=0.0001,
        decay_steps=DECAY_STEPS,
    ),
    loss=LossConfig(
        lambda_1=0.5,
        lambda_2=0.5,
    ),
)

lit_classifier = LitUNITEClassifier(unite_classifier_config)


lit_classifier = torch.compile(lit_classifier)


# profiler = AdvancedProfiler(".", "profile")


wandb_logger = WandbLogger(
    project="UNITE_deepfake_classification",
    name="run_name",
)

# ckpt = ModelCheckpoint(monitor="val/acc", mode="max", save_last=True)
ckpt_drive = ModelCheckpoint(
    dirpath="/content/drive/MyDrive/bootcamp_proj/final/gtav_loss_fix5_imgsz/",
    monitor="val/MulticlassAveragePrecision",
    mode="max",
    save_last=True,
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
visual = VisualizationCallback()

trainer = L.Trainer(
    max_epochs=20,
    # max_steps=10,
    # profiler=profiler,
    logger=wandb_logger,
    callbacks=[visual, ckpt_drive, lr_monitor],
    precision="bf16-mixed",
    log_every_n_steps=50,
    num_sanity_val_steps=0,
    accumulate_grad_batches=ACC_GRAD,
    # precision=16,
    # fast_dev_run=True,
)


wandb_logger.watch(lit_classifier)

trainer.fit(lit_classifier, datamodule=datamodule)
