import kagglehub
import lightning.pytorch as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import v2

from unite_detection.schemas import (
    ADLossConfig,
    ArchSchema,
    DataLoaderConfig,
    DataModuleConfig,
    DatasetConfig,
    EncoderConfig,
    OptimizerConfig,
    SamplerConfig,
    UNITEClassifierConfig,
    UNITEConfig,
)

print("Logging into kagglehub...")
kagglehub.login()
print("Logging into wandb...")
wandb.login()

# 1. Sweep 설정 정의
sweep_config = {
    "method": "bayes",  # 이전 결과를 바탕으로 최적값을 찾아가는 베이지안 탐색
    "metric": {"name": "val/MulticlassAveragePrecision", "goal": "maximize"},
    "parameters": {
        # 우리가 집중적으로 조절할 하이퍼파라미터
        "lambda_1": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.5,
        },  # CE 비중
        "lambda_2": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.5,
        },  # AD 비중
        "delta_between": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0,
        },  # Head 간 거리 마진
        "delta_within_fake": {
            "distribution": "uniform",
            "min": -2.0,
            "max": 0.5,
        },  # fake class에 대한 delta_within
        "delta_within_real": {
            "distribution": "uniform",
            "min": -2.0,
            "max": 0.5,
        },  # real class에 대한 delta_within
        "eta": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.3,
        },  # Center 업데이트 속도
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4,
        },  # Learning Rate
        "setting": {
            "values": ["long", "big"],
        },  # long: 32 frame 224x224 / big: 16 frame 384x384
        # 고정할 값들 (실험 편의를 위해 여기에 선언)
        "max_epochs": {"value": 3},  # 싹수 확인을 위한 짧은 학습
        "batch_size": {"value": 20},
        "acc_grad": {"value": 2},
    },
}

sweep_id = wandb.sweep(sweep_config, project="UNITE_hyper_tuning")

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


def sweep_train():
    with wandb.init() as run:
        config = wandb.config

        L.seed_everything(42, workers=True)

        # 이름 규칙 정하기 (대시보드에서 보기 편하게)
        run.name = (
            f"L1_{config.lambda_1}_L2_{config.lambda_2}_fake_{config.delta_within_fake}"
        )

        encoder_config = EncoderConfig(
            model=f"google/siglip2-base-patch16-{384 if config.setting == 'big' else 224}",
            use_auto_processor=True,
        )

        arch_schema = ArchSchema(
            num_frames=32 if config.setting == "long" else 16,
            img_size=[384, 384] if config.setting == "big" else [224, 224],
        )

        # 데이터 모듈 설정 (10,000개 샘플링 유지)
        # transform은 기존에 정의하신 것을 사용하세요.
        datamodule_config = DataModuleConfig(
            celeb_df_preprocess_path=Path("/content/preprocessed"),
            from_img=True,
            use_gta_v=True,
            gta_v_preprocess_path=Path("/content/preprocessed_gta"),
            do_preprocess=False,
            loader=DataLoaderConfig(
                batch_size=config.batch_size,
                num_workers=2,
                prefetch_factor=None,
                pin_memory=True,
                persistent_workers=True,
            ),
            dataset=DatasetConfig(
                arch=arch_schema,
                encoder=encoder_config,
                transform=transform,
            ),
            sampler=SamplerConfig(
                seed=42,
                run_sample=10000,
            ),
        )
        dm = DFDataModule(datamodule_config)
        # 중요: dm.setup()에서 사용하는 샘플링 로직은
        # train_dataloader의 WeightedRandomSampler가 처리합니다.

        # 모델 설정
        unite_classifier_config = UNITEClassifierConfig(
            arch=arch_schema,
            unite_model=UNITEConfig(
                dropout=0.1,
                use_bfloat=True,
                encoder=encoder_config,
            ),
            ad_loss=ADLossConfig(
                delta_within=(config.delta_within_real, config.delta_within_fake),
                delta_between=config.delta_between,
                eta=config.eta,
            ),
            optim=OptimizerConfig(
                lr=config.lr,
                decay_steps=(1000 * 32) // (config.batch_size * config.acc_grad),
            ),
            loss=LossConfig(
                lambda_1=config.lambda_1,
                lambda_2=config.lambda_2,
            ),
        )
        model = LitUNITEClassifier(unite_classifier_config)

        # 모델 컴파일 (속도 향상)
        # model = torch.compile(model)

        # 로거 및 트레이너 설정
        logger = WandbLogger(log_model="all")

        trainer = L.Trainer(
            max_epochs=config.max_epochs,
            logger=logger,
            accumulate_grad_batches=config.acc_grad,
            precision="bf16-mixed",
            log_every_n_steps=50,
            num_sanity_val_steps=0,  # 속도를 위해 생략
            deterministic=True,
        )

        # 학습 시작
        trainer.fit(model, datamodule=dm)

        wandb.agent(sweep_id, function=sweep_train, count=20)
