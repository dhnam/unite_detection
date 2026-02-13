from typing import TypedDict, override

import lightning.pytorch as L
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from unite_detection.dataset import CelebDFBaseDataset, SailVosDataset
from unite_detection.lit_modules.dataset_manager import CelebDFManager, GTAManager
from unite_detection.schemas import DataModuleConfig, DatasetConfig, SamplerConfig


class SamplerFactory:
    def __init__(self, config: SamplerConfig | None = None):
        self.config: SamplerConfig = config or SamplerConfig()

    def create_sampler(
        self, celeb_train: CelebDFBaseDataset, gta_train: SailVosDataset | None = None
    ) -> WeightedRandomSampler:
        celeb_counts = celeb_train.get_label_counter()
        celeb_weights = (
            self.config.real_weight / celeb_counts[0],
            self.config.fake_weight / celeb_counts[1],
        )
        sample_weights = [
            celeb_weights[sample["label"]] for sample in celeb_train.samples
        ]
        if gta_train is not None:
            gta_counts = gta_train.get_label_counter()
            gta_weight = self.config.gta_weight / gta_counts[1]
            sample_weights.extend([gta_weight] * len(gta_train))

        gen = None
        if self.config.seed is not None:
            gen = torch.Generator()
            gen.manual_seed(self.config.seed)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=min(len(sample_weights), self.config.run_sample),
            replacement=True,
            generator=gen,
        )
        return sampler


class DFDataModule(L.LightningDataModule):
    config: DataModuleConfig

    def __init__(self, config: DataModuleConfig | None = None):
        super().__init__()
        self.config = config or DataModuleConfig()
        self.save_hyperparameters(
            self.config.model_dump(exclude={"dataset": {"transform"}})
        )
        self.celeb_manager = CelebDFManager(self.config)
        self.gta_manager = GTAManager(self.config)
        self.sampler_factory = SamplerFactory(self.config.sampler)

        class LoaderParam(TypedDict):
            batch_size: int
            num_workers: int
            pin_memory: bool
            persistent_workers: bool
            prefetch_factor: int | None

    @override
    def prepare_data(self):
        self.celeb_manager.prepare()
        if self.config.use_gta_v:
            self.gta_manager.prepare()

    @override
    def setup(self, stage: str | None = None):
        test_val_config = self.config.dataset.model_copy()
        test_val_config.transform = None

        train_config = self.config.dataset.model_copy()

        if stage == "fit" or stage is None:
            self._setup_fit_data(train_config, test_val_config)
        if stage == "test" or stage is None:
            self._setup_test_data(test_val_config)

    def _setup_fit_data(
        self, train_config: DatasetConfig, val_config: DatasetConfig
    ) -> None:
        self.celeb_train = self.celeb_manager.dataset_cls(
            self.celeb_manager.train_paths, train_config
        )
        self.celeb_val = self.celeb_manager.dataset_cls(
            self.celeb_manager.val_paths, val_config
        )
        if self.config.use_gta_v:
            self.gta_train = SailVosDataset(
                self.gta_manager.train_paths, train_config, self.gta_manager.ext
            )
            self.gta_val = SailVosDataset(
                self.gta_manager.val_paths, val_config, self.gta_manager.ext
            )

    def _setup_test_data(self, test_config: DatasetConfig) -> None:
        self.celeb_test = self.celeb_manager.dataset_cls(
            self.celeb_manager.test_paths, test_config
        )
        if self.config.use_gta_v:
            self.gta_test = SailVosDataset(
                self.gta_manager.test_paths, test_config, self.gta_manager.ext
            )

    @override
    def train_dataloader(self):
        # 2. 현재 훈련 셋(Subset) 내의 클래스별 개수 계산
        gta = None
        dataset: Dataset = self.celeb_val
        if self.config.use_gta_v:
            gta = self.gta_train
            dataset = ConcatDataset([dataset, self.gta_val])
        sampler, dataset = self.sampler_factory.create_sampler(self.celeb_train, gta)
        return DataLoader(
            dataset,
            **self.config.loader.model_dump(),
            sampler=sampler,
        )

    @override
    def val_dataloader(self):
        dataset: Dataset = self.celeb_val
        if self.config.use_gta_v:
            dataset = ConcatDataset([dataset, self.gta_val])
        return DataLoader(
            dataset,
            **self.config.loader.model_dump(),
        )

    @override
    def test_dataloader(self):
        dataset: Dataset= self.celeb_test
        if self.config.use_gta_v:
            dataset = ConcatDataset([dataset, self.gta_test])
        return DataLoader(
            dataset,
            **self.config.loader.model_dump(),
        )
