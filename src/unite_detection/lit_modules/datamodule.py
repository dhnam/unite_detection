from typing import TypedDict, override
from pathlib import Path
import os

import lightning.pytorch as L
import kagglehub
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import WeightedRandomSampler, ConcatDataset, DataLoader
import torch
import gdown

from unite_detection.dataset import (
    CelebDFImageDataset,
    CelebDFVideoDataset,
    DeepFakeBaseDataset,
    SailVosDataset,
)
from unite_detection.schemas import DataModuleConfig
from unite_detection.utils import preprocess_celebdf, preprocess_celebdf_frames, unzip_by_size, preprocess_gta_v



class DFDataModule(L.LightningDataModule):
    config: DataModuleConfig
    def __init__(self, config: DataModuleConfig | None = None):
        super().__init__()
        if config is None:
            config = DataModuleConfig()

        self.config = config
        self.save_hyperparameters(config.model_dump(exclude={"dataset": {"transform"}}))

        self.dataset_cls: type[DeepFakeBaseDataset] = CelebDFImageDataset
        self.dataset_preprocess_method = preprocess_celebdf_frames
        if not config.from_img:
            self.dataset_cls = CelebDFVideoDataset
            self.dataset_preprocess_method = preprocess_celebdf
        self.celebdf_path: Path | None = None
        self.gta_v_path: Path | None = None

        class LoaderParam(TypedDict):
            batch_size: int
            num_workers: int
            pin_memory: bool
            persistent_workers: bool
            prefetch_factor: int | None

        self._loader_param = LoaderParam(**config.loader.model_dump())

    @override
    def prepare_data(self):
        path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
        if self.config.do_preprocess:
            self.dataset_preprocess_method(
                path, self.config.celeb_df_preprocess_path, size=self.config.dataset.size
            )
            self.celebdf_path = self.config.celeb_df_preprocess_path
        else:
            self.celebdf_path = Path(path)

        if self.config.use_gta_v:
            if self.config.gta_v_preprocess_path.exists():
                print(
                    f"GTA V data exists in {self.config.gta_v_preprocess_path}, skipping preprocessing."
                )
            else:
                if not self.config.gta_v_zip_path.exists():
                    print("get Sailvos dataset from drive")
                    # get SAIL-VOS
                    gdown.download(id=self.config.gta_v_gdrive_id, output=str(self.config.gta_v_zip_path))
                if not self.config.gta_v_down_path.exists():
                    unzip_by_size(self.config.gta_v_zip_path, self.config.gta_v_down_path)
                if self.config.do_preprocess:
                    preprocess_gta_v(
                        self.config.gta_v_down_path,
                        self.config.gta_v_preprocess_path,
                        self.config.dataset.size,
                    )
                    self.gta_v_path = self.config.gta_v_preprocess_path
                else:
                    self.gta_v_path = Path("./gta-v/mini-ref-sailvos/Images")

    def _setup_gta_path(self) -> tuple[list[Path], list[Path], list[Path]]:
        assert self.gta_v_path is not None
        all_folders_gta = [
            Path(x) for x in self.gta_v_path.glob("*") if os.path.isdir(x)
        ]
        if len(all_folders_gta) == 0:
            raise Exception(f"No file in {self.gta_v_path}")

        # 8:1:1 split
        train_vid, val_test_vid = train_test_split(
            all_folders_gta,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

        val_vid, test_vid = train_test_split(
            val_test_vid,
            test_size=0.5,
            random_state=42,
            shuffle=True,
        )

        return train_vid, val_vid, test_vid

    @override
    def setup(self, stage: str | None=None):
        assert self.celebdf_path is not None
        gta_train_path, gta_val_path, gta_test_path = None, None, None
        if self.config.use_gta_v:
            gta_train_path, gta_val_path, gta_test_path = self._setup_gta_path()

        test_val_config = self.config.dataset.model_copy()
        test_val_config.transform = None

        train_config = self.config.dataset.model_copy()

        if stage == "fit" or stage is None:
            # 1. 모든 비디오 폴더 경로 가져오기
            # 전처리된 이미지 폴더 구조: root/Celeb-real/id0_0000 ...
            # glob을 이용해 실제 폴더들을 다 찾습니다.

            all_folders: list[Path] = []
            if self.config.from_img:
                all_folders = [
                    Path(x)
                    for x in self.config.celeb_df_preprocess_path.glob("*/*")
                    if os.path.isdir(x)
                ]
            else:
                all_folders = [
                    Path(x) for x in self.config.celeb_df_preprocess_path.glob("*/*.mp4")
                ]

            # 2. Test Set 목록 로드 및 제외
            txt_path = self.celebdf_path / "List_of_testing_videos.txt"
            test_df = pd.read_csv(
                txt_path, sep=" ", header=None, names=["label", "path"]
            )
            test_paths_set: set[str] = set()
            if self.config.from_img:
                # 확장자를 떼고 비교해야 함 (이미지 폴더명은 확장자가 없으므로)
                test_paths_set = set(
                    test_df["path"]
                    .apply(
                        lambda x: str(
                            self.celebdf_path / Path(x).with_suffix("")
                        ).replace("\\", "/")
                    )
                    .values
                )
            else:
                test_paths_set = set(
                    test_df["path"]
                    .apply(lambda x: str(self.celebdf_path / x))
                    .replace("\\", "/")
                    .values
                )

            train_val_candidates = []
            for folder in all_folders:
                # 경로 정규화
                folder_str = str(folder).replace("\\", "/")
                # 테스트 셋에 포함되지 않은 것만 Train/Val 후보로 등록
                if folder_str not in test_paths_set:
                    train_val_candidates.append(folder_str)

            # 3. 비디오 단위로 Train / Val 분리 (가장 중요!)
            # 여기서 쪼개야 비디오 하나가 통째로 Train이나 Val 한쪽으로만 갑니다.
            train_videos, val_videos = train_test_split(
                train_val_candidates,
                test_size=self.config.val_split_ratio,
                random_state=42,
                shuffle=True,
            )

            print(f"Total Train/Val Videos: {len(train_val_candidates)}")
            print(
                f"Split result -> Train Videos: {len(train_videos)}, Val Videos: {len(val_videos)}"
            )

            # 4. 데이터셋 인스턴스 생성 (video_paths 주입)
            self.celebdf_train = self.dataset_cls(
                paths=train_videos,
                config=train_config,
            )

            self.celebdf_val = self.dataset_cls(
                paths=val_videos,
                config=test_val_config,
            )

            if self.config.use_gta_v:
                assert gta_train_path is not None
                assert gta_val_path is not None
                if not self.config.do_preprocess:
                    ext = ".bmp"
                else:
                    ext = ".png"
                self.gta_train = SailVosDataset(
                    paths=gta_train_path, config=train_config, ext=ext
                )
                self.gta_val = SailVosDataset(
                    paths=gta_val_path,
                    config=test_val_config,
                    ext=ext,
                )

        if stage == "test" or stage is None:
            # 테스트 셋은 기존 로직(txt 파일 기반)을 유지하거나,
            # 위에서 test_paths_set을 리스트로 변환해 넘겨줘도 됩니다.
            # 여기서는 편의상 기존 클래스가 is_test=True일 때 txt 파일을 읽는 로직을 그대로 쓴다고 가정하거나
            # 혹은 위와 똑같이 리스트를 만들어서 넘겨줍니다.

            # Test 리스트 생성 로직
            txt_path = self.config.celeb_df_preprocess_path / "List_of_testing_videos.txt"
            test_df = pd.read_csv(
                txt_path, sep=" ", header=None, names=["label", "path"]
            )
            test_videos = []
            if self.config.from_img:
                test_videos = [
                    str(self.config.celeb_df_preprocess_path / Path(x).with_suffix("")).replace(
                        "\\", "/"
                    )
                    for x in test_df["path"].values
                ]
            else:
                test_videos = [
                    str(self.config.celeb_df_preprocess_path / Path(x))
                    for x in test_df["path"].values
                ]
            # 실제 존재하는 폴더만 필터링
            test_videos = [v for v in test_videos if os.path.exists(v)]

            self.celebdf_test = self.dataset_cls(
                paths=test_videos,
                config=test_val_config,
            )
            if self.config.use_gta_v:
                assert gta_test_path is not None
                if not self.config.do_preprocess:
                    ext = ".bmp"
                else:
                    ext = ".png"
                self.gta_test = SailVosDataset(
                    paths=gta_test_path,
                    config=test_val_config,
                    ext=ext,
                )

    @override
    def train_dataloader(self):
        # 2. 현재 훈련 셋(Subset) 내의 클래스별 개수 계산
        label_counts = self.celebdf_train.get_label_counter()

        target_real_weight = 0.4
        target_fake_weight = 0.35
        target_gtav_weight = 0.25

        class_weights = (
            target_real_weight / label_counts[0],
            target_fake_weight / label_counts[1],
        )

        sample_weights = [
            class_weights[sample["label"]] for sample in self.celebdf_train.samples
        ]

        dataset = self.celebdf_train

        if self.config.use_gta_v:
            dataset = ConcatDataset([dataset, self.gta_train])
            label_counts_gta = self.gta_train.get_label_counter()
            gtav_weight = target_gtav_weight / label_counts_gta[1]
            sample_weights.extend([gtav_weight] * len(self.gta_train))

        gen = None
        if self.config.seed is not None:
            gen = torch.Generator()
            gen.manual_seed(42)

        # 5. 샘플러 생성
        # num_samples는 보통 학습 데이터 전체 길이로 설정합니다.
        # replacement=True여야 불균형 데이터에서 적은 쪽을 중복해서 뽑아 균형을 맞춥니다.
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=min(len(sample_weights), self.config.run_sample),
            replacement=True,
            generator=gen,
        )

        return DataLoader(
            dataset,
            **self._loader_param,
            sampler=sampler,
        )

    @override
    def val_dataloader(self):
        dataset = self.celebdf_val
        if self.config.use_gta_v:
            dataset = ConcatDataset([dataset, self.gta_val])
        return DataLoader(
            dataset,
            **self._loader_param,
        )

    @override
    def test_dataloader(self):
        dataset = self.celebdf_test
        if self.config.use_gta_v:
            dataset = ConcatDataset([dataset, self.gta_test])
        return DataLoader(
            dataset,
            **self._loader_param,
        )
