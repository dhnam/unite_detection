import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Protocol, cast, override

import gdown
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from unite_detection.dataset import (
    CelebDFImageDataset,
    CelebDFVideoDataset,
    DeepFakeBaseDataset,
)
from unite_detection.schemas import DataModuleConfig
from unite_detection.utils import (
    preprocess_celebdf,
    preprocess_celebdf_frames,
    preprocess_gta_v,
    unzip_by_size,
)


class AbstractDatasetManager(ABC):
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]

    def __init__(self, config: DataModuleConfig):
        self.config = config
        self.root: Path = Path()
        self.train_paths: list[Path]

    def prepare(self):
        self._prepare_files()
        self._get_splits()

    @abstractmethod
    def _prepare_files(self):
        pass

    @abstractmethod
    def _get_splits(self):
        pass


class PreprocessFunction(Protocol):
    def __call__(
        self,
        src_root: Path,
        dst_root: Path,
        size: tuple[int, int],
        max_workers: int = 8,
    ) -> None: ...


class CelebDFManager(AbstractDatasetManager):
    preprocess_fn: PreprocessFunction
    dataset_cls: type[DeepFakeBaseDataset]

    def __init__(self, config: DataModuleConfig):
        super().__init__(config)
        if self.config.from_img:
            self.preprocess_fn = preprocess_celebdf_frames
            self.dataset_cls = CelebDFImageDataset
        else:
            self.preprocess_fn = preprocess_celebdf
            self.dataset_cls = CelebDFVideoDataset

    @override
    def _prepare_files(self):
        path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
        if self.config.do_preprocess:
            self.preprocess_fn(
                Path(path),
                self.config.celeb_df_preprocess_path,
                size=self.config.dataset.size,
            )
            self.root = self.config.celeb_df_preprocess_path
        else:
            self.root = Path(path)

    @override
    def _get_splits(self):
        pattern = "*/*" if self.config.from_img else "*/*.mp4"
        all_paths = [
            Path(x)
            for x in self.root.glob(pattern)
            if os.path.isdir(x) or x.suffix == "mp4"
        ]
        txt_path = self.root / "List_of_testing_videos.txt"
        test_df = pd.read_csv(txt_path, sep=" ", header=None, names=["label", "path"])
        test_path_set: set[Path]
        if self.config.from_img:
            test_path_set = set(
                (self.root / Path(x).with_suffix("")) for x in test_df["path"]
            )
        else:
            test_path_set = set(self.root / x for x in test_df["path"])

        train_val_candidates = [x for x in all_paths if x not in test_path_set]
        train, val = cast(
            tuple[list[Path], list[Path]],
            train_test_split(
                train_val_candidates,
                test_size=self.config.val_split_ratio,
                random_state=42,
                shuffle=True,
            ),
        )
        self.train_paths = train
        self.val_paths = val
        self.test_paths = list(test_path_set)
        print("CelebDF Videos")
        print(f"Train: {len(train)}")
        print(f"Val: {len(val)}")
        print(f"Test: {len(test_path_set)}")


class GTAManager(AbstractDatasetManager):
    ext: Literal[".png", ".bmp"]

    def __init__(self, config: DataModuleConfig):
        super().__init__(config)
        self.ext = ".png" if self.config.do_preprocess else ".bmp"

    @override
    def _prepare_files(self):
        if not self.config.gta_v_zip_path.exists():
            print("get Sailvos dataset from drive")
            gdown.download(
                id=self.config.gta_v_gdrive_id, output=str(self.config.gta_v_zip_path)
            )

        if not self.config.gta_v_down_path.exists():
            unzip_by_size(self.config.gta_v_zip_path, self.config.gta_v_down_path)

        if self.config.do_preprocess:
            if self.config.gta_v_preprocess_path.exists():
                print(f"Already preprocessed in {self.config.gta_v_preprocess_path}")
            else:
                preprocess_gta_v(
                    self.config.gta_v_down_path,
                    self.config.gta_v_preprocess_path,
                    self.config.dataset.size,
                )
            self.root = self.config.gta_v_preprocess_path
        else:
            self.root = self.config.gta_v_down_path / "Image"

    @override
    def _get_splits(self):
        all_folders_gta = [Path(x) for x in self.root.glob("*") if os.path.isdir(x)]
        if len(all_folders_gta) == 0:
            raise Exception(f"No file in {self.root}")

        # 8:1:1 split
        train_vid, val_test_vid = cast(
            tuple[list[Path], list[Path]],
            train_test_split(
                all_folders_gta,
                test_size=0.2,
                random_state=42,
                shuffle=True,
            ),
        )

        val_vid, test_vid = cast(
            tuple[list[Path], list[Path]],
            train_test_split(
                val_test_vid,
                test_size=0.5,
                random_state=42,
                shuffle=True,
            ),
        )

        self.train_paths = train_vid
        self.val_paths = val_vid
        self.test_paths = test_vid
        print("GTA Videos")
        print(f"Train: {len(train_vid)}")
        print(f"Val: {len(val_vid)}")
        print(f"Test: {len(test_vid)}")
