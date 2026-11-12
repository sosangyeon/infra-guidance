from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from .lis_utils import LISMotionDataset
from .womd import WaymoMotionDataset

# dataset.name -> 데이터셋 클래스 매핑 (LIS는 40 step/TargetBuilder(10,30))
DATASET_REGISTRY = {
    "womd": WaymoMotionDataset,
    "argoverse2": WaymoMotionDataset,
    "lis": LISMotionDataset,
}


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super(DataModule, self).__init__()
        self.cfg = cfg["dataset"]
        self.data_dir = self.cfg["data_dir"]
        self.batch_size = self.cfg["batch_size"]
        self.num_workers = self.cfg["num_workers"]
        self.pin_memory = self.cfg["pin_memory"]
        self.data_len = self.cfg["data_len"]
        # config의 dataset.name 으로 LIS / Waymo 선택 (기본값 womd)
        name = self.cfg.get("name", "womd")
        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"unknown dataset name '{name}', choose from {list(DATASET_REGISTRY)}"
            )
        self.dataset_cls = DATASET_REGISTRY[name]

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.dataset_cls(
            root=self.data_dir, split="train", data_len=self.data_len["train"]
        )
        self.val_dataset = self.dataset_cls(
            root=self.data_dir, split="val", data_len=self.data_len["val"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],  # "train" -> "train"
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],  # "val" -> "val"
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
