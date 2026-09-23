from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from . import MODALITIES
from .data import SplitData


class SplitDataset(Dataset):
    def __init__(self, split: SplitData):
        if not split.has_labels:
            raise ValueError("SplitDataset requires classification and regression labels")
        self.split = split

    def __len__(self) -> int:
        return self.split.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": {
                modality: torch.from_numpy(self.split.features[modality][index]).float()
                for modality in MODALITIES
            },
            "masks": {
                modality: torch.from_numpy(self.split.masks[modality][index]).bool()
                for modality in MODALITIES
            },
            "classification": torch.tensor(int(self.split.classification[index]), dtype=torch.long),
            "regression": torch.tensor(float(self.split.regression[index]), dtype=torch.float32),
            "index": index,
        }


class InferenceDataset(Dataset):
    def __init__(self, split: SplitData):
        self.split = split

    def __len__(self) -> int:
        return self.split.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": {
                modality: torch.from_numpy(self.split.features[modality][index]).float()
                for modality in MODALITIES
            },
            "masks": {
                modality: torch.from_numpy(self.split.masks[modality][index]).bool()
                for modality in MODALITIES
            },
            "index": index,
        }


def make_loader(
    split: SplitData,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    require_labels: bool = True,
) -> DataLoader:
    dataset: Dataset = SplitDataset(split) if require_labels else InferenceDataset(split)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

