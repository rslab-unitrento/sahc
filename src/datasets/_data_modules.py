import os

import lightning as L
import torch
import torchvision
from torch.utils.data import DataLoader

from ._HierarchicalNWPU import HierarchicalNWPU
from ._utils import get_lookup_tables_and_mappings


class Random90Rot(torch.nn.Module):
    def forward(self, img):
        torch.randint(4, size=(1,))
        return torch.rot90(img, int(torch.randint(4, size=(1,))), (-1, -2))


transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        Random90Rot(),
        torchvision.transforms.Normalize(
            mean=[0.3684, 0.3810, 0.3435], std=[0.1450, 0.1354, 0.1319]
        ),
    ]
)

transform_eval = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.Normalize(
            mean=[0.3684, 0.3810, 0.3435], std=[0.1450, 0.1354, 0.1319]
        ),
    ]
)


class VHRDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        lut_path: str,
        batch_size: int,
        concurrent_trials: int = 1,
        debug: bool = False,
        transform: bool | None = True,
        n_folds: int = 1,
        seed: int = 42,
        num_workers: int | None = None,
        prefetch_factor: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.lut_path = lut_path
        self.batch_size = batch_size
        self.ignore_index = 255
        self.seed = seed
        self.n_folds = n_folds
        self.transform = transform
        if debug:
            self.num_workers = 0
        elif num_workers is None:
            self.num_workers = max((os.cpu_count() or 2) // 2, 1) // concurrent_trials
        else:
            self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def prepare_data(self) -> None:
        self.lookup_tensors, self.mappings = get_lookup_tables_and_mappings(
            self.lut_path
        )

    def setup(self, stage=None):
        self.train_ds = HierarchicalNWPU(
            self.data_dir,
            lookup_tensors=self.lookup_tensors,
            split="train",
            download=True,
            transforms=transform_train,
        )
        self.val_ds = HierarchicalNWPU(
            self.data_dir,
            split="val",
            lookup_tensors=self.lookup_tensors,
            transforms=transform_eval,
            download=True,
        )
        self.test_ds = HierarchicalNWPU(
            self.data_dir,
            lookup_tensors=self.lookup_tensors,
            split="test",
            transforms=transform_eval,
            download=True,
        )

    def train_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            **loader_kwargs,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            **loader_kwargs,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            **loader_kwargs,
            pin_memory=True,
            drop_last=False,
        )

    def _loader_kwargs(self) -> dict[str, int]:
        kwargs = {"num_workers": self.num_workers}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs
