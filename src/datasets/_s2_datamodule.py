import os
from collections import defaultdict

import lightning as L
import numpy as np
import torch
import xarray as xr
import zarr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ._s2_dataset import SentinelDataset, SentinelEvalDataset, SentinelPredictDataset
from ._utils import get_lookup_tables_and_mappings


class SpatialIndexGenerator:
    def __init__(self, zarr_path, patch_size, block_size=1000):
        self.ds = xr.open_zarr(zarr_path)
        self.gt = np.nan_to_num(self.ds["label"].values, nan=255)
        self.H, self.W = self.gt.shape
        self.patch_size = patch_size
        self.block_size = (
            block_size
        )

    def generate(self, max_samples_per_class: int, seed: int = 42):
        rows = np.arange(0, self.H, self.block_size)
        cols = np.arange(0, self.W, self.block_size)
        blocks = []
        for r in rows:
            for c in cols:
                blocks.append((r, c))

        train_blocks, val_test_blocks = train_test_split(
            blocks, train_size=0.7, random_state=seed
        )
        val_blocks, test_blocks = train_test_split(
            val_test_blocks, test_size=0.5, random_state=seed
        )

        splits = {
            "train": self._scan_blocks(train_blocks, max_samples_per_class),
            "val": self._scan_blocks(val_blocks, max_samples_per_class),
            "test": self._scan_blocks(test_blocks, max_samples_per_class),
        }
        return splits

    def _scan_blocks(self, blocks, max_per_class):
        class_coords = defaultdict(list)
        half_p = self.patch_size // 2

        for r_start, c_start in blocks:
            r_end = min(r_start + self.block_size, self.H)
            c_end = min(c_start + self.block_size, self.W)

            gt_block = self.gt[r_start:r_end, c_start:c_end]

            present_classes = np.unique(gt_block)

            for cls in present_classes:
                locs_y, locs_x = np.where(gt_block == cls)

                global_y = locs_y + r_start
                global_x = locs_x + c_start

                valid_mask = (
                    (global_y >= half_p)
                    & (global_y < self.H - half_p)
                    & (global_x >= half_p)
                    & (global_x < self.W - half_p)
                )

                if not np.any(valid_mask):
                    continue

                valid_coords = list(zip(global_y[valid_mask], global_x[valid_mask]))
                class_coords[cls].extend(valid_coords)

        final_list = []
        for cls, coords in class_coords.items():
            if len(coords) > max_per_class:
                indices = np.random.choice(len(coords), max_per_class, replace=False)
                final_list.extend([coords[i] for i in indices])
            else:
                final_list.extend(coords)
        print(len(class_coords.keys()))
        return final_list


class SentinelDataModule(L.LightningDataModule):
    def __init__(
        self,
        zarr_path: str,
        lut_path: str,
        split_path: str,
        patch_size: int = 64,
        batch_size: int = 32,
        samples_per_class: int = 1000,
        seed: int = 42,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.zarr_path = zarr_path
        self.split_path = split_path
        self.patch_size = patch_size
        self.lut_path = lut_path
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.ignore_index = 255
        self.seed = seed
        self.num_workers = (os.cpu_count() or 1) // 2 if num_workers is None else num_workers
        self.prefetch_factor = prefetch_factor
        self.save_hyperparameters()
        self.splits = None

    def prepare_data(self):
        self.lookup_tensors, self.mappings = get_lookup_tables_and_mappings(
            self.lut_path
        )
        if not os.path.exists(self.split_path):
            print("Generating Spatial Split Indices...")
            sampler = SpatialIndexGenerator(self.zarr_path, self.patch_size)
            self.splits = sampler.generate(max_samples_per_class=self.samples_per_class)
            torch.save(self.splits, self.split_path)

    def setup(self, stage=None):
        if self.splits is None:
            self.splits = torch.load(self.split_path, weights_only=False)

        self.train_ds = SentinelDataset(
            self.zarr_path,
            lookup_tensors=self.lookup_tensors,
            coords=self.splits["train"],
            patch_size=self.patch_size,
        )
        self.val_ds = SentinelEvalDataset(
            self.zarr_path,
            lookup_tensors=self.lookup_tensors,
            coords=self.splits["val"],
            patch_size=self.patch_size,
        )
        self.test_ds = SentinelEvalDataset(
            self.zarr_path,
            lookup_tensors=self.lookup_tensors,
            coords=self.splits["test"],
            patch_size=self.patch_size,
        )

        if stage == "predict":
            z = zarr.open(self.zarr_path)
            H, W = z["label"].shape
            half = self.patch_size // 2
            step_size = 3
            pred_coords = []
            for y in range(half, H - half, step_size):  # type:ignore
                for x in range(half, W - half, step_size):  # type:ignore
                    pred_coords.append((y, x))
            self.pred_ds = SentinelPredictDataset(
                self.zarr_path,
                coords=pred_coords,
                patch_size=self.patch_size,
            )

    def train_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
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
            drop_last=False,
        )

    def test_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            **loader_kwargs,
            drop_last=False,
        )

    def predict_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(
            self.pred_ds,
            batch_size=self.batch_size,
            **loader_kwargs,
            drop_last=False,
        )

    def _loader_kwargs(self) -> dict[str, int]:
        kwargs = {"num_workers": self.num_workers}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs
