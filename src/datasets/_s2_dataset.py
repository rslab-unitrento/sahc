from typing import Mapping

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class SentinelDataset(Dataset):
    def __init__(self, zarr_path, lookup_tensors, coords, patch_size=64):
        self.zarr_path = zarr_path
        self.lookup_tensors = lookup_tensors
        self.coords = coords
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.store = None

    def _init_store(self):
        z = zarr.open(self.zarr_path, mode="r")
        self.imagery = z["imagery"]
        self.labels = z["label"]

    def __len__(self):
        return len(self.coords)

    def _preprocess(
        self, sample: Mapping[str, torch.Tensor | np.ndarray]
    ) -> dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]:
        for key in self.lookup_tensors:
            if key in sample:
                temp: dict[str, torch.Tensor] = {}
                for level in self.lookup_tensors[key]:
                    lookup = self.lookup_tensors[key][level][0].to(sample[key].device)  # type: ignore
                    temp[level] = lookup[sample[key].long()]  # type: ignore
                sample[key] = temp  # type: ignore
        return sample  # type: ignore

    def __getitem__(self, idx):
        if self.store is None:
            self._init_store()

        y_c, x_c = self.coords[idx]

        y_min, y_max = y_c - self.half, y_c + self.half + 1
        x_min, x_max = x_c - self.half, x_c + self.half + 1

        img_patch = self.imagery[:, :, y_min:y_max, x_min:x_max]

        label = self.labels[y_c - 1 : y_c + 2, x_c - 1 : x_c + 2]

        img_tensor = (torch.from_numpy(img_patch).float() - 1000) / 10000
        label_tensor = {"ELU": torch.tensor(label)}

        sample = self._preprocess(label_tensor)
        sample["x"] = img_tensor

        return sample


class SentinelEvalDataset(SentinelDataset):
    def __init__(self, zarr_path, lookup_tensors, coords, patch_size=64):
        super().__init__(zarr_path, lookup_tensors, coords, patch_size)

    def __getitem__(self, idx):
        if self.store is None:
            self._init_store()

        y_c, x_c = self.coords[idx]

        y_min, y_max = y_c - self.half, y_c + self.half + 1
        x_min, x_max = x_c - self.half, x_c + self.half + 1

        img_patch = self.imagery[:, :, y_min:y_max, x_min:x_max]

        label = np.ones((3, 3)) * 255
        label[1, 1] = self.labels[y_c, x_c]

        img_tensor = (torch.from_numpy(img_patch).float() - 1000) / 10000
        label_tensor = {"ELU": torch.tensor(label)}

        sample = self._preprocess(label_tensor)
        sample["x"] = img_tensor

        return sample


class SentinelPredictDataset(SentinelDataset):
    def __init__(self, zarr_path, coords, patch_size=9):
        super().__init__(zarr_path, lookup_tensors={}, coords=coords, patch_size=patch_size)

    def __getitem__(self, idx):
        if self.store is None:
            self._init_store()

        y_c, x_c = self.coords[idx]

        y_min, y_max = y_c - self.half, y_c + self.half + 1
        x_min, x_max = x_c - self.half, x_c + self.half + 1

        img_patch = self.imagery[:, :, y_min:y_max, x_min:x_max]

        img_tensor = (torch.from_numpy(img_patch).float() - 1000) / 10000

        return {"x": img_tensor, "y_c": y_c, "x_c": x_c}
