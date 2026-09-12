from typing import Any

import numpy as np
import torch
import torchgeo
import torchgeo.datasets


class HierarchicalNWPU(torchgeo.datasets.RESISC45):
    def __init__(
        self,
        root,
        lookup_tensors,
        split="train",
        transforms=None,
        download=False,
        become_2d: bool = True,
    ):
        super().__init__(
            root=root,
            split=split,
            download=download,
        )
        self.lookup_tensors = lookup_tensors
        self.transform_NWPU = transforms
        self.become_2d = become_2d

    def __len__(self):
        return super().__len__()

    def _preprocess(
        self, sample: dict[str, torch.Tensor | np.ndarray]
    ) -> dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]:
        for key in self.lookup_tensors:
            if key in sample:
                temp: dict[str, torch.Tensor] = {}
                for level in self.lookup_tensors[key]:
                    lookup = self.lookup_tensors[key][level][0].to(sample[key].device)  # type: ignore
                    temp[level] = lookup[sample[key].long()]  # type: ignore
                sample[key] = temp  # type: ignore
        return sample  # type: ignore

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = super(HierarchicalNWPU, self).__getitem__(index)
        image = data["image"] / 255.0
        if self.transform_NWPU:
            image = self.transform_NWPU(np.moveaxis(image.numpy(), 0, 2))
        if self.become_2d:
            data["label"] = data["label"][..., None, None]
        labels = {
            "NWPU": data["label"],
        }
        # Generate target hierarchical labels
        sample = self._preprocess(labels)  # type:ignore
        sample["x"] = image
        return sample
