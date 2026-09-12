from abc import ABC

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import classification as cls_metrics

class BaseBackbone(nn.Module, ABC):
    n_features: int


class Projection(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden

        self.norm1 = nn.LayerNorm(self.n_features)
        self.norm2 = nn.LayerNorm(self.n_hidden)

        self.proj = nn.Sequential(
            nn.Linear(
                self.n_features,
                self.n_hidden,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm1(x)

        x = self.proj(x)

        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        return x


class VectorizedLevelHead(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        lut: dict[str, tuple[torch.Tensor, list[str]]],
        level_weights: dict[str, float],
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden

        self.head_info: list[dict[str, str | int]] = []
        self.head_map: dict[str, int] = {}

        idx_counter = 0
        max_classes = 0

        for level in sorted(level_weights):
            n_classes = len(lut[level][1])
            self.head_info.append({"level": level, "n_classes": n_classes})
            self.head_map[level] = idx_counter
            max_classes = max(max_classes, n_classes)
            idx_counter += 1

        self.num_heads = len(self.head_info)
        self.max_classes = max_classes

        self.proj = Projection(n_features, n_hidden)

        self.unified_head = nn.Conv2d(
            n_hidden, self.num_heads * self.max_classes, kernel_size=1
        )

        valid_mask = torch.zeros((self.num_heads, self.max_classes), dtype=torch.bool)
        n_classes_list = []
        for i, info in enumerate(self.head_info):
            valid_mask[i, : info["n_classes"]] = True
            n_classes_list.append(info["n_classes"])

        self.valid_mask: torch.Tensor
        self.register_buffer(
            name="valid_mask",
            tensor=valid_mask,
        )
        self.n_classes_tensor: torch.Tensor
        self.register_buffer(
            name="n_classes_tensor",
            tensor=torch.tensor(n_classes_list, dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:  # [B, C]
            x = x.view(*x.shape, 1, 1)
        x = self.proj(x)

        logits_5d: torch.Tensor = self.unified_head(x)
        valid_mask = self.valid_mask.view(1, self.num_heads * self.max_classes, 1, 1)
        logits_masked = logits_5d.masked_fill(~valid_mask, -1e9)

        B, _, H, W = logits_masked.shape
        logits = logits_masked.view(B, self.num_heads, self.max_classes, H, W)

        return logits


def build_metrics(
    num_classes, taxonomy, prefix, postfix
) -> torchmetrics.MetricCollection:
    metrics_dict = dict(
        OA=cls_metrics.MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
            ignore_index=255,
        ),
        mF1=cls_metrics.MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
            ignore_index=255,
        ),
        conf_mat=cls_metrics.MulticlassConfusionMatrix(
            num_classes=num_classes,
            ignore_index=255,
        ),
        mAP=cls_metrics.MulticlassAveragePrecision(
            num_classes=num_classes,
            ignore_index=255,
            average="macro",
        ),
    )

    if taxonomy == "ELU":
        metrics_dict.update(
            dict(
                Top3=cls_metrics.MulticlassAccuracy(
                    num_classes=num_classes,
                    average="micro",
                    ignore_index=255,
                    top_k=3,
                ),
                Top5=cls_metrics.MulticlassAccuracy(
                    num_classes=num_classes,
                    average="micro",
                    ignore_index=255,
                    top_k=5,
                ),
            )
        )

    return torchmetrics.MetricCollection(
        metrics_dict,  # type:ignore
        prefix=prefix,
        postfix=postfix,
        compute_groups=False,
    )
