import torch
import torch.nn as nn

import hrlc_utils

from ._stndrd_losses import BaseLoss

class SAHCLoss(BaseLoss):
    def __init__(
        self,
        taxonomy: hrlc_utils.Taxonomy,
        level_weights: dict[hrlc_utils.Level, float],
        class_weights: dict[hrlc_utils.Level, torch.Tensor],
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.taxonomy = taxonomy
        self.level_weights = level_weights

        self.ce_loss = nn.ModuleDict()
        for level_name in level_weights:
            self.ce_loss[f"{taxonomy}_{level_name}"] = nn.CrossEntropyLoss(
                weight=class_weights[level_name].float(),
                ignore_index=ignore_index,
            )

    def forward(
        self,
        pred: hrlc_utils.LevelPredictions,
        target: hrlc_utils.LevelTensor,
        consistency_weight: float = 0.0,
    ) -> dict[hrlc_utils.Level, torch.Tensor]:
        losses = {}
        for level_name in self.level_weights:
            current_target = target[level_name]
            if current_target.ndim == 1:
                current_target = current_target.view(*current_target.shape, 1, 1)
            criterion = self.ce_loss[f"{self.taxonomy}_{level_name}"]
            losses[level_name] = criterion(
                pred[level_name]["OG"], current_target
            ) + consistency_weight * criterion(
                pred[level_name]["SAHC"], current_target
            )
        return losses
