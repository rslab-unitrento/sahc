from abc import ABC
import torch
from torch import nn

import hrlc_utils


class BaseLoss(nn.Module, ABC):
    taxonomy: hrlc_utils.Taxonomy

    def forward(
        self,
        pred: hrlc_utils.LevelTensor,
        target: hrlc_utils.LevelTensor,
    ) -> hrlc_utils.LevelTensor: ...
