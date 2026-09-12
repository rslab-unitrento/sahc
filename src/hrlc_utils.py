import typing

import torch

Taxonomy = typing.Literal["NWPU", "ELU"]
Level = typing.Literal["Level_1", "Level_2", "Level_3", "Level_4"]
PredKind = typing.Literal["OG", "SAHC"]

LevelTensor = dict[Level, torch.Tensor]
LevelPredictions = dict[Level, dict[PredKind, torch.Tensor]]
