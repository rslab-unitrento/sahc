import logging
import typing

import einops
import lightning as L
import torch
import torchmetrics
import torchvision
from torch import nn

import hrlc_utils
import losses

from ._SAHC import SAHCModel, VectorizedSAHCHead
from ._RSTTEncoder import RSTTEncoder
from ._utils import BaseBackbone, build_metrics

_module_logger = logging.getLogger(__name__)

CONTEXT_SIZE = 9


class LightningSAHC(L.LightningModule):
    _class_logger = _module_logger.getChild(__qualname__)

    def __init__(
        self,
        mappings: dict[
            hrlc_utils.Taxonomy,
            dict[tuple[hrlc_utils.Level, hrlc_utils.Level], torch.Tensor],
        ],
        supervised_cce_cons: bool,
        consistency_weight: float = 1.0,
        final_metrics: torchmetrics.MetricCollection | None = None,
        ensemble_mode: typing.Literal["geometric"] = "geometric",
        lut: dict[
            hrlc_utils.Taxonomy,
            dict[hrlc_utils.Level, tuple[torch.Tensor, list[str]]],
        ] | None = None,
        taxonomy: hrlc_utils.Taxonomy | None = None,
        main_level: hrlc_utils.Level = "Level_2",
        level_weights: dict[hrlc_utils.Level, float] | None = None,
        class_weights: dict[hrlc_utils.Level, torch.Tensor] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        backbone_arch: typing.Literal["RSTTEncoder", "ResNet50"] = "RSTTEncoder",
        use_scheduler: bool = True,
        sched_pct: float = 0.3,
        inference_stride: int = 5,
        compile: bool = True,
        time_steps: int = 12,
        n_features: int = 24,
        use_2d_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        stale = {"main_" + "ta" + "sk", "ta" + "sk_" + "weights"}.intersection(kwargs)
        if stale:
            names = ", ".join(sorted(stale))
            raise TypeError(f"Unsupported configuration field(s): {names}")
        if lut is None:
            raise ValueError("lut is required")
        if taxonomy is None:
            raise ValueError("taxonomy is required")
        if taxonomy not in lut:
            raise ValueError(f"Unknown taxonomy: {taxonomy!r}")
        if taxonomy not in mappings:
            raise ValueError(f"Mappings are missing taxonomy: {taxonomy!r}")
        if level_weights is None:
            raise ValueError("level_weights is required")
        if len(level_weights) < 2:
            raise ValueError("At least two hierarchy levels are required")

        taxonomy_lut = lut[taxonomy]
        missing_levels = set(level_weights) - set(taxonomy_lut)
        if missing_levels:
            raise ValueError(f"Unknown levels for {taxonomy}: {sorted(missing_levels)}")
        if main_level not in level_weights:
            raise ValueError(f"main_level is not configured: {main_level!r}")

        self._instance_logger = self._class_logger.getChild(str(id(self)))
        self.supervised_cce_cons = bool(supervised_cce_cons)
        self.consistency_weight = consistency_weight
        self.ensemble_mode = ensemble_mode
        self.pred_kinds: list[hrlc_utils.PredKind] = ["OG", "SAHC"]
        self.n_features = n_features
        self.time_steps = time_steps
        self.compile = compile
        self.inference_stride = inference_stride
        self.central_pixel = CONTEXT_SIZE // 2
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.taxonomy = taxonomy
        self.main_level = main_level
        self.level_weights = dict(level_weights)
        self.sched_pct = sched_pct

        self.lut = {level: taxonomy_lut[level] for level in level_weights}
        self.mappings = {
            pair: mapping
            for pair, mapping in mappings[taxonomy].items()
            if pair[0] in level_weights and pair[1] in level_weights
        }
        main_level_lut = self.lut[self.main_level][0]
        self.labels = torch.zeros(
            (len(self.lut[self.main_level][1]),), dtype=torch.long
        )
        for code in torch.where(main_level_lut != 255):
            self.labels[main_level_lut[code]] = code
        self.sort_probs = sorted(range(len(self.labels)), key=lambda i: self.labels[i])

        if class_weights is None:
            class_weights = {
                level: torch.ones(len(self.lut[level][1])) for level in level_weights
            }
        missing_weights = set(level_weights) - set(class_weights)
        if missing_weights:
            raise ValueError(f"Class weights are missing levels: {sorted(missing_weights)}")
        self.class_weights = class_weights
        self.backbone_arch = backbone_arch

        match self.backbone_arch:
            case "RSTTEncoder":
                backbone = RSTTEncoder(
                    num_classes=None,
                    in_chans=self.n_features,
                    num_frames=self.time_steps,
                    **kwargs,
                )
            case "ResNet50":
                backbone = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                )
                if use_2d_backbone:
                    layers = list(backbone.children())[:-2]
                    backbone = nn.Sequential(
                        *layers,
                        nn.Conv2d(
                            layers[-1][-1].conv3.out_channels, 1024, kernel_size=1
                        ),
                        nn.ReLU(),
                    )
                    backbone.n_features = 1024
                else:
                    layers = list(backbone.children())[:-1]
                    backbone = nn.Sequential(
                        *layers,
                        nn.Conv2d(
                            layers[-2][-1].conv3.out_channels, 1024, kernel_size=1
                        ),
                        nn.ReLU(),
                    )
                    backbone.n_features = 1024
            case _:
                raise ValueError(
                    f"Unsupported backbone_arch={self.backbone_arch!r}; "
                    "use RSTTEncoder or ResNet50"
                )

        self.create_model(backbone, **kwargs)
        self.create_loss()
        self.create_metrics(final_metrics)

    def create_model(self, backbone: BaseBackbone, **kwargs) -> None:
        head = VectorizedSAHCHead(
            backbone.n_features,
            256,
            self.lut,
            self.level_weights,
            self.mappings,
            self.ensemble_mode,
            **kwargs,
        )
        self.head_info = head.head_info
        model = SAHCModel(backbone, head)
        self.model = model if not self.compile else torch.compile(model)  # type: ignore

    def create_loss(self) -> None:
        self.loss = losses.SAHCLoss(
            self.taxonomy,
            self.level_weights,
            self.class_weights,
            ignore_index=255,
        )

    def create_metrics(self, final_metrics: torchmetrics.MetricCollection | None) -> None:
        metrics = {}
        for level in self.level_weights:
            num_classes = len(self.lut[level][1])
            for step_key in ("train", "val", "test"):
                for pkind in self.pred_kinds:
                    key = f"{step_key}-{self.taxonomy}-{level}-{pkind}"
                    metrics[key] = build_metrics(
                        num_classes=num_classes,
                        taxonomy=self.taxonomy,
                        prefix=f"{step_key}-",
                        postfix=f"-{self.taxonomy}-{level}-{pkind}",
                    )
        self.metrics = nn.ModuleDict(metrics)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[hrlc_utils.LevelPredictions, hrlc_utils.LevelTensor]:
        if x.dim() == 5:
            x = einops.rearrange(x, "b c t h w -> b t c h w").contiguous()
        logits5d, level_jsd5d = self.model(x)

        predictions: hrlc_utils.LevelPredictions = {}
        level_jsd: hrlc_utils.LevelTensor = {}
        for idx, info in enumerate(self.head_info):
            level = info["level"]
            nc = info["n_classes"]
            predictions[level] = {
                "OG": logits5d["OG"][:, idx, :nc],
                "SAHC": logits5d["SAHC"][:, idx, :nc],
            }
            level_jsd[level] = level_jsd5d[idx]
        return predictions, level_jsd

    def training_step(self, train_batch: dict[str, typing.Any], batch_idx: int):
        x = train_batch["x"]
        preds, level_jsd = self(x)
        sup_losses = self.loss(
            preds,
            train_batch[self.taxonomy],
            self.consistency_weight if self.supervised_cce_cons else 0.0,
        )

        loss_sup = torch.scalar_tensor(0.0, device=x.device)
        loss_cons = torch.scalar_tensor(0.0, device=x.device)
        for level, level_weight in self.level_weights.items():
            loss_sup += level_weight * torch.nan_to_num(sup_losses[level], 0)
            loss_cons += level_jsd[level]

        return {
            "loss": loss_sup + self.consistency_weight * loss_cons,
            "sup_losses": sup_losses,
            "level_jsd": level_jsd,
            "preds": preds,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        log_dict = {}
        for level in self.level_weights:
            log_dict[f"train-sup_loss-{self.taxonomy}-{level}"] = outputs[
                "sup_losses"
            ][level]
            log_dict[f"train-level_jsd-{self.taxonomy}-{level}"] = outputs[
                "level_jsd"
            ][level]
            for pkind in self.pred_kinds:
                self.metrics[f"train-{self.taxonomy}-{level}-{pkind}"].update(
                    outputs["preds"][level][pkind],
                    batch[self.taxonomy][level],
                )
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log_dict(
            {
                "train-loss": outputs["loss"],
                "lr": current_lr,
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_train_epoch_end(self):
        log_dict = {}
        for level in self.level_weights:
            for pkind in self.pred_kinds:
                metric_key = f"train-{self.taxonomy}-{level}-{pkind}"
                metric_dict = self.metrics[metric_key].compute()
                del metric_dict[f"train-conf_mat-{self.taxonomy}-{level}-{pkind}"]
                log_dict.update(metric_dict)
                self.metrics[metric_key].reset()
        self.log_dict(log_dict, on_epoch=True, on_step=False, sync_dist=True)

    def evaluation_step(self, batch: dict[str, typing.Any], batch_idx: int):
        x = batch["x"]
        preds, _ = self(x)
        losses = self.loss(preds, batch[self.taxonomy])
        loss = torch.scalar_tensor(0.0, device=x.device)
        for level, level_weight in self.level_weights.items():
            loss += level_weight * torch.nan_to_num(losses[level])
        return {"loss": loss, "sup_losses": losses, "preds": preds}

    def on_evaluation_batch_end(self, outputs, batch, batch_idx, eval_step):
        log_dict = {}
        for level in self.level_weights:
            if (batch[self.taxonomy][level].unique() == 255).all():
                continue
            log_dict[f"{eval_step}-sup_loss-{self.taxonomy}-{level}"] = outputs[
                "sup_losses"
            ][level]
            for pkind in self.pred_kinds:
                self.metrics[f"{eval_step}-{self.taxonomy}-{level}-{pkind}"].update(
                    outputs["preds"][level][pkind],
                    batch[self.taxonomy][level],
                )
        log_dict[f"{eval_step}-loss"] = outputs["loss"]
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

    def on_evaluation_epoch_end(self, eval_step: typing.Literal["val", "test"]):
        log_dict = {}
        for level in self.level_weights:
            for pkind in self.pred_kinds:
                metric_key = f"{eval_step}-{self.taxonomy}-{level}-{pkind}"
                metric_dict = self.metrics[metric_key].compute()
                del metric_dict[f"{eval_step}-conf_mat-{self.taxonomy}-{level}-{pkind}"]
                log_dict.update(metric_dict)
        self.log_dict(log_dict, on_epoch=True, on_step=False, sync_dist=True)

    def on_evaluation_epoch_start(self, eval_step: typing.Literal["val", "test"]):
        for level in self.level_weights:
            for pkind in self.pred_kinds:
                self.metrics[
                    f"{eval_step}-{self.taxonomy}-{level}-{pkind}"
                ].reset()

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.on_evaluation_batch_end(outputs, batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self.on_evaluation_epoch_end("val")

    def on_validation_epoch_start(self):
        self.on_evaluation_epoch_start("val")

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        self.on_evaluation_batch_end(outputs, batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self.on_evaluation_epoch_end("test")

    def on_test_epoch_start(self):
        self.on_evaluation_epoch_start("test")

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        preds, _ = self(batch["x"])
        return preds, batch["y_c"], batch["x_c"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if not self.use_scheduler:
            return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=round((self.trainer.max_epochs or 1) * 0.8),
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]


class ConsistencyWarmupCallback(L.Callback):
    def __init__(self, max_weight: float):
        super().__init__()
        self.max_weight = max_weight
        self.hpo_epoch_start = 5
        self.hpo_epoch_end = 15

    def on_train_batch_start(self, trainer, pl_module: LightningSAHC, batch, batch_idx):
        steps_per_epoch = trainer.num_training_batches
        start_step = self.hpo_epoch_start * steps_per_epoch
        end_step = self.hpo_epoch_end * steps_per_epoch
        current_step = trainer.global_step
        if current_step < start_step:
            pl_module.consistency_weight = 0.0
        elif current_step >= end_step:
            pl_module.consistency_weight = self.max_weight
        else:
            pct = (current_step - start_step) / (end_step - start_step)
            pl_module.consistency_weight = self.max_weight * pct
        pl_module.log("consistency_weight", pl_module.consistency_weight)
