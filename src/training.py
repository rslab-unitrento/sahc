from __future__ import annotations

import copy
import json
import logging
import pathlib
import typing
from collections.abc import Mapping

import lightning as L
import yaml
from lightning.pytorch import callbacks, loggers

import datasets
import models

_LOG = logging.getLogger(__name__)


def _write_resolved_config(path: pathlib.Path, config: Mapping[str, typing.Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )


def _build_logger(
    run_dir: pathlib.Path,
    experiment_name: str,
    trial_id: str,
    mlflow_config: Mapping[str, typing.Any],
):
    if mlflow_config.get("enabled", False):
        from lightning.pytorch.loggers import MLFlowLogger

        return MLFlowLogger(
            experiment_name=experiment_name,
            run_name=trial_id,
            tracking_uri=mlflow_config.get("tracking_uri"),
        )
    return loggers.CSVLogger(str(run_dir / "logs"), name="lightning", version="")


def train(
    *,
    seed: int,
    trial_id: str,
    save_path: str,
    experiment_name: str,
    dataset_config: Mapping[str, typing.Any],
    model_config: Mapping[str, typing.Any],
    trainer_config: Mapping[str, typing.Any],
    metaparameters: Mapping[str, typing.Any],
    mlflow_config: Mapping[str, typing.Any] | None = None,
    ray_tune_config: Mapping[str, typing.Any] | None = None,
    final_test: bool = True,
    use_ray: bool = False,
    resolved_config: Mapping[str, typing.Any] | None = None,
) -> pathlib.Path:
    """Train one seed/trial and return its isolated run directory."""

    epochs = int(metaparameters["epochs"])
    if epochs < 1:
        raise ValueError("metaparameters.epochs must be a positive integer")

    L.seed_everything(seed, workers=True)
    run_dir = pathlib.Path(save_path) / experiment_name / trial_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_config = copy.deepcopy(dict(dataset_config))
    datamodule_name = data_config.pop("datamodule")
    if "num_workers" not in data_config and "num_workers" in trainer_config:
        data_config["num_workers"] = trainer_config["num_workers"]
    dm_class = getattr(datasets, datamodule_name)
    dm = dm_class(
        batch_size=int(metaparameters["batch_size"]),
        seed=seed,
        debug=bool(trial_id == "debug"),
        **data_config,
    )
    dm.prepare_data()
    dm.setup()

    model_config = copy.deepcopy(dict(model_config))
    model_name = model_config.pop("lightning_module")
    model_config.setdefault("compile", False if trial_id == "debug" else True)
    resolved_model_config = copy.deepcopy(model_config)
    resolved_model_config["lightning_module"] = model_name
    model_class = getattr(models, model_name)
    model_config.update(lut=dm.lookup_tensors, mappings=dm.mappings)
    model = model_class(**dict(metaparameters), **model_config)

    mlflow_config = mlflow_config or {}
    logger = _build_logger(run_dir, experiment_name, trial_id, mlflow_config)
    callback_list: list[typing.Any] = [
        models.ConsistencyWarmupCallback(float(metaparameters["consistency_weight"]))
    ]
    if model_config.get("swa", False):
        callback_list.append(
            callbacks.StochasticWeightAveraging(
                swa_epoch_start=0.8,
                swa_lrs=0.2 * float(metaparameters["lr"]),
                annealing_epochs=5,
                annealing_strategy="cos",
            )
        )

    trainer_kwargs = {
        "accelerator": trainer_config.get("accelerator", "auto"),
        "strategy": trainer_config.get("strategy", "auto"),
        "devices": trainer_config.get("devices", "auto"),
        "num_nodes": trainer_config.get("num_nodes", 1),
        "precision": trainer_config.get("precision", "32-true"),
        "gradient_clip_val": trainer_config.get("gradient_clip_val", 10),
        "logger": logger,
        "callbacks": callback_list,
        "max_epochs": epochs,
        "max_steps": trainer_config.get("max_steps", -1),
        "limit_train_batches": trainer_config.get("limit_train_batches", 1.0),
        "limit_val_batches": trainer_config.get("limit_val_batches", 1.0),
        "limit_test_batches": trainer_config.get("limit_test_batches", 1.0),
        "num_sanity_val_steps": trainer_config.get("num_sanity_val_steps", 0),
        "log_every_n_steps": trainer_config.get("log_every_n_steps", 1),
        "enable_progress_bar": not use_ray,
        "default_root_dir": str(run_dir),
    }
    fast_dev_run = trainer_config.get("fast_dev_run", False)
    if fast_dev_run:
        trainer_kwargs["fast_dev_run"] = fast_dev_run
    trainer = L.Trainer(**trainer_kwargs)

    resolved = copy.deepcopy(dict(resolved_config or {}))
    resolved.setdefault("seed", seed)
    resolved.setdefault("trial_id", trial_id)
    resolved.setdefault(
        "dataset_config", {"datamodule": datamodule_name, **data_config}
    )
    resolved.setdefault("model_config", resolved_model_config)
    resolved.setdefault("metaparameters", dict(metaparameters))
    resolved.setdefault("trainer_config", dict(trainer_config))
    _write_resolved_config(run_dir / "resolved_config.yaml", resolved)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    trainer.fit(model, datamodule=dm)
    checkpoint_path = checkpoint_dir / "last-swa.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))
    if final_test:
        trainer.test(model, datamodule=dm, ckpt_path=str(checkpoint_path))
    return run_dir


def _run_ray(
    *,
    seed: int,
    experiment_name: str,
    ray_tune_config: dict[str, typing.Any],
    metaparameters: dict[str, typing.Any],
    config: dict[str, typing.Any],
):
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
    import optuna

    started_here = not ray.is_initialized()
    if started_here:
        ray.init()
    try:
        search_space = {}
        for key, spec in ray_tune_config.get("search_space", {}).items():
            kind = spec["type"]
            if kind == "loguniform":
                search_space[key] = tune.loguniform(float(spec["min"]), float(spec["max"]))
            elif kind == "uniform":
                search_space[key] = tune.uniform(float(spec["min"]), float(spec["max"]))
            elif kind == "choice":
                search_space[key] = tune.choice(spec["values"])
            else:
                raise ValueError(f"Unsupported Ray search-space type: {kind}")

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(ray_tune_config.get("startup_trials", 0)),
            seed=seed,
        )
        search = OptunaSearch(
            metric="target_metric",
            mode=ray_tune_config.get("mode", "max"),
            sampler=sampler,
            seed=seed,
        )

        def trainable(selected):
            params = dict(metaparameters)
            params.update(selected)
            run_config = copy.deepcopy(config)
            run_config["trial_id"] = f"trial_{tune.get_trial_id()}"
            run_dir = train(
                seed=seed,
                trial_id=run_config["trial_id"],
                save_path=run_config["save_path"],
                experiment_name=experiment_name,
                dataset_config=run_config["dataset_config"],
                model_config=run_config["model_config"],
                trainer_config=run_config["trainer_config"],
                metaparameters=params,
                mlflow_config=run_config.get("mlflow_config", {}),
                final_test=False,
                use_ray=True,
                resolved_config=run_config,
            )
            metric_name = ray_tune_config.get("target_metric")
            metric_value = _read_metric(run_dir, metric_name)
            tune.report(target_metric=metric_value)

        num_samples = int(ray_tune_config.get("tune_trials", 1))
        if num_samples < 1:
            raise ValueError("ray_tune_config.tune_trials must be positive")
        tuner = tune.Tuner(
            tune.with_resources(trainable, {"cpu": 1, "gpu": 1}),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                search_alg=search,
                num_samples=num_samples,
            ),
            run_config=ray.air.RunConfig(name=experiment_name),
        )
        result = tuner.fit()
        best = result.get_best_result(
            metric="target_metric",
            mode=ray_tune_config.get("mode", "max"),
        )
        return dict(best.config)
    finally:
        if started_here:
            ray.shutdown()


def _read_metric(run_dir: pathlib.Path, metric_name: str | None) -> float:
    """Read the final validation metric emitted by the local CSV logger."""

    if not metric_name:
        raise ValueError("ray_tune_config.target_metric is required for Ray tuning")
    metrics_path = run_dir / "logs" / "lightning" / "metrics.csv"
    if not metrics_path.is_file():
        raise RuntimeError(f"CSV logger did not produce {metrics_path}")
    values: list[float] = []
    import csv

    with metrics_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            value = row.get(metric_name)
            if value not in (None, ""):
                values.append(float(value))
    if not values:
        raise RuntimeError(f"Metric {metric_name!r} was not logged in {metrics_path}")
    return values[-1]


def run_from_config(
    *,
    seed: int | list[int],
    use_ray: bool,
    debug: bool,
    metaparameters: dict[str, typing.Any],
    ray_tune_config: dict[str, typing.Any] | None = None,
    **config: typing.Any,
) -> list[pathlib.Path]:
    """Run optional tuning, then final/SWA evaluation for scalar or list seeds."""

    ray_tune_config = copy.deepcopy(ray_tune_config or {})
    selected_seed = seed[0] if isinstance(seed, list) else seed
    if use_ray:
        metaparameters.update(
            _run_ray(
                seed=selected_seed,
                experiment_name=config["experiment_name"],
                ray_tune_config=ray_tune_config,
                metaparameters=metaparameters,
                config=config,
            )
        )
        metaparameters["epochs"] = int(
            ray_tune_config.get("final_epochs", metaparameters["epochs"])
        )

    seeds = [selected_seed] if debug else (
        seed if isinstance(seed, list) else [seed]
    )
    runs = []
    for run_seed in seeds:
        trial_id = "debug" if debug else f"final_run_{run_seed}"
        runs.append(
            train(
                seed=run_seed,
                trial_id=trial_id,
                save_path=config["save_path"],
                experiment_name=config["experiment_name"],
                dataset_config=config["dataset_config"],
                model_config=config["model_config"],
                trainer_config=config["trainer_config"],
                metaparameters=copy.deepcopy(metaparameters),
                mlflow_config=config.get("mlflow_config", {}),
                ray_tune_config=ray_tune_config,
                final_test=True,
                use_ray=False,
                resolved_config={
                    **config,
                    "metaparameters": copy.deepcopy(metaparameters),
                    "seed": run_seed,
                    "trial_id": trial_id,
                },
            )
        )
    return runs
