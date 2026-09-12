from __future__ import annotations

import json
import pathlib
import typing

import numpy as np
import rioxarray  # noqa: F401
import torch
import xarray as xr

import datasets
import models
import scripting


def _normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove only the compile wrapper used by the saved Lightning models."""

    normalized = {}
    prefix = "model._orig_mod."
    for key, value in state_dict.items():
        normalized_key = (
            "model." + key[len(prefix) :]
            if key.startswith(prefix)
            else key
        )
        if normalized_key in normalized:
            raise RuntimeError(f"Checkpoint key normalization collision: {normalized_key}")
        normalized[normalized_key] = value
    return normalized


def _matching_state_dict(
    state_dict: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized = _normalize_state_dict_keys(state_dict)
    if set(normalized) == set(expected):
        return normalized
    missing = sorted(set(expected) - set(normalized))
    unexpected = sorted(set(normalized) - set(expected))
    raise RuntimeError(
        "Checkpoint keys do not match the uncompiled model. "
        f"missing={missing[:5]} unexpected={unexpected[:5]}"
    )


def select_prediction(
    predictions: dict,
    level: str,
    prediction_kind: typing.Literal["OG", "SAHC"],
) -> torch.Tensor:
    if prediction_kind not in {"OG", "SAHC"}:
        raise ValueError("prediction_kind must be OG or SAHC")
    try:
        return predictions[level][prediction_kind]
    except KeyError as error:
        raise ValueError(
            f"Prediction is missing level={level!r}, kind={prediction_kind!r}"
        ) from error


def _labels_sidecar(path: pathlib.Path, class_names: list[str]) -> None:
    path.with_suffix(path.suffix + ".labels.json").write_text(
        json.dumps(
            {
                "class_id_semantics": "zero-based model class ID",
                "nodata": 255,
                "taxonomy_code": None,
                "classes": [
                    {"class_id": index, "name": name}
                    for index, name in enumerate(class_names)
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def predict(
    *,
    checkpoint_path: str,
    output_path: str,
    level: str,
    prediction_kind: typing.Literal["OG", "SAHC"],
    dataset_config: dict,
    model_config: dict,
    metaparameters: dict | None = None,
    max_batches: int | None = None,
    device: str = "cpu",
    crs: str | None = None,
    nodata: int = 255,
) -> pathlib.Path:
    if prediction_kind not in {"OG", "SAHC"}:
        raise ValueError("prediction_kind must be OG or SAHC")
    if not 0 <= nodata <= 255:
        raise ValueError("nodata must be in [0, 255]")
    if max_batches is not None and max_batches < 0:
        raise ValueError("max_batches must be non-negative")

    dataset_config = dict(dataset_config)
    dm_name = dataset_config.pop("datamodule")
    dm_class = getattr(datasets, dm_name)
    dm = dm_class(**dataset_config)
    dm.prepare_data()
    dm.setup(stage="predict")
    if not hasattr(dm, "predict_dataloader"):
        raise ValueError(f"{dm_name} does not provide a prediction dataloader")
    model_config = dict(model_config)
    if model_config.pop("lightning_module", "LightningSAHC") != "LightningSAHC":
        raise ValueError("Inference supports LightningSAHC only")
    taxonomy = model_config.get("taxonomy")
    if taxonomy not in dm.lookup_tensors:
        raise ValueError(f"Unknown taxonomy selection: taxonomy={taxonomy!r}")
    if level not in dm.lookup_tensors[taxonomy]:
        raise ValueError(f"Unknown level selection: level={level!r}")
    model_config.update(lut=dm.lookup_tensors, mappings=dm.mappings, compile=False)
    model = models.LightningSAHC(
        **(metaparameters or {}),
        **model_config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _matching_state_dict(state_dict, model.state_dict())
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    zarr_path = dataset_config.get("zarr_path")
    if zarr_path is None:
        raise ValueError("dataset_config.zarr_path is required for geospatial output")
    ds = xr.open_zarr(zarr_path)
    if "label" not in ds:
        raise ValueError("Input Zarr metadata must contain a label array")
    height, width = ds["label"].shape[-2:]
    y_coords = ds.coords.get("y", np.arange(height))
    x_coords = ds.coords.get("x", np.arange(width))
    input_crs = getattr(getattr(ds, "rio", None), "crs", None)
    crs = str(input_crs) if input_crs is not None else crs
    if crs is None:
        raise ValueError("Input metadata has no CRS; set an explicit crs")

    full_map = np.full((height, width), nodata, dtype=np.uint8)
    loader = dm.predict_dataloader()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            outputs, *_ = model(batch["x"].to(device))
            logits = select_prediction(outputs, level, prediction_kind)
            predictions = logits.argmax(dim=1).cpu().numpy()
            y_batch = np.asarray(batch["y_c"])
            x_batch = np.asarray(batch["x_c"])
            for prediction, y, x in zip(predictions, y_batch, x_batch):
                full_map[int(y) - 1 : int(y) + 2, int(x) - 1 : int(x) + 2] = prediction

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(
        full_map,
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        name=f"{taxonomy}_{level}_{prediction_kind}",
    )
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_nodata(nodata, inplace=True)
    da.rio.to_raster(path, compress="deflate")
    _labels_sidecar(path, dm.lookup_tensors[taxonomy][level][1])
    return path


def main(**config):
    for key in ("yaml_filepath", "log_dir", "log_level", "log_artifacts"):
        config.pop(key, None)
    return predict(**config)


if __name__ == "__main__":
    scripting.logged_main("SAHC checkpoint inference", main)
