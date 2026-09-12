import glob
import os

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from numcodecs import Blosc

import scripting

BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def remap_labels(gt_da, possible_labels, no_data_val=255):
    valid_labels = possible_labels[possible_labels != no_data_val]
    max_val = max(int(possible_labels.max()), int(np.nanmax(gt_da.values)))
    lut = np.full(max_val + 1, no_data_val, dtype=np.int32)
    for i, val in enumerate(valid_labels):
        lut[val] = i
    remapped_data = lut[gt_da.values.astype(int)]
    return gt_da.copy(data=remapped_data)


def convert_to_zarr(source_folder, gt_path, output_zarr_path, lut_path, **kwargs):
    print("Loading Ground Truth...")
    df = pd.read_excel(lut_path)
    possible_labels = df["Code"]

    print("Loading Time Series...")
    months = [f"{i:02d}" for i in range(1, 13)]
    monthly_composites = []

    for m in months:
        search_pattern = os.path.join(
            source_folder, f"M10_2020{m}01_2020{m}*_32tpq_elu"
        )
        matches = glob.glob(search_pattern)
        if not matches:
            raise FileNotFoundError(
                f"Could not find folder matching pattern: {search_pattern}"
            )
        month_dir = matches[0]
        subdirs = [
            d
            for d in os.listdir(month_dir)
            if os.path.isdir(os.path.join(month_dir, d))
        ]
        inner_folder = subdirs[0]
        tile_path = os.path.join(month_dir, inner_folder, "tile_0")
        band_das = []
        for b in BAND_NAMES:
            tif_path = os.path.join(tile_path, f"{b}.tif")
            if os.path.exists(tif_path):
                da = rioxarray.open_rasterio(tif_path, chunks={"x": 1024, "y": 1024})
                band_das.append(da)
            else:
                print(f"Warning: Band {b} missing in {tile_path}")
        month_ds = xr.concat(band_das, dim="band")
        month_ds = month_ds.assign_coords(band=BAND_NAMES[: len(band_das)])
        monthly_composites.append(month_ds)

    ds_ts = xr.concat(monthly_composites, dim="time")  # type:ignore

    gt = rioxarray.open_rasterio(
        gt_path, chunks={"x": 1024, "y": 1024}
    ).rio.reproject_match(ds_ts)  # type: ignore
    gt = remap_labels(gt, possible_labels, 255)
    ds = xr.Dataset({"imagery": ds_ts, "label": gt.drop_vars("band").squeeze("band")})

    ds = ds.chunk({"time": -1, "band": -1, "y": 32, "x": 32})
    print("Writing to Zarr (this may take time)...")
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    encoding = {v: {"compressor": compressor} for v in ds.data_vars}

    ds.to_zarr(output_zarr_path, mode="w", consolidated=True, encoding=encoding)
    print("Done.")


if __name__ == "__main__":
    scripting.logged_main(
        "Zarr conversion from Geotiff",
        convert_to_zarr,
    )
