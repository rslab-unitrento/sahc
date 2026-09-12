import pathlib

import numpy as np
import pandas as pd
import rasterio as rio

import scripting


def remapper(
    source_folder: str | pathlib.Path,
    output_folder: str | pathlib.Path,
    lut_path: str | pathlib.Path,
    out_no_data_val=255,
    **kwargs,
):
    if isinstance(source_folder, str):
        source_folder = pathlib.Path(source_folder)

    if isinstance(output_folder, str):
        output_folder = pathlib.Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    lut_path = pathlib.Path(lut_path)
    df = pd.read_excel(lut_path, sheet_name="ELU")

    level_name = "Level_1"
    if not 0 <= out_no_data_val <= 255:
        raise ValueError("out_no_data_val must be in [0, 255]")
    if "Code" not in df or level_name not in df:
        raise ValueError(f"{lut_path} must contain Code and {level_name} columns")

    pairs = df.dropna(subset=["Code", level_name])[["Code", level_name]].copy()
    pairs["Code"] = pairs["Code"].astype(int)
    duplicate_names = pairs.groupby(level_name)["Code"].nunique()
    if (duplicate_names > 1).any():
        names = duplicate_names[duplicate_names > 1].index.tolist()
        raise ValueError(f"Ambiguous original codes for {names[:5]}")
    class_names = sorted(pairs[level_name].unique().tolist())
    code_by_name = pairs.drop_duplicates(level_name).set_index(level_name)["Code"]
    direct_reverse_lut = np.full(256, out_no_data_val, dtype=np.uint16)
    for network_class, class_name in enumerate(class_names):
        if class_name not in code_by_name:
            raise ValueError(f"Missing {level_name} mapping for {class_name!r}")
        direct_reverse_lut[network_class] = code_by_name[class_name]

    # The round-trip protects class ordering: network id -> original code -> name
    # must recover the same network id for every retained class.
    name_by_code = dict(zip(pairs["Code"], pairs[level_name]))
    for network_class, original_code in enumerate(direct_reverse_lut[: len(class_names)]):
        if name_by_code[int(original_code)] != class_names[network_class]:
            raise ValueError("Taxonomy remapping round-trip failed")

    tif_list = list(source_folder.glob("*.tif"))
    for img_path in tif_list:
        with rio.open(img_path) as src:
            data = src.read()
            profile = src.profile
            if data.max(initial=0) >= len(direct_reverse_lut):
                raise ValueError("Prediction contains a class outside the [0, 255] range")
            remapped_data = direct_reverse_lut[data]
            profile.update(dtype=rio.uint16, nodata=out_no_data_val)
            out_path = output_folder / img_path.name
            with rio.open(out_path, "w", **profile) as dst:
                dst.write(remapped_data)


if __name__ == "__main__":
    scripting.logged_main(
        "Remapper from architecture output to original",
        remapper,
    )
