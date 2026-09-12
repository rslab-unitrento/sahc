from itertools import permutations

import pandas as pd
import torch

import hrlc_utils


def get_lookup_tables_and_mappings(
    path: str,
) -> tuple[
    dict[hrlc_utils.Taxonomy, dict[hrlc_utils.Level, tuple[torch.Tensor, list[str]]]],
    dict[
        hrlc_utils.Taxonomy,
        dict[tuple[hrlc_utils.Level, hrlc_utils.Level], torch.Tensor],
    ],
]:
    """Parse taxonomy lookup tables and mappings between its hierarchy levels."""
    xls_dict = pd.read_excel(path, sheet_name=None)

    lookup_tables = {}
    mapping_tables = {}
    taxonomy_sheet_names = [s for s in xls_dict.keys() if not s.startswith("Level_")]

    for sheet_name in taxonomy_sheet_names:
        df = xls_dict[sheet_name]

        if "Code" not in df.columns:
            print(f"Skipping taxonomy sheet '{sheet_name}': 'Code' column missing.")
            continue

        df = df.dropna(subset=["Code"]).copy()
        df["Code"] = df["Code"].astype(int)

        if df["Code"].max() >= 256 or df["Code"].min() < 0:
            raise ValueError(f"Sheet '{sheet_name}' contains Codes outside [0, 255].")

        level_cols = [col for col in df.columns if col.startswith("Level_")]

        sheet_lookup = {}
        sheet_class_map = {}

        for level in level_cols:
            unique_labels = sorted(df[level].unique().tolist())
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            sheet_class_map[level] = label_to_id

            lookup_tensor = torch.full((256,), 255, dtype=torch.long)
            codes = df["Code"].values
            ids = [label_to_id[label] for label in df[level]]
            lookup_tensor[codes] = torch.tensor(ids, dtype=torch.long)  # type: ignore

            sheet_lookup[level] = (lookup_tensor, unique_labels)

        lookup_tables[sheet_name] = sheet_lookup
        sheet_hierarchy = {}

        for src_level, dst_level in permutations(level_cols, 2):
            src_map = sheet_class_map[src_level]
            dst_map = sheet_class_map[dst_level]

            n_src = len(src_map)
            n_dst = len(dst_map)

            mat = torch.zeros((n_src, n_dst), dtype=torch.float32)

            for s_label, d_label in zip(df[src_level], df[dst_level]):
                s_id = src_map[s_label]
                d_id = dst_map[d_label]
                mat[s_id, d_id] = 1.0

            sheet_hierarchy[(src_level, dst_level)] = mat

        mapping_tables[sheet_name] = sheet_hierarchy

    return (lookup_tables, mapping_tables)
