"""Filters the input of one sample: drops regions with too few reads and the variants in them.

    python filtering.py --sample_dir AML/AML-59 --sample AML-59 --out_dir rerun/AML-59/SCATI

Reads <sample_dir>/input/<sample>_{variants,regions}.csv and writes <out_dir>/<sample>_{variants,regions}_filtered.csv.
A region is dropped if its mean read count over all cells is below --min_avg_region_count (default 1). A variant is dropped if its region is.
The regions file is written without a header row, which is what SCATI reads.
"""
import argparse
import os
import sys

import pandas as pd

META_COLS = ["CHR", "POS", "REF", "ALT", "REGION", "NAME", "FREQ"]


def read_regions(path):
    """The regions file (first column "chr_regionName", then one count per cell); returns the table and the cell names if it has a header row."""
    with open(path) as f:
        first = f.readline().rstrip("\n").split(",")
    has_header = False
    for v in first[1:]:                       # a header row holds cell names, a data row only numbers
        try:
            float(v or 0)
        except ValueError:
            has_header = True
    if has_header:
        df = pd.read_csv(path, index_col=0)
        return df, list(df.columns)
    df = pd.read_csv(path, header=None, index_col=0)
    df.columns = range(df.shape[1])
    return df, None


def filter_sample(sample_dir, sample, out_dir, min_avg_region_count=1.0):
    base = os.path.join(sample_dir, "input", sample)
    variants = pd.read_csv(base + "_variants.csv", dtype=str)
    cells = [c for c in variants.columns if c not in META_COLS]
    regions, region_cells = read_regions(base + "_regions.csv")

    # both files must hold the same cells in the same order (SCATI reads the regions file by position)
    if regions.shape[1] != len(cells):
        sys.exit(f"{sample}: {len(cells)} cells in the variants file but {regions.shape[1]} in the regions file")
    if region_cells is not None and region_cells != cells:
        sys.exit(f"{sample}: the cells of the regions file differ from those of the variants file (names or order)")
    print(f"{sample}: {len(variants)} variants, {len(regions)} regions, {len(cells)} cells")

    mean_count = regions.mean(axis=1)
    drop = mean_count[(mean_count < min_avg_region_count) | (mean_count == 0)]
    for region, m in drop.items():
        print(f"  drop region {region}: mean count {m:.4f}")
    kept = regions.drop(index=drop.index)

    kept_names = {r.partition("_")[2] for r in kept.index}          # "chr_regionName": the name is what follows the first underscore
    keep_variant = variants["REGION"].isin(kept_names)
    for _, v in variants[~keep_variant].iterrows():
        print(f"  drop variant {v['NAME']} (region {v['REGION']})")
    variants = variants[keep_variant]

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, sample)
    kept.index.name = "region_id"
    kept.to_csv(out + "_regions_filtered.csv", header=False, lineterminator="\n")         # LF: the C++ reader splits lines on \n only
    variants.to_csv(out + "_variants_filtered.csv", index=False, lineterminator="\n")
    print(f"{sample}: kept {len(kept)}/{len(regions)} regions, {len(variants)} variants")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample_dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_avg_region_count", type=float, default=1.0)
    a = ap.parse_args()
    filter_sample(a.sample_dir, a.sample, a.out_dir, a.min_avg_region_count)
