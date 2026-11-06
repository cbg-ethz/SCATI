"""Inputs of COMPASS and of LoPhy from the filtered SCATI input of one sample, without any germline information.

    python baseline_inputs.py <base of the filtered files> <COMPASS folder> <LoPhy folder>
    e.g. python baseline_inputs.py AML/AML-23/SCATI/AML-23 baselines/AML-23/COMPASS baselines/AML-23/LoPhy

COMPASS: data_variants.csv / data_regions.csv, the filtered files (FREQ dropped if present, so no variant is pinned to the root; the regions file
without a header row). LoPhy: character_matrix.csv (genotype 0 / 1 / 2 / 3 of the variants file as 0 / 1 / 1 / -1), variant_reads.csv,
total_reads.csv, region_reads.csv (cells x columns), cell_samples.txt (one sample) and meta.csv with SNP = 0 for every variant (SBS = single-base
substitution).
"""
import os
import shutil
import sys

import pandas as pd

META_COLS = ["CHR", "POS", "REF", "ALT", "REGION", "NAME", "FREQ"]


def write_baseline_inputs(base, compass_dir, lophy_dir):
    vpath = base + "_variants_filtered.csv" if os.path.exists(base + "_variants_filtered.csv") else base + "_variants.csv"
    rpath = base + "_regions_filtered.csv" if os.path.exists(base + "_regions_filtered.csv") else base + "_regions.csv"
    var = pd.read_csv(vpath, dtype=str)
    cells = [c for c in var.columns if c not in META_COLS]
    n_loci, n_cells = len(var), len(cells)
    parts = var[cells].stack().str.split(":", n=2, expand=True)
    ref = parts[0].astype(int).unstack()[cells].to_numpy()
    alt = parts[1].astype(int).unstack()[cells].to_numpy()
    gt = parts[2].astype(int).unstack()[cells].to_numpy()
    with open(rpath) as f:
        has_header = f.readline().split(",")[0].strip() == ""      # a header row of cell names starts with an empty field
    reg = pd.read_csv(rpath, header=0 if has_header else None, index_col=0)
    region_names = [str(r) for r in reg.index]
    depths = reg.values.astype(int)
    if depths.shape[1] != n_cells:
        sys.exit(f"{n_cells} cells in the variants file but {depths.shape[1]} in the regions file")

    # COMPASS
    os.makedirs(compass_dir, exist_ok=True)
    var.drop(columns=[c for c in ("FREQ",) if c in var.columns]).to_csv(
        os.path.join(compass_dir, "data_variants.csv"), index=False, lineterminator="\n")
    if has_header:
        reg.to_csv(os.path.join(compass_dir, "data_regions.csv"), header=False, lineterminator="\n")
    else:
        shutil.copyfile(rpath, os.path.join(compass_dir, "data_regions.csv"))

    # LoPhy (LF line endings: its csv reader does not strip a carriage return)
    os.makedirs(lophy_dir, exist_ok=True)
    calls = gt.copy()
    calls[calls == 2] = 1
    calls[calls == 3] = -1
    names = var["NAME"].tolist()

    def write(mat, cols, fname):
        pd.DataFrame(mat.T, index=cells, columns=cols).to_csv(os.path.join(lophy_dir, fname), lineterminator="\n")

    write(calls, names, "character_matrix.csv")
    write(alt, names, "variant_reads.csv")
    write(ref + alt, names, "total_reads.csv")
    write(depths, region_names, "region_reads.csv")
    with open(os.path.join(lophy_dir, "cell_samples.txt"), "w", newline="\n") as f:
        f.write("".join("0\n" for _ in cells))
    region_index = {r[r.find("_") + 1:]: k for k, r in enumerate(region_names)}   # a variant's REGION = region name without "<chr>_"
    rows = []
    for i in range(n_loci):
        region = var["REGION"].iloc[i]
        if region not in region_index:
            sys.exit(f"variant {names[i]}: region {region} not in the regions file")
        is_sbs = len(var["REF"].iloc[i]) == 1 and len(var["ALT"].iloc[i]) == 1
        rows.append({"CHR": var["CHR"].iloc[i], "REGION_INDEX": region_index[region], "SNP": 0, "SBS": int(is_sbs), "SAMPLE": 0,
                     "NAME": names[i]})
    pd.DataFrame(rows).to_csv(os.path.join(lophy_dir, "meta.csv"), index=False, lineterminator="\n")
    return n_loci, len(region_names), n_cells


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    n_loci, n_reg, n_cells = write_baseline_inputs(*sys.argv[1:])
    print(f"{n_loci} variants, {n_reg} regions, {n_cells} cells -> {sys.argv[2]}, {sys.argv[3]}")
