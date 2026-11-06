"""Writes COMPASS inputs with germline information: <setting>/rep*/COMPASS_snp/, next to COMPASS/.

COMPASS reads an optional FREQ column (population frequency). Here FREQ is the alternative allele fraction at the root of the simulated tree,
read from SCATI/data_n_alt_root.csv and data_n_ref_root.csv: 0.5 (heterozygous), 1.0 (homozygous alternative) or 0 (not at the root).
Inputs go to COMPASS_snp/data_{variants,regions}.csv (multi-sample: COMPASS_snp/sampleN/), results to COMPASS_snp/results/.

Usage, from simulations/:
    python generate/make_compass_snp_inputs.py --sim_root simulated_data
Existing COMPASS_snp files are kept unless --force is given; --dry_run only counts.
"""
import argparse
import os
import shutil
import sys

import pandas as pd


def convert_one(src_dir, dst_dir, freq, force):
    """Write one COMPASS_snp input in dst_dir from the COMPASS input in src_dir, adding the FREQ column after NAME.
    Returns (n_variants, n_germline), or None if the output exists and force is not set."""
    out_var = os.path.join(dst_dir, "data_variants.csv")
    if os.path.exists(out_var) and not force:
        return None
    var = pd.read_csv(os.path.join(src_dir, "data_variants.csv"), dtype=str)
    if len(var) != len(freq):
        raise ValueError(f"{src_dir}: {len(var)} variants but {len(freq)} root allele counts")
    if "FREQ" in var.columns:
        raise ValueError(f"{src_dir}: the variants file already has a FREQ column")
    var.insert(list(var.columns).index("NAME") + 1, "FREQ", [f"{f:g}" for f in freq])
    os.makedirs(dst_dir, exist_ok=True)
    var.to_csv(out_var, index=False, lineterminator="\n")
    shutil.copyfile(os.path.join(src_dir, "data_regions.csv"), os.path.join(dst_dir, "data_regions.csv"))
    return len(var), int((freq > 0).sum())


def convert_rep(rep_dir, force=False, dry_run=False):
    """Write <rep_dir>/COMPASS_snp/ from <rep_dir>/COMPASS/ (one input, or one per sampleN/).
    Returns one convert_one() result per input, or [] if there is no COMPASS/ folder."""
    comp = os.path.join(rep_dir, "COMPASS")
    if not os.path.isdir(comp):
        return []
    alt_file, ref_file = os.path.join(rep_dir, "SCATI", "data_n_alt_root.csv"), os.path.join(rep_dir, "SCATI", "data_n_ref_root.csv")
    for f in (alt_file, ref_file):
        if not os.path.exists(f):
            sys.exit(f"missing {f}")
    n_alt, n_ref = pd.read_csv(alt_file, header=None)[0], pd.read_csv(ref_file, header=None)[0]
    total = (n_alt + n_ref).astype(float)
    freq = (n_alt / total.where(total > 0)).fillna(0.0)        # alternative allele fraction at the root
    units = [("", comp)] if os.path.isfile(os.path.join(comp, "data_variants.csv")) else \
        [(d, os.path.join(comp, d)) for d in sorted(os.listdir(comp)) if d.startswith("sample") and os.path.isdir(os.path.join(comp, d))]
    return [(0, 0) if dry_run else convert_one(src, os.path.join(rep_dir, "COMPASS_snp", sub), freq, force) for sub, src in units]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim_root", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    made = skipped = 0
    n_var = n_germ = 0
    for setting in sorted(os.listdir(a.sim_root)):
        sdir = os.path.join(a.sim_root, setting)
        if not os.path.isdir(sdir):
            continue
        for rep in sorted(os.listdir(sdir)):
            for res in convert_rep(os.path.join(sdir, rep), a.force, a.dry_run):
                if res is None:
                    skipped += 1
                else:
                    made += 1
                    n_var += res[0]
                    n_germ += res[1]
    what = "would be written" if a.dry_run else "written"
    print(f"{made} COMPASS_snp input(s) {what}, {skipped} already existed" + (f"; {n_germ} of {n_var} variants with FREQ > 0 (germline)" if not a.dry_run and n_var else ""))


if __name__ == "__main__":
    main()
