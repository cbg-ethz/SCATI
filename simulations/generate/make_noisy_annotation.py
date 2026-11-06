"""Write a copy of <base>.cell_annotation.csv (columns barcode,cluster,assignment; assignment is healthy or tumor)
with some labels flipped.
  --healthy_as_cancer f :      a fraction f of the healthy cells is labelled tumor
  --cancer_in_healthy_set c :  tumor cells (singlets and doublets) are labelled healthy until a fraction c of the healthy-labelled cells are cancer
Cells are flipped in random order given by --seed, so the same dataset and seed give the same file and
a lower error level flips a subset of the cells flipped at a higher level.

Usage: python make_noisy_annotation.py --annotation data.cell_annotation.csv --out data_annot_x.cell_annotation.csv \
           --cancer_in_healthy_set 0.05 --healthy_as_cancer 0.10 --seed 1
"""
import argparse

import numpy as np
import pandas as pd


def noisy_annotation(df, cancer_in_healthy_set, healthy_as_cancer, seed):
    out = df.copy()
    healthy = (df["assignment"] == "healthy").to_numpy()
    rng = np.random.default_rng(seed)
    # one uniform draw per cell; a higher level flips a superset of the cells flipped at a lower level
    u = rng.random(len(df))
    flip_to_tumor = healthy & (u < healthy_as_cancer)
    n_kept_healthy = int(healthy.sum() - flip_to_tumor.sum())
    # number k of cancer cells added to the healthy set: c = k / (kept healthy + k), so k = c / (1 - c) * kept healthy
    k = int(round(cancer_in_healthy_set / (1.0 - cancer_in_healthy_set) * n_kept_healthy)) if cancer_in_healthy_set > 0 else 0
    cancer_idx = np.flatnonzero(~healthy)
    chosen = cancer_idx[np.argsort(u[cancer_idx])[:k]]
    flip_to_healthy = np.zeros(len(df), dtype=bool)
    flip_to_healthy[chosen] = True
    out.loc[flip_to_healthy, "assignment"] = "healthy"
    out.loc[flip_to_tumor, "assignment"] = "tumor"
    n_h = int((out["assignment"] == "healthy").sum())
    stats = dict(n_cells=len(df), n_healthy_true=int(healthy.sum()), n_cancer_true=int((~healthy).sum()),
                 n_cancer_as_healthy=int(flip_to_healthy.sum()), n_healthy_as_cancer=int(flip_to_tumor.sum()),
                 n_labelled_healthy=n_h,
                 contamination=float(flip_to_healthy.sum() / n_h) if n_h else float("nan"))
    return out, stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotation", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cancer_in_healthy_set", type=float, default=0.0)
    p.add_argument("--healthy_as_cancer", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    df = pd.read_csv(a.annotation, dtype={"barcode": str})
    out, st = noisy_annotation(df, a.cancer_in_healthy_set, a.healthy_as_cancer, a.seed)
    out.to_csv(a.out, index=False, lineterminator="\n")
    print(f"{a.out}: {st['n_healthy_true']} healthy and {st['n_cancer_true']} cancer cells; relabelled {st['n_cancer_as_healthy']} "
          f"cancer -> healthy and {st['n_healthy_as_cancer']} healthy -> tumor; {st['n_labelled_healthy']} cells labelled healthy, "
          f"{100 * st['contamination']:.0f}% of them are cancer")
