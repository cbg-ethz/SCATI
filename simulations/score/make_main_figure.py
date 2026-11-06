"""Main simulation-benchmark figure: SCATI (without annotation), SCATI-WA (true healthy-cell annotation), COMPASS (germline frequencies), LoPhy.

    python score/make_main_figure.py --metrics analysis_out/sweep/metrics.csv --out figures

One row of six panels: CN MAE, CN MAE with copy numbers clipped to 1-3, VAF MAE, ARI, cell-pair accuracy (--pair_metric f1 gives the macro F1)
and runtime. All cells are scored. Boxes show median and IQR, whiskers the most extreme value within 1.5 IQR, dots the values beyond.
Only datasets that all four methods finished are used.
Writes fig_simulation_benchmark.pdf and .png to --out and prints the datasets used.
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

METHODS = ["scati_noann", "scati", "compass_snp", "lophy"]
LABELS = {"scati_noann": "SCATI", "scati": "SCATI-WA", "compass_snp": "COMPASS", "lophy": "LoPhy"}
# SCATI-WA is a light version of the SCATI blue
SHORT = {"scati_noann": "SCATI", "scati": "SCATI-WA", "compass_snp": "COMPASS", "lophy": "LoPhy"}      # tick labels
COLORS = {"scati_noann": "#2a78d6", "scati": "#9bbbe8", "compass_snp": "#eb6834", "lophy": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEY = ["setting", "rep", "axis", "level", "n_samples"]
BOX_FILL_ALPHA = 0.22
TEXT_WIDTH_IN = 6.3

mpl.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 110, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.facecolor": "white", "legend.edgecolor": GRID,
})


def wide(df, metric, methods):
    """One row per dataset, one column per method; only datasets where all listed methods have a value."""
    d = df[(df["status"] == "ok") & df["method"].isin(methods)]
    w = d.pivot_table(index=KEY, columns="method", values=metric, aggfunc="first")
    return w.reindex(columns=[m for m in methods if m in w.columns]).dropna()


def panel(ax, w, ylabel):
    x = np.arange(len(w.columns))
    for i, m in enumerate(w.columns):
        ax.boxplot(w[m].to_numpy(), positions=[i], widths=0.5, showfliers=True, manage_ticks=False, zorder=3, patch_artist=True,
                   flierprops=dict(marker="o", markersize=2.2, markerfacecolor=COLORS[m], markeredgecolor="none", alpha=0.8),
                   boxprops=dict(facecolor=mpl.colors.to_rgba(COLORS[m], BOX_FILL_ALPHA), edgecolor=COLORS[m], lw=1.1),
                   whiskerprops=dict(color=COLORS[m], lw=1.1), capprops=dict(color=COLORS[m], lw=1.1),
                   medianprops=dict(color=INK, lw=1.4))
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in w.columns], rotation=40, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True, help="sweep/metrics.csv written by run_analysis.sh")
    ap.add_argument("--out", required=True, help="folder for the figure files")
    ap.add_argument("--pair_metric", choices=["acc", "f1"], default="acc", help="cell-pair relations: accuracy or macro F1 (default acc)")
    a = ap.parse_args()
    df = pd.read_csv(a.metrics)
    os.makedirs(a.out, exist_ok=True)

    cols = [("cn_mae", "CN MAE"), ("cn_mae_clip", "CN MAE, CN 1-3"), ("vaf_mae", "VAF MAE"), ("ari", "ARI")]
    pair_col, pair_label = ("pair_acc", "Cell-pair accuracy") if a.pair_metric == "acc" else ("pair_f1_macro", "Pair F1 (macro)")
    fig = plt.figure(figsize=(TEXT_WIDTH_IN * 1.35, 2.3))
    gs = GridSpec(1, 6, figure=fig, wspace=1.0)
    letters = iter("abcdef")
    for c, (stem, ylab) in enumerate(cols):
        w = wide(df, f"{stem}_all", METHODS)
        n_used = len(w)
        ax = fig.add_subplot(gs[0, c])
        panel(ax, w, ylab)
        ax.set_title(next(letters), loc="left", fontweight="bold", fontsize=8)
    ax = fig.add_subplot(gs[0, 4])
    panel(ax, wide(df, pair_col, METHODS), pair_label)
    ax.set_title(next(letters), loc="left", fontweight="bold", fontsize=8)
    ax = fig.add_subplot(gs[0, 5])
    panel(ax, wide(df, "wall_seconds", METHODS) / 60.0, "Runtime (min)")      # seconds to minutes
    ax.set_title(next(letters), loc="left", fontweight="bold", fontsize=8)
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], lw=3, label=LABELS[m].replace("\n", " ")) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=len(METHODS), frameon=False, bbox_to_anchor=(0.5, 1.12))
    for ext in ("pdf", "png"):
        path = os.path.join(a.out, f"fig_simulation_benchmark.{ext}")
        fig.savefig(path, dpi=300)
        print("saved", path)
    print(f"datasets in the panels: {n_used} (datasets that all of {', '.join(METHODS)} finished; LoPhy missing: "
          f"{df[(df.method == 'lophy') & (df.status != 'ok')]['setting'].astype(str).add('/').add(df[(df.method == 'lophy') & (df.status != 'ok')]['rep']).tolist()})")


if __name__ == "__main__":
    main()
