"""PDF figures of the real-data comparison (vector graphics, exact size, no cropping).

    python make_paper_figures.py --cohort AML --example AML-59 --out figures

Writes into --out:
    main_figure.pdf          rows a-d (VAF error, region error, summed error minus SCATI, nodes; style of the simulation figure) above row e (SCATI
                                        tree, observed and predicted region counts / copy numbers / VAF)
    appendix/<sample>.pdf    SCATI, COMPASS and LoPhy of every sample (tree, counts, copy numbers, VAF)
    metrics.csv              the numbers of the panels

The three methods are scored with the distances of plot_methods.py (region error, VAF error, number of nodes), every method of a sample on the
same cells: all cells except the doublets of COMPASS, which reports no genotype for them. SCATI is scored on its doublets as well (with the summed
genotype of the two nodes of the pair); LoPhy has no doublet model. A method without a result in a sample (e.g. a failed LoPhy run) is left out of
that sample
"""
import argparse
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "simulations", "score"))
import make_main_figure as mf              # noqa: E402  (palette and panel style of the simulation figure)
import plot_methods as pm                  # noqa: E402
plt.rcParams["savefig.bbox"] = "standard"  # exact figure size (make_main_figure sets "tight")

NAMES = {"compass_snp": "COMPASS", "lophy": "LoPhy"}


def patient_sex(res_dir, sample):
    """Sex of a sample from <cohort>/patients.tsv (res_dir = <cohort>/<sample>); None if not listed."""
    path = os.path.join(res_dir, "..", "patients.tsv")
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path, sep="\t").set_index("patient")
    return str(t.loc[sample, "sex"]) if sample in t.index else None


def load_sample(res_dir, sample):
    """The data of a sample and the loaded results (keys scati, compass_snp, lophy) of every method that has one."""
    data = pm.load_data(os.path.join(res_dir, "SCATI", sample))
    sex = patient_sex(res_dir, sample)
    lophy_meta = os.path.join(res_dir, "LoPhy", "meta.csv")
    todo = [("scati", os.path.join(res_dir, "SCATI", "results"), "best_tree_strand_genotypes.json", lambda d: pm.load_scati(d, data)),
            ("compass_snp", os.path.join(res_dir, "COMPASS", "results"), "data_nodes_genotypes.tsv", lambda d: pm.load_compass(d, data, sex)),
            ("lophy", os.path.join(res_dir, "LoPhy", "results"), "data_0.gv", lambda d: pm.load_lophy(d, lophy_meta, data, sex))]
    keys, results = [], []
    for key, d, marker, load in todo:
        if os.path.exists(os.path.join(d, marker)):
            keys.append(key)
            results.append(load(d))
    return data, keys, results


def common_cells(keys, results):
    """Cells on which all methods of a sample are scored: all cells except the doublets of COMPASS (it reports no genotype for them)."""
    n = len(results[0].cell_node)
    return results[keys.index("compass_snp")].singlet_mask() if "compass_snp" in keys else np.ones(n, dtype=bool)


def sample_results(res_dir, sample, min_depth):
    """Metrics of every method that has a result in this sample, on the cells of common_cells()."""
    data, keys, results = load_sample(res_dir, sample)
    common = common_cells(keys, results)
    rows = []
    for key, r in zip(keys, results):
        m = pm.compute_metrics(r, data, common, min_depth)
        rows.append(dict(sample=sample, model=key, n_nodes=m["n_nodes"], n_cells_scored=m["n_cells_scored"], n_cells=len(common),
                         n_doublets_scored=m["n_doublets_scored"],
                         region_err=m["region_err"], vaf_err=m["VAF_MAE"], vaf_err_root_only=m["VAF_MAE_root_only"], ref_node=m["ref_node"]))
    return rows


def box_panel(ax, w, ylabel, models, s, zero_line=False):
    """Box plot as in make_main_figure.panel (box = median and IQR, whiskers within 1.5 IQR, dots beyond), line widths and marker sizes times s."""
    for i, m in enumerate(models):
        v = w[m].dropna().to_numpy()
        if zero_line and np.allclose(v, 0):
            ax.plot([i - 0.25, i + 0.25], [0, 0], color=mf.COLORS[m], lw=2.2 * s, zorder=4)
            continue
        ax.boxplot(v, positions=[i], widths=0.5, showfliers=True, manage_ticks=False, zorder=3, patch_artist=True,
                   flierprops=dict(marker="o", markersize=2.2 * s, markerfacecolor=mf.COLORS[m], markeredgecolor="none", alpha=0.8),
                   boxprops=dict(facecolor=mf.mpl.colors.to_rgba(mf.COLORS[m], mf.BOX_FILL_ALPHA), edgecolor=mf.COLORS[m], lw=1.1 * s),
                   whiskerprops=dict(color=mf.COLORS[m], lw=1.1 * s), capprops=dict(color=mf.COLORS[m], lw=1.1 * s),
                   medianprops=dict(color=mf.INK, lw=1.4 * s))
    if zero_line:
        ax.axhline(0, color=mf.INK2, lw=0.6 * s, ls=":", zorder=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([mf.SHORT[m] for m in models], rotation=40, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)


def metric_row(sub, df, models, ref, s):
    """The four metric panels in the (sub)figure `sub`; returns the x position (figure fraction) of the left edge of panel a."""
    def wide(col):
        return df.pivot(index="sample", columns="model", values=col).reindex(columns=models)
    gs = sub.add_gridspec(1, 4, left=0.074, right=0.985, top=0.84, bottom=0.30, wspace=0.75)
    tot = wide("total_err")
    panels = [(wide("vaf_err"), "VAF error", False), (wide("region_err"), "Region error", False),
              (tot.sub(tot[ref], axis=0), "Combined error\ndifference to SCATI", True), (wide("n_nodes"), "Number of nodes", False)]
    for c, (w, ylab, zl) in enumerate(panels):
        ax = sub.add_subplot(gs[0, c])
        box_panel(ax, w, ylab, models, s, zl)
        ax.set_title("abcd"[c], loc="left", fontweight="bold", fontsize=8 * s)
    handles = [plt.Line2D([], [], color=mf.COLORS[m], lw=3 * s, label=mf.SHORT[m]) for m in models]
    sub.legend(handles=handles, loc="upper center", ncol=len(models), frameon=False, bbox_to_anchor=(0.5, 1.0))
    return gs[0, 0].get_position(sub).x0


def sample_rows(res_dir, sample, methods):
    data, keys, results = load_sample(res_dir, sample)
    common = common_cells(keys, results)
    chosen = [(k, r) for k in methods for kk, r in zip(keys, results) if kk == k]
    metrics = []
    for k, r in chosen:
        r.label = "SCATI" if k.startswith("scati") else NAMES[k]
        metrics.append(pm.compute_metrics(r, data, common, 5))
    return data, chosen, metrics


def sample_dirs(cohort):
    for sample in sorted(os.listdir(cohort)):
        if os.path.isdir(os.path.join(cohort, sample, "SCATI")):
            yield sample, os.path.join(cohort, sample)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default="AML")
    ap.add_argument("--example", default="AML-59", help="sample of row e of the main figure")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row_h", type=float, default=5.6, help="height of a method row in the appendix figures (native layout, inches)")
    ap.add_argument("--pack", type=int, default=8, help="a tree node with more event lines than this shows 2-3 events per line (all events are shown)")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--only", choices=["main", "appendix"], default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "appendix"), exist_ok=True)
    scati = "scati"
    models = [scati, "compass_snp", "lophy"]
    mf.COLORS[scati], mf.SHORT[scati] = "#2a78d6", "SCATI"

    rows, dirs = [], {}
    for sample, d in sample_dirs(a.cohort):
        if sample in a.exclude or not os.path.exists(os.path.join(d, "SCATI", "results", "best_tree_strand_genotypes.json")):
            continue
        dirs[sample] = d
        rows += sample_results(d, sample, 5)
    df = pd.DataFrame(rows)
    df["total_err"] = df["region_err"] + df["vaf_err"]
    df.to_csv(os.path.join(a.out, "metrics.csv"), index=False)

    pm.TREE_PACK = a.pack
    if a.only != "appendix":                                   # main figure: metrics row above the SCATI row of the example sample
        data, chosen, metrics = sample_rows(dirs[a.example], a.example, [scati])
        pm.ROW_H_NATIVE, pm.TREE_W_NATIVE, pm.RIGHT_NATIVE = 8.0, 11.0, 2.0      # the standard layout of plot_methods.py
        chosen[0][1].label = ""                                 # row e has no title: the letter e stands under the letter a
        with tempfile.TemporaryDirectory() as tmp:
            W, H = pm.make_figure([chosen[0][1]], data, metrics, None, "light", tmp, 100, dry_run=True)    # size of row e
            s = W / (mf.TEXT_WIDTH_IN * 1.45)                                                           # factor from the simulation-style size of the metric panels
            H_top = 2.4 * s
            fig = plt.figure(figsize=(W, H_top + H), facecolor="white")
            top, bottom = fig.subfigures(2, 1, height_ratios=[H_top, H], hspace=0)
            with plt.rc_context({"font.size": 7.5 * s, "axes.labelsize": 7.5 * s, "xtick.labelsize": 6.5 * s, "ytick.labelsize": 7 * s,
                                 "legend.fontsize": 7 * s, "axes.linewidth": 0.6 * s, "xtick.major.width": 0.6 * s, "ytick.major.width": 0.6 * s,
                                 "xtick.major.size": 3.5 * s, "ytick.major.size": 3.5 * s, "grid.linewidth": 0.5 * s}):
                xa = metric_row(top, df, models, scati, s)
            pm.make_figure([chosen[0][1]], data, metrics, None, "light", tmp, 100, fig=bottom)
            bottom.text(xa, 0.995, "e", fontsize=8 * s, fontweight="bold", va="top", ha="left")
            fig.savefig(os.path.join(a.out, "main_figure.pdf"), facecolor="white")
            plt.close(fig)
        print("main figure:", round(W, 1), "x", round(H_top + H, 1), "in")
    if a.only != "main":                                       # appendix: the three methods of every sample
        pm.ROW_H_NATIVE, pm.TREE_W_NATIVE, pm.RIGHT_NATIVE = a.row_h, 11.0, 1.2     # a little narrower, so that two figures fit on a page
        for sample, d in dirs.items():
            data, chosen, metrics = sample_rows(d, sample, models)
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(a.out, "appendix", f"{sample}.png")
                W, H = pm.make_figure([r for _, r in chosen], data, metrics, out, "light", tmp, 100)
            os.remove(out)                                      # the pdf written next to it is the figure
            print(f"{sample}: {W:.1f} x {H:.1f} in (aspect {W / H:.2f}; {5.17 * H / W:.2f} in high at the text width of 5.17 in)")


if __name__ == "__main__":
    main()
