"""Scores every run in a simulation folder against the ground truth (simulation_metrics.py) and writes a summary.

    python score/analyze_simulation_results.py --sim_root simulated_data --out analysis_out/sweep
    python score/analyze_simulation_results.py --sim_root simulated_robust --out analysis_out/robust --methods scati scati_noann compass_snp lophy

Writes into --out:
  metrics.csv            one row per (dataset, method); input of make_main_figure.py, make_benchmark_tables.py
  summary.md             coverage, medians per method, SCATI vs the others per dataset and setting, node and doublet counts, runtime
  paired_by_dataset.csv  SCATI and the other methods side by side per dataset
  runtime_scati.csv      SCATI run_summary.json fields per dataset
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulation_metrics as sm  # noqa: E402

PRIMARY = ["cn_mae_singlets", "cn_mae_all", "vaf_mae_singlets", "vaf_mae_all", "ari_all",
           "balanced_accuracy", "recall_gain", "recall_loss", "pair_acc", "pair_f1_macro"]
LOWER_IS_BETTER = ("cn_", "vaf_")


def fmt(df, **kw):
    return "```\n" + df.round(kw.pop("digits", 3)).to_string(**kw) + "\n```\n"


def section(title):
    return f"\n## {title}\n\n"


def coverage(m):
    t = m.groupby(["method", "status"]).size().unstack(fill_value=0)
    errs = m[m.status == "error"]
    out = fmt(t)
    if len(errs):
        e = errs.groupby(["method", "error"]).size().reset_index(name="n").sort_values("n", ascending=False).head(8)
        out += "\nMost common errors:\n" + fmt(e, digits=0, index=False)
    return out


def per_method(m):
    ok = m[m.status == "ok"]
    cols = [c for c in PRIMARY + ["n_pred_nodes", "wall_seconds"] if c in ok]
    return fmt(ok.groupby("method")[cols].median().T) + "(medians over the datasets each method finished)\n"


def paired(m, other):
    ok = m[m.status == "ok"].copy()
    ok["ds"] = ok.setting + "/" + ok.rep
    a = ok[ok.method == "scati"].set_index("ds")
    b = ok[ok.method == other].set_index("ds")
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return f"no dataset scored for both scati and {other}\n", None
    a, b = a.loc[idx], b.loc[idx]
    rows = []
    for c in PRIMARY:
        if c not in a:
            continue
        ok_pair = a[c].notna() & b[c].notna()
        better = (a[c] < b[c]) if c.startswith(LOWER_IS_BETTER) else (a[c] > b[c])
        rows.append(dict(metric=c, scati_median=a[c].median(), other_median=b[c].median(), scati_mean=a[c].mean(),
                         other_mean=b[c].mean(), scati_better=f"{int((better & ok_pair).sum())}/{int(ok_pair.sum())}"))
    side = a[PRIMARY + ["n_true_nodes", "n_pred_nodes", "n_true_doublets", "n_pred_doublets", "n_cells"]].add_prefix("scati_") \
        .join(b[PRIMARY + ["n_pred_nodes", "n_pred_doublets"]].add_prefix(f"{other}_"))
    return f"{len(idx)} datasets scored for both.\n" + fmt(pd.DataFrame(rows), index=False), side


def per_setting(m, other):
    ok = m[m.status == "ok"].copy()
    ok["ds"] = ok.setting + "/" + ok.rep
    a = ok[ok.method == "scati"].set_index("ds")
    b = ok[ok.method == other].set_index("ds")
    idx = a.index.intersection(b.index)
    rows = []
    for s, g in a.loc[idx].groupby("setting"):
        c = b.loc[g.index]
        rows.append({"setting": s, "n": len(g),
                     "ARI scati/other": f"{g.ari_all.median():.2f} / {c.ari_all.median():.2f}",
                     "CN_MAE scati/other": f"{g.cn_mae_singlets.median():.3f} / {c.cn_mae_singlets.median():.3f}",
                     "pred-true nodes scati/other": f"{(g.n_pred_nodes - g.n_true_nodes).mean():+.1f} / {(c.n_pred_nodes - c.n_true_nodes).mean():+.1f}",
                     "doublets pred/true scati": f"{g.n_pred_doublets.sum() / max(g.n_true_doublets.sum(), 1):.2f}",
                     "doublets pred/true other": f"{c.n_pred_doublets.sum() / max(g.n_true_doublets.sum(), 1):.2f}"})
    return fmt(pd.DataFrame(rows), index=False)


def diagnostics(m):
    ok = m[m.status == "ok"].copy()
    ok["ds"] = ok.setting + "/" + ok.rep
    out = ""
    rows = []
    for meth, g in ok.groupby("method"):
        rows.append(dict(method=meth, n=len(g), true_nodes=g.n_true_nodes.mean(), pred_nodes=g.n_pred_nodes.mean(),
                         mean_abs_node_error=(g.n_pred_nodes - g.n_true_nodes).abs().mean(),
                         true_doublet_frac=(g.n_true_doublets / g.n_cells).mean(), pred_doublet_frac=(g.n_pred_doublets / g.n_cells).mean()))
    out += fmt(pd.DataFrame(rows), index=False)
    s = ok[ok.method == "scati"].copy()
    s["dbl_true"] = s.n_true_doublets / s.n_cells
    s["dbl_pred"] = s.n_pred_doublets / s.n_cells
    flag = s[(s.dbl_pred > 1.5 * s.dbl_true) & (s.dbl_pred - s.dbl_true > 0.1)]
    out += f"\nSCATI datasets whose predicted doublet fraction is >1.5x the true one and >0.1 higher ({len(flag)}):\n"
    out += fmt(flag.sort_values("dbl_pred", ascending=False)[["ds", "n_cells", "n_true_nodes", "n_pred_nodes", "ari_all",
                                                              "n_true_doublets", "n_pred_doublets"]].head(25), index=False, digits=3)
    deficit = s[(s.n_true_nodes - s.n_pred_nodes) >= 3]
    out += f"\nSCATI datasets with >=3 fewer nodes than true ({len(deficit)}):\n"
    out += fmt(deficit.sort_values("ari_all")[["ds", "n_true_nodes", "n_pred_nodes", "ari_all", "cn_mae_singlets"]].head(25), index=False)
    worst = s.nsmallest(10, "ari_all")[["ds", "n_cells", "n_true_nodes", "n_pred_nodes", "ari_all", "cn_mae_singlets", "recall_gain", "recall_loss"]]
    out += "\nSCATI 10 lowest ARI:\n" + fmt(worst, index=False)
    for pm in ("compass_snp",):
      p = ok[ok.method == pm]
      if len(p):
        out += (f"\n{pm}: {int((p.n_pred_nodes > 3 * p.n_true_nodes).sum())}/{len(p)} datasets have more than 3x the true node count; "
                f"{int((p.n_pred_doublets / p.n_cells > 0.5).sum())} flag more than half of the cells as doublets.\n")
    return out


def runtime_scati(sim_root):
    rows = []
    for f in sorted(glob.glob(os.path.join(sim_root, "*", "rep*", "SCATI", "results", "run_summary.json"))):
        parts = f.replace("\\", "/").split("/")
        setting, rep = parts[-5], parts[-4]
        try:
            s = json.load(open(f))
            ch = s["chains"]
            rows.append(dict(setting=setting, rep=rep, ds=f"{setting}/{rep}", n_samples=s.get("n_samples"), n_cells=s.get("n_cells"),
                             wall_h=s["total_wall_seconds"] / 3600, steps_per_s=s["mean_steps_per_second_per_chain"],
                             peak_rss_gb=s.get("peak_rss_gb"), lambda_rounding=s.get("lambda_rounding_precision"), cache_gb=s.get("cache_gb"),
                             nodes_mean=np.mean([c["n_nodes"] for c in ch]), nodes_max=max(c["n_nodes"] for c in ch),
                             cache_hit_min=min(c["cna_cache_hit_rate"] for c in ch),
                             cache_clears=sum(sum(c.get("cache_clears", {}).values()) for c in ch),
                             peak_cache_gb=max(c["peak_cache_gb"] for c in ch)))
        except Exception as e:  # skip unreadable summaries
            print(f"  (skipping {f}: {e})", file=sys.stderr)
    return pd.DataFrame(rows)


def runtime_text(df, m):
    if df.empty:
        return "no SCATI run_summary.json found\n"
    out = f"{len(df)} SCATI run summaries. wall hours: " + ", ".join(f"{k}={v:.2f}" for k, v in df.wall_h.describe()[["min", "25%", "50%", "75%", "max"]].items()) + "\n"
    out += f"lambda_rounding_precision: {df.lambda_rounding.value_counts().to_dict()}  cache_gb: {df.cache_gb.value_counts().to_dict()}  peak_rss recorded: {int(df.peak_rss_gb.notna().sum())}\n\n"
    by = df.groupby("setting").agg(n=("ds", "count"), wall_h_median=("wall_h", "median"), wall_h_max=("wall_h", "max"), steps_per_s_median=("steps_per_s", "median"),
                                   final_nodes_mean=("nodes_mean", "mean"), cache_clears_mean=("cache_clears", "mean"))
    out += fmt(by)
    out += "\nSlowest 12 runs:\n" + fmt(df.nlargest(12, "wall_h")[["ds", "n_samples", "wall_h", "steps_per_s", "nodes_mean", "nodes_max", "cache_hit_min", "cache_clears", "peak_cache_gb"]], index=False)
    ok = m[(m.status == "ok") & m.wall_seconds.notna()]
    if len(ok):
        t = ok.groupby("method").wall_seconds.agg(["count", "median", "max"]) / [1, 3600, 3600]
        t.columns = ["datasets_with_timing", "median_h", "max_h"]
        t["datasets_with_timing"] = ok.groupby("method").wall_seconds.count()
        out += "\nWall time per method from the timing files (benchmark.tsv / run_summary.json / run_info.txt):\n" + fmt(t)
    else:
        out += "\nNo method has timing files here (no benchmark.tsv / run_info.txt).\n"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim_root", default="simulated_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["scati", "scati_noann", "compass_snp", "lophy"], choices=list(sm.METHOD_CONFIG))
    ap.add_argument("--metrics_csv", help="use this metrics.csv instead of scoring again")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.metrics_csv:
        m = pd.read_csv(args.metrics_csv)
        for c in ("pair_acc", "pair_f1_same", "pair_f1_related", "pair_f1_apart", "pair_f1_macro"):   # columns missing from older metrics.csv files
            if c not in m:
                m[c] = np.nan
    else:
        m = sm.run(args.sim_root, args.methods)
    m.to_csv(os.path.join(args.out, "metrics.csv"), index=False)

    md = [f"# Simulation results: {os.path.abspath(args.sim_root)}\n"]
    md.append(section("Coverage (runs scored per method)") + coverage(m))
    md.append(section("Per-method medians") + per_method(m))
    side_all = []
    for other in [x for x in m.method.unique() if x != "scati" and x in ("compass_snp", "lophy", "scati_noann")]:
        text, side = paired(m, other)
        md.append(section(f"SCATI vs {other}, paired by dataset") + text)
        if side is not None:
            md.append("\nPer setting (median ARI, median CN MAE, mean predicted-minus-true nodes, predicted/true doublets):\n" + per_setting(m, other))
            side_all.append(side.add_prefix("") if len(side_all) == 0 else side[[c for c in side.columns if c.startswith(other)]])
    if side_all:
        pd.concat(side_all, axis=1).to_csv(os.path.join(args.out, "paired_by_dataset.csv"))
    md.append(section("Node counts and doublets") + diagnostics(m))
    rt = runtime_scati(args.sim_root)
    rt.to_csv(os.path.join(args.out, "runtime_scati.csv"), index=False)
    md.append(section("Runtime") + runtime_text(rt, m))
    open(os.path.join(args.out, "summary.md"), "w", encoding="utf8").write("\n".join(md))
    print(f"wrote {args.out}/summary.md and tables ({os.path.getsize(os.path.join(args.out, 'summary.md')) // 1024} KB summary)")


if __name__ == "__main__":
    main()
