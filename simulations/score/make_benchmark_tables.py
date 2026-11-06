"""LaTeX tables of the simulation benchmark.

Print the tables to stdout:
    python score/make_benchmark_tables.py --analysis analysis_out --sim_roots simulated_data simulated_robust > benchmark_tables.tex

Also write the tables and the numbers quoted in the text to a folder:
    python score/make_benchmark_tables.py --analysis analysis_out --sim_roots simulated_data simulated_robust --out analysis_out/tables

Take the captions from the manuscript (a caption that disagrees with the data gives a warning on stderr):
    python score/make_benchmark_tables.py --analysis analysis_out --captions manuscript --tex <file.tex>

Input, all written by run_analysis.sh:
    <analysis>/sweep/metrics.csv          main benchmark, 11 settings x 10 datasets (its default setting is the reference of every robustness condition)
    <analysis>/robust/metrics.csv         robustness runs
    <analysis>/robust/metrics_annot.csv   annotation-error runs
    --sim_roots                           simulation folders, used to read whether a dataset has an FGD

Output:
    stdout (and --out)                    tab_sim_settings, tab_sim_time, tab_sim_robust, tab_sim_annot as LaTeX, then the numbers quoted in the appendix
                                          text as comments (sim_text_numbers.txt with --out)
    <analysis>/robust/conditions.csv      median metrics per robustness condition
    <analysis>/robust/summary.md          the same with paired differences to the default

Methods: SCATI without annotation, SCATI-WA, COMPASS with germline frequencies (compass_snp) and LoPhy.

Cells are medians over datasets with the exact distribution-free 95% CI of the median: order statistics x_(l) and x_(n+1-l), l the largest
integer with P(Bin(n, 1/2) <= l-1) <= 0.025, which gives 97.9% coverage for n = 10. With fewer than 6 datasets the data range is shown.
Paired counts ("datasets in which SCATI is better") use the datasets that both methods finished. Only runs with status ok and a wall time
within --max_hours are used.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

METHODS = [("scati", "SCATI"), ("compass", "COMPASS"), ("lophy", "LoPhy")]     # the SCATI column holds the runs without annotation, see remap_noann
SETTINGS = [("other", "default", "Default configuration"),
            ("n_nodes", "3", "Clones: 3"), ("n_nodes", "10", "Clones: 10"),
            ("n_SNVs", "10", "SNVs: 10"), ("n_SNVs", "50", "SNVs: 50"),
            ("n_CNAs", "10", "CNAs: 10"), ("n_CNAs", "15", "CNAs: 15"),
            ("event_type", "SNV_only", "SNVs only"), ("event_type", "local_events", "Local events"),
            ("n_samples", "2", "Samples: 2"), ("n_samples", "3", "Samples: 3")]
ROBUST = [("default", "Default"), ("fgd100", "Whole-genome duplication in every dataset"), ("dbl0", "Doublet rate 0\\%"), ("dbl0.25", "Doublet rate 25\\%"),
          ("healthy0.05", "Healthy cells 5\\%"), ("healthy0.50", "Healthy cells 50\\%"), ("backmut20", "Back mutations 20\\%"),
          ("false25", "Loci without somatic mutation 25\\%"), ("false50", "Loci without somatic mutation 50\\%"),
          ("cont30", "T 30\\%"), ("cont50", "T 50\\%"), ("h2c30", "H 30\\%"), ("h2c50", "H 50\\%"),
          ("cont30_h2c30", "T 30\\% + H 30\\%"), ("cont50_h2c50", "T 50\\% + H 50\\%")]
ALPHA = 0.05
MAX_HOURS = {"value": 24.0}      # runs longer than this count as not finished (--max_hours; 0 = no limit)
PAIR = {"metric": "pair_acc", "title": "Cell-pair accuracy"}      # --pair_metric f1 switches to the macro F1


# ------------------------------------------------------------------------------------------------ data
def load(path):
    d = pd.read_csv(path)
    d = d[d["status"] == "ok"].copy()
    if MAX_HOURS["value"] and "wall_seconds" in d:
        late = pd.to_numeric(d["wall_seconds"], errors="coerce") / 3600.0 > MAX_HOURS["value"]
        for _, r in d[late].iterrows():
            print(f"NOTE {path}: {r['method']} {r.get('setting', '')}/{r.get('rep', '')} took {float(r['wall_seconds']) / 3600:.1f} h, "
                  f"longer than {MAX_HOURS['value']:g} h: counted as not finished", file=sys.stderr)
        d = d[~late].copy()
    d["method"] = d["method"].replace({"compass_snp": "compass"})
    if "wall_seconds" in d:
        d["wall_min"] = pd.to_numeric(d["wall_seconds"], errors="coerce") / 60.0
    if {"n_pred_nodes", "n_true_nodes"} <= set(d.columns):
        d["node_diff"] = d["n_pred_nodes"] - d["n_true_nodes"]
        d["node_abs_err"] = d["node_diff"].abs()
    if {"n_pred_doublets", "n_cells"} <= set(d.columns):
        d["doublet_share_called"] = 100.0 * d["n_pred_doublets"] / d["n_cells"]
        d.loc[d["method"] == "lophy", "doublet_share_called"] = np.nan        # LoPhy has no doublet model
    return d


# ------------------------------------------------------------------------------------ robustness conditions
GEN_CONDITIONS = ["default", "fgd100", "dbl0", "dbl0.25", "healthy0.05", "healthy0.50", "backmut20", "false25", "false50"]
ANNOT_CONDITIONS = ["cont5", "cont10", "h2c10", "h2c20", "cont5_h2c10", "cont10_h2c20", "cont30", "cont50", "h2c30", "h2c50",
                    "cont30_h2c50", "cont30_h2c30", "cont50_h2c50"]
LABEL = {"default": "default (0.08 doublets, 25% healthy, no FGD, clean labels)",
         "fgd100": "every dataset has an FGD",
         "dbl0": "doublet rate 0", "dbl0.25": "doublet rate 0.25",
         "healthy0.05": "5% healthy cells", "healthy0.50": "50% healthy cells", "backmut20": "20% back mutations",
         "false25": "25% of all loci are false", "false50": "50% of all loci are false",
         "cont5": "5% of the healthy-labelled cells are cancer", "cont10": "10% of the healthy-labelled cells are cancer",
         "h2c10": "10% of healthy cells labelled cancer", "h2c20": "20% of healthy cells labelled cancer",
         "cont5_h2c10": "5% contamination and 10% healthy->cancer", "cont10_h2c20": "10% contamination and 20% healthy->cancer",
         "cont30": "30% of the healthy-labelled cells are cancer", "cont50": "50% of the healthy-labelled cells are cancer",
         "h2c30": "30% of healthy cells labelled cancer", "h2c50": "50% of healthy cells labelled cancer",
         "cont30_h2c50": "30% contamination and 50% healthy->cancer", "cont30_h2c30": "30% contamination and 30% healthy->cancer",
         "cont50_h2c50": "50% contamination and 50% healthy->cancer"}


ROBUST_METHODS = ("scati", "scati_noann", "compass_snp", "lophy")



def has_fgd(setting, rep, roots):
    """Whether the dataset's event file in the first root that has it contains an FGD; NaN if no root has the dataset."""
    for root in roots:
        path = os.path.join(root, str(setting), str(rep), "SCATI", "data_events_ordered.json")
        if os.path.exists(path):
            with open(path) as f:
                return '"FGD"' in f.read()
    return float("nan")


def conditions_table(df, roots=()):
    """One row per (condition, dataset)."""
    rows = []
    for _, r in df.iterrows():
        method = r["method"]
        if method in ROBUST_METHODS and r["setting"] in GEN_CONDITIONS:
            cond = r["setting"]
        elif str(r["method"]).startswith("scati_annot_") and r["setting"] == "default":
            cond = r["method"][len("scati_annot_"):]
        else:
            continue
        rows.append(dict(method=method if method in ROBUST_METHODS else "scati", condition=cond, rep=r["rep"], setting=r["setting"], has_fgd=has_fgd(r["setting"], r["rep"], roots), status=r["status"], ari_all=r.get("ari_all"), cn_mae_all=r.get("cn_mae_all"), cn_mae_clip_all=r.get("cn_mae_clip_all"),
                         vaf_mae_all=r.get("vaf_mae_all"), cn_mae_singlets=r.get("cn_mae_singlets"), cn_mae_clip_singlets=r.get("cn_mae_clip_singlets"),
                         vaf_mae_singlets=r.get("vaf_mae_singlets"), recall_gain=r.get("recall_gain"), recall_loss=r.get("recall_loss"),
                         balanced_accuracy=r.get("balanced_accuracy"), pair_acc=r.get("pair_acc"), pair_f1_macro=r.get("pair_f1_macro"),
                         n_true_nodes=r.get("n_true_nodes"), n_pred_nodes=r.get("n_pred_nodes"),
                         n_true_doublets=r.get("n_true_doublets"), n_pred_doublets=r.get("n_pred_doublets"),
                         n_cells=r.get("n_cells"), wall_seconds=r.get("wall_seconds")))
    t = pd.DataFrame(rows)
    num = [c for c in t.columns if c not in ("method", "condition", "rep", "status", "setting", "has_fgd")]
    t[num] = t[num].apply(pd.to_numeric, errors="coerce")
    t["doublet_ratio"] = t.n_pred_doublets / t.n_true_doublets.clip(lower=1)
    t["node_diff"] = t.n_pred_nodes - t.n_true_nodes
    t["doublet_frac_pred"] = t.n_pred_doublets / t.n_cells
    t["doublet_frac_true"] = t.n_true_doublets / t.n_cells
    return t



def robustness_summary(t, out_dir):
    """Writes summary.md to out_dir: per method, the median metrics of every robustness condition and the paired difference to the default."""
    os.makedirs(out_dir, exist_ok=True)
    ok_all = t[t.status == "ok"]
    fence = "```"
    md = ["# Robustness (SCATI with and without annotation, COMPASS with germline frequencies, LoPhy)\n"]
    for method in [m for m in ROBUST_METHODS if m in set(t.method)]:
        tm, ok = t[t.method == method], ok_all[ok_all.method == method]
        order = [c for c in GEN_CONDITIONS + ANNOT_CONDITIONS if c in set(tm.condition)]
        rows = []
        for c in order:
            g, all_ = ok[ok.condition == c], tm[tm.condition == c]
            clip = g["cn_mae_clip_all"].median() if "cn_mae_clip_all" in g.columns else float("nan")
            rows.append(dict(condition=c, datasets=f"{len(g)} / {len(all_)}",
                             ARI_median=g.ari_all.median(), ARI_q25=g.ari_all.quantile(.25), ARI_min=g.ari_all.min(),
                             CN_MAE_median=g.cn_mae_all.median(), CN_MAE_clip_median=clip,
                             gain_recall=g.recall_gain.median(), loss_recall=g.recall_loss.median(),
                             doublet_frac_pred=g.doublet_frac_pred.median(), doublet_frac_true=g.doublet_frac_true.median(),
                             nodes_pred_minus_true=g.node_diff.mean()))
        tab = pd.DataFrame(rows).round(3)
        md += [f"\n## {method}\n",
               "Conditions (median over the replicates that finished; `doublet_frac_*` = share of cells called / truly doublets):\n",
               fence + "\n" + tab.to_string(index=False) + "\n" + fence + "\n"]

        fg = ok[ok.condition.isin(["default", "fgd100"]) & ok.has_fgd.notna()]
        if len(fg) and fg.has_fgd.astype(bool).any():
            fr = []
            for flag, g in fg.groupby(fg.has_fgd.astype(bool)):
                fr.append(dict(dataset_has_FGD=bool(flag), datasets=len(g), ARI_median=g.ari_all.median(), ARI_q25=g.ari_all.quantile(.25),
                               CN_MAE_median=g.cn_mae_all.median(), CN_MAE_mean=g.cn_mae_all.mean(), gain_recall=g.recall_gain.median(),
                               nodes_pred_minus_true=g.node_diff.mean()))
            md += ["\nBy whether the dataset really contains an FGD (conditions default and fgd100 pooled):\n",
                   fence + "\n" + pd.DataFrame(fr).round(3).to_string(index=False) + "\n" + fence + "\n"]

        base = ok[ok.condition == "default"].set_index("rep")
        pr = []
        for c in order:
            if c == "default":
                continue
            g = ok[ok.condition == c].set_index("rep")
            idx = base.index.intersection(g.index)
            if len(idx) == 0:
                continue
            d_ari = (g.loc[idx, "ari_all"] - base.loc[idx, "ari_all"])
            d_cn = (g.loc[idx, "cn_mae_all"] - base.loc[idx, "cn_mae_all"])
            pr.append(dict(condition=c, paired_n=len(idx), ARI_diff_median=d_ari.median(), ARI_diff_worst=d_ari.min(),
                           datasets_ARI_worse_by_0p05=int((d_ari < -0.05).sum()), CN_MAE_diff_median=d_cn.median()))
        if pr:
            md += ["\nDifference to the `default` run of the same replicate (negative ARI difference = worse than default). For the "
                   "annotation-error conditions the data are identical; for the doublet-rate / healthy-fraction conditions they share the "
                   "seed and differ through the changed parameter:\n",
                   fence + "\n" + pd.DataFrame(pr).round(3).to_string(index=False) + "\n" + fence + "\n"]

        # the three lowest-ARI datasets per condition
        worst = ok.sort_values("ari_all").groupby("condition").head(3)[["condition", "rep", "ari_all", "cn_mae_all", "node_diff", "doublet_ratio"]]
        md += ["\nThree lowest-ARI datasets per condition:\n", fence + "\n" + worst.round(3).to_string(index=False) + "\n" + fence + "\n"]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ------------------------------------------------------------------------------------------ statistics
def median_ci(values):
    """Returns (median, lower, upper, n, exact): the distribution-free 95% CI of the median, or the data range if n is too small."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, 0, False
    l = 0
    for k in range(1, n + 1):
        if binom.cdf(k - 1, n, 0.5) <= ALPHA / 2:
            l = k
        else:
            break
    if l == 0:
        return float(np.median(v)), float(v[0]), float(v[-1]), n, False
    return float(np.median(v)), float(v[l - 1]), float(v[n - l]), n, True


FORMATS = {"ari": 3, "bacc": 3, "rec": 2}


def num(x, kind):
    """Format a number by metric kind. Errors get three decimals below 0.1 and two above, and '<0.001' for tiny values."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if kind in ("ari", "bacc", "rec"):
        return f"{x:.3f}".replace("-", "$-$")
    if kind == "err":
        if abs(x) < 0.0005:
            return "$<$0.001" if x >= 0 else "$>-$0.001"
        return (f"{x:.3f}" if abs(x) < 0.1 else f"{x:.2f}").replace("-", "$-$")
    if kind == "signed":
        return (f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}").replace("-", "$-$")
    if kind == "time":
        return (f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}").replace("-", "$-$")
    if kind in ("mem", "pct"):
        return f"{x:.1f}".replace("-", "$-$")
    return f"{x:.3g}".replace("-", "$-$")


def median_cell(values, kind, bold=False):
    """Median without interval."""
    m = median_ci(values)[0]
    if np.isnan(m):
        return "--"
    return f"\\textbf{{{num(m, kind)}}}" if bold else num(m, kind)


def cell(values, kind, bold=False):
    m, lo, hi, n, exact = median_ci(values)
    if n == 0 or np.isnan(m):
        return "--"
    lo_s, hi_s = num(lo, kind), num(hi, kind)
    m_s = f"\\textbf{{{num(m, kind)}}}" if bold else num(m, kind)
    return f"{m_s} {{\\scriptsize({lo_s}, {hi_s})}}" + ("" if exact else "$^\\dagger$")


# ---------------------------------------------------------------------------------------------- tables
HIGHER_IS_BETTER = {"ari": True, "bacc": True, "rec": True, "err": False, "time": False, "mem": False, "signed": False}    # 'pct' (doublet share) has no direction


def shown(x, kind):
    """The value rounded as printed, so that medians that print equal are both marked."""
    if kind == "err":
        return 0.0 if abs(x) < 0.0005 else round(x, 3 if abs(x) < 0.1 else 2)
    if kind in ("ari", "bacc", "rec"):
        return round(x, 3)
    if kind == "time":
        return round(x, 0 if abs(x) >= 10 else 1)
    return round(x, 1)


def block_rows(d, rows, metric, kind, select, methods=None, show_ci=True, min_n=None):
    """LaTeX rows of one metric block. rows is [(key, label)] and select(d, key) gives the datasets of a row.
    The best median of each row is bold."""
    out = []
    methods = methods or METHODS
    for key, label in rows:
        sub = select(d, key)
        vals = [sub.loc[sub["method"] == m, metric] if metric in sub else pd.Series(dtype=float) for m, _ in methods]
        meds = [median_ci(v)[0] if len(v) else np.nan for v in vals]
        best = None
        if kind in HIGHER_IS_BETTER and np.isfinite(meds).any():
            sh = [shown(x, kind) if np.isfinite(x) else np.nan for x in meds]
            best = np.nanmax(sh) if HIGHER_IS_BETTER[kind] else np.nanmin(sh)
        cells = []
        for v, mdn in zip(vals, meds):
            is_best = best is not None and np.isfinite(mdn) and shown(mdn, kind) == best
            if min_n and 0 < len(v) < min_n and pd.to_numeric(v, errors="coerce").notna().any():      # some datasets did not finish: median of the others, marked with *
                cells.append((cell if show_ci else median_cell)(v, kind, bold=is_best) + "$^{*}$")
                continue
            cells.append((cell if show_ci else median_cell)(v, kind, bold=is_best) if len(v) else "--")
        out.append(label + " & " + " & ".join(cells) + "\\\\")
    return out


def stacked_table(d, rows, blocks, select, caption, label, expected=(), suffix="", force_generated=False, methods=None, show_ci=True, min_n=None):
    methods = methods or METHODS
    caption = use_caption(label, caption, expected, suffix, force_generated)
    lines =["\\begin{table}[!htbp]", f"\\caption{{{caption}}}\\label{{{label}}}", "\\scriptsize\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{@{}l" + "c" * len(methods) + "@{}}", "\\toprule", " & " + " & ".join(n for _, n in methods) + "\\\\"]
    for title, metric, kind in blocks:
        lines += ["\\midrule", f"\\multicolumn{{{len(methods) + 1}}}{{@{{}}l}}{{\\textbf{{{title}}}}}\\\\"]
        lines += block_rows(d, rows, metric, kind, select, methods, show_ci, min_n)
    lines += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def scati_weak_sentence(sweep):
    """Caption sentence on the datasets where SCATI has an ARI below 0.95, counted by setting."""
    s = sweep[sweep["method"] == "scati"]
    n = len(s)
    low = s[s["ari_all"] < 0.95]
    if not len(low):
        return f"SCATI had an ARI of at least $0.95$ in all ${n}$ datasets."
    names = {(ax, lv): lab.replace(" (default)", "") for ax, lv, lab in SETTINGS}
    counts = low.groupby(["axis", "level"]).size().sort_values(ascending=False, kind="stable")
    parts = ", ".join(f"{names[(a, str(l))]} ({c})" for (a, l), c in counts.items())
    below9 = int((s["ari_all"] < 0.9).sum())
    fewer = int((low["n_pred_nodes"] < low["n_true_nodes"]).sum())
    return (f"SCATI had an ARI below $0.95$ in ${len(low)}$ of ${n}$ datasets ({'none' if below9 == 0 else below9} below $0.90$), in the settings {parts}; "
            f"in ${fewer}$ of these datasets it inferred fewer clones than simulated.")


def dataset_note(d, n_ds):
    """Caption text on the number of datasets per setting; names the settings where a method has fewer."""
    labels = {(a, str(l)): lab.replace(" (default)", "") for a, l, lab in SETTINGS}
    short = []
    for (a, l), g in d.groupby(["axis", "level"]):
        for m, name in METHODS:
            n = g.loc[g["method"] == m, "rep"].nunique()
            if n < n_ds and (a, str(l)) in labels:
                short.append(f"{name} has {n} at {labels[(a, str(l))]}")
    note = f"{n_ds} datasets per setting"
    if short:
        note += f", except that {'; '.join(short)}, where a run did not finish"
    if d.groupby(["axis", "level", "method"])["rep"].nunique().min() < 6:
        note += "; $^\\dagger$: fewer than $6$ datasets, so the range of the data is shown instead of the confidence interval"
    return note


CAPTIONS = {}
CAPTION_MODE = {"generated": False}      # --captions generated: use the captions of this script, not the manuscript's


def read_captions(tex_path):
    """{label: caption text} of the tables in the manuscript. The file is only read."""
    t = open(tex_path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"\}\\label\{(tab:[\w]+)\}", t):
        a = t.rfind("\\caption{", 0, m.start())
        if a >= 0 and t.rfind("\\begin{table}", 0, m.start()) < a:
            out[m.group(1)] = t[a + len("\\caption{"):m.start()]
    return out


def use_caption(label, generated, expected=(), suffix="", force_generated=False):
    """The manuscript caption verbatim; the generated one only if the manuscript has none.
    Warns on stderr for each text in `expected` that the caption lacks."""
    if CAPTION_MODE["generated"] or force_generated:
        return generated
    cap = CAPTIONS.get(label)
    if cap is None:
        print(f"WARNING {label}: no caption in the manuscript, using the generated one", file=sys.stderr)
        return generated
    for e in expected:
        if e not in cap:
            print(f"WARNING {label}: the caption of the manuscript does not contain '{e}' (the data give this; the caption was not changed)", file=sys.stderr)
    return cap + suffix


def settings_table(sweep, n_ds, methods=None, show_ci=True):
    def select(d, key):
        return d[(d["axis"] == key[0]) & (d["level"].astype(str) == key[1])]
    rows = [((a, l), lab) for a, l, lab in SETTINGS]
    blocks = [("ARI", "ari_all", "ari"), (PAIR["title"], PAIR["metric"], "ari"), ("CN MAE (all cells)", "cn_mae_all", "err"),
              ("VAF MAE (all cells)", "vaf_mae_all", "err")]
    cap = (f"Accuracy by setting of the main benchmark. Median over the datasets of the setting with the exact 95\\% confidence interval of the median "
           f"in parentheses ({dataset_note(sweep, n_ds)}); "
           "the better median of each row is in bold. "
           "ARI: adjusted Rand index of the cell assignments, higher is better; CN MAE and VAF MAE: mean absolute error of the copy number and the variant allele frequency over all cells, lower is better. "
           "SCATI-NA: SCATI run without the healthy-cell annotation. "
           "The methods are run as described in Section~\\ref{sec:app_sim_data}. " + scati_weak_sentence(sweep))
    if not show_ci:     # median-only version: its own caption, without confidence intervals
        cap = (f"Accuracy by setting of the main benchmark. Median over the datasets of the setting ({dataset_note(sweep, n_ds)}). "
               "The better median of each row is in bold. ARI: adjusted Rand index of the cell assignments. "
               "CN MAE and VAF MAE: mean absolute error of the copy number and the variant allele frequency over all cells. "
               "SCATI-wA: SCATI with the true healthy-cell annotation.")
        return stacked_table(sweep, rows, blocks, select, cap, "tab:sim_settings_results", force_generated=True, methods=methods, show_ci=False)
    return stacked_table(sweep, rows, blocks, select, cap, "tab:sim_settings_results", expected=[dataset_note(sweep, n_ds)], suffix=" SCATI was run without the healthy-cell annotation.")


TIME_METHODS = [("scati_noann", "SCATI"), ("scati", "SCATI-wA"), ("compass", "COMPASS"), ("lophy", "LoPhy")]     # SCATI = without annotation, SCATI-wA = with the true annotation


def time_table(sweep, n_ds):
    def select(d, key):
        return d[(d["axis"] == key[0]) & (d["level"].astype(str) == key[1])]
    rows = [((a, l), lab) for a, l, lab in SETTINGS]
    blocks = [("Wall-clock time per dataset (min)", "wall_min", "time"), ("Peak memory (GB)", "max_rss_gb", "mem")]
    cap = (f"Run time and memory by setting of the main benchmark. Median over the datasets of the setting with the exact 95\\% confidence interval of the median in parentheses "
           f"({dataset_note(sweep, n_ds)}); the lower median of each row is in bold. "
           "SCATI runs four chains in parallel; SCATI and COMPASS were run for different numbers of iterations, so the times are those to a result with the settings used here, not costs per iteration. "
           "SCATI-wA: SCATI with the true healthy-cell annotation.")
    return stacked_table(sweep, rows, blocks, select, cap, "tab:sim_time", expected=[dataset_note(sweep, n_ds)],
                         suffix=" SCATI-wA: SCATI with the true healthy-cell annotation.", methods=TIME_METHODS)


ROBUST_GEN = [("default", "Default"), ("fgd100", "Whole-genome duplication (WGD)"), ("dbl0", "Doublet rate 0\\%"), ("dbl0.25", "Doublet rate 25\\%"),
              ("healthy0.05", "Healthy cells 5\\%"), ("healthy0.50", "Healthy cells 50\\%"), ("backmut20", "Back mutations 20\\%"),
              ("false25", "Loci without somatic mutation 25\\%"), ("false50", "Loci without somatic mutation 50\\%")]
ROBUST_ANNOT = [("default", "True annotation (default)"), ("cont30", "T 30\\%"), ("cont50", "T 50\\%"), ("h2c30", "H 30\\%"), ("h2c50", "H 50\\%"),
                ("cont30_h2c30", "T 30\\% + H 30\\%"), ("cont50_h2c50", "T 50\\% + H 50\\%")]


def robust_table(robust, methods=None, show_ci=True):
    d = robust.copy()
    # conditions where a method finished fewer datasets than the others: median over those, marked with *
    nfin = d.groupby(["condition", "method"])["rep"].nunique()
    partial = nfin[nfin < nfin.max()]
    names = {"lophy": "LoPhy", "compass": "COMPASS", "scati": "SCATI", "scati_noann": "SCATI", "scatiwa": "SCATI-wA"}
    labels = dict(ROBUST_GEN)
    note = "".join(f" $^{{*}}$ Median over the {n} of {nfin.max()} datasets that {names.get(m, m)} finished within 24 h (row {labels.get(c, c)})." for (c, m), n in partial.items())

    def select(dd, key):
        return dd[dd["condition"] == key]
    blocks = [("ARI", "ari_all", "ari"), (PAIR["title"], PAIR["metric"], "ari"), ("CN MAE (all cells)", "cn_mae_all", "err"),
              ("VAF MAE (all cells)", "vaf_mae_all", "err"), ("Cells called doublets (\\%)", "doublet_share_called", "pct")]
    cap = ("Robustness to whole-genome duplications, the doublet rate, the fraction of healthy cells, back mutations and loci without a somatic mutation. Median over $10$ datasets per condition with the 95\\% confidence interval of the median in parentheses. "
           "The better median of each row is in bold. WGD: a whole-genome duplication in every dataset. In the conditions with another doublet rate, SCATI and COMPASS keep their doublet rate of $8\\%$. Loci without somatic mutation: the stated share of all candidate loci carries no somatic mutation (germline variants or positions without any variant)."
           "SCATI was run without the healthy-cell annotation." + note)
    if not show_ci:
        cap = ("Robustness to whole-genome duplications, the doublet rate, the fraction of healthy cells, back mutations and loci without a somatic mutation. Median over $10$ datasets per condition. "
               "The better median of each row is in bold. WGD: a whole-genome duplication in every dataset. In the conditions with another doublet rate, SCATI and COMPASS keep their doublet rate of $8\\%$. Loci without somatic mutation: the stated share of all candidate loci carries no somatic mutation (germline variants or positions without any variant)."
               "SCATI-wA: SCATI with the true healthy-cell annotation." + note)
    return stacked_table(d, ROBUST_GEN, blocks, select, cap, "tab:sim_robust", expected=[n_per_condition(d, [k for k, _ in ROBUST_GEN])], force_generated=True,
                         methods=methods, show_ci=show_ci, min_n=int(d.groupby(["condition", "method"])["rep"].nunique().max()))


def n_per_condition(d, conditions):
    """Caption text for the number of datasets per condition; a range if the numbers differ."""
    n = d[d["condition"].isin(conditions)].groupby(["condition", "method"])["rep"].nunique()
    return f"${int(n.max())}$ datasets per condition" if n.min() == n.max() else f"between ${int(n.min())}$ and ${int(n.max())}$ datasets per condition"


def annotation_table(robust):
    """SCATI-WA only: errors in the healthy-cell annotation, which the other methods do not use."""
    d = robust[robust["method"] == "scati"]          # 'scati' is the annotated run here, this table is not remapped
    cols = [("ARI", "ari_all", "ari"), ("CN MAE", "cn_mae_all", "err"), ("VAF MAE", "vaf_mae_all", "err")]
    cap = ("Errors in the healthy-cell annotation given to SCATI-wA. T: share of the cells labelled healthy that are tumor cells; H: share of the healthy cells labelled as tumor. "
           "The datasets are those of the default configuration of the main benchmark. Median over $10$ datasets per condition with the 95\\% confidence interval of the median in parentheses. "
           "No annotation: SCATI (without annotation) on the same datasets.")
    cap = use_caption("tab:sim_annot", cap, [n_per_condition(d, [k for k, _ in ROBUST_ANNOT])], force_generated=True)
    lines = ["\\begin{table}[!htbp]", f"\\caption{{{cap}}}\\label{{tab:sim_annot}}",
             "\\scriptsize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
             "Annotation & " + " & ".join(c[0] for c in cols) + "\\\\", "\\midrule"]
    for key, label in ROBUST_ANNOT:
        sub = d[d["condition"] == key]
        lines.append(label + " & " + " & ".join(cell(sub[c], k) if c in sub and sub[c].notna().any() else "--" for _, c, k in cols) + "\\\\")
    ref = robust[(robust["method"] == "scati_noann") & (robust["condition"] == "default")]
    if len(ref):
        lines += ["\\midrule", "No annotation & " + " & ".join(cell(ref[c], k) if c in ref and ref[c].notna().any() else "--" for _, c, k in cols) + "\\\\"]
    lines += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def iqr_cell(values, kind, bold=False):
    """Median with the 25th and 75th percentile."""
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if not len(v):
        return "--"
    m, lo, hi = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
    m_s = f"\\textbf{{{num(m, kind)}}}" if bold else num(m, kind)
    return f"{m_s} {{\\tiny({num(lo, kind)}, {num(hi, kind)})}}"


POOLED = [("ARI", "ari_all", "ari"), ("CN MAE (all cells)", "cn_mae_all", "err"), ("CN MAE (singlets)", "cn_mae_singlets", "err"),
          ("CN MAE, clipped to 1--3 (all cells)", "cn_mae_clip_all", "err"), ("VAF MAE (all cells)", "vaf_mae_all", "err"),
          ("Balanced accuracy", "balanced_accuracy", "bacc"), ("Recall of gains", "recall_gain", "rec"),
          ("Recall of losses", "recall_loss", "rec"), ("Error in the number of clones", "node_abs_err", "signed"),
          ("Cells called doublets (\\%)", "doublet_share_called", "pct"), ("Wall-clock time (min)", "wall_min", "time"),
          ("Peak memory (GB)", "max_rss_gb", "mem")]


def pooled_table(sweep):
    """All datasets of the main benchmark pooled: median and IQR per method, and the number of datasets in which SCATI is better."""
    n_all = sweep[["setting", "rep"]].drop_duplicates().shape[0]
    cap = (f"Results of the main benchmark pooled over all {n_all} datasets. Median over the datasets with the interquartile range in parentheses; the better median of each row is in bold "
           "(not for the share of called doublets, whose true value is $8\\%$). SCATI-NA: SCATI without the healthy-cell annotation. "
           "Recall of gains and losses: share of the regions with a true copy-number gain or loss that are called as such. Error in the number of clones: absolute difference between the inferred and the true number of clones. "
           "Lower block: number of datasets in which SCATI is better than the method in the column, among the datasets that both finished.")
    cap = use_caption("tab:sim_pooled", cap)
    lines = ["\\begin{table}[!htbp]", f"\\caption{{{cap}}}\\label{{tab:sim_pooled}}", "\\scriptsize\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{@{}l" + "c" * len(METHODS) + "@{}}", "\\toprule", " & " + " & ".join(n for _, n in METHODS) + "\\\\", "\\midrule"]
    for title, metric, kind in POOLED:
        vals = [sweep.loc[sweep["method"] == m, metric] if metric in sweep else pd.Series(dtype=float) for m, _ in METHODS]
        meds = [pd.to_numeric(v, errors="coerce").median() if len(v) else np.nan for v in vals]
        best = None
        if kind in HIGHER_IS_BETTER and np.isfinite(meds).any():
            sh = [shown(x, kind) if np.isfinite(x) else np.nan for x in meds]
            best = np.nanmax(sh) if HIGHER_IS_BETTER[kind] else np.nanmin(sh)
        cells = [iqr_cell(v, kind, bold=(best is not None and np.isfinite(md) and shown(md, kind) == best)) for v, md in zip(vals, meds)]
        lines.append(title + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\midrule", f"\\multicolumn{{{len(METHODS) + 1}}}{{@{{}}l}}{{\\textbf{{Datasets in which SCATI is better}}}}\\\\"]
    for title, metric, higher in [("ARI", "ari_all", True), ("CN MAE (all cells)", "cn_mae_all", False), ("VAF MAE (all cells)", "vaf_mae_all", False)]:
        w = sweep.pivot_table(index=["setting", "rep"], columns="method", values=metric, aggfunc="first")
        cells = ["--"]
        for m, _ in METHODS[1:]:
            d = w[["scati", m]].dropna()
            better = int((d["scati"] > d[m]).sum()) if higher else int((d["scati"] < d[m]).sum())
            cells.append(f"{better}/{len(d)}")
        lines.append(title + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- macros
def macros(sweep, methods=None):
    methods = methods or METHODS
    out = ["% generated by make_benchmark_tables.py -- do not edit"]

    def add(name, value):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    ds = sweep[["setting", "rep"]].drop_duplicates()
    add("SimNDatasets", str(len(ds)))
    for m, nm in methods:
        sub = sweep[sweep["method"] == m]
        dec = 3 if m.startswith("scati") else 2
        nm = nm.replace(" ", "").replace("/", "").replace("-", "")
        add(f"SimNDatasets{nm}", str(len(sub)))
        add(f"SimTime{nm}Median", f"{sub['wall_min'].median():.0f}")
        add(f"SimTime{nm}MaxH", f"{sub['wall_min'].max() / 60:.1f}")
        add(f"SimMemory{nm}Median", f"{sub['max_rss_gb'].median():.1f}")
        add(f"SimAri{nm}Median", f"{sub['ari_all'].median():.3f}")
        if "pair_acc" in sub:
            add(f"SimPairAcc{nm}Median", f"{sub['pair_acc'].median():.3f}")
            add(f"SimPairF1{nm}Median", f"{sub['pair_f1_macro'].median():.3f}")
        add(f"SimCn{nm}Median", f"{sub['cn_mae_all'].median():.3f}")
        # numbers for the failure modes in the main text
        add(f"SimAriBelowHalf{nm}Percent", f"{100 * (sub['ari_all'] < 0.5).mean():.0f}")
        add(f"SimPredOverTrueClones{nm}Median", f"{(sub['n_pred_nodes'] / sub['n_true_nodes']).median():.1f}")
        add(f"SimRecallGain{nm}Median", f"{sub['recall_gain'].median():.2f}")
        add(f"SimRecallLoss{nm}Median", f"{sub['recall_loss'].median():.2f}")
        add(f"SimCnClipped{nm}Median", f"{sub['cn_mae_clip_all'].median():.3f}")
        add(f"SimVaf{nm}Median", f"{sub['vaf_mae_all'].median():.4f}")
        # range of the per-setting median ARI over the settings
        per = sub.groupby(["axis", "level"])["ari_all"].median()
        add(f"SimAri{nm}SettingsMin", f"{per.min():.{dec}f}")
        add(f"SimAri{nm}SettingsMax", f"{per.max():.{dec}f}")
        names = {(ax, lv): lab.replace(" (default)", "") for ax, lv, lab in SETTINGS}
        add(f"SimAri{nm}SettingsMinSetting", names[tuple(map(str, per.idxmin()))].replace("&", "\&"))
        cn = sub[sub["axis"] == "n_CNAs"].groupby("level")["cn_mae_all"].median()
        for lev in ("5", "10", "15"):
            if lev in cn.index.astype(str):
                add(f"SimCn{nm}Cnas{'Five' if lev == '5' else 'Ten' if lev == '10' else 'Fifteen'}", f"{cn[cn.index.astype(str) == lev].iloc[0]:.3f}")
    return "\n".join(out) + "\n"


def robust_macros(robust):
    """Lowest median ARI of SCATI over the conditions, and the median ARI of COMPASS and LoPhy at doublet rates 0 and 0.25."""
    out = []
    labels = dict(ROBUST)
    d = robust[robust["method"] == "scati"]
    per = d.groupby("condition")["ari_all"].median()
    per = per[per.index.isin(labels)]
    worst = per.idxmin()
    out.append(f"\\newcommand{{\\SimRobustScatiAriMin}}{{{per.min():.3f}}}")
    out.append(f"\\newcommand{{\\SimRobustScatiAriMinCondition}}{{{labels[worst]}}}")
    for m, nm in METHODS[1:]:
        s = robust[(robust["method"] == m) & robust["condition"].isin(["dbl0", "dbl0.25"])].groupby("condition")["ari_all"].median()
        if len(s):
            out.append(f"\\newcommand{{\\SimRobust{nm.replace(' ', '').replace('/', '')}AriLowDbl}}{{{s.get('dbl0', np.nan):.2f}}}")
            out.append(f"\\newcommand{{\\SimRobust{nm.replace(' ', '').replace('/', '')}AriHighDbl}}{{{s.get('dbl0.25', np.nan):.2f}}}")
    return "\n".join(out) + "\n"


def plain_numbers(macro_text):
    """The macros as 'name: value' lines, to check the hand-written text against."""
    import re
    lines = ["# numbers quoted in the appendix text, generated by make_benchmark_tables.py"]
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", macro_text):
        lines.append(f"{m.group(1)}: {m.group(2).replace(chr(92) + '%', '%')}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True, help="folder written by run_analysis.sh (contains sweep/ and robust/)")
    ap.add_argument("--out", default=None, help="also write the tables and the numbers quoted in the text to this folder")
    ap.add_argument("--sim_roots", nargs="*", default=[], help="simulation folders, used to read whether a dataset has an FGD")
    ap.add_argument("--captions", default="generated", choices=["manuscript", "generated"],
                    help="generated (default): captions of this script; manuscript: captions of the --tex file, verbatim")
    ap.add_argument("--max_hours", type=float, default=24.0, help="runs longer than this count as not finished (default 24; 0 = no limit)")
    ap.add_argument("--pair_metric", choices=["acc", "f1"], default="acc", help="cell-pair relations: accuracy (default) or macro F1")
    ap.add_argument("--tex", default=None, help="manuscript .tex file to read the captions from with --captions manuscript (read only)")
    a = ap.parse_args()
    MAX_HOURS["value"] = a.max_hours
    if a.pair_metric == "f1":
        PAIR.update(metric="pair_f1_macro", title="Cell-pair F1")
    if a.captions == "manuscript":
        if not a.tex:
            ap.error("--captions manuscript needs --tex")
        CAPTIONS.update(read_captions(a.tex))
    CAPTION_MODE["generated"] = a.captions == "generated"
    sweep = load(os.path.join(a.analysis, "sweep", "metrics.csv"))
    robust_dir = os.path.join(a.analysis, "robust")
    metric_files = [os.path.join(robust_dir, "metrics.csv"), os.path.join(a.analysis, "sweep", "metrics.csv"), os.path.join(robust_dir, "metrics_annot.csv")]
    conditions = conditions_table(pd.concat([pd.read_csv(m) for m in metric_files], ignore_index=True), a.sim_roots)
    conditions.to_csv(os.path.join(robust_dir, "conditions.csv"), index=False)
    robustness_summary(conditions, robust_dir)
    robust = load(os.path.join(robust_dir, "conditions.csv"))

    def remap_noann(d):
        """The SCATI column of the tables holds the runs without annotation; the annotated runs only go into the annotation table."""
        if "scati_noann" not in set(d["method"]):
            return d
        d = d[d["method"] != "scati"].copy()
        d["method"] = d["method"].replace({"scati_noann": "scati"})
        return d
    robust_annot = robust.copy()
    sweep_both = sweep.copy()                      # SCATI with and without annotation, for the time table
    sweep, robust = remap_noann(sweep), remap_noann(robust)
    n_ds = int(sweep.groupby(["axis", "level"])["rep"].nunique().max())
    three = sweep[sweep["method"].isin([m for m, _ in METHODS])]

    tables = {"tab_sim_settings": settings_table(sweep_both[sweep_both["method"].isin([m for m, _ in TIME_METHODS])], n_ds, TIME_METHODS, show_ci=False), "tab_sim_time": time_table(sweep_both[sweep_both["method"].isin([m for m, _ in TIME_METHODS])], n_ds), "tab_sim_robust": robust_table(robust_annot[robust_annot["method"].isin([m for m, _ in TIME_METHODS])], TIME_METHODS, show_ci=False),
              "tab_sim_annot": annotation_table(robust_annot)}
    wa = sweep_both[sweep_both["method"] == "scati"].copy()
    wa["method"] = "scatiwa"
    numbers = plain_numbers(macros(pd.concat([sweep, wa]), METHODS + [("scatiwa", "SCATI-wA")]) + robust_macros(robust))
    # print the LaTeX; the manuscript is not edited
    for name, text in tables.items():
        print(f"% ===== {name} =====")
        print(text)
    print("% ===== numbers quoted in the text (the text is written by hand) =====")
    print("% " + scati_weak_sentence(three))
    print("\n".join("% " + l for l in numbers.splitlines()))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        files = {f"{k}.tex": v for k, v in tables.items()}
        files["sim_text_numbers.txt"] = numbers
        for name, text in files.items():
            with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)


if __name__ == "__main__":
    main()
