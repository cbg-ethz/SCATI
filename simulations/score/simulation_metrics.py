"""Accuracy and runtime metrics of the simulation benchmark, one row per (dataset, method).

Metrics are computed per cell, so they mean the same for every method.

  cn_mae_*       mean |predicted CN - true CN| over cells x regions
  cn_mae_clip_*  same, with both CNs clipped to 1..3 (loss / neutral / gain)
  vaf_mae_*      mean |predicted VAF - true VAF| over cells x SNV loci
  ari_*          ARI of cell labels: one class per clone plus one shared doublet class; mean over
                 samples, "_pooled" over all cells (only for methods run jointly on all samples)
  doublet_precision/_recall/_f1, singlets_called_doublet, doublet_recall_mixed
                 doublet detection; "mixed" = doublets of two different clones
  recall_gain/loss/neutral, balanced_accuracy
                 CN class of each singlet cell x region relative to the root CN of truth and method
  pair_acc, pair_f1_*  cell-pair relations in the tree (section "cell-pair relations" below), on the true singlets that
                 no method calls a doublet; n_pair_cells is their number
  *_all / *_singlets  over all cells (truth of a doublet = sum of its two clones) or true singlets only

Also dataset sizes, true and predicted doublet and node counts, and runtime and memory from
benchmark.tsv, run_summary.json and run_info.txt.

CLI:
    python score/simulation_metrics.py --sim_root simulated_data --methods scati scati_noann compass_snp lophy \
        --out simulation_metrics.csv
    python score/simulation_metrics.py --rep_dir simulated_data/1sample/rep0 --methods scati
"""
import argparse
import glob
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:  # no scikit-learn: same formula in numpy/pandas
    def adjusted_rand_score(labels_true, labels_pred):
        """Adjusted Rand index (Hubert & Arabie), same as sklearn's."""
        t, p = np.asarray(labels_true), np.asarray(labels_pred)
        n = len(t)
        if n < 2:
            return 1.0
        table = pd.crosstab(pd.Series(t, name="t"), pd.Series(p, name="p")).to_numpy().astype(np.float64)
        comb2 = lambda x: x * (x - 1.0) / 2.0
        sum_ij = comb2(table).sum()
        sum_a = comb2(table.sum(axis=1)).sum()
        sum_b = comb2(table.sum(axis=0)).sum()
        total = comb2(float(n))
        expected = sum_a * sum_b / total
        max_index = (sum_a + sum_b) / 2.0
        if max_index == expected:  # both clusterings trivial
            return 1.0
        return float((sum_ij - expected) / (max_index - expected))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import method_adapters as ma  # noqa: E402 
from genotype_comparison import cn_from_strands  # noqa: E402


# ---- cell-pair relations
# The relation of two cells in nodes a and b is same (a = b), above (a is an ancestor of b), below (a is a descendant of b) or apart
# (different branches). Pairs count in both directions. The confusion matrix over all pairs follows from the cell counts per
# (true node, inferred node), so no pairs are sampled. Scores: accuracy, the F1 of same / related (above + below) / apart, and their mean
# (macro F1). Unlike the ARI they see the topology: swapping a parent and a child leaves the ARI at 1 but lowers the pair accuracy.

CLASS_NAMES = ("same", "above", "below", "apart")


def node_relation_matrix(parent):
    """parent[v] is the parent of node v, -1 for a root. Returns R (n x n) with R[a, b] the class of a relative to b."""
    parent = np.asarray(parent, dtype=int)
    n = len(parent)
    anc = np.zeros((n, n), dtype=bool)          # anc[a, b]: a is a proper ancestor of b
    for b in range(n):
        v, seen = parent[b], 0
        while v >= 0 and seen <= n:
            anc[v, b] = True
            v, seen = parent[v], seen + 1
    R = np.full((n, n), 3, dtype=int)
    R[anc] = 1
    R[anc.T] = 2
    np.fill_diagonal(R, 0)
    return R


def read_tree_parents(path, n_nodes=None, clone_ids=None):
    """Parent of every node (-1 for the root) from the edges 'a -> b' of a .gv tree file ('Clone_a -> Clone_b' for LoPhy).
    Dashed edges (cell bubbles) and dir=none edges are skipped.
    clone_ids: LoPhy clone ids by matrix row (row r = clone_ids[r]); parents are then given as row numbers."""
    edges = []
    with open(path) as f:
        text = f.read()
    for line in text.split("\n"):
        if "->" not in line or "dashed" in line or "dir=none" in line:
            continue
        m = re.match(r"\s*(?:Clone_)?(\d+)\s*->\s*(?:Clone_)?(\d+)", line)
        if m:
            edges.append((int(m.group(1)), int(m.group(2))))
    if clone_ids is not None:
        row = {c: r for r, c in enumerate(clone_ids)}
        edges = [(row[a], row[b]) for a, b in edges if a in row and b in row]
    n = n_nodes if n_nodes is not None else (max([max(e) for e in edges] + [0]) + 1)
    parent = np.full(n, -1, dtype=int)
    for a, b in edges:
        if b < n and a < n:
            parent[b] = a
    return parent


def pair_confusion(true_node, pred_node, R_true, R_pred):
    """4x4 confusion matrix over all ordered cell pairs i != j, rows true class, columns inferred class.
    true_node, pred_node: node of each cell in the two trees; every cell must have a node."""
    true_node = np.asarray(true_node, dtype=int)
    pred_node = np.asarray(pred_node, dtype=int)
    nt, npred = R_true.shape[0], R_pred.shape[0]
    table = np.zeros((nt, npred))
    np.add.at(table, (true_node, pred_node), 1.0)
    a, x = np.nonzero(table)
    w = table[a, x]
    Rt = R_true[a[:, None], a[None, :]]
    Rp = R_pred[x[:, None], x[None, :]]
    conf = np.zeros((4, 4))
    for ct in range(4):
        mt = Rt == ct
        for cp in range(4):
            conf[ct, cp] = w @ (mt & (Rp == cp)) @ w
    conf[0, 0] -= w.sum()       # remove the pairs of a cell with itself
    return conf


def pair_scores(conf):
    """Accuracy over the 4 classes and the F1 of same, related (above + below) and apart.
    A related pair is correct only if the direction is right. Macro F1 is the mean of the three F1 values,
    skipping relations that occur in neither tree."""
    total = conf.sum()
    out = {"pair_acc": float(np.trace(conf) / total) if total > 0 else np.nan}
    groups = {"same": [0], "related": [1, 2], "apart": [3]}
    f1 = []
    for name, cls in groups.items():
        tp = sum(conf[c, c] for c in cls)
        denom = conf[cls, :].sum() + conf[:, cls].sum()
        v = 2.0 * tp / denom if denom > 0 else np.nan
        out[f"pair_f1_{name}"] = float(v)
        f1.append(v)
    out["pair_f1_macro"] = float(np.nanmean(f1)) if np.isfinite(f1).any() else np.nan
    return out

# method key -> adapter in method_adapters.ADAPTERS, results folder, has copy numbers, run per sample
METHOD_CONFIG = {
    "scati":               dict(adapter="scati", dirname="SCATI", results_name="results", has_cn=True, per_sample=False),
    "scati_noann":         dict(adapter="scati", dirname="SCATI", results_name="results_noann", has_cn=True, per_sample=False),   # no healthy/tumor annotation
    "compass_snp":         dict(adapter="compass", dirname="COMPASS_snp", results_name="results", has_cn=True, per_sample=True),
    "lophy":               dict(adapter="lophy", dirname="LoPhy", results_name="results", has_cn=True, per_sample=False),
}
# SCATI runs with an erroneous healthy/tumor annotation (Snakefile_robustness): SCATI/results_annot_<tag>/
ANNOTATION_ERROR_TAGS = ("cont30", "cont50", "h2c30", "h2c50", "cont30_h2c30", "cont50_h2c50")
for _tag in ANNOTATION_ERROR_TAGS:
    METHOD_CONFIG[f"scati_annot_{_tag}"] = dict(adapter="scati", dirname="SCATI", results_name=f"results_annot_{_tag}",
                                               has_cn=True, per_sample=False)
CLASSES = ("loss", "neutral", "gain")  # row/column order of the 3x3 confusion matrices


# ── Per-unit evaluation (one method run on a set of cells) ───────────────────

def _true_labels(truth_ca):
    """True label of each cell: 's<node>' for singlets, 'd<a>_<b>' (sorted clone pair) for doublets."""
    labels = []
    for node, node2, is_d in zip(truth_ca["node"], truth_ca["node2"], truth_ca["is_doublet"]):
        labels.append(f"d{min(node, node2)}_{max(node, node2)}" if is_d else f"s{node}")
    return np.array(labels)


def _pred_labels(group_boundaries, cell_order, n_cells):
    """Predicted label of each cell in original order, from the groups of order_cells."""
    labels = np.empty(n_cells, dtype=object)
    order = np.asarray(cell_order)
    for _, start, end, is_doublet, key in group_boundaries:
        lab = f"d{min(key)}_{max(key)}" if is_doublet else f"s{key}"
        labels[order[start:end]] = lab
    return labels


def _collapse_doublets(labels):
    """Merge all doublet labels into one class "D", so every method is scored on the same label space.
    SCATI and the truth name the clone pair, COMPASS only flags a doublet, LoPhy has none."""
    return np.array(["D" if str(l)[:1] in ("d", "D") else l for l in labels], dtype=object)


def _predicted_root_cn(cfg, results_dir, input_dir, n_regions):
    """Root copy number per region of the method's tree, None if not available."""
    if not cfg["has_cn"]:
        return None
    if cfg["adapter"] == "scati":
        path = os.path.join(results_dir, "best_tree_strand_genotypes.json")
        with open(path) as f:
            strand = json.load(f)
        root = cn_from_strands(strand["0"]["strands_regions"], n_regions)
        return root if len(root) == n_regions else None
    if cfg["adapter"] == "compass":
        # same table as the adapter uses for its predictions
        n_nodes = len(pd.read_csv(os.path.join(results_dir, "data_nodes_genotypes.tsv"), sep="\t", index_col=0))
        return ma.load_compass_cn_table(results_dir, input_dir, n_nodes).iloc[0].to_numpy(dtype=float)
    if cfg["adapter"] == "lophy":
        # the LoPhy root is clone 0, the first row of its matrices
        with open(os.path.join(results_dir, "data_0.gv")) as f:
            total_cn = np.array(ma._parse_gv_attr(f.read(), "total_copy_numbers"), dtype=float)
        return total_cn[0] if total_cn.shape[1] == n_regions else None
    return None


def _cn_class(cn, root):
    """Class of each copy number relative to the root: -1 loss, 0 neutral, +1 gain; NaN if unknown."""
    out = np.sign(np.round(cn) - np.round(root)[None, :])
    out[~np.isfinite(cn)] = np.nan
    return out


def _node_of_label(labels):
    """Node of each cell from its label 's<node>'; -1 for a doublet."""
    return np.array([int(str(l)[1:]) if str(l)[:1] == "s" else -1 for l in labels], dtype=int)


def _pad(parent, n):
    parent = np.asarray(parent, dtype=int)
    return np.concatenate([parent, np.full(max(0, n - len(parent)), -1, dtype=int)])


def _tree_inputs(cfg, rep_dir, results_dir, true_labels, pred_labels, tree_path):
    """Cell nodes and node parents of the true and the inferred tree, as input for the pair scores.
    A parent array is None if its tree file cannot be read; the pair metrics are then NaN."""
    true_node, pred_node = _node_of_label(true_labels), _node_of_label(pred_labels)
    out = dict(true_node=true_node, pred_node=pred_node, true_parent=None, pred_parent=None)
    try:
        gv = os.path.join(rep_dir, "SCATI", "data_treeTRUE.gv")
        out["true_parent"] = _pad(read_tree_parents(gv), int(true_node.max()) + 1)
        if cfg["adapter"] == "lophy":
            # LoPhy cells are assigned to matrix rows (kept clones by ascending id, root first); the tree names clone ids
            with open(os.path.join(results_dir, "data_0.gv")) as f:
                text = f.read()
            n_rows = len(ma._parse_gv_attr(text, "total_copy_numbers"))
            body = re.sub(r"^[ \t]*graph \[.*?^\];", "", text, count=1, flags=re.DOTALL | re.MULTILINE)
            ids = sorted({0} | {int(i) for i in re.findall(r"^\s*Clone_(\d+)\s*\[label", body, re.MULTILINE)})
            if len(ids) != n_rows:
                ids = list(range(n_rows))
            out["pred_parent"] = read_tree_parents(tree_path, n_nodes=max(n_rows, int(pred_node.max()) + 1), clone_ids=ids)
        else:
            out["pred_parent"] = _pad(read_tree_parents(tree_path), int(pred_node.max()) + 1)
    except Exception as e:  # the other metrics stay valid
        print(f"  warning: tree not read, no pair metrics: {type(e).__name__}: {e}", file=sys.stderr)
        out["pred_parent"] = None
    return out


def add_pair_metrics(rep_rows):
    """Adds the pair scores of all methods of one dataset to the rows in place.
    All methods are scored on the same cells: true singlets that no method calls a doublet.
    Scores are computed per sample and averaged, since per-sample runs have one tree per sample."""
    ok = [r for r in rep_rows if r.get("status") == "ok" and r.get("_units")]
    keep = {id(r): np.concatenate([(~u["is_doublet"]) & (u["pred_node"] >= 0) & (u["true_node"] >= 0) for u in r["_units"]]) for r in ok}
    if len({len(k) for k in keep.values()}) != 1:
        return
    common = np.logical_and.reduce(list(keep.values())) if keep else None
    for r in ok:
        confs, offset = [], 0
        for u in r["_units"]:
            n = len(u["is_doublet"])
            cm, offset = common[offset:offset + n], offset + n
            if u["true_parent"] is None or u["pred_parent"] is None:
                continue
            Rt, Rp = node_relation_matrix(u["true_parent"]), node_relation_matrix(u["pred_parent"])
            for k in np.unique(u["sample_id"]):
                m = cm & (u["sample_id"] == k)
                if m.sum() >= 2:
                    confs.append(pair_confusion(u["true_node"][m], u["pred_node"][m], Rt, Rp))
        names = ["pair_acc", "pair_f1_same", "pair_f1_related", "pair_f1_apart", "pair_f1_macro"]
        scores = [pair_scores(c) for c in confs]
        for name in names:
            vals = [s[name] for s in scores if np.isfinite(s[name])]
            r[name] = float(np.mean(vals)) if vals else np.nan
        r["n_pair_cells"] = int(common.sum())


def evaluate_unit(method, rep_dir, sample_subdir=None):
    """Evaluates one run of `method` on the cells of `sample_subdir` (None = all samples).
    Returns per-cell arrays in original cell order and the 3x3 CN class confusion of true singlets."""
    cfg = METHOD_CONFIG[method]
    method_dir = os.path.join(rep_dir, cfg["dirname"])
    results_dir = os.path.join(method_dir, cfg["results_name"], sample_subdir) if sample_subdir \
        else os.path.join(method_dir, cfg["results_name"])
    input_dir = os.path.join(method_dir, sample_subdir) if sample_subdir else method_dir

    bases = ma.sample_basenames(rep_dir, sample_subdir)
    strand_gt, n_regions, n_loci = ma.load_ground_truth(rep_dir, bases)
    cn_pred, vaf_pred, cell_order, groups, tree_path = ma.ADAPTERS[cfg["adapter"]](
        results_dir, input_dir, n_regions, n_loci)

    truth_ca = ma._load_ground_truth_cell_assignments(rep_dir, bases)
    n_cells = len(truth_ca)
    if len(cell_order) != n_cells:
        raise ValueError(f"{method}: {len(cell_order)} cells in the results but {n_cells} in the data")
    order = np.asarray(cell_order)

    # restore the original cell order (row i of the adapter output is cell order[i])
    cn_p = np.empty_like(cn_pred, dtype=float)
    cn_p[order] = cn_pred
    vaf_p = np.empty_like(vaf_pred, dtype=float)
    vaf_p[order] = vaf_pred
    cn_t = ma.ground_truth_matrix_per_cell(rep_dir, bases, strand_gt, n_regions, "cn")
    vaf_t = ma.ground_truth_matrix_per_cell(rep_dir, bases, strand_gt, n_loci, "vaf")

    is_doublet = truth_ca["is_doublet"].to_numpy(dtype=bool)
    true_labels = _true_labels(truth_ca)
    pred_labels = _pred_labels(groups, cell_order, n_cells)
    if cfg["adapter"] == "compass":
        # COMPASS only flags doublets: a flagged cell gets the label "D<node>", which matches no true label
        ca = pd.read_csv(os.path.join(results_dir, "data_cellAssignments.tsv"), sep="\t")
        flagged = ca["doublet"].astype(str).str.lower().eq("yes").to_numpy()
        pred_labels[flagged] = [f"D{n}" for n in ca["node"].to_numpy()[flagged]]

    # sample index of each cell, global also for per-sample runs
    if sample_subdir:
        first = int(re.fullmatch(r"sample(\d+)", sample_subdir).group(1))
        sample_id = np.full(n_cells, first)
    else:
        lengths = [len(pd.read_csv(os.path.join(rep_dir, "SCATI", b + "_cellAssignmentsTRUE.tsv"), sep="\t"))
                   for b in bases]
        sample_id = np.repeat(np.arange(len(bases)), lengths)

    out = dict(sample_id=sample_id, is_doublet=is_doublet, true_label=true_labels, pred_label=pred_labels)
    out.update(_tree_inputs(cfg, rep_dir, results_dir, true_labels, pred_labels, tree_path))

    def abs_err(pred, true):
        ok = np.isfinite(pred) & np.isfinite(true)
        diff = np.where(ok, np.abs(np.where(ok, pred, 0.0) - np.where(ok, true, 0.0)), 0.0)
        return diff.sum(axis=1), ok.sum(axis=1)

    out["vaf_err_sum"], out["vaf_n"] = abs_err(vaf_p, vaf_t)
    conf = np.full((3, 3), np.nan)
    if cfg["has_cn"]:
        out["cn_err_sum"], out["cn_n"] = abs_err(cn_p, cn_t)
        # CN error with both sides clipped to 1..3
        out["cnclip_err_sum"], out["cnclip_n"] = abs_err(np.clip(cn_p, 1, 3), np.clip(cn_t, 1, 3))
        pred_root = _predicted_root_cn(cfg, results_dir, input_dir, n_regions)
        true_root = cn_from_strands(strand_gt["0"]["strands_regions"], n_regions)
        if pred_root is not None:
            tc, pc = _cn_class(cn_t, true_root), _cn_class(cn_p, pred_root)
            ok = np.isfinite(tc) & np.isfinite(pc) & (~is_doublet)[:, None]
            conf = np.zeros((3, 3))
            for i, t in enumerate((-1, 0, 1)):
                for j, p in enumerate((-1, 0, 1)):
                    conf[i, j] = np.sum(ok & (tc == t) & (pc == p))
    else:
        out["cn_err_sum"] = np.full(n_cells, np.nan)
        out["cn_n"] = np.zeros(n_cells)
        out["cnclip_err_sum"] = np.full(n_cells, np.nan)
        out["cnclip_n"] = np.zeros(n_cells)
    out["conf"] = conf
    out["n_true_nodes"] = len(set(l for l in true_labels if l.startswith("s")))
    return out


# ── Summaries over the runs of one dataset ───────────────────────────────────

def _mae(units, mask_fn, kind):
    num = sum(np.nansum(u[f"{kind}_err_sum"][mask_fn(u)]) for u in units)
    den = sum(u[f"{kind}_n"][mask_fn(u)].sum() for u in units)
    return float(num / den) if den > 0 else np.nan


def _mean_sample_ari(units, singlets_only):
    """ARI per sample, averaged over samples."""
    t = _collapse_doublets(np.concatenate([u["true_label"] for u in units]))
    p = _collapse_doublets(np.concatenate([u["pred_label"] for u in units]))
    s = np.concatenate([u["sample_id"] for u in units])
    d = np.concatenate([u["is_doublet"] for u in units])
    keep = ~d if singlets_only else np.ones(len(t), bool)
    aris = [adjusted_rand_score(t[keep & (s == k)], p[keep & (s == k)])
            for k in np.unique(s) if (keep & (s == k)).sum() > 1]
    return float(np.mean(aris)) if aris else np.nan


def _pooled_ari(units, singlets_only):
    t = _collapse_doublets(np.concatenate([u["true_label"] for u in units]))
    p = _collapse_doublets(np.concatenate([u["pred_label"] for u in units]))
    d = np.concatenate([u["is_doublet"] for u in units])
    keep = ~d if singlets_only else np.ones(len(t), bool)
    return float(adjusted_rand_score(t[keep], p[keep])) if keep.sum() > 1 else np.nan


def summarize(units, per_sample_method, has_doublet_model=True):
    row = {}
    for scope, fn in (("all", lambda u: np.ones(len(u["is_doublet"]), bool)),
                      ("singlets", lambda u: ~u["is_doublet"])):
        row[f"cn_mae_{scope}"] = _mae(units, fn, "cn")
        row[f"cn_mae_clip_{scope}"] = _mae(units, fn, "cnclip")
        row[f"vaf_mae_{scope}"] = _mae(units, fn, "vaf")
    # ARI over all cells only
    row["ari_all"] = _mean_sample_ari(units, False)
    # clusters of per-sample runs differ between samples, so no pooled ARI
    row["ari_all_pooled"] = (np.nan if per_sample_method and len(units) > 1 else _pooled_ari(units, False))
    conf = sum(np.nan_to_num(u["conf"]) for u in units) if all(np.isfinite(u["conf"]).all() for u in units) \
        else np.full((3, 3), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(conf) / conf.sum(axis=1)
    for name, r in zip(CLASSES, recall):
        row[f"recall_{name}"] = float(r) if np.isfinite(r) else np.nan
    row["balanced_accuracy"] = float(np.nanmean(recall)) if np.isfinite(recall).any() else np.nan
    # 3x3 CN class confusion, rows true and columns predicted
    for i, t_name in enumerate(CLASSES):
        for j, p_name in enumerate(CLASSES):
            row[f"conf_{t_name}_{p_name}"] = float(conf[i, j])

    d = np.concatenate([u["is_doublet"] for u in units])
    p = np.concatenate([u["pred_label"] for u in units])
    row["n_cells"] = int(len(d))
    row["n_true_doublets"] = int(d.sum())
    row["n_pred_doublets"] = int(sum(1 for l in p if l[0] in "dD"))
    # doublet detection, for methods with a doublet model; "mixed" = doublet of two different clones,
    # since a same-clone doublet looks like a singlet
    pred_d = np.array([l[0] in "dD" for l in p])
    t_lab = np.concatenate([u["true_label"] for u in units])
    pairs = [re.fullmatch(r"d(\d+)_(\d+)", str(l)) for l in t_lab]
    mixed = np.array([m is not None and m.group(1) != m.group(2) for m in pairs])
    tp, fp, fn = int((d & pred_d).sum()), int((~d & pred_d).sum()), int((d & ~pred_d).sum())
    n_single = int((~d).sum())
    nan = float("nan")
    if has_doublet_model:
        row["doublet_precision"] = tp / (tp + fp) if (tp + fp) else nan
        row["doublet_recall"] = tp / (tp + fn) if (tp + fn) else nan
        row["doublet_f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan
        row["doublet_recall_mixed"] = float((pred_d & mixed).sum() / mixed.sum()) if mixed.sum() else nan
        row["singlets_called_doublet"] = fp / n_single if n_single else nan
    else:
        for k in ("doublet_precision", "doublet_recall", "doublet_f1", "doublet_recall_mixed", "singlets_called_doublet"):
            row[k] = nan
    row["n_true_nodes"] = int(max(u["n_true_nodes"] for u in units))
    # per-sample runs have one tree per sample: take the largest count, as for the truth
    if per_sample_method:
        row["n_pred_nodes"] = int(max(len({l for l in u["pred_label"] if l[0] == "s"}) for u in units))
    else:
        row["n_pred_nodes"] = int(len({l for u in units for l in u["pred_label"] if l[0] == "s"}))
    return row


# ── Runtime and memory ───────────────────────────────────────────────────────

def _read_kv(path):
    out = {}
    if os.path.isfile(path):
        with open(path) as f:
            for tok in f.read().split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k] = v
    return out


def runtime_info(method, rep_dir, sample_subdir=None):
    """Runtime, peak memory and MCMC steps of one run, read from benchmark.tsv (all methods),
    run_summary.json (SCATI) and run_info.txt (COMPASS, LoPhy)."""
    cfg = METHOD_CONFIG[method]
    results = os.path.join(rep_dir, cfg["dirname"], cfg["results_name"])
    if sample_subdir:
        results = os.path.join(results, sample_subdir)
    info = {}
    bench = os.path.join(results, "benchmark.tsv")
    if os.path.isfile(bench):
        b = pd.read_csv(bench, sep="\t").iloc[0]
        info.update(wall_seconds=float(b["s"]), cpu_seconds=float(b.get("cpu_time", np.nan)),
                    max_rss_gb=float(b.get("max_rss", np.nan)) / 1024.0, runtime_source="benchmark")
    summary = os.path.join(results, "run_summary.json")
    if os.path.isfile(summary):
        with open(summary) as f:
            s = json.load(f)
        info.setdefault("wall_seconds", s.get("total_wall_seconds"))
        info.setdefault("max_rss_gb", s.get("peak_rss_gb"))
        info.setdefault("runtime_source", "run_summary")
        info.update(n_chains=s.get("n_chains"), mcmc_steps_per_chain=s.get("mcmc_steps_per_chain"),
                    steps_per_second=s.get("mean_steps_per_second_per_chain"),
                    cache_gb=s.get("cache_gb"), lambda_rounding_precision=s.get("lambda_rounding_precision"))
    run_info = _read_kv(os.path.join(results, "run_info.txt"))
    if run_info:
        info.setdefault("n_chains", int(run_info["n_chains"]) if "n_chains" in run_info else None)
        if "chain_length" in run_info:
            info.setdefault("mcmc_steps_per_chain", int(run_info["chain_length"]))
        if "iters" in run_info:
            info.setdefault("lophy_iters", int(run_info["iters"]))
    info.setdefault("runtime_source", "none")
    return info


# ── Whole datasets ───────────────────────────────────────────────────────────

_SETTING_PATTERNS = [
    (r"nodes(\d+)", "n_nodes"), (r"SNVs(\d+)", "n_SNVs"), (r"CNAs(\d+)", "n_CNAs"),
    (r"(\d+)samples?", "n_samples"),
]


def parse_setting(name):
    """Sweep axis and level of a setting folder name; the level of 'event_type' is text."""
    for pattern, axis in _SETTING_PATTERNS:
        m = re.fullmatch(pattern, name)
        if m:
            return axis, int(m.group(1))
    if name in ("SNV_only", "local_events", "local_and_large_scale_events"):
        return "event_type", name
    return "other", name


def discover_runs(sim_root):
    runs = []
    for setting in sorted(os.listdir(sim_root)):
        sdir = os.path.join(sim_root, setting)
        reps = [r for r in glob.glob(os.path.join(sdir, "rep*")) if os.path.isdir(r)]
        for rep_dir in sorted(reps, key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
            runs.append((setting, os.path.basename(rep_dir), rep_dir))
    return runs


def _unit_subdirs(method, rep_dir):
    """[None] for a method run once on all samples, else its sampleN folders."""
    cfg = METHOD_CONFIG[method]
    bases = ma.sample_basenames(rep_dir)
    if not cfg["per_sample"] or len(bases) == 1:
        return [None]
    res = os.path.join(rep_dir, cfg["dirname"], cfg["results_name"])
    subs = sorted(d for d in (os.listdir(res) if os.path.isdir(res) else []) if re.fullmatch(r"sample\d+", d))
    return subs or [f"sample{k}" for k in range(len(bases))]


def evaluate_dataset(method, rep_dir):
    """All metrics of one method on one dataset, as one row."""
    cfg = METHOD_CONFIG[method]
    row = {"method": method}
    subs = [None]  # kept if the ground truth is missing; only the runtime is then reported
    try:
        subs = _unit_subdirs(method, rep_dir)
        units = [evaluate_unit(method, rep_dir, s) for s in subs]
        row.update(summarize(units, cfg["per_sample"], has_doublet_model=cfg["adapter"] != "lophy"))
        row["_units"] = units          # input of add_pair_metrics, removed in run()
        row["status"] = "ok"
    except FileNotFoundError as e:
        row.update(status="missing", error=str(e)[:200])
    except Exception as e:  # a broken run does not stop the sweep
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        traceback.print_exc(file=sys.stderr)
    # runtime: sum over the per-sample jobs; memory: maximum
    infos = [runtime_info(method, rep_dir, s) for s in subs]
    walls = [i.get("wall_seconds") for i in infos if i.get("wall_seconds") is not None]
    rss = [i.get("max_rss_gb") for i in infos if i.get("max_rss_gb") is not None]
    row["wall_seconds"] = float(np.sum(walls)) if walls else np.nan
    row["max_rss_gb"] = float(np.nanmax(rss)) if rss else np.nan
    first = infos[0]
    for key in ("cpu_seconds", "n_chains", "mcmc_steps_per_chain", "steps_per_second", "lophy_iters",
                "cache_gb", "lambda_rounding_precision", "runtime_source"):
        row[key] = first.get(key)
    return row


def run(sim_root, methods, rep_dirs=None):
    rows = []
    runs = [(os.path.basename(os.path.dirname(r)), os.path.basename(r), r) for r in rep_dirs] if rep_dirs \
        else discover_runs(sim_root)
    for setting, rep, rep_dir in runs:
        axis, level = parse_setting(setting)
        try:
            n_samples = len(ma.sample_basenames(rep_dir))
        except FileNotFoundError:  # no ground truth: all methods are reported as missing
            n_samples = np.nan
        rep_rows = []
        for method in methods:
            row = dict(setting=setting, axis=axis, level=level, rep=rep, n_samples=n_samples)
            row.update(evaluate_dataset(method, rep_dir))
            rep_rows.append(row)
            print(f"{setting}/{rep} {method}: {row['status']}", flush=True)
        add_pair_metrics(rep_rows)
        for row in rep_rows:
            row.pop("_units", None)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim_root", default="simulated_data")
    parser.add_argument("--rep_dir", nargs="*", help="score only these rep folders")
    parser.add_argument("--methods", nargs="+", default=["scati", "scati_noann", "compass_snp", "lophy"],
                        choices=list(METHOD_CONFIG))
    parser.add_argument("--out", default="simulation_metrics.csv")
    args = parser.parse_args()
    df = run(args.sim_root, args.methods, args.rep_dir)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
