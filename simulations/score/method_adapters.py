"""Converts the results of SCATI, COMPASS and LoPhy on a simulated dataset into
per-cell copy-number and VAF matrices and plots them against the ground truth.
For one method's results folder it writes region_comparison.png (observed
region counts, predicted CN, true CN), vaf_comparison.png (same for VAF) and
predicted_tree.png (the method's own tree, drawn with `dot`).

Layout: <rep_dir>/<Method>/<results_name>/ holds a method's results. The
simulated reads and the ground truth are always read from <rep_dir>/SCATI/
(data_*.csv, data_strand_genotypes.json, data_*_cellAssignmentsTRUE.tsv).

Samples: a multi-sample rep has files data_1_*, data_2_*, ... (1-based). A run
without --sample (SCATI, LoPhy) covers all samples, with cells concatenated in
sample order. --sample sampleN (0-based folder, one run per sample) uses data_{N+1}_*.

CLI:
    python score/method_adapters.py --method scati --rep_dir <rep_dir>
    python score/method_adapters.py --method compass --method_dir COMPASS_snp --rep_dir <rep_dir> --sample sample0
"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from genotype_comparison import (
    cn_from_strands, cn_from_strands_doublet, true_vaf_from_strands, true_vaf_doublet,
    assign_cells, order_cells)

METHOD_DIRNAMES = {
    "scati": "SCATI",
    "compass": "COMPASS",
    "compass_snp": "COMPASS_snp",       # COMPASS run with germline variants flagged
    "lophy": "LoPhy",
}
FIGSIZE = (16, 10)
_CMAP = LinearSegmentedColormap.from_list("obs_cn", ["#021a24", "#0d4f6c", "#ffffff"])
_BG = "#0e0e12"
# VAF colors: blue at 0 (hom-ref), white at 0.5 (het), red at 1 (hom-alt)
_VAF_CMAP = plt.cm.RdBu_r.copy()
_VAF_CMAP.set_bad(_BG)
_TITLE_COLOR = "#e8e0d0"
_LABEL_COLOR = "#c8b89a"
_TICK_COLOR = "#9aabbc"
_SPINE_COLOR = "#2a3a4a"
_GRID_COLOR = "#3a5a7a"
_BOUNDARY_COLOR = "#ff9f4a"
_DOUBLET_SEP_COLOR = "#4ade80"
_FONT = "monospace"


# ── Ground truth (always from <rep_dir>/SCATI/) ─────────────────────────────

def sample_basenames(rep_dir, sample_subdir=None):
    """File basenames in <rep_dir>/SCATI/ covered by this comparison, in cell
    order: ["data"] (single sample), ["data_1", ..., "data_S"] (all samples)
    or ["data_{N+1}"] (--sample sampleN)."""
    scati_dir = os.path.join(rep_dir, "SCATI")
    if os.path.isfile(os.path.join(scati_dir, "data_variants.csv")) \
            or os.path.isfile(os.path.join(scati_dir, "data_cellAssignmentsTRUE.tsv")):
        return ["data"]
    n_samples = max(sum(1 for f in os.listdir(scati_dir) if re.fullmatch(r"data_\d+_variants\.csv", f)),
                    sum(1 for f in os.listdir(scati_dir) if re.fullmatch(r"data_\d+_cellAssignmentsTRUE\.tsv", f)))
    if n_samples == 0:
        raise FileNotFoundError(f"No data_variants.csv, data_<k>_variants.csv or cellAssignmentsTRUE files in {scati_dir}")
    if sample_subdir:
        m = re.fullmatch(r"sample(\d+)", sample_subdir)
        if not m:
            raise ValueError(f"--sample must look like sample<N>, got {sample_subdir!r}")
        return [f"data_{int(m.group(1)) + 1}"]
    return [f"data_{k}" for k in range(1, n_samples + 1)]


def load_ground_truth(rep_dir, bases):
    """Returns (strand_data, n_regions, n_loci) from data_strand_genotypes.json.
    The panel size is read from the root node's strands."""
    scati_dir = os.path.join(rep_dir, "SCATI")
    with open(os.path.join(scati_dir, "data_strand_genotypes.json")) as f:
        strand_data = json.load(f)
    root = strand_data["0"]
    n_regions = len(root["strands_regions"][0])
    n_loci = len(root["strands_loci_ref"][0])
    return strand_data, n_regions, n_loci


def _load_ground_truth_cell_assignments(rep_dir, bases):
    """True cell assignments, samples concatenated. "cell" is renumbered to the
    row in the concatenation (each file numbers its cells from 0)."""
    parts = [pd.read_csv(os.path.join(rep_dir, "SCATI", b + "_cellAssignmentsTRUE.tsv"), sep="\t")
             for b in bases]
    ca = pd.concat(parts, ignore_index=True)
    ca["cell"] = np.arange(len(ca))
    return ca


def load_ground_truth_root_mask(rep_dir, bases):
    """Boolean mask over cells, True where the true node is the root (node 0).
    These cells are the baseline for normalizing region counts."""
    return (_load_ground_truth_cell_assignments(rep_dir, bases)["node"] == 0).to_numpy()


def load_observed(rep_dir, bases):
    """Observed reads, samples concatenated along cells. Returns regions_df
    (n_regions x n_cells), ref and alt (n_loci x n_cells) and the variant
    metadata columns of the first sample."""
    scati_dir = os.path.join(rep_dir, "SCATI")
    regions = [pd.read_csv(os.path.join(scati_dir, b + "_regions.csv"), index_col=0, header=None)
               for b in bases]
    regions_df = pd.concat(regions, axis=1, ignore_index=True)
    refs, alts, meta = [], [], None
    for b in bases:
        variants_df = pd.read_csv(os.path.join(scati_dir, b + "_variants.csv"), dtype=str)
        meta_cols = [c for c in ["CHR", "POS", "REF", "ALT", "REGION", "NAME", "FREQ"] if c in variants_df.columns]
        cell_cols = [c for c in variants_df.columns if c not in meta_cols]
        split = variants_df[cell_cols].stack().str.split(":", n=2, expand=True)
        refs.append(split[0].astype(int).unstack()[cell_cols].to_numpy())
        alts.append(split[1].astype(int).unstack()[cell_cols].to_numpy())
        if meta is None:
            meta = variants_df[meta_cols]
    return regions_df, np.concatenate(refs, axis=1), np.concatenate(alts, axis=1), meta


def ground_truth_matrix_per_cell(rep_dir, bases, strand_data_gt, n_cols, kind):
    """True CN or VAF per cell, shape (n_cells, n_cols), in original cell order
    (index with a method's cell_order to match its rows). Each cell uses its
    true node or doublet pair, since predicted node indices do not match the
    true tree's. kind is "cn" or "vaf"."""
    ca = _load_ground_truth_cell_assignments(rep_dir, bases)
    n_cells = len(ca)
    out = np.zeros((n_cells, n_cols))
    for node, group in ca.groupby("node"):
        singlets = group[~group["is_doublet"]]
        if len(singlets):
            row = (cn_from_strands(strand_data_gt[str(node)]["strands_regions"], n_cols) if kind == "cn"
                   else true_vaf_from_strands(strand_data_gt, node, n_cols))
            out[singlets["cell"].to_numpy()] = row
    doublets = ca[ca["is_doublet"]]
    for (n1, n2), group in doublets.groupby(["node", "node2"]):
        row = (cn_from_strands_doublet(strand_data_gt, n1, n2, n_cols) if kind == "cn"
               else true_vaf_doublet(strand_data_gt, n1, n2, n_cols))
        out[group["cell"].to_numpy()] = row
    return out


def expand_from_node_table(node_table, group_boundaries, n_cols):
    """Like matrix_from_strand_data, for a (n_nodes, n_cols) table with one row
    per node. A doublet row is the sum of its two nodes' rows."""
    out = np.zeros((group_boundaries[-1][2] if group_boundaries else 0, n_cols))
    for label, start, end, is_doublet, key in group_boundaries:
        if is_doublet:
            n1, n2 = key
            row = node_table[n1] + node_table[n2]
        else:
            row = node_table[key]
        out[start:end] = np.tile(row, (end - start, 1))
    return out


def matrix_from_strand_data(strand_data, group_boundaries, n_cols, kind):
    """Matrix (n_cells_ordered, n_cols) of CN or VAF (kind "cn" or "vaf") from
    per-strand genotypes, repeating each group's node or doublet row over its cells."""
    out = np.zeros((group_boundaries[-1][2] if group_boundaries else 0, n_cols))
    for label, start, end, is_doublet, key in group_boundaries:
        if is_doublet:
            n1, n2 = key
            row = (cn_from_strands_doublet(strand_data, n1, n2, n_cols) if kind == "cn"
                   else true_vaf_doublet(strand_data, n1, n2, n_cols))
        else:
            row = (cn_from_strands(strand_data[str(key)]["strands_regions"], n_cols) if kind == "cn"
                   else true_vaf_from_strands(strand_data, key, n_cols))
        out[start:end] = np.tile(row, (end - start, 1))
    return out


# ── Generic N-series comparison plot ────────────────────────────────────────

def plot_matrix_comparison(series, group_boundaries, col_boundaries_labels, title, out_path,
                            value_label, is_cn):
    """Draws the matrices in series side by side with one shared colorbar.
    series is a list of (panel_title, matrix), each (n_cells_ordered, n_cols)
    and row-ordered by group_boundaries."""
    n_panels = len(series)
    width_ratios = [1] * n_panels + [0.045]
    fig, axes = plt.subplots(1, n_panels + 1, figsize=FIGSIZE, facecolor=_BG,
                              gridspec_kw={"width_ratios": width_ratios, "wspace": 0.12})
    ax_cbar = axes[-1]
    axes = axes[:-1]

    if is_cn:
        cmap = _CMAP
        # vmax comes from the non-observed series only, because noisy observed
        # counts have a heavy right tail. All panels share it.
        is_observed = [label.lower().startswith("observed") for label, _ in series]
        shared_vals = np.concatenate([m[np.isfinite(m)].ravel()
                                       for (label, m), obs in zip(series, is_observed) if not obs and m.size])
        vmin, vmax = 0.0, (max(1.0, np.nanpercentile(shared_vals, 99)) if shared_vals.size else 1.0)
    else:
        cmap = _VAF_CMAP
        vmin, vmax = 0.0, 1.0

    im = None
    for ax, (panel_title, m) in zip(axes, series):
        im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(panel_title, color=_TITLE_COLOR, fontsize=13, fontweight="600", fontfamily=_FONT)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(value_label, color=_LABEL_COLOR, fontsize=9, fontfamily=_FONT, labelpad=8)
    if is_cn:
        cbar.set_ticks(range(int(vmax) + 1))
    cbar.ax.yaxis.set_tick_params(color=_LABEL_COLOR, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_LABEL_COLOR, fontfamily=_FONT)
    cbar.outline.set_edgecolor(_SPINE_COLOR)

    col_boundaries, col_ticks = col_boundaries_labels
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.set_xticks([t for t, _ in col_ticks])
        ax.set_xticklabels([c for _, c in col_ticks], fontsize=7, color=_TICK_COLOR,
                           fontfamily=_FONT, rotation=45, ha="right")
        ax.tick_params(axis="x", length=3, color=_TICK_COLOR, pad=2)
        for spine in ax.spines.values():
            spine.set_edgecolor(_SPINE_COLOR)
            spine.set_linewidth(0.5)
        for b in col_boundaries[1:]:
            ax.axvline(b - 0.5, color=_GRID_COLOR, lw=0.5, alpha=0.6)

    axes[0].set_yticks([(start + end) / 2 for _, start, end, _, _ in group_boundaries])
    axes[0].set_yticklabels([label for label, *_ in group_boundaries], fontsize=8,
                             color=_LABEL_COLOR, fontfamily=_FONT)
    axes[0].tick_params(axis="y", length=0, pad=4)
    axes[0].set_ylabel("cells (predicted node order, doublets last)", color=_TICK_COLOR,
                        fontsize=9, fontfamily=_FONT, labelpad=6)
    for ax in axes[1:]:
        ax.set_yticks([])

    first_doublet = next((start for _, start, _, is_d, _ in group_boundaries if is_d), None)
    for _, start, _, _, _ in group_boundaries[1:]:
        is_sep = start == first_doublet
        color = _DOUBLET_SEP_COLOR if is_sep else _BOUNDARY_COLOR
        lw = 1.3 if is_sep else 0.6
        for ax in axes:
            ax.axhline(start - 0.5, color=color, lw=lw, alpha=0.8)

    fig.suptitle(title, color=_TITLE_COLOR, fontsize=11, fontfamily=_FONT)
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.16, top=0.90)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _chrom_boundaries(labels):
    """Column indices where the chromosome changes, and (index, chromosome) x-ticks.
    The chromosome is the part of each label before the first underscore."""
    chroms = [str(label).split("_")[0] for label in labels]
    boundaries, ticks, prev = [], [], None
    for i, c in enumerate(chroms):
        if c != prev:
            boundaries.append(i)
            ticks.append((i, c))
            prev = c
    return boundaries, ticks


def render_tree(dot_path, out_png):
    """Renders a .gv file to PNG with `dot`. Returns the path, or None if dot or the file is missing or dot fails."""
    dot = shutil.which("dot")
    if not dot or not os.path.isfile(dot_path):
        return None
    try:
        # COMPASS writes width=-nan height=-nan when no singlet is attached to any node.
        # That is invalid DOT, so a copy with a fixed size is rendered instead.
        with open(dot_path) as f:
            text = f.read()
        sanitized = re.sub(r"\b(width|height)\s*=\s*-?nan\b", r"\1=0.3", text)
        src = dot_path
        if sanitized != text:
            src = out_png + ".sanitized.gv"
            with open(src, "w") as f:
                f.write(sanitized)
        try:
            subprocess.run([dot, "-Tpng", src, "-o", out_png], check=True, capture_output=True)
        finally:
            if src != dot_path and os.path.isfile(src):
                os.remove(src)
        return out_png
    except (subprocess.CalledProcessError, OSError) as e:
        stderr = (getattr(e, "stderr", b"") or b"").decode(errors="replace").strip()
        print(f"  (skipping {out_png}: {e}" + (f" -- dot said: {stderr[:500]}" if stderr else "") + ")")
        return None


_QUOTED_EDGE_RE = re.compile(r'^\s*".*?"\s*->\s*".*?";?\s*$')


def strip_cell_attachment_edges(dot_path, filtered_dot_path):
    """Writes a copy of a LoPhy .gv file without the per-cell edges, so only the
    tree topology and node labels are drawn. Cell edges are the quoted
    "Clone_K" -> "cell" lines; topology edges are unquoted."""
    with open(dot_path) as f:
        text = f.read()
    # Drop LoPhy's `graph [ ... ];` block of result matrices. Its strings exceed
    # the 16384 character limit of dot, which then refuses the file.
    text = re.sub(r"^[ 	]*graph \[.*?^\];", "", text, count=1, flags=re.DOTALL | re.MULTILINE)
    kept = [line for line in text.splitlines(keepends=True) if not _QUOTED_EDGE_RE.match(line)]
    with open(filtered_dot_path, "w") as f:
        f.writelines(kept)
    return filtered_dot_path


# ── Per-method adapters ──────────────────────────────────────────────────────
# Each adapter takes (results_dir, input_dir, n_regions, n_loci) and returns
# (cn_pred, vaf_pred, cell_order, group_boundaries, tree_path). cn_pred and
# vaf_pred are (n_cells_ordered, n_regions) and (n_cells_ordered, n_loci);
# cn_pred may be None. input_dir is the method's input folder, not results.

def scati_predicted(results_dir, input_dir, n_regions, n_loci):
    """SCATI writes best_tree_* files directly into results_dir."""
    out_dir = results_dir
    with open(os.path.join(out_dir, "best_tree_strand_genotypes.json")) as f:
        strand_data = json.load(f)
    n_nodes = len(strand_data)
    cell_assignments = pd.read_csv(os.path.join(out_dir, "best_tree_cellAssignments.tsv"), sep="\t")
    probs_path = os.path.join(out_dir, "best_tree_cellAssignmentProbs.tsv")
    use_doublets = os.path.exists(probs_path)
    probs_df = pd.read_csv(probs_path, sep="\t") if use_doublets else None
    cell_to_node, cell_to_doublet = assign_cells(cell_assignments, probs_df, n_nodes, use_doublets)
    cell_order, group_boundaries = order_cells(len(cell_assignments), n_nodes, cell_to_node, cell_to_doublet)
    cn_pred = matrix_from_strand_data(strand_data, group_boundaries, n_regions, "cn")
    vaf_pred = matrix_from_strand_data(strand_data, group_boundaries, n_loci, "vaf")
    return cn_pred, vaf_pred, cell_order, group_boundaries, os.path.join(out_dir, "best_tree.gv")


def load_compass_cn_table(results_dir, input_dir, n_nodes):
    """COMPASS's per-node copy-number table, one row per node.

    COMPASS writes no table when too few cells are attached to the root to
    estimate region weights, as in many per-sample runs. Then every node gets the
    root copy number (2, or 1 on male chrX) taken from the ground-truth root
    node; without ground truth a male simulation is assumed. A table whose row
    count differs from n_nodes is treated as missing and a warning is printed."""
    cn_path = os.path.join(results_dir, "data_nodes_copynumbers.tsv")
    if os.path.exists(cn_path):
        cn_df = pd.read_csv(cn_path, sep="\t", index_col=0)
        if len(cn_df) == n_nodes:
            return cn_df
        print(f"WARNING: {cn_path} has {len(cn_df)} nodes but the genotype table has {n_nodes}: stale file "
              f"from an earlier run, ignored (root copy number used for every node)", file=sys.stderr)
    root_cn = None
    try:
        try:
            compass_root = _find_method_root(input_dir, "COMPASS")
        except ValueError:
            compass_root = _find_method_root(input_dir, "COMPASS_snp")
        scati_dir = os.path.join(os.path.dirname(compass_root), "SCATI")
        with open(os.path.join(scati_dir, "data_strand_genotypes.json")) as f:
            strand_gt = json.load(f)
        n_reg = len(strand_gt["0"]["strands_regions"][0])
        root_cn = np.asarray(cn_from_strands(strand_gt["0"]["strands_regions"], n_reg), dtype=float)
    except (OSError, ValueError, KeyError, IndexError):
        root_cn = None
    if root_cn is None:  # no ground truth: assume male (chrX copy number 1)
        regions_csv = os.path.join(input_dir, "data_regions.csv")
        chroms = pd.read_csv(regions_csv, header=None, usecols=[0])[0].astype(str).str.split("_").str[0]
        root_cn = np.where(chroms.to_numpy() == "X", 1.0, 2.0)
    return pd.DataFrame(np.tile(root_cn, (n_nodes, 1)))


def compass_predicted(results_dir, input_dir, n_regions, n_loci):
    """COMPASS writes data_* files into results_dir. Its per-node CN and genotype
    tables are used directly. Doublet pairs are not available, so cells are
    ordered by hard node assignment. Genotypes {0,1,2} are mapped to VAF
    {0,0.5,1}, an approximation."""
    base = os.path.join(results_dir, "data")
    geno_df = pd.read_csv(base + "_nodes_genotypes.tsv", sep="\t", index_col=0)
    n_nodes = len(geno_df)
    cn_df = load_compass_cn_table(results_dir, input_dir, n_nodes)
    cell_assignments = pd.read_csv(base + "_cellAssignments.tsv", sep="\t")
    cell_to_node, cell_to_doublet = assign_cells(cell_assignments, None, n_nodes, False)
    cell_order, group_boundaries = order_cells(len(cell_assignments), n_nodes, cell_to_node, cell_to_doublet)
    cn_pred = expand_from_node_table(cn_df.to_numpy(dtype=float), group_boundaries, n_regions)
    vaf_pred = expand_from_node_table(geno_df.to_numpy(dtype=float) / 2.0, group_boundaries, n_loci)
    return cn_pred, vaf_pred, cell_order, group_boundaries, base + "_tree.gv"


def _parse_gv_attr(text, key):
    """Parses the list in a `key="[...]"` graph attribute of LoPhy's .gv text."""
    m = re.search(rf'{re.escape(key)}="(\[.*?\])"', text, re.DOTALL)
    if not m:
        raise ValueError(f"Attribute {key!r} not found in LoPhy .gv output")
    return ast.literal_eval(m.group(1))


def lophy_predicted(results_dir, input_dir, n_regions, n_loci):
    """LoPhy writes one data_0.gv holding the tree, per-clone copy numbers and cell
    assignments. Total CN is per region. VAF per clone is mutant CN divided by
    the total CN of the locus's region (meta.csv REGION_INDEX), 0 if that is 0."""
    gv_path = os.path.join(results_dir, "data_0.gv")
    with open(gv_path) as f:
        text = f.read()

    # (n_clones, n_regions), columns in the order of data_regions.csv
    total_cn = np.array(_parse_gv_attr(text, "total_copy_numbers"), dtype=float)
    # (n_clones, n_loci), alt allele copy number at each locus
    mutant_cn = np.array(_parse_gv_attr(text, "mutant_copy_numbers"), dtype=float)
    cell_assignments = np.array(_parse_gv_attr(text, "cell_assignments"), dtype=int)

    meta_df = pd.read_csv(os.path.join(input_dir, "meta.csv"))
    region_index = meta_df["REGION_INDEX"].to_numpy()[:n_loci]
    with np.errstate(invalid="ignore", divide="ignore"):
        region_total_per_locus = total_cn[:, region_index]  # (n_kept_clones, n_loci)
        vaf_per_clone = np.where(region_total_per_locus > 0,
                                  mutant_cn / np.where(region_total_per_locus == 0, 1, region_total_per_locus),
                                  0.0)

    n_clones_kept = total_cn.shape[0]
    cell_assignments_df = pd.DataFrame({"node": cell_assignments})
    cell_to_node, cell_to_doublet = assign_cells(cell_assignments_df, None, n_clones_kept, False)
    cell_order, group_boundaries = order_cells(len(cell_assignments), n_clones_kept, cell_to_node, cell_to_doublet)
    cn_pred = expand_from_node_table(total_cn, group_boundaries, n_regions)
    vaf_pred = expand_from_node_table(vaf_per_clone, group_boundaries, n_loci)

    filtered_gv = os.path.join(results_dir, "data_0_tree_only.gv")
    strip_cell_attachment_edges(gv_path, filtered_gv)
    return cn_pred, vaf_pred, cell_order, group_boundaries, filtered_gv


def _find_method_root(a_dir, method_dirname):
    """Walks up from a_dir (<Method> or <Method>/sample{N}) to the <Method> folder."""
    d = a_dir
    while os.path.basename(d) != method_dirname:
        parent = os.path.dirname(d)
        if parent == d:
            raise ValueError(f"Could not locate a {method_dirname!r} folder above {a_dir!r}")
        d = parent
    return d


ADAPTERS = {
    "scati": scati_predicted,
    "compass": compass_predicted,
    "compass_snp": compass_predicted,
    "lophy": lophy_predicted,
}


def generate_comparison_plots(method, rep_dir, sample_subdir=None, results_name="results", method_dirname=None):
    """Writes the comparison plots into the method's results folder.
    method is a key of ADAPTERS. sample_subdir (e.g. "sample0") is for per-sample
    runs. results_name is the results folder, e.g. "results_noann". method_dirname
    overrides the folder in rep_dir, e.g. "COMPASS_snp" with method="compass"."""
    dirname = method_dirname or METHOD_DIRNAMES[method]
    method_dir = os.path.join(rep_dir, dirname)
    results_dir = os.path.join(method_dir, results_name, sample_subdir) if sample_subdir \
        else os.path.join(method_dir, results_name)
    input_dir = os.path.join(method_dir, sample_subdir) if sample_subdir else method_dir

    bases = sample_basenames(rep_dir, sample_subdir)
    strand_data_gt, n_regions, n_loci = load_ground_truth(rep_dir, bases)

    if method not in ADAPTERS:
        raise NotImplementedError(f"No adapter registered yet for method={method!r}")
    cn_pred, vaf_pred, cell_order, group_boundaries, tree_gv = ADAPTERS[method](
        results_dir, input_dir, n_regions, n_loci)
    regions_df, ref, alt, variants_meta = load_observed(rep_dir, bases)
    if len(cell_order) != regions_df.shape[1]:
        raise ValueError(f"{method}: {len(cell_order)} cells in the results but {regions_df.shape[1]} "
                         f"in the simulated data for {bases}")

    raw_counts = regions_df.values.astype(float)
    # Scale counts so the true root cells' per-region mean is CN 2 (all cells if there are none)
    root_mask = load_ground_truth_root_mask(rep_dir, bases)
    baseline_counts = raw_counts[:, root_mask] if root_mask.any() else raw_counts
    per_region_baseline = baseline_counts.mean(axis=1)
    per_region_baseline[per_region_baseline == 0] = np.nan
    region_counts = (raw_counts / per_region_baseline[:, None] * 2.0)[:, cell_order].T  # (n_cells_ordered, n_regions)

    with np.errstate(invalid="ignore", divide="ignore"):
        total = ref + alt
        obs_vaf = np.where(total >= 5, alt / np.where(total == 0, 1, total), np.nan)  # NaN below depth 5
    obs_vaf = obs_vaf[:, cell_order].T  # (n_cells_ordered, n_loci)

    os.makedirs(results_dir, exist_ok=True)

    if cn_pred is not None:
        cn_true = ground_truth_matrix_per_cell(rep_dir, bases, strand_data_gt, n_regions, "cn")[cell_order]
        col_boundaries_labels = _chrom_boundaries(regions_df.index)
        plot_matrix_comparison(
            [("Observed region counts", region_counts),
             ("Predicted copy number", cn_pred),
             ("Ground-truth copy number", cn_true)],
            group_boundaries, col_boundaries_labels,
            f"{dirname}: region/copy-number comparison",
            os.path.join(results_dir, "region_comparison.png"),
            value_label="Copy number", is_cn=True)

    vaf_true = ground_truth_matrix_per_cell(rep_dir, bases, strand_data_gt, n_loci, "vaf")[cell_order]
    locus_labels = list(variants_meta.get("NAME", [str(i) for i in range(n_loci)]))
    col_boundaries_labels = (list(range(0, n_loci + 1, max(1, n_loci // 20))),
                             [(i, locus_labels[i]) for i in range(0, n_loci, max(1, n_loci // 20))])
    plot_matrix_comparison(
        [("Observed VAF", obs_vaf), ("Predicted VAF", vaf_pred), ("Ground-truth VAF", vaf_true)],
        group_boundaries, col_boundaries_labels,
        f"{dirname}: VAF comparison",
        os.path.join(results_dir, "vaf_comparison.png"),
        value_label="VAF", is_cn=False)

    render_tree(tree_gv, os.path.join(results_dir, "predicted_tree.png"))


def generate_simulation_check(rep_dir):
    """Compares observed simulated reads with the true CN and VAF for all cells of
    rep_dir, ordered by true node with doublets last. Writes to <rep_dir>/SCATI/:
      simulation_check_region.png  region counts vs true CN
      simulation_check_vaf.png     VAF (depth >= 5) vs true VAF
      simulation_check.json        Pearson r of each pair and cell counts per sample
    Counts are divided by the true root cells' per-region mean and multiplied by
    the root copy number. Returns (r_cn, r_vaf), computed on singlets."""
    scati_dir = os.path.join(rep_dir, "SCATI")
    bases = sample_basenames(rep_dir)
    strand_data_gt, n_regions, n_loci = load_ground_truth(rep_dir, bases)
    regions_df, ref, alt, variants_meta = load_observed(rep_dir, bases)
    ca = _load_ground_truth_cell_assignments(rep_dir, bases)

    cell_order, group_boundaries = [], []
    singlets = ca[~ca["is_doublet"]]
    for node in sorted(singlets["node"].unique()):
        start = len(cell_order)
        cell_order.extend(singlets.loc[singlets["node"] == node, "cell"].tolist())
        group_boundaries.append((f"node {node}", start, len(cell_order), False, int(node)))
    for (n1, n2), group in ca[ca["is_doublet"]].groupby(["node", "node2"]):
        start = len(cell_order)
        cell_order.extend(group["cell"].tolist())
        group_boundaries.append((f"doublet {n1}+{n2}", start, len(cell_order), True, (int(n1), int(n2))))
    singlet_rows = np.array([i for i, c in enumerate(cell_order) if not ca["is_doublet"].iloc[c]])

    root_cn = cn_from_strands(strand_data_gt["0"]["strands_regions"], n_regions)
    raw_counts = regions_df.values.astype(float)
    root_cells = ((ca["node"] == 0) & ~ca["is_doublet"]).to_numpy()
    baseline = (raw_counts[:, root_cells] if root_cells.any() else raw_counts).mean(axis=1)
    baseline[baseline == 0] = np.nan
    obs_counts = (raw_counts / baseline[:, None] * root_cn[:, None])[:, cell_order].T
    cn_true = ground_truth_matrix_per_cell(rep_dir, bases, strand_data_gt, n_regions, "cn")[cell_order]

    with np.errstate(invalid="ignore", divide="ignore"):
        total = ref + alt
        obs_vaf = np.where(total >= 5, alt / np.where(total == 0, 1, total), np.nan)[:, cell_order].T
    vaf_true = ground_truth_matrix_per_cell(rep_dir, bases, strand_data_gt, n_loci, "vaf")[cell_order]

    def pearson(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else float("nan")

    r_cn = pearson(obs_counts[singlet_rows].ravel(), cn_true[singlet_rows].ravel())
    r_vaf = pearson(obs_vaf[singlet_rows].ravel(), vaf_true[singlet_rows].ravel())

    plot_matrix_comparison(
        [("Observed region counts (root-normalized)", obs_counts), ("Ground-truth copy number", cn_true)],
        group_boundaries, _chrom_boundaries(regions_df.index),
        f"simulation check: region counts vs copy number (singlet cells, Pearson r = {r_cn:.3f})",
        os.path.join(scati_dir, "simulation_check_region.png"), value_label="Copy number", is_cn=True)
    locus_labels = list(variants_meta.get("NAME", [str(i) for i in range(n_loci)]))
    step = max(1, n_loci // 20)
    plot_matrix_comparison(
        [("Observed VAF", obs_vaf), ("Ground-truth VAF", vaf_true)],
        group_boundaries,
        (list(range(0, n_loci + 1, step)), [(i, locus_labels[i]) for i in range(0, n_loci, step)]),
        f"simulation check: VAF (singlet cells, depth >= 5, Pearson r = {r_vaf:.3f})",
        os.path.join(scati_dir, "simulation_check_vaf.png"), value_label="VAF", is_cn=False)

    sample_sizes = [len(pd.read_csv(os.path.join(scati_dir, b + "_cellAssignmentsTRUE.tsv"), sep="\t")) for b in bases]
    starts = np.concatenate([[0], np.cumsum(sample_sizes)])
    per_sample = [{"cells": int(sample_sizes[s]),
                   "fraction_true_root_singlets": float(root_cells[starts[s]:starts[s + 1]].mean()),
                   "n_true_nodes_present": int(ca["node"].iloc[starts[s]:starts[s + 1]].nunique())}
                  for s in range(len(bases))]
    with open(os.path.join(scati_dir, "simulation_check.json"), "w") as f:
        json.dump({"region_counts_vs_true_CN_pearson_r": r_cn, "vaf_vs_true_VAF_pearson_r": r_vaf,
                   "n_regions": int(n_regions), "n_loci": int(n_loci),
                   "n_cells": int(len(ca)), "n_doublet_cells": int(ca["is_doublet"].sum()),
                   "per_sample": per_sample}, f, indent=2)
    return r_cn, r_vaf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation_check", action="store_true",
                        help="Only write SCATI/simulation_check_*.png and .json for --rep_dir; --method is not needed.")
    parser.add_argument("--method", required=False, choices=list(METHOD_DIRNAMES.keys()))
    parser.add_argument("--rep_dir", required=True)
    parser.add_argument("--sample", default=None, help='sample folder of a per-sample run, e.g. "sample0"')
    parser.add_argument("--results_name", default="results",
                        help='results folder inside the method folder, e.g. "results_noann"')
    parser.add_argument("--method_dir", default=None,
                        help='method folder in --rep_dir if not the default, e.g. "COMPASS_snp" with --method compass')
    args = parser.parse_args()
    if args.simulation_check:
        generate_simulation_check(args.rep_dir)
    else:
        if not args.method:
            parser.error("--method is required unless --simulation_check is given")
        generate_comparison_plots(args.method, args.rep_dir, args.sample, args.results_name, args.method_dir)
