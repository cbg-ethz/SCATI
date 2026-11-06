"""Trees, observed vs predicted region counts and observed vs predicted VAF of SCATI, COMPASS and LoPhy on one sample, one
row per method, and a simple distance between prediction and data for each method. make_paper_figures.py uses it for all samples.

    python plot_methods.py --data_base AML/AML-23/SCATI/AML-23 --scati AML/AML-23/SCATI/results --out AML-23_SCATI.png
    python plot_methods.py --data_base ... --compass .../COMPASS/results --lophy .../LoPhy/results --lophy_meta .../LoPhy/meta.csv --out ...
Only the methods given are drawn. The trees are rendered with `dot` from graphviz, which must be on the PATH.

Inputs:
    --data_base  <base>_{variants,regions}_filtered.csv (or without _filtered): the files the methods were run on
    --scati      best_tree_{cellAssignments,cellAssignmentProbs}.tsv, best_tree_strand_genotypes.json, best_tree.gv
    --compass    data_{cellAssignments,nodes_genotypes,nodes_copynumbers}.tsv, data_tree.gv
    --lophy      data_0.gv (all matrices and the tree), and --lophy_meta (its meta.csv)
Outputs: --out (.png, plus the same name as .pdf), <out>_metrics.csv with the distances below, and the tree renderings in <out dir>/trees/.

Panels. Cells are ordered by the node they are attached to (doublets last). The observed region counts of a method are divided by the mean count
of the cells that this method attaches to its root as singlets and multiplied by the root copy number of the region, so that a region with
the root copy number has the value 2 and the colours are comparable with the predicted copy numbers (same scale in all rows)

Observed VAF = alt / (ref + alt) in cells with at least --min_depth reads at the locus. Predicted VAF: SCATI: alt copies / all copies of the locus at the node
(strand genotypes); LoPhy: mutant copies / total copies of the region; COMPASS: its genotype state 0 / 1 / 2 as 0 / 0.5 / 1

Distances. They assume no read-count model. They are computed on the SHARED set of cells, the cells that none of the
methods calls a doublet (SCATI and COMPASS doublets are excluded, so every method is scored on the same cells). The VAF error is also given for
the trivial tree in which every cell is a root cell ("root only"):
    region error  mean |observed / expected - 1| over cells and regions: the relative deviation of the observed region count from its expectation.
                  expected = (reads of the cell / mean reads of the root cells) * rel / Lambda, in units of the mean count of the cells that the method
                  attaches to its root as singlets (the "root cells" below; if fewer than 10 cells are attached to the root, as often in COMPASS and
                  LoPhy, the reference node is the populated node closest to the root, with most cells among equals, and "root" means that node),
                  with rel = copy number of the cell's node / copy number of the root in that region and
                  Lambda = sum_r rho_r rel_r (rho_r: share of region r in the root cells), i.e. the read share of each region under the copy numbers of
                  the node, scaled by the depth of the cell. The unit cancels in the ratio, so the error does not depend on the depth of the root cells.
                  Regions with copy number 0 in the node of the cell are not scored (the ratio is undefined there).
    VAF error     mean |observed VAF - predicted VAF| over cell x locus entries with enough reads.
A tree with more nodes has more freedom to fit the data and neither distance penalises that: read them together with the number of nodes.
"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulations", "score"))
from genotype_comparison import (assign_cells, cn_from_strands, cn_from_strands_doublet,  # noqa: E402
                                        true_vaf_doublet, true_vaf_from_strands)

META_COLS = ["CHR", "POS", "REF", "ALT", "REGION", "NAME", "FREQ"]

# Layout of the figure (inches unless noted); make_paper_figures.py changes some of these
TREE_DPI = 100          # resolution of the tree renderings (pixels per inch)
TREE_WRAP = 40          # event lines of the tree nodes are wrapped at this many characters
TREE_PEN = 4            # width of the coloured node borders
TREE_NODESEP = 0.35     # horizontal gap between neighbouring nodes
TREE_FONT = 24          # font size of the tree graphs (pt, before scaling into the figure)
TREE_SHORT_EVENTS = True   # shorter copy-number event names ("Loss chr7", "Gain chr11 (1-2)", "CNLOH chr6 (2)") instead of SCATI's own wording
TREE_VCENTER = True     # a tree is centred vertically in its row (False: top-aligned)
TREE_PACK = None        # a node with more than TREE_PACK event lines shows 2 (3 above twice that many) events per line (None: one event per line)
TREE_W_NATIVE = 11.0    # width of the tree column
RIGHT_NATIVE = 2.0      # right margin
ROW_H_NATIVE = 8.0      # height of a method row

THEMES = {
    "light": dict(bg="white", fg="#111111", fg2="#555555", grid="#999999", bound="#e8590c", dbl="#2b9348", bad="#d0d0d0",
                  cn=LinearSegmentedColormap.from_list("cn", ["#021a24", "#0d4f6c", "#ffffff"])),
    "dark": dict(bg="#0e0e12", fg="#e8e0d0", fg2="#9aabbc", grid="#3a5a7a", bound="#ff9f4a", dbl="#4ade80", bad="#33333a",
                 cn=LinearSegmentedColormap.from_list("cn", ["#021a24", "#0d4f6c", "#ffffff"])),
}


# ── data ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def load_data(base):
    vpath = base + "_variants_filtered.csv" if os.path.exists(base + "_variants_filtered.csv") else base + "_variants.csv"
    rpath = base + "_regions_filtered.csv" if os.path.exists(base + "_regions_filtered.csv") else base + "_regions.csv"
    var = pd.read_csv(vpath, dtype=str)
    meta = [c for c in META_COLS if c in var.columns]
    cells = [c for c in var.columns if c not in meta]
    parts = var[cells].stack().str.split(":", n=2, expand=True)
    ref = parts[0].astype(int).unstack()[cells].to_numpy()
    alt = parts[1].astype(int).unstack()[cells].to_numpy()
    with open(rpath) as f:
        has_header = f.readline().split(",")[0].strip() == ""          # a regions file with a header row starts with an empty cell
    reg = pd.read_csv(rpath, header=0 if has_header else None, index_col=0)
    counts = reg.values.astype(float)
    if counts.shape[1] != len(cells):
        sys.exit(f"{len(cells)} cells in the variants file but {counts.shape[1]} in the regions file")
    return dict(cells=cells, ref=ref, alt=alt, counts=counts, region_names=[str(r) for r in reg.index],
                loci_chrom=var["CHR"].astype(str).tolist(), loci_names=var["NAME"].tolist())


class Result:
    """What the figure and the distances need from one method (all per node, nodes numbered 0 = root)."""

    def __init__(self, name, node_cn, node_vaf, cell_node, cell_doublet, node_labels, tree_gv, doublet_cn=None, doublet_vaf=None, note=""):
        self.name, self.node_cn, self.node_vaf = name, node_cn, node_vaf
        self.cell_node, self.cell_doublet = np.asarray(cell_node, dtype=int), list(cell_doublet)   # None | (n1, n2) | "?"
        self.node_labels, self.tree_gv, self.note = node_labels, tree_gv, note
        self.doublet_cn, self.doublet_vaf = doublet_cn, doublet_vaf
        self.n_nodes = len(node_cn)
        self.label = name

    def singlet_mask(self):
        return np.array([d is None for d in self.cell_doublet])


def _read_tsv(path):
    """A tab-separated table; a stray carriage return inside a field (the cell name of the last row when the input had Windows line endings) is removed."""
    df = pd.read_csv(path, sep="\t", lineterminator="\n")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("\r", "", regex=False)
    df.columns = [str(c).replace("\r", "") for c in df.columns]
    return df


def _align(df, key, cells, what):
    if key in df.columns:
        df = df.copy()
        df[key] = df[key].astype(str)                      # cell names may be numbers (0, 1, 2, ...)
        cells = [str(c) for c in cells]
    if key in df.columns and set(df[key]) == set(cells):
        return df.set_index(key).reindex(cells).reset_index()
    print(f"  WARNING: {what}: cell names do not match the data, assuming the same cell order", file=sys.stderr)
    return df.reset_index(drop=True)


def load_scati(d, data):
    cells = data["cells"]
    with open(os.path.join(d, "best_tree_strand_genotypes.json")) as f:
        sd = json.load(f)
    n_nodes = len(sd)
    n_reg, n_loci = len(sd["0"]["strands_regions"][0]), len(sd["0"]["strands_loci_ref"][0])
    node_cn = np.array([cn_from_strands(sd[str(k)]["strands_regions"], n_reg) for k in range(n_nodes)])
    node_vaf = np.array([true_vaf_from_strands(sd, k, n_loci) for k in range(n_nodes)])
    ca = _align(_read_tsv(os.path.join(d, "best_tree_cellAssignments.tsv")), "cell", cells, "SCATI assignments")
    probs_path = os.path.join(d, "best_tree_cellAssignmentProbs.tsv")
    probs = _align(_read_tsv(probs_path), "cell", cells, "SCATI probabilities") if os.path.exists(probs_path) else None
    cell_node, cell_dbl = assign_cells(ca, probs, n_nodes, probs is not None)
    doublet = [cell_dbl.get(j) for j in range(len(cells))]
    return Result("SCATI", node_cn, node_vaf, [cell_node[j] for j in range(len(cells))], doublet, [f"node {k}" for k in range(n_nodes)],
                  os.path.join(d, "best_tree.gv"),
                  doublet_cn=lambda a, b: cn_from_strands_doublet(sd, a, b, n_reg), doublet_vaf=lambda a, b: true_vaf_doublet(sd, a, b, n_loci))


def load_compass(d, data, sex=None):
    cells = data["cells"]
    ca = _align(_read_tsv(os.path.join(d, "data_cellAssignments.tsv")), "cell", cells, "COMPASS assignments")
    geno = pd.read_csv(os.path.join(d, "data_nodes_genotypes.tsv"), sep="\t", index_col=0)
    n_nodes = len(geno)
    node_vaf = geno.to_numpy(dtype=float) / 2.0           # 0 / 1 / 2 -> 0 / 0.5 / 1 (no continuous VAF in COMPASS)
    n_reg = len(data["region_names"])
    cn_path, note = os.path.join(d, "data_nodes_copynumbers.tsv"), ""
    node_cn = None
    if os.path.exists(cn_path):
        cn = pd.read_csv(cn_path, sep="\t", index_col=0)
        if len(cn) == n_nodes and cn.shape[1] == n_reg:
            node_cn = cn.to_numpy(dtype=float)
        else:
            print(f"  WARNING: {cn_path} has {cn.shape} (nodes x regions), expected ({n_nodes}, {n_reg}): ignored", file=sys.stderr)
    if node_cn is None:
        # COMPASS skips its copy-number phase (and writes no table) when too few cells attach to its root; its tree has no CNAs then,
        # so every node gets the germline copy number
        germline = np.full(n_reg, 2.0)
        if sex == "male":                                       # no CNA inferred: the germline copy numbers of the sex, as COMPASS assumes with --sex
            germline[[r.split("_")[0] in ("X", "Y") for r in data["region_names"]]] = 1.0
        node_cn, note = np.tile(germline, (n_nodes, 1)), (note + ", " if note else "") + "no copy numbers"
    doublet = ["?" if str(v).lower() == "yes" else None for v in ca["doublet"]]
    return Result("COMPASS", node_cn, node_vaf, ca["node"].to_numpy(), doublet, [f"node {k}" for k in range(n_nodes)],
                  os.path.join(d, "data_tree.gv"), note=note)


def _gv_attr(text, key):
    m = re.search(rf'{re.escape(key)}="(\[.*?\])"', text, re.DOTALL)
    if not m:
        raise ValueError(f"attribute {key!r} not found in the LoPhy .gv file")
    return ast.literal_eval(m.group(1))


def lophy_dfs_order(text, ids):
    """LoPhy clone ids in depth-first (pre-order) order from the root 0; children by ascending id; clones the tree does not reach follow."""
    body = re.sub(r"^[ \t]*graph \[.*?^\];", "", text, count=1, flags=re.DOTALL | re.MULTILINE)
    children = {}
    for a, b in re.findall(r"^\s*Clone_(\d+) -> Clone_(\d+);", body, re.MULTILINE):
        children.setdefault(int(a), []).append(int(b))
    order, stack = [], [0]
    while stack:
        c = stack.pop()
        if c in order:
            continue
        order.append(c)
        stack.extend(sorted(children.get(c, []), reverse=True))
    return [c for c in order if c in ids] + [c for c in ids if c not in order]


def load_lophy(d, meta_csv, data, sex=None):
    cells = data["cells"]
    gv = os.path.join(d, "data_0.gv")
    with open(gv) as f:
        text = f.read()
    total = np.array(_gv_attr(text, "total_copy_numbers"), dtype=float)      # clones x regions
    mutant = np.array(_gv_attr(text, "mutant_copy_numbers"), dtype=float)    # clones x loci
    assign = np.array(_gv_attr(text, "cell_assignments"), dtype=int)
    names = _gv_attr(text, "cells")
    order = pd.Series(assign, index=names)
    if set(order.index) == set(cells):
        assign = order.reindex(cells).to_numpy()
    else:
        print("  WARNING: LoPhy cell names do not match the data, assuming the same cell order", file=sys.stderr)
    meta = pd.read_csv(meta_csv)
    region_index = meta["REGION_INDEX"].to_numpy()[:mutant.shape[1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = total[:, region_index]
        vaf = np.where(tot > 0, mutant / np.where(tot == 0, 1, tot), np.nan)
    # rows are the clones that are kept (those with a label), in ascending clone id
    ids = sorted({0} | {int(i) for i in re.findall(r"^\s*Clone_(\d+)\s*\[label", text, re.MULTILINE)})
    if len(ids) != len(total):
        print(f"  WARNING: {len(ids)} clones in the tree but {len(total)} rows in the matrices: clone names in the labels may be off", file=sys.stderr)
        ids = list(range(len(total)))
    # The clones are renumbered in depth-first order of the tree (the root is 0, then 1, 2, ... as the tree is walked down, children by ascending
    # LoPhy id): LoPhy's own ids (3, 12, 23, ...) say nothing about the position in the tree. Rows of all tables, the cell assignments, the
    # heatmap blocks and the tree labels all use this numbering.
    dfs_ids = lophy_dfs_order(text, ids)
    perm = [ids.index(i) for i in dfs_ids]                  # new row k = old row perm[k]
    inv = np.empty(len(perm), dtype=int)
    inv[perm] = np.arange(len(perm))
    total, vaf, assign = total[perm], vaf[perm], inv[assign]
    if sex == "male":                                          # LoPhy has no sex input and keeps chrX / chrY diploid: where it infers no change at all, show the
        xy = np.array([r.split("_")[0] in ("X", "Y") for r in data["region_names"]])     # single copy of a male (a region with a loss or gain is left as LoPhy gives it)
        if len(xy) == total.shape[1]:
            same = np.all(total == 2.0, axis=0)
            total = total.copy()
            total[:, xy & same] = 1.0
    res = Result("LoPhy", total, vaf, assign, [None] * len(cells), [f"node {k}" for k in range(len(dfs_ids))], gv)
    res.lophy_rank = {cid: k for k, cid in enumerate(dfs_ids)}      # LoPhy clone id -> number in the figure
    return res


# ── trees ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# Every method's tree is read from its own .gv file (topology and node labels) and redrawn in one common style, so that the trees can be
# compared directly; SNV names are written the same way ("GENE ref>alt"), the copy-number events keep the wording of the method.

def _clean_event(s):
    s = re.sub(r"<[^>]+>", "", s).strip()

    def snv(m):
        chrom, pos, ref, alt, gene = m.groups()
        return f"{chrom + ':' + pos if gene.startswith('chr') else gene} {ref}>{alt}"
    s = re.sub(r"(chr[0-9XY]+)\.(\d+)\.([ACGT]+)_([ACGT]+)\(([^)]*)\)(?:\s*\(chr[0-9XY]+\))?", snv, s)
    s = re.sub(r"\s*\(chr[0-9XY]+\)$", "", s)                      # COMPASS: " (chr6)" after a copy-number event
    s = re.sub(r"[: ](REF|ALT)(,(REF|ALT))*$", "", s)                 # COMPASS ":REF,REF" / LoPhy " REF": which allele (not drawn)
    if TREE_SHORT_EVENTS:                                              # SCATI: a whole-chromosome event is just "Loss chr7"; a local one carries its regions in brackets
        s = re.sub(r"^Aneuploidy (Gain|Loss) (chr\w+) \(\d+ regions?\)$", r"\1 \2", s)
        s = re.sub(r"^(Gain|Loss|CNLOH) (chr\w+) regions? (\d+(?:-\d+)?)$", r"\1 \2 (\3)", s)
    return s


def parse_tree(res):
    """(nodes {label: [event lines]}, edges [(parent label, child label)]) from the method's own .gv file."""
    with open(res.tree_gv) as f:
        text = f.read()
    nodes, edges = {}, []
    if res.name == "SCATI":
        for k, lab in re.findall(r'^(\d+) \[label="(Node .*?)"', text, re.MULTILINE):
            nodes[f"node {k}"] = [_clean_event(x) for x in lab.split("\\n")[1:] if x]
        edges = [(f"node {a}", f"node {b}") for a, b, rest in re.findall(r"^(\d+) -> (\d+)(.*)$", text, re.MULTILINE) if "dashed" not in rest]
    elif res.name == "COMPASS":
        for k, lab in re.findall(r"^(\d+)\[label=<(.*?)>\];", text, re.MULTILINE | re.DOTALL):
            nodes[f"node {k}"] = [x for x in (_clean_event(y) for y in lab.split("<br/>")) if x]
        edges = [(f"node {a}", f"node {b}") for a, b, rest in re.findall(r"^(\d+) -> (\d+) \[(.*?)\];", text, re.MULTILINE) if "dashed" not in rest]
    else:                                                  # LoPhy: drop the result matrices (`graph [ ... ];`) first
        text = re.sub(r"^[ \t]*graph \[.*?^\];", "", text, count=1, flags=re.DOTALL | re.MULTILINE)
        rank = res.lophy_rank
        for k, lab in re.findall(r'^Clone_(\d+) \[label="(.*?)"\];', text, re.MULTILINE | re.DOTALL):
            nodes[f"node {rank[int(k)]}"] = [x for x in (_clean_event(y) for y in lab.split("\\n")) if x]
        nodes.setdefault(f"node {rank[0]}", [])
        edges = [(f"node {rank[int(a)]}", f"node {rank[int(b)]}") for a, b in re.findall(r"^Clone_(\d+) -> Clone_(\d+);", text, re.MULTILINE)]
    for a, b in edges:
        nodes.setdefault(a, [])
        nodes.setdefault(b, [])
    return nodes, edges


NODE_COLORS = ["lightcoral", "skyblue3", "sandybrown", "paleturquoise3", "thistle", "darkolivegreen3", "lightpink", "mediumpurple",
               "khaki3", "salmon", "cadetblue3", "plum", "darkseagreen", "tan", "lightsteelblue", "peachpuff3", "orchid", "goldenrod"]


def render_tree(res, out_png, th):
    """The tree in the style of the SCATI / COMPASS graphs (coloured node outlines), rendered to a png with graphviz."""
    dot = shutil.which("dot")
    if not dot or not os.path.isfile(res.tree_gv):
        return None
    try:
        nodes, edges = parse_tree(res)
    except Exception as e:                                  # an unexpected .gv layout
        print(f"  WARNING: tree of {res.name} not parsed: {e}", file=sys.stderr)
        return None
    dark = th["bg"] != "white"
    ink, line = ("#e8e0d0", "#9aabbc") if dark else ("black", "dimgray")
    out = ["digraph G{", f'graph [bgcolor="{th["bg"]}" pad=0.03 nodesep={TREE_NODESEP} ranksep=0.3];', f'node [shape=box style=rounded color={line} fontsize={TREE_FONT} fontcolor="{ink}" fontname=Helvetica penwidth=4 margin="0.08,0.04"];']
    for a_, b_ in edges:
        out.append(f'"{a_}" -> "{b_}" [color="{line}" penwidth=3 weight=2];')
    for lab, events in nodes.items():
        k = res.node_labels.index(lab) if lab in res.node_labels else len(res.node_labels)
        color = NODE_COLORS[k % len(NODE_COLORS)]
        title = lab[0].upper() + lab[1:]
        wrap_w = TREE_WRAP
        if TREE_PACK and len(events) > TREE_PACK:                       # all events, several per line
            per_line = 2 if len(events) <= 2 * TREE_PACK else 3
            events = ["   ".join(events[i:i + per_line]) for i in range(0, len(events), per_line)]
            wrap_w = 200
        wrapped = [w_ for e in events for w_ in textwrap.wrap(e.replace('"', "'"), width=wrap_w, break_long_words=False, break_on_hyphens=False) or [""]]
        text = "\\n".join([title] + wrapped) + "\\n"
        out.append(f'"{lab}" [label="{text}" color={color} penwidth={TREE_PEN}];')
    out.append("}")
    src = out_png + ".gv"
    with open(src, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    try:
        subprocess.run([dot, "-Tpng", f"-Gdpi={TREE_DPI}", src, "-o", out_png], check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  (tree of {res.name} not rendered: {(getattr(e, 'stderr', b'') or b'').decode(errors='replace')[:300]})", file=sys.stderr)
        return None
    finally:
        os.remove(src)
    return out_png


# ── distances ─────────────────────────────────────────────────────────────────────────────────────────────────────

MIN_REF_CELLS = 10


def reference_node(res):
    """Node whose singlet cells are the reference (unit) of the region counts: the root if it has at least MIN_REF_CELLS singlets, otherwise the
    populated node (at least MIN_REF_CELLS singlets) closest to the root in the tree, the one with most cells among equals (COMPASS and LoPhy
    often attach no cell to the root: the first cells attach below it). None if the tree cannot be read or no node is populated enough."""
    n_cells = np.bincount(res.cell_node[res.singlet_mask()], minlength=res.n_nodes)
    if n_cells[0] >= MIN_REF_CELLS:
        return 0
    try:
        _, edges = parse_tree(res)
    except Exception:
        return None
    parent = {int(b.split()[1]): int(a.split()[1]) for a, b in edges}

    def depth(k):
        d = 0
        while k in parent and k != 0 and d <= len(parent):
            k, d = parent[k], d + 1
        return d
    cand = [k for k in range(res.n_nodes) if n_cells[k] >= MIN_REF_CELLS]
    return min(cand, key=lambda k: (depth(k), -n_cells[k])) if cand else None


def reference_cells(res):
    """(node, boolean mask of the reference singlet cells); falls back to all singlets (node 0) if no reference node is found."""
    k = reference_node(res)
    if k is None:
        print(f"  WARNING: {res.name} has no node with {MIN_REF_CELLS} singlets: all singlets used as the reference", file=sys.stderr)
        return 0, res.singlet_mask()
    return k, (res.cell_node == k) & res.singlet_mask()


def compute_metrics(res, data, common, min_depth):
    """Region error and VAF error of one method on the cells in the boolean mask `common`. A cell is scored with the genotype of its node; a cell that
    SCATI calls a doublet with the summed genotype of its two nodes. A cell without a prediction (the doublets of COMPASS) must not be in `common`."""
    D = data["counts"].T.astype(float)                       # cells x regions
    ref_node, root = reference_cells(res)
    mu = D[root].mean(axis=0)
    mu[mu == 0] = np.nan
    obs_rel = D / mu                                          # observed region counts relative to the root cells
    depth = D.sum(axis=1)
    scale = (depth / depth[root].mean())[:, None]             # reads of the cell relative to the root cells
    rho = mu / np.nansum(mu)                                  # share of every region in the root cells
    root_cn = res.node_cn[ref_node].astype(float)            # copy numbers of the reference node (the root unless it has too few cells)
    root_cn[root_cn == 0] = np.nan
    cn_cell, vaf_cell = predicted(res, D.shape[1], data["ref"].shape[0])   # copy numbers and VAF of the genotype of every cell (NaN: no prediction)
    rel = cn_cell / root_cn
    lam = np.nansum(rho * rel, axis=1)                        # normaliser: the read shares of all regions of a genotype add up to one
    with np.errstate(invalid="ignore", divide="ignore"):
        pred_tree = scale * rel / lam[:, None]                # expected relative count of every cell under the copy numbers of its genotype
    out = dict(method=res.name, n_nodes=res.n_nodes, ref_node=ref_node, n_root_cells=int(root.sum()), n_cells_scored=int(common.sum()),
               n_doublets_scored=int((common & ~res.singlet_mask()).sum()))
    # Region error: relative error |observed / expected - 1| of the region count, both in units of the root cells' typical count. The unit cancels,
    # so the error does not depend on how many reads the reference (root) cells have. Regions with copy number 0 in the genotype of the cell are left out
    # (expected count 0). No reference tree: the error needs no comparison with an all-root tree.
    ok = common[:, None] & np.isfinite(obs_rel) & np.isfinite(pred_tree) & (cn_cell > 0) & (pred_tree > 0)
    out["region_err"] = float(np.mean(np.abs(obs_rel[ok] / pred_tree[ok] - 1)))
    ref, alt = data["ref"].T, data["alt"].T                  # cells x loci
    n = ref + alt
    obs = np.where(n >= min_depth, alt / np.maximum(n, 1), np.nan)
    for tag, v in (("", vaf_cell), ("_root_only", np.tile(res.node_vaf[0], (len(res.cell_node), 1)))):
        ok = common[:, None] & (n >= min_depth) & np.isfinite(v)
        out["VAF_MAE" + tag] = float(np.mean(np.abs(obs - v)[ok]))
    out["VAF_MAE_gain"] = 1 - out["VAF_MAE"] / out["VAF_MAE_root_only"]
    return out


# ── figure ────────────────────────────────────────────────────────────────────────────────────────────────────────

def predicted(res, n_reg, n_loci):
    n = len(res.cell_node)
    cn, vaf = np.full((n, n_reg), np.nan), np.full((n, n_loci), np.nan)
    single = res.singlet_mask()
    cn[single], vaf[single] = res.node_cn[res.cell_node[single]], res.node_vaf[res.cell_node[single]]
    for j, d in enumerate(res.cell_doublet):
        if isinstance(d, tuple) and res.doublet_cn is not None:   # SCATI: sum of the genotypes of the two nodes; COMPASS: no prediction (NaN, grey)
            cn[j], vaf[j] = res.doublet_cn(*d), res.doublet_vaf(*d)
    return cn, vaf


def cell_order(res):
    groups, order = [], []
    single = res.singlet_mask()
    for k in range(res.n_nodes):
        idx = np.where(single & (res.cell_node == k))[0]
        if len(idx):
            groups.append((f"{res.node_labels[k]} ({len(idx)})", len(order), len(order) + len(idx), False))
            order.extend(idx)
    idx = sorted((j for j, d in enumerate(res.cell_doublet) if d is not None),
                 key=lambda j: (res.cell_doublet[j] if isinstance(res.cell_doublet[j], tuple) else (10 ** 6, 10 ** 6), j))
    if idx:
        groups.append((f"doublets ({len(idx)})", len(order), len(order) + len(idx), True))
        order.extend(idx)
    return np.array(order), groups


def chrom_ticks(labels):
    ticks, prev = [], None
    for i, c in enumerate(labels):
        if c != prev:
            ticks.append((i, c))
            prev = c
    return ticks


def chrom_label_rows(reg_chrom, width_in, fontsize, gap_in=0.16):
    """Chromosome labels below the region columns: [(label, centre of the chromosome in region units, x of the label in inches, row)].
    The labels alternate between two rows; within a row they are pushed apart until they do not overlap (and kept inside the axis), and a leader line joins every
    label to its chromosome, so that chromosomes of a single region (a few hundredths of an inch) stay legible."""
    ticks = chrom_ticks(reg_chrom)
    n = len(reg_chrom)
    ends = [t[0] for t in ticks[1:]] + [n]
    items = []
    for i, ((s, c), e) in enumerate(zip(ticks, ends)):
        centre = (s + e - 1) / 2.0
        items.append(dict(label=c, centre=centre, x0=(centre + 0.5) / n * width_in, w=0.62 * fontsize / 72.0 * len(c) + gap_in, row=i % 2))
    for row in (0, 1):
        its = [it for it in items if it["row"] == row]
        for it in its:
            it["x"] = it["x0"]
        for _ in range(500):
            moved = False
            for a_, b_ in zip(its, its[1:]):
                need = (a_["w"] + b_["w"]) / 2
                if b_["x"] - a_["x"] < need - 1e-9:
                    push = (need - (b_["x"] - a_["x"])) / 2
                    a_["x"] -= push
                    b_["x"] += push
                    moved = True
            for it in its:
                it["x"] = min(max(it["x"], it["w"] / 2), width_in - it["w"] / 2)
            if not moved:
                break
    return [(it["label"], it["centre"], it["x"], it["row"]) for it in items]


def spread_labels(pos, gap, lo, hi, iters=500):
    """Positions of ordered labels moved apart to at least `gap` from each other and kept within [lo, hi]; labels that already fit do not move."""
    pos = list(pos)
    if len(pos) * gap > hi - lo:                                      # cannot all fit: spread evenly
        return [lo + (i + 0.5) * (hi - lo) / len(pos) for i in range(len(pos))]
    for _ in range(iters):
        moved = False
        for i in range(len(pos) - 1):
            d = pos[i + 1] - pos[i]
            if d < gap - 1e-9:
                push = (gap - d) / 2
                pos[i] -= push
                pos[i + 1] += push
                moved = True
        pos = [min(max(p, lo + gap / 2), hi - gap / 2) for p in pos]
        if not moved:
            break
    return pos


def style(ax, th):
    ax.set_facecolor(th["bg"])
    for s in ax.spines.values():
        s.set_edgecolor(th["grid"])
        s.set_linewidth(0.5)
    ax.tick_params(length=2, color=th["grid"], labelsize=6.5, labelcolor=th["fg2"])


def make_figure(results, data, metrics, out_png, theme, trees_dir, dpi, fig=None, dry_run=False, min_depth=5):
    """One row per result: tree, observed and predicted region counts / copy numbers, observed and predicted VAF. With dry_run only the size of the
    figure (width, height in inches) is returned; with `fig` (a figure or subfigure) the rows are drawn into it instead of a new figure."""
    th = THEMES[theme]
    n_reg, n_loci = len(data["region_names"]), len(data["loci_names"])
    reg_chrom = [r.split("_")[0] for r in data["region_names"]]
    vaf_cmap = plt.cm.RdBu_r.copy()
    vaf_cmap.set_bad(th["bad"])      # no prediction (COMPASS doublets): grey
    cn_cmap = th["cn"].copy()
    cn_cmap.set_bad(th["bad"])
    vmax_cn = float(np.clip(np.nanmax([np.nanmax(r.node_cn) for r in results]), 4, 8))
    n = len(results)
    F_TITLE, F_HEAD, F_LAB, F_TICK, F_CB = 28, 22, 18, 15, 32   # font sizes (pt) of the method names, column headers, node labels, ticks, colour bars

    # all trees are drawn in the same style and at the same scale; the tree column has a fixed width
    tree_imgs = []
    for res in results:
        png = render_tree(res, os.path.join(trees_dir, f"tree_{res.name}.png"), th)
        tree_imgs.append(plt.imread(png) if png else None)
    sizes = [im.shape[:2] for im in tree_imgs if im is not None]
    hmax, wmax = (max(s[0] for s in sizes), max(s[1] for s in sizes)) if sizes else (1, 1)
    row_h, tree_w = ROW_H_NATIVE, TREE_W_NATIVE
    pad_w, pad_h = 0.5, 0.15
    scale = min((tree_w - pad_w) / (wmax / TREE_DPI), (row_h - pad_h) / (hmax / TREE_DPI))   # picture pixels -> inches
    print(f"trees: {tree_w:.1f} in wide column, drawn at {scale:.2f} x their size (graph font {TREE_FONT} pt -> {TREE_FONT * scale:.0f} pt)")
    scales = [scale] * n
    widths = [tree_w, 1.6, 4.8, 0.2, 4.8, 0.38, 1.9, 2.0, 0.2, 2.0, 0.38]    # tree | node labels | observed counts | predicted CN | bar | gap | observed VAF | predicted VAF | bar
    F_CHR, LEVEL_PT, STRIP_PT = 22, 28, 8                                        # chromosome labels below the lowest row: font size and row spacing (pt)
    chrom_labels = chrom_label_rows(reg_chrom, widths[2], F_CHR)
    n_levels = 1 + max(r for _, _, _, r in chrom_labels)
    left, right, top, bottom = 0.25, RIGHT_NATIVE, 1.35, 0.35
    bottom += (n_levels * LEVEL_PT + STRIP_PT + 6) / 72.0    # margins in inches (right: tick labels and title of the last colour bar; bottom: chromosome labels)
    W, H = sum(widths) + left + right, n * row_h + top + bottom
    if dry_run:                                                    # only the size of the figure (inches)
        return W, H
    own_fig = fig is None
    if own_fig:
        fig = plt.figure(figsize=(W, H), facecolor=th["bg"])
    gs = fig.add_gridspec(n, 11, width_ratios=widths, wspace=0.0, hspace=0.13, left=left / W, right=1 - right / W, top=1 - top / H, bottom=bottom / H)
    heads = ["Tree", "Observed" + chr(10) + "region counts", "Predicted" + chr(10) + "copy numbers", "Observed" + chr(10) + "VAF", "Predicted" + chr(10) + "VAF"]
    for row, res in enumerate(results):
        order, groups = cell_order(res)
        cn_pred, vaf_pred = predicted(res, n_reg, n_loci)
        D = data["counts"].T.astype(float)
        ref_node, root = reference_cells(res)
        mu = D[root].mean(axis=0)
        mu[mu == 0] = np.nan
        root_cn = res.node_cn[ref_node].astype(float)
        obs_cn = D / mu * root_cn
        ref, alt = data["ref"].T, data["alt"].T
        tot = ref + alt
        obs_vaf = np.where(tot >= min_depth, alt / np.maximum(tot, 1), np.nan)
        axes = [fig.add_subplot(gs[row, c]) for c in (0, 2, 4, 7, 9)]
        # tree: every tree at the same scale, top-aligned and centred in its column
        ax = axes[0]
        ax.set_facecolor(th["bg"])
        ax.axis("off")
        img = tree_imgs[row]
        if img is not None:
            h, w = img.shape[:2]
            sc = scales[row]
            y0 = (row_h / sc * TREE_DPI - h) / 2 if TREE_VCENTER else 0.0
            ax.imshow(img, extent=((tree_w / sc * TREE_DPI - w) / 2, (tree_w / sc * TREE_DPI + w) / 2, y0 + h, y0))
            ax.set_xlim(0, tree_w / sc * TREE_DPI)
            ax.set_ylim(row_h / sc * TREE_DPI, 0)
            ax.set_aspect("auto")
        else:
            ax.text(0.5, 0.5, "tree not rendered\n(graphviz `dot` missing?)", ha="center", va="center", color=th["fg2"], transform=ax.transAxes, fontsize=F_LAB)
        ref_node = reference_cells(res)[0]
        sub = f"reference: node {ref_node}" if ref_node != 0 else ""     # the node count and notes are in the caption / text
        ax.text(0.0, 1.0, res.label, transform=ax.transAxes, ha="left", va="bottom", color=th["fg"], fontsize=F_TITLE, fontweight="bold")
        if sub:
            ax.set_title(sub, loc="right", color=th["fg2"], fontsize=F_LAB - 2, pad=6)
        # heatmaps
        panels = [(obs_cn[order], cn_cmap, 0, vmax_cn), (cn_pred[order], cn_cmap, 0, vmax_cn),
                  (obs_vaf[order], vaf_cmap, 0, 1), (vaf_pred[order], vaf_cmap, 0, 1)]
        for ax, (mat, cmap, lo, hi) in zip(axes[1:], panels):
            style(ax, th)
            ax.tick_params(labelsize=F_TICK)
            ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            for lab, s, e, is_d in groups[1:]:
                ax.axhline(s - 0.5, color=th["dbl"] if is_d and s == next(g[1] for g in groups if g[3]) else th["bound"],
                           lw=1.4 if is_d else 0.8, alpha=0.85)
        for ax in axes[1:]:
            ax.set_xticks([])
        if row == n - 1:                                            # chromosome of every region column, below the lowest plot: alternating bars over the span of each chromosome, names below
            ticks_c = chrom_ticks(reg_chrom)
            spans = [(s, e) for (s, _), e in zip(ticks_c, [x[0] for x in ticks_c[1:]] + [n_reg])]
            for ax in axes[1:3]:
                h_in = ax.get_position().height * H
                y0, y1 = -2.0 / 72 / h_in, -(2.0 + STRIP_PT) / 72 / h_in          # strip below the axis, in axes fractions
                tr = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
                for i, (s, e) in enumerate(spans):
                    ax.add_patch(matplotlib.patches.Rectangle((s - 0.5, y1), e - s, y0 - y1, transform=tr, facecolor=th["fg2"] if i % 2 == 0 else th["grid"],
                                                              edgecolor="none", clip_on=False))
                for c, centre, x_in, lv in chrom_labels:
                    dx = (x_in - (centre + 0.5) / n_reg * widths[2]) * 72          # label displaced from its chromosome (points)
                    ax.annotate(c, xy=(centre, y1), xycoords=("data", "axes fraction"), xytext=(dx, -4 - lv * LEVEL_PT), textcoords="offset points",
                                ha="center", va="top", fontsize=F_CHR, color=th["fg"],
                                arrowprops=dict(arrowstyle="-", lw=0.6, color=th["fg2"], shrinkA=1, shrinkB=0, relpos=(0.5, 1.0)), annotation_clip=False)
            axes[1].annotate("Chr", xy=(0.0, y1), xycoords=("axes fraction", "axes fraction"), xytext=(-8, -3), textcoords="offset points", ha="right", va="top",
                             fontsize=F_CHR, color=th["fg"], fontweight="bold", annotation_clip=False)
        for ax in axes[1:3]:
            for b, _ in chrom_ticks(reg_chrom)[1:]:
                ax.axvline(b - 0.5, color="#9a9a9a", lw=0.9, alpha=0.95)
        tot_cells = max(1, len(order))
        # every node is labelled, also the small ones: labels of neighbouring small blocks are pushed apart (they stay in the order of the blocks)
        h_rows = (F_LAB * 1.15 / 72.0) / (axes[1].get_position().height * H) * tot_cells        # height of a label in heatmap rows
        pos = spread_labels([(s + e) / 2 for _, s, e, _ in groups], h_rows, -0.5, tot_cells - 0.5)
        lab = [g[0] for g in groups]
        axes[1].set_yticks(pos)
        axes[1].set_yticklabels(lab, color=th["fg"], fontsize=F_LAB)
        axes[1].tick_params(axis="y", length=0, pad=6)
        for ax in axes[2:]:
            ax.set_yticks([])
        for gcol, cmap, hi, lab in ((5, cn_cmap, vmax_cn, "copy number"), (10, vaf_cmap, 1, "VAF")):
            cax = fig.add_subplot(gs[row, gcol])
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, hi)), cax=cax)
            cb.set_label(lab, color=th["fg"], fontsize=F_CB, labelpad=12)
            cb.ax.tick_params(colors=th["fg"], labelsize=22, length=6)
            cb.outline.set_edgecolor(th["grid"])
        if row == 0:
            for ax, h in zip(axes[1:], heads[1:]):
                ax.set_title(h, color=th["fg"], fontsize=F_HEAD, fontweight="600", loc="center", pad=10)
    if own_fig:
        if out_png:
            fig.savefig(out_png, dpi=dpi, facecolor=th["bg"])
            fig.savefig(os.path.splitext(out_png)[0] + ".pdf", facecolor=th["bg"])
        plt.close(fig)
    return W, H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_base", required=True, help="base name of the filtered input files")
    ap.add_argument("--scati", default=None)
    ap.add_argument("--scati_label", default="SCATI", help="row title of the SCATI run")
    ap.add_argument("--compass", default=None)
    ap.add_argument("--lophy", default=None)
    ap.add_argument("--lophy_meta", default=None, help="LoPhy's meta.csv (default: next to the results folder)")
    ap.add_argument("--out", required=True, help="output figure (.png; a .pdf is written next to it)")
    ap.add_argument("--theme", choices=list(THEMES), default="light")
    ap.add_argument("--min_depth", type=int, default=5, help="minimum reads at a locus for the VAF of a cell to be used")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    trees_dir = os.path.join(out_dir, "trees", os.path.splitext(os.path.basename(a.out))[0])
    os.makedirs(trees_dir, exist_ok=True)
    lophy_meta = a.lophy_meta or (os.path.join(os.path.dirname(os.path.normpath(a.lophy)), "meta.csv") if a.lophy else None)

    data = load_data(a.data_base)
    print(f"{len(data['cells'])} cells, {len(data['region_names'])} regions, {len(data['loci_names'])} variants")
    loaders = [("SCATI", a.scati, lambda d: load_scati(d, data), "best_tree_strand_genotypes.json"),
               ("COMPASS", a.compass, lambda d: load_compass(d, data), "data_nodes_genotypes.tsv"),
               ("LoPhy", a.lophy, lambda d: load_lophy(d, lophy_meta, data), "data_0.gv")]
    results = []
    for name, d, load, marker in loaders:
        if d is None:
            continue
        if not os.path.exists(os.path.join(d, marker)):
            sys.exit(f"[{name}] no result in {d} ({marker} missing)")
        print(f"[{name}] {d}")
        res = load(d)
        if name == "SCATI":
            res.label = a.scati_label
        results.append(res)
    if not results:
        sys.exit("no method given (--scati, --compass or --lophy)")
    common = np.all([r.singlet_mask() for r in results], axis=0)
    print(f"distances on the {int(common.sum())} cells that no method calls a doublet")
    metrics = [compute_metrics(r, data, common, a.min_depth) for r in results]
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.splitext(a.out)[0] + "_metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(3).to_string(index=False))
    make_figure(results, data, metrics, a.out, a.theme, trees_dir, a.dpi, min_depth=a.min_depth)
    print("written:", a.out)


if __name__ == "__main__":
    main()
