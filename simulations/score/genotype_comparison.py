"""Copy numbers and VAFs from strand genotypes, and cell-to-node assignment.
Used by real_data/plot_methods.py and simulations/method_adapters.py.

SCATI's predicted and the simulator's true *_strand_genotypes.json have the same format: a dict keyed
by node id string. Each node has `strands_regions` (per strand, per region copy number) and
`strands_loci_ref` / `strands_loci_alt` (per strand, per locus allele counts).
"""
from collections import defaultdict

import numpy as np


def cn_from_strands(strands_regions, n_regions):
    """Total copy number per region of a node: sum over its strands. Zeros if it has no strands."""
    if not strands_regions:
        return np.zeros(n_regions)
    return np.array(strands_regions, dtype=float).sum(axis=0)


def cn_from_strands_doublet(strand_data, n1, n2, n_regions):
    return (cn_from_strands(strand_data[str(n1)]["strands_regions"], n_regions)
            + cn_from_strands(strand_data[str(n2)]["strands_regions"], n_regions))


def true_vaf_from_strands(strand_data, node, n_loci):
    """Genotype VAF per locus of a node: alt allele count / total allele count over its strands.
    Entries of -1 are skipped. NaN where the total is 0, e.g. a homozygous deletion."""
    slr = strand_data[str(node)]["strands_loci_ref"]
    sla = strand_data[str(node)]["strands_loci_alt"]
    total_ref = np.zeros(n_loci)
    total_alt = np.zeros(n_loci)
    for s in range(len(slr)):
        for i in range(n_loci):
            v_r, v_a = slr[s][i], sla[s][i]
            if v_r >= 0:
                total_ref[i] += v_r
            if v_a >= 0:
                total_alt[i] += v_a
    total = total_ref + total_alt
    return np.where(total > 0, total_alt / total, np.nan)


# ── Cell assignment and ordering ─────────────────────────────────────────────

def assign_cells(cell_assignments_df, probs_df, n_nodes, use_doublets):
    """Assigns a cell to a doublet if its best doublet probability beats its best singlet probability
    in the *_cellAssignmentProbs.tsv table. Returns (cell_to_node, cell_to_doublet)."""
    cell_to_node = dict(enumerate(cell_assignments_df["node"]))
    cell_to_doublet = {}
    if not use_doublets or probs_df is None:
        return cell_to_node, cell_to_doublet

    node_cols = [c for c in probs_df.columns if c.startswith("Node ")]
    doublet_cols = [c for c in probs_df.columns if c.startswith("Doublet_")]
    for cell_idx in range(len(probs_df)):
        row = probs_df.iloc[cell_idx]
        best_singlet_col = max(node_cols, key=lambda c: row[c])
        best_singlet_p = row[best_singlet_col]
        if doublet_cols:
            best_doublet_col = max(doublet_cols, key=lambda c: row[c])
            best_doublet_p = row[best_doublet_col]
        else:
            best_doublet_col, best_doublet_p = None, -1.0
        if best_doublet_p > best_singlet_p:
            parts = best_doublet_col.split("_")
            n1, n2 = int(parts[1]), int(parts[2])
            cell_to_doublet[cell_idx] = (n1, n2)
    return cell_to_node, cell_to_doublet


def order_cells(n_cells, n_nodes, cell_to_node, cell_to_doublet):
    """Orders cells as singlet blocks by node, then doublet blocks by (n1, n2); original order within a block.
    Returns (cell_order, group_boundaries), the latter a list of (label, start, end, is_doublet, key)."""
    node_to_cells = defaultdict(list)
    doublet_to_cells = defaultdict(list)
    for c in range(n_cells):
        if c in cell_to_doublet:
            doublet_to_cells[cell_to_doublet[c]].append(c)
        else:
            node_to_cells[cell_to_node.get(c)].append(c)

    active_nodes = [n for n in range(n_nodes) if node_to_cells[n]]
    active_doublets = sorted(doublet_to_cells.keys())

    cell_order = []
    group_boundaries = []
    for node in active_nodes:
        start = len(cell_order)
        cell_order.extend(node_to_cells[node])
        group_boundaries.append((f"node {node}", start, len(cell_order), False, node))
    for (n1, n2) in active_doublets:
        start = len(cell_order)
        cell_order.extend(doublet_to_cells[(n1, n2)])
        group_boundaries.append((f"doublet {n1}+{n2}", start, len(cell_order), True, (n1, n2)))

    return cell_order, group_boundaries


def true_vaf_doublet(strand_data, n1, n2, n_loci):
    slr1, sla1 = strand_data[str(n1)]["strands_loci_ref"], strand_data[str(n1)]["strands_loci_alt"]
    slr2, sla2 = strand_data[str(n2)]["strands_loci_ref"], strand_data[str(n2)]["strands_loci_alt"]
    total_ref = np.zeros(n_loci)
    total_alt = np.zeros(n_loci)
    for slr, sla in [(slr1, sla1), (slr2, sla2)]:
        for s in range(len(slr)):
            for i in range(n_loci):
                v_r, v_a = slr[s][i], sla[s][i]
                if v_r >= 0:
                    total_ref[i] += v_r
                if v_a >= 0:
                    total_alt[i] += v_a
    total = total_ref + total_alt
    return np.where(total > 0, total_alt / total, np.nan)
