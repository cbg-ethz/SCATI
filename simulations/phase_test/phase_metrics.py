"""Scores the phase of a SCATI result against the ground truth of a phase-test dataset.
Definitions of informative event, linked and components are in phase_common.py.

    python phase_metrics.py --rep_dir data/rep0 --results some/results [--out metrics.json]

--results is the folder with best_tree_strand_genotypes.json and best_tree_cellAssignments.tsv. All metrics are independent of strand labels.
  pair accuracy      for pairs of het loci, whether SCATI's root and the truth agree on same or different haplotype. Strata: linked in the same
                     region, linked in different regions, unlinked (about 0.5 unless the phase is given). A locus that is not het in SCATI's root counts as wrong.
  switch error rate  within each component, the share of adjacent het loci (ordered by region) whose relation differs from the truth.
  entries            the (informative event, informed locus) pairs. An entry is correct if SCATI's (ref, alt) copy numbers at the locus equal the
                     truth in the SCATI node matched to the event's node. Reported for het loci, for somatic SNVs that are hom-ref at the root, and per event.
Each true node is matched to the SCATI node holding most of its true singlets, if that is more than half of them.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from phase_common import Truth, identifiability


def load_result(results_dir):
    with open(os.path.join(results_dir, "best_tree_strand_genotypes.json")) as f:
        strand = json.load(f)
    ca = pd.read_csv(os.path.join(results_dir, "best_tree_cellAssignments.tsv"), sep="\t")
    return strand, ca


def ref_alt(strand_node):
    return (np.sum(strand_node["strands_loci_ref"], axis=0).astype(int), np.sum(strand_node["strands_loci_alt"], axis=0).astype(int))


def alt_strand_at_root(strand_root, locus):
    """Strand carrying the alt allele of a het locus (one ref and one alt copy in total); None if the locus is not het."""
    ref = sum(s[locus] for s in strand_root["strands_loci_ref"])
    alt = sum(s[locus] for s in strand_root["strands_loci_alt"])
    if ref != 1 or alt != 1:
        return None
    for s, v in enumerate(strand_root["strands_loci_alt"]):
        if v[locus] == 1:
            return s
    return None


def match_nodes(truth, ca):
    """Returns {true node: (matched SCATI node or None, share of the node's true singlets in it)}."""
    tca = pd.read_csv(truth.base + "_cellAssignmentsTRUE.tsv", sep="\t")
    tsing = ~tca["is_doublet"].to_numpy().astype(bool)
    s_node = ca.set_index("cell").reindex(tca["cell"].astype(str).to_numpy() if ca["cell"].dtype == object else tca["cell"].to_numpy())["node"].to_numpy()
    out = {}
    for n in range(truth.n_nodes):
        idx = np.where(tsing & (tca["node"].to_numpy() == n))[0]
        c = Counter(int(x) for x in s_node[idx] if not pd.isna(x))
        if not c:
            out[n] = (None, 0.0)
            continue
        m, k = c.most_common(1)[0]
        out[n] = (m if k / len(idx) > 0.5 else None, k / len(idx))
    return out


def evaluate(rep_dir, results_dir):
    t = Truth(rep_dir)
    ident = identifiability(t)
    strand, ca = load_result(results_dir)
    root = strand["0"]
    het = ident["het"]
    pred_alt = {i: alt_strand_at_root(root, i) for i in het}
    out = {"n_het": len(het), "n_het_called_het": int(sum(v is not None for v in pred_alt.values()))}

    # pair accuracy by stratum
    strata = defaultdict(lambda: [0, 0])               # name -> [correct, total]
    for p in ident["pairs"]:
        i, j = p["i"], p["j"]
        name = ("linked_same_region" if p["same_region"] else "linked_diff_region") if p["linked"] else "unlinked"
        ok = pred_alt[i] is not None and pred_alt[j] is not None and ((pred_alt[i] == pred_alt[j]) == p["same_haplotype"])
        strata[name][0] += int(ok)
        strata[name][1] += 1
    out["pairs"] = {k: {"correct": v[0], "n": v[1], "accuracy": v[0] / v[1] if v[1] else None} for k, v in strata.items()}
    for k in ("linked_same_region", "linked_diff_region", "unlinked"):
        out["pairs"].setdefault(k, {"correct": 0, "n": 0, "accuracy": None})

    # switch error rate within components
    sw_err = sw_n = 0
    for c, members in ident["by_comp"].items():
        mem = sorted(members, key=lambda i: (int(t.locus_region[i]), i))
        for a, b in zip(mem[:-1], mem[1:]):
            truth_same = t.alt_strand[a] == t.alt_strand[b]
            ok = pred_alt[a] is not None and pred_alt[b] is not None and ((pred_alt[a] == pred_alt[b]) == truth_same)
            sw_err += int(not ok)
            sw_n += 1
    out["switch"] = {"errors": sw_err, "n": sw_n, "rate": sw_err / sw_n if sw_n else None}

    # entries: allele-specific copy numbers in the matched node
    match = match_nodes(t, ca)
    out["node_match"] = {str(n): {"scati_node": m, "share": round(s, 3)} for n, (m, s) in match.items()}
    het_set = set(het)
    entries = []
    for e in ident["informed"]:
        n = e["node"]
        m = match[n][0]
        t_ref, t_alt = t.node_ref_alt(n)
        p_ref, p_alt = ref_alt(strand[str(m)]) if m is not None else (None, None)
        for i in e["loci"]:
            ok = m is not None and t_ref[i] == p_ref[i] and t_alt[i] == p_alt[i]
            kind = "het" if i in het_set else ("snv_homref" if t.root_alt[i] == 0 and i < t.n_snvs else "other")
            entries.append(dict(node=n, idx=e["idx"], type=e["type"], locus=int(i), kind=kind, ok=bool(ok)))
    df = pd.DataFrame(entries)
    out["entries"] = {}
    for kind in ("het", "snv_homref"):
        d = df[df.kind == kind] if len(df) else df
        out["entries"][kind] = {"correct": int(d.ok.sum()) if len(d) else 0, "n": int(len(d)), "accuracy": float(d.ok.mean()) if len(d) else None}
    ev_rows = []
    if len(df):
        for (n, idx), d in df.groupby(["node", "idx"]):
            ev_rows.append(dict(node=int(n), idx=int(idx), type=d.type.iloc[0], n_loci=len(d), ok=bool(d.ok.all())))
    out["events"] = {"correct": int(sum(r["ok"] for r in ev_rows)), "n": len(ev_rows), "detail": ev_rows}
    out["n_snv_evaluable"] = len(ident["snv_eval"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep_dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = evaluate(a.rep_dir, a.results)
    s = json.dumps(r, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
