"""Truth side of the phase test: loads the ground truth of a simulated dataset, replays its events and finds the phase-identifiable loci.

Definitions (ground truth only):
  het locus          a locus with one ref and one alt copy at the root.
  informative event  a copy-number event (gain, loss, CNLOH) informs a locus if it changes the allelic ratio alt / (ref + alt) there, the locus
                     has copy number > 0 at the end of the node, and its ratio differs from the parent's. Changes that later events in the
                     same node undo do not count.
  linked             two loci are linked if some event informs both. The connected components of this graph are the blocks within which the
                     relative phase is identifiable; loci in different components, or in none, are not linked.
Somatic SNV sites are treated like any other locus once their alt allele is present.
"""
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generate"))
try:
    import scipy.stats  # noqa: F401
except ImportError:      # stub scipy: the generator imports betabinom, which replaying events does not need
    import types
    _st = types.ModuleType("scipy.stats")
    _st.betabinom = None
    _sp = types.ModuleType("scipy")
    _sp.stats = _st
    sys.modules["scipy"], sys.modules["scipy.stats"] = _sp, _st
from generate_synthetic_data_new import apply_event, propagate_from_parent  # noqa: E402

CNA_TYPES = ("local_gain", "local_loss", "CNLOH", "aneuploidy_gain", "aneuploidy_loss", "FGD")


class Truth:
    """Ground truth of one dataset (files rep_dir/SCATI/data_*)."""

    def __init__(self, rep_dir, base="data"):
        d = os.path.join(rep_dir, "SCATI")
        b = os.path.join(d, base)
        self.rep_dir, self.base = rep_dir, b
        with open(b + "_strand_genotypes.json") as f:
            self.strand = json.load(f)
        with open(b + "_events_ordered.json") as f:
            self.events = {int(k): v for k, v in json.load(f).items()}
        with open(b + "_true_model_params.json") as f:
            self.params = json.load(f)
        self.n_nodes = len(self.strand)
        self.locus_region = np.array(self.params["all_locus_to_region"], dtype=int)
        self.n_loci = len(self.locus_region)
        self.n_regions = len(self.strand["0"]["strands_regions"][0])
        self.n_snvs, self.n_het_dedicated = int(self.params["n_SNVs"]), int(self.params["n_het"])
        self.parents = self._parents(b + "_treeTRUE.gv")
        self.names = [f"SNV{i}" if i < self.n_snvs else f"Het{i - (self.n_loci - self.n_het_dedicated)}" for i in range(self.n_loci)]
        root = self.strand["0"]
        ref0 = np.sum(root["strands_loci_ref"], axis=0)
        alt0 = np.sum(root["strands_loci_alt"], axis=0)
        self.root_ref, self.root_alt = ref0, alt0
        self.het = np.where((ref0 == 1) & (alt0 == 1))[0]
        # strand carrying the alt allele of each het locus at the root; strand labels are arbitrary, so only compare pairs
        alt_strand = {int(i): int(np.argmax([root["strands_loci_alt"][s][i] for s in range(len(root["strands_loci_alt"]))])) for i in self.het}
        self.alt_strand = alt_strand

    def _parents(self, gv):
        text = open(gv).read()
        parents = {0: -1}
        for a, b in re.findall(r"^(\d+) -> (\d+) \[color=dimgray", text, re.MULTILINE):
            parents[int(b)] = int(a)
        return [parents[n] for n in range(self.n_nodes)]

    def children(self):
        ch = [[] for _ in range(self.n_nodes)]
        for n in range(1, self.n_nodes):
            ch[self.parents[n]].append(n)
        return ch

    def order(self):
        ch, out, stack = self.children(), [], [0]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(ch[n])
        return out

    def replay(self):
        """Replays the events of every node in order and checks that the result matches the stored genotypes.
        Returns one record per event: dict(node, idx, ev, before, after), with (ref, alt) copy numbers per locus before and after the event."""
        root = self.strand["0"]
        sr = [None] * self.n_nodes
        sref, salt = [None] * self.n_nodes, [None] * self.n_nodes
        sr[0] = [np.array(s) for s in root["strands_regions"]]
        sref[0] = [np.array(s) for s in root["strands_loci_ref"]]
        salt[0] = [np.array(s) for s in root["strands_loci_alt"]]
        records = []
        for n in self.order():
            if n == 0:
                continue
            propagate_from_parent(n, self.parents, sr, sref, salt)
            for k, ev in enumerate(self.events[n]):
                before = self._ref_alt(sref[n], salt[n])
                apply_event(n, ev, sr, sref, salt, self.locus_region.tolist(), self.n_loci, self.n_regions)
                after = self._ref_alt(sref[n], salt[n])
                records.append(dict(node=n, idx=k, ev=ev, before=before, after=after))
            truth_n = self.strand[str(n)]
            ok = (len(sr[n]) == len(truth_n["strands_regions"]) and all(np.array_equal(a, b) for a, b in zip(sr[n], truth_n["strands_regions"]))
                  and all(np.array_equal(a, b) for a, b in zip(sref[n], truth_n["strands_loci_ref"]))
                  and all(np.array_equal(a, b) for a, b in zip(salt[n], truth_n["strands_loci_alt"])))
            if not ok:
                raise RuntimeError(f"replay of node {n} differs from the ground truth")
        return records

    @staticmethod
    def _ref_alt(sref, salt):
        return np.sum(sref, axis=0).astype(int), np.sum(salt, axis=0).astype(int)

    def node_ref_alt(self, n):
        t = self.strand[str(n)]
        return self._ref_alt(t["strands_loci_ref"], t["strands_loci_alt"])


def write_haplotype_file(truth, path):
    """Writes the true root genotype as the haplotype annotation read by `--use_haplotype_annotation 1` (<prefix>_haplotypes.tsv).
    Three rows (regions, ref, alt); strands are separated by ';' and values by ','."""
    root = truth.strand["0"]
    rows = {"root_strands_regions": root["strands_regions"], "root_strands_loci_ref": root["strands_loci_ref"], "root_strands_loci_alt": root["strands_loci_alt"]}
    with open(path, "w", newline="\n") as f:
        for key, strands in rows.items():
            f.write(key + "\t" + ";".join(",".join(str(int(v)) for v in s) for s in strands) + "\n")


def ratio(ref, alt):
    tot = ref + alt
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, alt / np.where(tot > 0, tot, 1), np.nan)


def informed_loci(truth, records):
    """For every CNA event record, the loci it informs (see the module docstring)."""
    out = []
    for rec in records:
        ev = rec["ev"]
        if ev["type"] in ("SNV", "FGD"):
            continue
        n = rec["node"]
        r_b, r_a = ratio(*rec["before"]), ratio(*rec["after"])
        ref_n, alt_n = truth.node_ref_alt(n)
        ref_p, alt_p = truth.node_ref_alt(truth.parents[n])
        r_n, r_p = ratio(ref_n, alt_n), ratio(ref_p, alt_p)
        changed = np.isfinite(r_b) & np.isfinite(r_a) & (np.abs(r_b - r_a) > 1e-9)
        observable = (ref_n + alt_n > 0) & np.isfinite(r_n) & np.isfinite(r_p) & (np.abs(r_n - r_p) > 1e-9)
        out.append(dict(node=n, idx=rec["idx"], type=ev["type"], strand=ev.get("strand", ev.get("donor_strand")),
                        regions=sorted(ev["regions"]), loci=np.where(changed & observable)[0].tolist()))
    return out


def components(n_loci, informed):
    """Union-find over loci informed by a common event. Returns {locus: component id} for loci informed by at least one event."""
    parent = list(range(n_loci))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    touched = set()
    for e in informed:
        for i in e["loci"]:
            touched.add(i)
        for i, j in zip(e["loci"][:-1], e["loci"][1:]):
            parent[find(i)] = find(j)
    roots = {}
    comp = {}
    for i in sorted(touched):
        comp[i] = roots.setdefault(find(i), len(roots))
    return comp


def identifiability(truth):
    """Replays the events and returns the informed loci, components, het loci and het pairs with their linked / same-haplotype status."""
    records = truth.replay()
    informed = informed_loci(truth, records)
    comp = components(truth.n_loci, informed)
    het = [int(i) for i in truth.het]
    by_comp = defaultdict(list)
    for i in het:
        if i in comp:
            by_comp[comp[i]].append(i)
    pairs = []
    for i, j in combinations(het, 2):
        same_region = bool(truth.locus_region[i] == truth.locus_region[j])
        linked = i in comp and j in comp and comp[i] == comp[j]
        same_hap = truth.alt_strand[i] == truth.alt_strand[j]
        pairs.append(dict(i=i, j=j, same_region=same_region, linked=linked, same_haplotype=bool(same_hap)))
    # evaluable somatic SNVs: hom-ref at the root, informed by an event, in a component with a het locus
    snv_eval = []
    for s in range(truth.n_snvs):
        if truth.root_alt[s] == 0 and s in comp and any(comp[h] == comp[s] for h in het if h in comp):
            snv_eval.append(s)
    return dict(records=records, informed=informed, comp=comp, het=het, by_comp=dict(by_comp), pairs=pairs, snv_eval=snv_eval)
