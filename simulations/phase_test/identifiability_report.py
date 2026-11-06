"""Prints which loci and pairs of loci are phase-identifiable in one replicate (definitions in phase_common.py).

    python identifiability_report.py --rep_dir data/rep0 [--out identifiability.json]
"""
import argparse
import json

from phase_common import Truth, identifiability


def report(rep_dir):
    t = Truth(rep_dir)
    res = identifiability(t)
    het, comp, pairs = res["het"], res["comp"], res["pairs"]
    informed_het = [i for i in het if i in comp]
    in_pair_comp = [i for c, m in res["by_comp"].items() if len(m) >= 2 for i in m]
    lines = []
    lines.append(f"{rep_dir}: {t.n_loci} loci, {len(het)} heterozygous at the root ({t.n_het_dedicated} dedicated + {len(het) - t.n_het_dedicated} SNV sites), {t.n_snvs} somatic SNV sites")
    lines.append("informative events (event: loci it informs):")
    for e in res["informed"]:
        lines.append(f"  node {e['node']} {e['type']:11s} strand {e['strand']} regions {e['regions'][0]}-{e['regions'][-1]}: " + (", ".join(t.names[i] for i in e["loci"]) or "none (no observable allelic change)"))
    lines.append("linked components (het loci only):")
    for c, members in sorted(res["by_comp"].items()):
        lines.append(f"  component {c}: " + ", ".join(t.names[i] for i in members) + f"  ({len(members)} hets, {len(members) * (len(members) - 1) // 2} pairs)")
    lines.append(f"het loci informed by at least one event: {len(informed_het)} of {len(het)}")
    lines.append(f"het loci in a component with another het (phase relative to it identifiable): {len(in_pair_comp)} of {len(het)}")
    n_l = sum(p["linked"] for p in pairs)
    lines.append(f"het pairs: {len(pairs)} in total; linked {n_l} ({sum(p['linked'] and p['same_region'] for p in pairs)} in the same region, "
                 f"{sum(p['linked'] and not p['same_region'] for p in pairs)} in different regions); unlinked {len(pairs) - n_l} ({sum((not p['linked']) and p['same_region'] for p in pairs)} in the same region)")
    lines.append(f"events that inform at least one het (event haplotype metric): {sum(any(i in set(het) for i in e['loci']) for e in res['informed'])} of {len(res['informed'])} copy-number events")
    lines.append("somatic SNVs evaluable (hom-ref at the root, informed by an event, component with a het): " + (", ".join(t.names[s] for s in res["snv_eval"]) or "none"))
    return "\n".join(lines), dict(n_het=len(het), n_het_informed=len(informed_het), n_het_in_pair_component=len(in_pair_comp), n_pairs=len(pairs), n_pairs_linked=int(n_l),
                                  n_snv_evaluable=len(res["snv_eval"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rep_dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    text, summary = report(a.rep_dir)
    print(text)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
