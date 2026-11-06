"""Pools the phase metrics (phase_metrics.evaluate) of all replicates, per mode.

    python collect_phase_results.py --data data [--out summary.csv]

For each mode it prints the pooled counts (correct / total) of the pair classes, switch errors, het and somatic SNV entries and
events, and the per-replicate accuracy of the unlinked pairs. --out gets one row per (replicate, mode).
--pattern sets the result folder of a run; the placeholders are {rep_dir}, {rep} and {mode}.
"""
import argparse
import glob
import json
import os

import pandas as pd

from phase_metrics import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pattern", default="{rep_dir}/SCATI/results_phase_{mode}")
    ap.add_argument("--modes", default="ABC")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json_dir", default=None, help="also write the full metrics of every run to <json_dir>/rep<k>_<mode>.json")
    a = ap.parse_args()
    if a.json_dir:
        os.makedirs(a.json_dir, exist_ok=True)
    rows = []
    for rep_dir in sorted(glob.glob(os.path.join(a.data, "rep*")), key=lambda p: int(os.path.basename(p)[3:])):
        rep = os.path.basename(rep_dir)[3:]
        for mode in a.modes:
            res_dir = a.pattern.format(rep_dir=rep_dir, rep=rep, mode=mode)
            if not os.path.exists(os.path.join(res_dir, "best_tree_strand_genotypes.json")):
                print(f"rep{rep} mode {mode}: no result in {res_dir}")
                continue
            r = evaluate(rep_dir, res_dir)
            if a.json_dir:
                with open(os.path.join(a.json_dir, f"rep{rep}_{mode}.json"), "w") as fh:
                    json.dump(r, fh, indent=1)
            row = dict(rep=int(rep), mode=mode, n_het=r["n_het"], n_het_called_het=r["n_het_called_het"],
                       switch_err=r["switch"]["errors"], switch_n=r["switch"]["n"],
                       het_entries_ok=r["entries"]["het"]["correct"], het_entries_n=r["entries"]["het"]["n"],
                       snv_entries_ok=r["entries"]["snv_homref"]["correct"], snv_entries_n=r["entries"]["snv_homref"]["n"],
                       events_ok=r["events"]["correct"], events_n=r["events"]["n"])
            for k in ("linked_same_region", "linked_diff_region", "unlinked"):
                row[f"{k}_ok"], row[f"{k}_n"] = r["pairs"][k]["correct"], r["pairs"][k]["n"]
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no results found")
        return
    if a.out:
        df.to_csv(a.out, index=False)
    f = lambda ok, n: f"{ok}/{n} = {ok / n:.3f}" if n else "n/a"
    for mode, d in df.groupby("mode"):
        s = d.sum(numeric_only=True)
        print(f"\n=== mode {mode}: {len(d)} replicates")
        print(f"  hets called heterozygous: {int(s.n_het_called_het)}/{int(s.n_het)}")
        print(f"  pairs linked, same region:       {f(int(s.linked_same_region_ok), int(s.linked_same_region_n))}")
        print(f"  pairs linked, different regions: {f(int(s.linked_diff_region_ok), int(s.linked_diff_region_n))}")
        print(f"  pairs unlinked (expected ~0.5 unless phase is given): {f(int(s.unlinked_ok), int(s.unlinked_n))}; per replicate: "
              + " ".join(f"{x:.2f}" for x in (d.unlinked_ok / d.unlinked_n)))
        print(f"  switch errors: {int(s.switch_err)}/{int(s.switch_n)}")
        print(f"  event haplotype (het entries): {f(int(s.het_entries_ok), int(s.het_entries_n))}; somatic SNV entries: {f(int(s.snv_entries_ok), int(s.snv_entries_n))}; "
              f"events with all entries right: {f(int(s.events_ok), int(s.events_n))}")


if __name__ == "__main__":
    main()
