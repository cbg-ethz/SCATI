"""Simulates the datasets for the phase test of SCATI.

Each replicate has 1 chromosome with 50 regions, 20 germline heterozygous loci with random phase, 10 somatic SNVs, 5 nodes, 5000 cells,
4 local gains/losses and 1 CNLOH. There is no aneuploidy and no FGD. All other settings are those of the simulation sweep
(run_simulation_sweep.FIXED_KWARGS).

    python phase_test/make_phase_data.py --out phase_test/data --reps 10

Writes <out>/rep<k>/SCATI/data_*. The truth is in data_strand_genotypes.json, data_events_ordered.json,
data_germline_het_loci.csv and data_true_model_params.json. data_haplotypes.tsv holds the true haplotypes, the input of the control run C.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generate"))   # simulation scripts
from generate_synthetic_data_new import generate_data
import run_simulation_sweep as sweep
from phase_common import Truth, write_haplotype_file

N_CELLS, N_NODES, N_SNVS, N_HET, N_REGIONS = 5000, 5, 10, 20, 50
N_LOCAL_CNAS, N_CNLOH = 4, 1


def make(out, rep, base_seed=20261103):
    seed = sweep.stable_seed(base_seed, "phase_test", rep)
    rng = np.random.default_rng(seed + 7919)               # gain/loss split as in the sweep (draw_event_counts)
    n_gain = int(rng.binomial(N_LOCAL_CNAS, rng.uniform(*sweep.GAIN_FRACTION_RANGE)))
    rep_dir = os.path.join(out, f"rep{rep}")
    kwargs = dict(sweep.FIXED_KWARGS)
    kwargs.update(
        basename=os.path.join(rep_dir, "SCATI", "data"),
        n_cells=N_CELLS, n_nodes=N_NODES, n_SNVs=N_SNVS, n_false_SNVs=0,
        n_local_CNAs=N_LOCAL_CNAS, n_local_gain=n_gain, n_local_loss=N_LOCAL_CNAS - n_gain,
        n_aneuploidy=0, n_aneuploidy_gain=0, n_aneuploidy_loss=0, n_FGD=0, n_CNLOH=N_CNLOH,
        region_calibration=sweep.load_region_calibration(),
        mean_total_count=sweep.draw_mean_total_count(seed),
        n_regions=N_REGIONS, chromosome_sizes=[N_REGIONS], sex="female", healthy_fraction=0.25,
        n_het=N_HET, random_phase=True, seed=seed,
        max_ordering_attempts=5000)      # the events often conflict on one chromosome, so many attempts are needed
    generate_data(**kwargs)
    write_haplotype_file(Truth(rep_dir), os.path.join(rep_dir, "SCATI", "data_haplotypes.tsv"))
    return rep_dir, seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--first_rep", type=int, default=0)
    a = ap.parse_args()
    for r in range(a.first_rep, a.first_rep + a.reps):
        d, s = make(a.out, r)
        print(f"rep{r}: seed {s} -> {d}")
