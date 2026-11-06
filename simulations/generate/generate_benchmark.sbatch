#!/bin/bash
#SBATCH --job-name=sim_generate
#SBATCH --array=0-11
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=02:00:00
#SBATCH --output=sim_generate_%A_%a.log
#
# Generates the simulated datasets, one array task per group of settings, NTESTS replicates each.
#   tasks 0-5   main benchmark into simulated_data/: one task per swept axis (its two non-default values)
#               and one for the all-defaults setting "default".
#   tasks 6-11  robustness datasets into simulated_robust/: one task per setting, each changing one
#               parameter away from the defaults: fgd100 (FGD in every dataset), dbl0 and dbl0.25
#               (doublet rate), healthy0.05 and healthy0.50 (healthy fraction), backmut20 (20% back mutations).
# The settings false25 and false50 are not generated here; run them separately, e.g.
#   python generate/run_simulation_sweep.py --output_root simulated_robust --ntests 10 --setting_name false25 --false_snv_fraction 0.25
#
# One-time environment (graphviz provides `dot` for ground_truth_tree.png):
#   conda create -n simulations -c conda-forge python=3.11 numpy scipy pandas matplotlib graphviz
#
# Run from simulations/ (about 50 MB per dataset):
#   sbatch generate/generate_benchmark.sbatch                  # all 12 tasks, 10 replicates per setting
#   sbatch --array=0-5 generate/generate_benchmark.sbatch      # main benchmark only (--array=6-11: robustness only)
#   NTESTS=3 sbatch generate/generate_benchmark.sbatch         # other number of replicates
#   MAIN_ROOT=... ROBUST_ROOT=... CONDA_ENV=... sbatch ...     # other output folders or conda environment
set -eo pipefail

NTESTS=${NTESTS:-10}
MAIN_ROOT=${MAIN_ROOT:-simulated_data}
ROBUST_ROOT=${ROBUST_ROOT:-simulated_robust}

# Output folder and run_simulation_sweep.py arguments of task i.
ROOTS=(
    "$MAIN_ROOT" "$MAIN_ROOT" "$MAIN_ROOT" "$MAIN_ROOT" "$MAIN_ROOT" "$MAIN_ROOT"
    "$ROBUST_ROOT" "$ROBUST_ROOT" "$ROBUST_ROOT" "$ROBUST_ROOT" "$ROBUST_ROOT" "$ROBUST_ROOT"
)
ARGS=(
    "--n_nodes 3 10"
    "--n_SNVs 10 50"
    "--n_CNAs 10 15"
    "--event_types SNV_only local_events"
    "--n_samples 2 3"
    ""
    "--setting_name fgd100 --fgd_probability 1.0"
    "--setting_name dbl0 --doublet_rate 0"
    "--setting_name dbl0.25 --doublet_rate 0.25"
    "--setting_name healthy0.05 --healthy_fraction 0.05"
    "--setting_name healthy0.50 --healthy_fraction 0.50"
    "--setting_name backmut20 --alt_to_ref_fraction 0.20"
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-simulations}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"      # simulations/

i=${SLURM_ARRAY_TASK_ID}
echo "task ${i}: python generate/run_simulation_sweep.py --output_root ${ROOTS[$i]} --ntests ${NTESTS} ${ARGS[$i]}"
python -u generate/run_simulation_sweep.py --output_root "${ROOTS[$i]}" --ntests "${NTESTS}" ${ARGS[$i]}
