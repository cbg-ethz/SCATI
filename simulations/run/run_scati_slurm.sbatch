#!/bin/bash
#SBATCH --job-name=sim_driver
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=120:00:00
#SBATCH --output=sim_driver_%j.log
#
# Driver: a light snakemake process that submits each inference job (SCATI with and without annotation, COMPASS_snp, LoPhy)
# as its own Slurm job, with cores, memory and runtime from run/Snakefile_run_methods.
# Setup in the snakemake env: pip install snakemake-executor-plugin-slurm
#
# Run it on the login node (in tmux or screen), from simulations/, not as a Slurm job:
#   LOPHY_CONDA_ENV=lophy bash run/run_scati_slurm.sbatch > driver.log 2>&1                          # main benchmark
#   LOPHY_CONDA_ENV=lophy SIM_ROOT=simulated_robust bash run/run_scati_slurm.sbatch > driver.log 2>&1
#   SNAKEFILE=run/Snakefile_robustness TARGET=robust_annot bash run/run_scati_slurm.sbatch > driver.log 2>&1   # annotation errors
#   SNAKEFILE=phase_test/Snakefile_phase TARGET=phase_all SIM_ROOT=phase_test bash run/run_scati_slurm.sbatch      # phase test
#
# Environment variables (snakemake arguments after the script name are passed on):
#   SCATI_CHAIN_LENGTH=300 COMPASS_CHAIN_LENGTH=300 LOPHY_ITERS=2   short test run
#   JOBS=60                       maximum number of jobs in the queue
#   SCATI_EXTRA_ARGS="--cache_gb 6"   extra SCATI flags
#   SCATI_MEM_MB=32000            memory in MB of a one-sample SCATI job (default 24000, +20000 per further sample)
#   SIM_SLURM_ACCOUNT=<acct>      Slurm account for the jobs (default: none given)
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-snakemake}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"      # simulations/, the data folders are relative to it

ACCOUNT_ARG=(--slurm-no-account)
[ -n "${SIM_SLURM_ACCOUNT:-}" ] && ACCOUNT_ARG=(--slurm-account "${SIM_SLURM_ACCOUNT}")

# Memory (MB) and runtime (min) for rules that set none.
snakemake -s "${SNAKEFILE:-run/Snakefile_run_methods}" "${TARGET:-all}" \
    --executor slurm --jobs "${JOBS:-50}" --latency-wait 120 \
    --default-resources mem_mb=4000 runtime=60 \
    "${ACCOUNT_ARG[@]}" \
    --config compare_python="$(which python)" \
        scati_chain_length="${SCATI_CHAIN_LENGTH:-100000}" \
        compass_chain_length="${COMPASS_CHAIN_LENGTH:-20000}" \
        lophy_iters="${LOPHY_ITERS:-200}" \
        scati_mem_mb="${SCATI_MEM_MB:-24000}" \
        scati_extra_args="${SCATI_EXTRA_ARGS:-}" \
        sim_root="${SIM_ROOT:-simulated_data}" \
        sim_sex="${SIM_SEX:-female}" \
        scati_node_cost="${SCATI_NODE_COST:-500}" \
        scati_doublet_omega="${SCATI_DOUBLET_OMEGA:-}" \
        scati_doublet_rate="${SCATI_DOUBLET_RATE:-0.08}" \
        compass_node_cost="${COMPASS_NODE_COST:-50}" \
        compass_doublet_rate="${COMPASS_DOUBLET_RATE:-0.08}" \
        compass_extra_args="${COMPASS_EXTRA_ARGS:-}" \
        lophy_conda_env="${LOPHY_CONDA_ENV:-}" \
    --keep-going --rerun-incomplete "$@"
