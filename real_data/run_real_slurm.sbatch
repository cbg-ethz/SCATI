#!/bin/bash
#SBATCH --job-name=real_driver
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=120:00:00
#SBATCH --output=real_driver_%j.log

set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-snakemake}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

ACCOUNT_ARG=(--slurm-no-account)
[ -n "${SIM_SLURM_ACCOUNT:-}" ] && ACCOUNT_ARG=(--slurm-account "${SIM_SLURM_ACCOUNT}")

snakemake --configfile config_aml.yaml \
    --executor slurm --jobs "${JOBS:-14}" --latency-wait 120 \
    --default-resources mem_mb=4000 runtime=60 \
    "${ACCOUNT_ARG[@]}" \
    --config python="${PYTHON:-python}" \
    --keep-going --rerun-incomplete "$@"
