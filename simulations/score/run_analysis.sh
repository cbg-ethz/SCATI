#!/bin/bash
# Scores the finished runs on the cluster, then makes the main figure and the tables. Nothing is rerun.
# From simulations/:
#     bash score/run_analysis.sh                       # scores simulated_data (SCATI with/without annotation, COMPASS_snp, LoPhy) and simulated_robust
#     CONDA_ENV=myenv bash score/run_analysis.sh       # other conda environment
#     ROBUST_ROOT=none bash score/run_analysis.sh      # main sweep only
# Writes analysis_out/sweep/ (metrics.csv, summary.md, figures/), analysis_out/robust/, analysis_out/tables/
# and analysis_out.tgz with all of it, the one file to download.
set -uo pipefail
cd "$(dirname "$0")/.."      # simulations/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-snakemake}"

OUT="${ANALYSIS_OUT:-analysis_out}"
SWEEP_ROOT="${SWEEP_ROOT:-simulated_data}"
mkdir -p "$OUT"

# run <name> <sim_root> <methods...>
run() {
    local name=$1 root=$2
    shift 2
    if [ ! -d "$root" ]; then echo "$name: skipped ($root not found)"; return; fi
    if python score/analyze_simulation_results.py --sim_root "$root" --out "$OUT/$name" --methods "$@" > "$OUT/$name.log" 2>&1; then
        echo "$name: ok -> $OUT/$name/summary.md"
    else
        echo "$name: FAILED, see $OUT/$name.log"; tail -5 "$OUT/$name.log"
    fi
}

run sweep "$SWEEP_ROOT" ${SWEEP_METHODS:-scati scati_noann compass_snp lophy}

# Robustness: scores the runs in ROBUST_ROOT and the SCATI annotation-error runs on the default datasets of the main sweep.
# ROBUST_ROOT=none skips it. The robustness summary and conditions.csv are written by the tables step below.
ROBUST_ROOT="${ROBUST_ROOT:-simulated_robust}"
if [ -d "$ROBUST_ROOT" ]; then
    mkdir -p "$OUT/robust"
    ANNOT_METHODS="scati_annot_cont30 scati_annot_cont50 scati_annot_h2c30 scati_annot_h2c50 scati_annot_cont30_h2c30 scati_annot_cont50_h2c50"
    {
        python score/simulation_metrics.py --sim_root "$ROBUST_ROOT" --out "$OUT/robust/metrics.csv" --methods ${ROBUST_METHODS:-scati scati_noann compass_snp lophy}
        python score/simulation_metrics.py --sim_root "$SWEEP_ROOT" --rep_dir "$SWEEP_ROOT"/default/rep* --out "$OUT/robust/metrics_annot.csv" --methods $ANNOT_METHODS
    } > "$OUT/robust.log" 2>&1 && echo "robust: ok -> $OUT/robust/metrics.csv, metrics_annot.csv" || { echo "robust: FAILED, see $OUT/robust.log"; tail -5 "$OUT/robust.log"; }
fi

# Main figure (fig_simulation_benchmark.pdf) from the metrics of the main benchmark
if [ -f "$OUT/sweep/metrics.csv" ]; then
    python score/make_main_figure.py --metrics "$OUT/sweep/metrics.csv" --out "$OUT/sweep/figures" > "$OUT/figure.log" 2>&1 \
        && echo "figure: ok -> $OUT/sweep/figures" || { echo "figure: FAILED, see $OUT/figure.log"; tail -3 "$OUT/figure.log"; }
fi

# LaTeX tables of the benchmark (printed to $OUT/tables.log and written to $OUT/tables/); also writes $OUT/robust/conditions.csv and summary.md
python score/make_benchmark_tables.py --analysis "$OUT" --out "$OUT/tables" --sim_roots "$SWEEP_ROOT" "$ROBUST_ROOT" > "$OUT/tables.log" 2>&1 && echo "tables: ok -> $OUT/tables" || { echo "tables: FAILED, see $OUT/tables.log"; tail -3 "$OUT/tables.log"; }

tar czf "$OUT.tgz" -C "$OUT" .
ls -lh "$OUT.tgz"
