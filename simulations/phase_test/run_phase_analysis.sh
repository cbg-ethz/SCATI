#!/bin/bash
# Scores the phase-test runs on the cluster and packs the results into one small file.
# Run from simulations/ after the runs of phase_test/Snakefile_phase are finished:
#     bash phase_test/run_phase_analysis.sh
#     CONDA_ENV=myenv bash phase_test/run_phase_analysis.sh      # other conda environment
#     ANALYSIS_OUT=analysis_out_phase2 bash phase_test/run_phase_analysis.sh
# Writes into analysis_out_phase/ and packs it into analysis_out_phase.tgz:
#     phase_summary.txt, phase_summary.csv    metrics pooled per mode, and one row per (replicate, mode)
#     metrics/rep<k>_<mode>.json              full metrics of each run
#     identifiability/rep<k>.txt              identifiable loci and pairs of each replicate
#     missing.txt                             runs without done.txt
#     results/rep<k>/SCATI/results_phase_<mode>/   small result files of each run, same layout as phase_test/data/
# Existing results are scored, nothing is rerun.
set -uo pipefail
cd "$(dirname "$0")/.."      # simulations/

if [ -z "${PYTHON:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-snakemake}"
fi
PY="${PYTHON:-python}"
OUT="${ANALYSIS_OUT:-analysis_out_phase}"
DATA="${DATA:-phase_test/data}"
mkdir -p "$OUT/metrics" "$OUT/identifiability" "$OUT/results"

$PY -c "import numpy, pandas" 2> /dev/null || { echo "python needs numpy and pandas (CONDA_ENV=... or PYTHON=...)"; exit 1; }

# runs without a result
: > "$OUT/missing.txt"
for rep_dir in "$DATA"/rep*; do
    for m in A B C; do
        [ -f "$rep_dir/SCATI/results_phase_$m/done.txt" ] || echo "$(basename "$rep_dir") $m" >> "$OUT/missing.txt"
    done
done
echo "runs without done.txt: $(wc -l < "$OUT/missing.txt") (see $OUT/missing.txt)"

# metrics
$PY phase_test/collect_phase_results.py --data "$DATA" --out "$OUT/phase_summary.csv" --json_dir "$OUT/metrics" > "$OUT/phase_summary.txt" 2> "$OUT/phase_summary.err" \
    && echo "metrics: ok -> $OUT/phase_summary.txt" || { echo "metrics: FAILED, see $OUT/phase_summary.err"; tail -5 "$OUT/phase_summary.err"; }

# identifiability report of each replicate
for rep_dir in "$DATA"/rep*; do
    r=$(basename "$rep_dir")
    $PY phase_test/identifiability_report.py --rep_dir "$rep_dir" > "$OUT/identifiability/$r.txt" 2>&1 \
        || echo "identifiability of $r: FAILED, see $OUT/identifiability/$r.txt"
done
echo "identifiability: $(ls "$OUT/identifiability" | wc -l) replicates"

# copy the small result files of each run
for d in "$DATA"/rep*/SCATI/results_phase_*; do
    [ -d "$d" ] || continue
    rel="${d#"$DATA"/}"
    mkdir -p "$OUT/results/$rel"
    for f in best_tree_strand_genotypes.json best_tree_cellAssignments.tsv best_tree.json best_tree.gv best_tree_parameters.json run_summary.json benchmark.tsv; do
        [ -f "$d/$f" ] && cp "$d/$f" "$OUT/results/$rel/"
    done
done

tar czf "$OUT.tgz" -C "$OUT" .
ls -lh "$OUT.tgz"
echo; cat "$OUT/phase_summary.txt"
