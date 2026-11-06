# SCATI

SCATI infers a mutation tree with SNVs and copy number events (local gains and losses, 
copy-neutral loss of heterozygosity (CNLOH), chromosome gains and
losses (aneuploidy), whole genome duplications) from targeted single-cell DNA sequencing data. In addition, it outputs the attachment 
of every cell to a clone or as a doublet to two clones, and the inferred germline alleles. It can analyse several samples or
time points of the same patient jointly.

SCATI builds on [COMPASS](https://github.com/cbg-ethz/COMPASS).

## Contents

| folder | content                                                                                               |
|---|-------------------------------------------------------------------------------------------------------|
| `src/` | the source code of SCATI                                                                              |
| `real_data/` | 14 AML samples: input, the results of SCATI, COMPASS and LoPhy, and the scripts for the figures       |
| `simulations/generate/` | the simulator and the converters to the input formats of COMPASS and LoPhy                            |
| `simulations/run/` | pipelines to run SCATI, COMPASS and LoPhy on the simulated datasets                                   |
| `simulations/score/` | scoring against the ground truth, and the scripts for plotting, the tables and the robustness summary |
| `simulations/phase_test/` | phasing/haplotype test: data generator, pipeline, metrics                                             |
| `simulations/example/` | one dataset generated with the default settings, with SCATI results with and without annotation       |

## Build

A C++20 compiler with OpenMP is required, for example g++ 11 or newer.

```bash
mkdir -p build
g++ -std=c++20 -O3 -fopenmp -Isrc -o build/SCATI src/SCATI.cpp src/Inference.cpp src/Node.cpp src/Scores.cpp src/input.cpp src/Tree.cpp src/Tree_moves.cpp
```

or with CMake:

```bash
cmake -S . -B build && cmake --build build
```

The scripts need Python 3 with numpy, graphviz, pandas, scipy and matplotlib, and Snakemake.
Create an environment for example with
`conda create -n scati -c conda-forge python=3.11 numpy scipy pandas matplotlib graphviz snakemake`.

## Run SCATI on the example dataset

All commands in this README are run from the repository root:

```bash
build/SCATI -i simulations/example/default/rep0/SCATI/data -o simulations/example/default/rep0/SCATI/results
```

`-i` is the input (the prefix of the dataset's files: SCATI reads `data_variants.csv`, `data_regions.csv` and `data.cell_annotation.csv` from that folder)
and `-o` is the output folder, which SCATI creates (here the `results/` folder of the example). The inferred tree is written to `results/best_tree.gv`.
To draw it as a picture run:

```bash
dot -Tpng -o simulations/example/default/rep0/SCATI/results/best_tree.png simulations/example/default/rep0/SCATI/results/best_tree.gv
```

The defaults are 4 chains of 100,000 steps each, with the healthy-cell annotation `data.cell_annotation.csv` that lies next to the data.
The dataset has 5000 cells and 200 regions, so a full run takes some time; add `--chain_length 200` for a short test run. 
The true tree is in `simulations/example/default/rep0/SCATI/ground_truth_tree.png`.


## Input

`-i <prefix>` names the input files of one sample:

* `<prefix>_variants.csv`: one row per variant.
* `<prefix>_regions.csv`: one row per region.
* `<prefix>.cell_annotation.csv`: cells with `assignment` = `healthy` are the labelled healthy cells.
  Only needed with `--use_healthy_annotation 1` (the default).

Optional inputs:
* When `<prefix>_variants_filtered.csv` or `<prefix>_regions_filtered.csv` exist, they are used instead of the plain
files.

* Several samples of one patient: `--n_samples n` reads `<prefix>_1_...` to `<prefix>_n_...` and infers one tree for
all of them; the cell attachments, region parameters and dropout are fitted per sample.

* `--use_haplotype_annotation 1` reads `<prefix>_haplotypes.tsv`, which fixes the root genotype. 
* `--region_reference <file>` gives the region read fractions and dispersions from a reference built from the healthy
  cells of other samples. Only used with `--use_healthy_annotation 0`.


## Main options

| option | default | meaning                                                                                                                               |
|---|---|---------------------------------------------------------------------------------------------------------------------------------------|
| `-i`, `-o` | | input prefix, output directory                                                                                                        |
| `--n_chains` | 4 | MCMC chains, run in parallel; the best tree over all chains is reported                                                               |
| `--chain_length` | 100000 | MCMC steps per chain                                                                                                                  |
| `--burn_in` | chain_length / 2 | steps before the best tree is recorded                                                                                                |
| `--random_seed` | 42 | seed; each chain gets its own stream                                                                                                  |
| `--sex` | female | `male` or `female`: number of chrX and chrY copies at the root                                                                        |
| `--use_doublets` | 1 | model doublets                                                                                                                        |
| `--use_healthy_annotation` | 1 | fit the region parameters and the root genotype from the labelled healthy cells; with 0 they are fitted from the cells attached to the root |
| `--refit_after_burnin` | 1 | with the healthy annotation: refit the region parameters from the root cells at the end of burn-in and stop using the labels          |
| `--node_cost` | 500 | cost of a node (higher: fewer clones)                                                                                                 |
| `--Gain_cost`, `--Loss_cost` | 500 | cost of a local gain or loss (an aneuploidy gain or loss costs 0.8x)                                                                  |
| `--CNLOH_cost` | 500 | cost of a CNLOH                                                                                                                       |
| `--FGD_cost` | 2500 | cost of a whole genome duplication                                                                                                    |
| `--target_doublet_rate` | 0.08 | expected doublet fraction                                                                                                             |
| `--dropoutrate` | 0.03 | per-copy dropout rate: prior mean of the fitted dropout, and the value used when too few healthy cells are available                  |
| `--allelic_error_rate` | 0.02 | VAF at a homozygous locus                                                                                                             |
| `--fit_region_dispersion_mle` | 0 | fit the region dispersions by maximum likelihood instead of the method of moments                                                     |
| `--cache_gb` | 4 | memory budget of the likelihood caches per chain and sample                                                                           |
| `--verbose` | 0 | print every MCMC step                                                                                                                 |


## Output

In the output directory:

* `best_tree.gv`: the tree (graphviz), with the events of each clone and its number of cells
* `best_tree.json`: the tree with all events
* `best_tree_cellAssignments.tsv`: for each cell its sample, most likely clone and whether it is more likely a doublet
* `best_tree_cellAssignmentProbs.tsv`: posterior probability of each cell for each clone and doublet
* `best_tree_strand_genotypes.json`: copy number of every region and allele copies of every locus, per strand, in every clone
* `best_tree_parameters.json`: fitted region parameters and dropout of each sample
* `run_summary.json`: run time, memory, settings and the score of each chain
* `chain_<i>/mcmc_trace.txt`: log-likelihood, log-prior and score after every step; `chain_<i>/chain_summary.json`

## COMPASS and LoPhy

The benchmarks also run [COMPASS](https://github.com/cbg-ethz/COMPASS) and [LoPhy](https://doi.org/10.1101/2025.09.16.676596), which are not part of this
repository. Clone and build them in a new folder `external/` in the repository root.

## Run on a real sample

The 14 AML samples are in `real_data/AML/<sample>/`: the filtered input (`SCATI/<sample>_{variants,regions}_filtered.csv`) and the results of SCATI, COMPASS and
LoPhy. The sex of each patient is in `real_data/AML/patients.tsv`. To run SCATI on one sample (AML-59, female):

```bash
build/SCATI -i real_data/AML/AML-59/SCATI/AML-59 -o AML-59_run --n_chains 4 --chain_length 200000 --sex female --target_doublet_rate 0.08 --cache_gb 8 --use_healthy_annotation 0 --node_cost 250 --cn_neutral_node_cost_factor 1.0 --Gain_cost 250 --Loss_cost 250 --CNLOH_cost 250 --FGD_cost 2500 --alt_to_ref_cost 250 --ref_to_alt_cost 50
```

The output is written to `AML-59_run/` (the original results are in `real_data/AML/AML-59/SCATI/results/`).

## Simulate a new dataset

```bash
python simulations/generate/generate_synthetic_data_new.py -o my_sim/data --seed 1 --ncells 5000 --nnodes 5 --nSNVs 20 --nLocalCNAs 3 --nAneuploidy 1 --nFGD 0 --nCNLOH 1 --nregions 200 --nchromosomes 10 --sex female --healthy_fraction 0.25 --doubletrate 0.08
```

This writes the input of SCATI (`my_sim/data_variants.csv`, `data_regions.csv`, `data.cell_annotation.csv`) and the ground truth (`data_treeTRUE.gv`,
`data_strand_genotypes.json`, `data_cellAssignmentsTRUE.tsv`, ...). `--help` lists all settings. Run SCATI on it as in the example above, with `-i my_sim/data`.

## Reproduce the results

* **Real data:** the results of the 14 samples are included. 
To generate the main figure, the figure of every sample and the metrics run:
  `python real_data/make_paper_figures.py --cohort real_data/AML --example AML-59 --out figures`.
  To rerun SCATI on all 14 samples (filtering and SCATI, into `real_data/rerun/`), run `cd real_data && snakemake --configfile config_aml.yaml --cores 4` (`real_data/Snakefile`);
  `real_data/run_real_slurm.sbatch` does the same on a Slurm cluster with one job per sample. COMPASS and LoPhy were run on the input made by `real_data/baseline_inputs.py` with
  `--nchains 4 --chainlength 20000 --CNA 1 --sex <sex>` (COMPASS) and `-seed 0 -iters 200` (LoPhy), otherwise with their default settings.
* **Simulation benchmark** (110 datasets; Slurm; COMPASS and LoPhy installed in `external/`):

  ```bash
  cd simulations
  sbatch --array=0-5 generate/generate_benchmark.sbatch                                                 # generates the datasets, into simulated_data/
  bash run/run_scati_slurm.sbatch > driver.log                                                          # runs SCATI, COMPASS and LoPhy on all of them
  bash score/run_analysis.sh                                                                            # metrics, main figure, tables
  ```

  The robustness runs and the phase test are described at the top of `simulations/run/Snakefile_robustness` and `simulations/phase_test/Snakefile_phase`.
