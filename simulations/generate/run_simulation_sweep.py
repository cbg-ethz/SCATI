"""
Settings-sweep driver for generate_synthetic_data_new.py.

Each setting changes one axis (nodes, SNVs, CNAs, event types, samples) away from its
default, with --ntests random replicates. Every dataset is also converted to the
other methods' input formats via convert_to_other_methods.py.

Defaults: 5 nodes, 20 SNVs, 5 CNAs, local_and_large_scale_events, 1 sample, 5000 cells,
200 regions on 10 chromosomes, female, 25% healthy cells, doublet rate 0.08, noise weight 1%.

Example:
    python generate/run_simulation_sweep.py --n_SNVs 10 20 50 --event_types SNV_only local_events --ntests 5

Output, with settings named e.g. default, nodes10, SNVs50, SNV_only, 2samples:
    <output_root>/<setting>/rep{i}/{SCATI,COMPASS,COMPASS_snp,LoPhy}/
        input files, ground_truth_tree.gv/.png, empty results/
The seed depends on --base_seed, the setting name and the replicate index. Ground truth other than the tree is kept only in SCATI/.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

from generate_synthetic_data_new import generate_data, n_false_snvs_for
from convert_to_other_methods import convert_all, METHOD_DIRNAMES
from make_compass_snp_inputs import convert_rep as write_compass_snp_inputs


# Parameters that are not sweep axes.
FIXED_KWARGS = dict(
    allow_empty_regions=True,
    root_het_fraction=0.2,
    root_hom_alt_fraction=0.05,
    alt_to_ref_fraction=0.0,     # fraction of real SNVs that are alt->ref back mutations
    nodeprobs_concentration=10.0,
    dropout_rate=0.03,
    region_dispersion_min=50.0,      # used only without the RCC calibration file
    region_dispersion_max=2000.0,
    region_weight_sigma=0.3,
    noise_weight=0.01,           # share of region counts drawn from the outlier component
    background_rate=0.04,        # equals the fixed value in the C++ code
    mean_total_count=50000.0,        # overridden per dataset by draw_mean_total_count
    theta_total_count=10.0,
    mean_snv_depth=100.0,
    theta_snv_depth=10.0,
    doublet_rate=0.08,           # also the target doublet rate given to SCATI and COMPASS
    root_cycling_prob=0.0,       # no cycling cells
    cycling_prob_min=0.0,
    cycling_prob_max=0.0,
    cycling_cell_factor=2.0,
    allelic_error_rate=0.02,  # equals the fixed value in the C++ code
    max_ordering_attempts=200,
)

# Random copy-number event mix, drawn per dataset by draw_event_counts.
# Local events are n_CNAs gains/losses plus CNLOH; large-scale events are aneuploidy and FGD.
# SNV_only has no copy-number events, so n_CNAs is ignored.
FALSE_SNV_FRACTION = 0.0          # share of all loci that are false (never mutated)
FGD_PROBABILITY = 0.0             # share of datasets with a whole-genome duplication
ANEUPLOIDY_PER_LOCAL_CNA = 0.5    # Poisson mean of the aneuploidy count = this * n_CNAs
CNLOH_MEAN = 0.7                  # Poisson mean of the CNLOH count
GAIN_FRACTION_RANGE = (0.2, 0.8)  # uniform range of P(event is a gain), drawn per dataset for local and aneuploidy events

EVENT_TYPE_CHOICES = ["SNV_only", "local_events", "local_and_large_scale_events"]
METHOD_CHOICES = list(METHOD_DIRNAMES.keys())

# Value of each axis when it is not the swept one.
DEFAULT_N_NODES = 5
DEFAULT_N_SNVS = 20
DEFAULT_N_CNAS = 5
DEFAULT_EVENT_TYPE = "local_and_large_scale_events"
DEFAULT_N_SAMPLES = 1


def stable_seed(base_seed, *parts):
    """Deterministic seed from base_seed and the identifying parts of a dataset."""
    h = hashlib.sha256("|".join(str(p) for p in (base_seed,) + parts).encode()).hexdigest()
    return base_seed + int(h[:8], 16) % 1_000_000


MIN_REGIONS_PER_CHROMOSOME = 10
EQUAL_REGION_WEIGHTS = False       # no depth variation between regions
DEFAULT_SETTING_NAME = "default"   # folder name of the all-defaults setting


def random_chromosome_sizes(n_regions, n_chromosomes, seed):
    """Regions per chromosome: a minimum each, the rest spread by Dirichlet(5) weights.
    Depends only on `seed`, so all datasets of a sweep share one panel."""
    import numpy as np
    rng = np.random.default_rng(seed)
    spare = n_regions - MIN_REGIONS_PER_CHROMOSOME * n_chromosomes
    if spare < 0:
        raise ValueError(f"{n_regions} regions cannot give {n_chromosomes} chromosomes "
                         f"{MIN_REGIONS_PER_CHROMOSOME} regions each")
    extra = rng.multinomial(spare, rng.dirichlet(np.full(n_chromosomes, 5.0)))
    return [MIN_REGIONS_PER_CHROMOSOME + int(e) for e in extra]


AXIS_DEFAULTS = dict(n_nodes=DEFAULT_N_NODES, n_SNVs=DEFAULT_N_SNVS, n_CNAs=DEFAULT_N_CNAS,
                     event_type=DEFAULT_EVENT_TYPE, n_samples=DEFAULT_N_SAMPLES)


def load_region_calibration():
    """Per-region (relative weight, dispersion) pairs from rcc_region_calibration.json, or None if the file is missing."""
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rcc_region_calibration.json")
    if not os.path.isfile(path):
        print(f"WARNING: {path} not found -- using the synthetic region weight/dispersion draws")
        return None
    with open(path) as f:
        return json.load(f)


# Range of summed region counts per diploid cell, as in the real RCC samples.
MEAN_TOTAL_COUNT_RANGE = (30000.0, 85000.0)


def draw_mean_total_count(seed):
    """Per-dataset mean total count, log-uniform over MEAN_TOTAL_COUNT_RANGE."""
    import numpy as np
    rng = np.random.default_rng(seed + 104729)
    return float(np.exp(rng.uniform(np.log(MEAN_TOTAL_COUNT_RANGE[0]), np.log(MEAN_TOTAL_COUNT_RANGE[1]))))


def draw_event_counts(seed, n_CNAs, is_snv_only, is_large_scale):
    """Random copy-number event mix as kwargs for generate_data.
    The n_CNAs local events are gains with a per-dataset probability from GAIN_FRACTION_RANGE.
    Large-scale settings add Poisson aneuploidies and an FGD with probability FGD_PROBABILITY.
    Every setting with copy-number events adds Poisson(CNLOH_MEAN) CNLOH events."""
    import numpy as np
    rng = np.random.default_rng(seed + 7919)   # separate stream from generate_data's
    counts = dict(n_local_CNAs=0, n_local_gain=0, n_local_loss=0, n_aneuploidy=0,
                  n_aneuploidy_gain=0, n_aneuploidy_loss=0, n_FGD=0, n_CNLOH=0)
    if is_snv_only:
        return counts
    n_gain = int(rng.binomial(n_CNAs, rng.uniform(*GAIN_FRACTION_RANGE)))
    counts.update(n_local_CNAs=n_CNAs, n_local_gain=n_gain, n_local_loss=n_CNAs - n_gain)
    if is_large_scale:
        n_aneuploidy = int(rng.poisson(ANEUPLOIDY_PER_LOCAL_CNA * n_CNAs))
        n_again = int(rng.binomial(n_aneuploidy, rng.uniform(*GAIN_FRACTION_RANGE)))
        counts.update(n_aneuploidy=n_aneuploidy, n_aneuploidy_gain=n_again,
                      n_aneuploidy_loss=n_aneuploidy - n_again,
                      n_FGD=int(rng.random() < FGD_PROBABILITY))
    counts["n_CNLOH"] = int(rng.poisson(CNLOH_MEAN))
    return counts


def setting_name(axis, value):
    """Folder name of the setting that sets `axis` to `value`, e.g. "nodes10" or "2samples"; axis None gives the default name."""
    if axis is None:
        return DEFAULT_SETTING_NAME
    return {
        "n_nodes": lambda v: f"nodes{v}",
        "n_SNVs": lambda v: f"SNVs{v}",
        "n_CNAs": lambda v: f"CNAs{v}",
        "event_type": lambda v: str(v),
        "n_samples": lambda v: f"{v}sample" + ("s" if v != 1 else ""),
    }[axis](value)


def build_jobs(n_nodes_options, n_SNVs_options, n_CNAs_options, event_types,
               n_samples_options, n_tests, base_seed, output_root,
               n_cells, n_regions, n_chromosomes, sex, healthy_fraction):
    """Return (rep_dir, seed, generate_data_kwargs) jobs, n_tests replicates per setting.
    An axis whose options differ from just its default is swept with all other axes at their defaults.
    With no swept axis there is one default setting. Does not generate data."""
    options = dict(n_nodes=n_nodes_options, n_SNVs=n_SNVs_options, n_CNAs=n_CNAs_options,
                   event_type=event_types, n_samples=n_samples_options)
    settings = []  # (axis, value); axis None means all defaults
    for axis, opts in options.items():
        if list(opts) != [AXIS_DEFAULTS[axis]]:
            settings.extend((axis, v) for v in opts)
    if not settings:
        settings = [(None, None)]

    chromosome_sizes = random_chromosome_sizes(n_regions, n_chromosomes, base_seed)
    region_calibration = None if EQUAL_REGION_WEIGHTS else load_region_calibration()

    jobs = []
    for axis, value in settings:
        combo = dict(AXIS_DEFAULTS)
        if axis is not None:
            combo[axis] = value
        n_nodes, n_SNVs, n_CNAs = combo["n_nodes"], combo["n_SNVs"], combo["n_CNAs"]
        event_type, n_samples = combo["event_type"], combo["n_samples"]
        is_large_scale = (event_type == "local_and_large_scale_events")
        is_snv_only = (event_type == "SNV_only")
        setting = setting_name(axis, value)

        for rep in range(n_tests):
            # The all-defaults seed ignores --setting_name, so renamed default runs draw the same data.
            seed = stable_seed(base_seed, "default" if axis is None else setting, n_nodes, n_SNVs, n_CNAs, event_type, n_samples, rep)
            rep_dir = os.path.join(output_root, setting, f"rep{rep}")
            kwargs = dict(FIXED_KWARGS)
            kwargs.update(
                basename=os.path.join(rep_dir, "SCATI", "data"),
                n_cells=n_cells,
                n_nodes=n_nodes,
                n_SNVs=n_SNVs,
                n_false_SNVs=n_false_snvs_for(n_SNVs, FALSE_SNV_FRACTION),
                **draw_event_counts(seed, n_CNAs, is_snv_only, is_large_scale),
                _redraw=dict(n_CNAs=n_CNAs, is_snv_only=is_snv_only, is_large_scale=is_large_scale),
                region_calibration=region_calibration,
                mean_total_count=draw_mean_total_count(seed),
                n_regions=n_regions,
                chromosome_sizes=chromosome_sizes,
                sex=sex,
                healthy_fraction=healthy_fraction,
                n_samples=n_samples,
                seed=seed,
            )
            jobs.append((rep_dir, seed, kwargs))
    return jobs


def _propagate_ground_truth_tree(rep_dir, method_dirs):
    """Copy the ground-truth tree into every method folder and render a PNG if `dot` is available."""
    src_gv = os.path.join(rep_dir, "SCATI", "data_treeTRUE.gv")
    if not os.path.isfile(src_gv):
        return
    dot = shutil.which("dot")
    for method_dir in method_dirs:
        if method_dir == "SCATI":
            dst_gv = src_gv
        else:
            dst_gv = os.path.join(rep_dir, method_dir, "data_treeTRUE.gv")
            os.makedirs(os.path.dirname(dst_gv), exist_ok=True)
            shutil.copyfile(src_gv, dst_gv)
        if dot:
            png_path = os.path.join(os.path.dirname(dst_gv), "ground_truth_tree.png")
            try:
                subprocess.run([dot, "-Tpng", dst_gv, "-o", png_path], check=True,
                               capture_output=True)
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"  (skipping ground_truth_tree.png for {method_dir}: {e})")


MAX_EVENT_REDRAWS = 20


def generate_with_redraw(seed, kwargs):
    """Call generate_data; if the event mix cannot be ordered, redraw it with a derived seed, up to MAX_EVENT_REDRAWS times.
    Returns the seed used."""
    kwargs = dict(kwargs)
    redraw = kwargs.pop("_redraw")
    for k in range(MAX_EVENT_REDRAWS + 1):
        try:
            generate_data(**kwargs)
            return kwargs["seed"]
        except RuntimeError as e:
            if k == MAX_EVENT_REDRAWS:
                raise
            new_seed = seed + 999983 * (k + 1)
            print(f"  REDRAW {k + 1}: event mix could not be ordered ({str(e).split('.')[0]}); "
                  f"redrawing with seed {new_seed}")
            kwargs.update(seed=new_seed, **draw_event_counts(new_seed, **redraw))


def run_jobs(jobs, methods=METHOD_CHOICES):
    for i, (rep_dir, seed, kwargs) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}] {rep_dir} (seed={seed})")
        generate_with_redraw(seed, kwargs)

        n_samples = kwargs.get("n_samples", 1)
        convert_all(kwargs["basename"], rep_dir, n_samples=n_samples, methods=methods)

        method_dirs = ["SCATI"] + [METHOD_DIRNAMES[m] for m in methods]
        if "compass" in methods:     # COMPASS_snp: COMPASS with germline variants as population frequencies
            write_compass_snp_inputs(rep_dir, force=True)
            method_dirs.append("COMPASS_snp")
        _propagate_ground_truth_tree(rep_dir, method_dirs)

        # Compare observed read counts and VAF with the ground truth, written to SCATI/simulation_check_*.png and .json.
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "score"))
            from method_adapters import generate_simulation_check
            generate_simulation_check(rep_dir)
        except Exception as e:  # plot errors must not stop data generation
            print(f"  (skipping simulation check plots for {rep_dir}: {e})")

        # Empty results/ folders for the inference output.
        for method_dir in method_dirs:
            os.makedirs(os.path.join(rep_dir, method_dir, "results"), exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_root', type=str, default="simulated_data")
    parser.add_argument('--base_seed',   type=int, default=1)
    parser.add_argument('--ntests',      type=int, default=5,
                        help="Replicates per setting.")
    parser.add_argument('--n_nodes',     type=int, nargs='+', default=[DEFAULT_N_NODES])
    parser.add_argument('--n_SNVs',      type=int, nargs='+', default=[DEFAULT_N_SNVS])
    parser.add_argument('--n_CNAs',      type=int, nargs='+', default=[DEFAULT_N_CNAS])
    parser.add_argument('--event_types', type=str, nargs='+', default=[DEFAULT_EVENT_TYPE],
                        choices=EVENT_TYPE_CHOICES,
                        help="SNV_only: no copy-number events. "
                             f"local_events: n_CNAs gains/losses plus CNLOH (mean {CNLOH_MEAN}). "
                             "local_and_large_scale_events: also aneuploidies "
                             f"(mean {ANEUPLOIDY_PER_LOCAL_CNA} x n_CNAs) and FGD (see --fgd_probability).")
    parser.add_argument('--false_snv_fraction', type=float, default=0.0,
                        help="Share of loci that are false (never mutated).")
    parser.add_argument('--fgd_probability', type=float, default=0.0,
                        help="Share of datasets with a whole-genome duplication (only with local_and_large_scale_events).")
    parser.add_argument('--n_samples',   type=int, nargs='+', default=[DEFAULT_N_SAMPLES],
                        help="Samples (time points) per dataset.")
    parser.add_argument('--n_cells',     type=int, default=5000,
                        help="Cells per sample.")
    parser.add_argument('--n_regions',      type=int, default=200)
    parser.add_argument('--n_chromosomes',  type=int, default=10,
                        help="Chromosomes 1..n-1 plus chrX (haploid if male); no chrY.")
    parser.add_argument('--min_regions_per_chromosome', type=int, default=10,
                        help="Minimum regions per chromosome.")
    parser.add_argument('--setting_name', type=str, default="default",
                        help="Folder name of the all-defaults setting; does not change the seeds.")
    parser.add_argument('--doublet_rate', type=float, default=0.08,
                        help="Fraction of doublet cells.")
    parser.add_argument('--equal_region_weights', action='store_true',
                        help="Equal expected read share per region instead of the RCC-calibrated weights and dispersions.")
    parser.add_argument('--sex', type=str, default="female", choices=["male", "female"])
    parser.add_argument('--alt_to_ref_fraction', type=float, default=0.0,
                        help="Fraction of real SNVs that are alt->ref back mutations.")
    parser.add_argument('--noise_weight', type=float, default=0.01,
                        help="Share of region counts drawn from the outlier component.")
    parser.add_argument('--healthy_fraction', type=float, default=0.25,
                        help="Fraction of healthy cells before doublets are added.")
    parser.add_argument('--dry_run', action='store_true',
                        help="Print the job list without generating data.")
    parser.add_argument('--methods', type=str, nargs='*', default=["compass", "lophy"],
                        choices=METHOD_CHOICES,
                        help="Other methods to write inputs for (compass also writes COMPASS_snp); pass no values to write only SCATI.")
    args = parser.parse_args()
    MIN_REGIONS_PER_CHROMOSOME = args.min_regions_per_chromosome
    FIXED_KWARGS['doublet_rate'] = args.doublet_rate
    FGD_PROBABILITY = args.fgd_probability
    FALSE_SNV_FRACTION = args.false_snv_fraction
    FIXED_KWARGS['alt_to_ref_fraction'] = args.alt_to_ref_fraction
    FIXED_KWARGS['noise_weight'] = args.noise_weight
    DEFAULT_SETTING_NAME = args.setting_name
    EQUAL_REGION_WEIGHTS = args.equal_region_weights
    if EQUAL_REGION_WEIGHTS:
        FIXED_KWARGS.update(region_weight_sigma=0.0, region_dispersion_min=2000.0, region_dispersion_max=2000.0)

    jobs = build_jobs(
        n_nodes_options=args.n_nodes,
        n_SNVs_options=args.n_SNVs,
        n_CNAs_options=args.n_CNAs,
        event_types=args.event_types,
        n_samples_options=args.n_samples,
        n_tests=args.ntests,
        base_seed=args.base_seed,
        output_root=args.output_root,
        n_cells=args.n_cells,
        n_regions=args.n_regions,
        n_chromosomes=args.n_chromosomes,
        sex=args.sex,
        healthy_fraction=args.healthy_fraction,
    )

    print(f"Prepared {len(jobs)} simulation jobs under {args.output_root}")
    if args.dry_run:
        for rep_dir, seed, _ in jobs:
            print(f"  {rep_dir} (seed={seed})")
    else:
        run_jobs(jobs, methods=args.methods)
