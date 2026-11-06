"""
Converts one simulated dataset (as written by generate_synthetic_data_new.py) into
the input formats of COMPASS and LoPhy.

Usage: python convert_to_other_methods.py <basename> [--out_root DIR] [--n_samples N]
Reads <basename>_variants.csv, _regions.csv, _genotypes.csv, _n_alt_root.csv,
_events_ordered.json and _true_model_params.json (multi-sample: <basename>_1, _2, ...).
Writes <out_root>/COMPASS/ and <out_root>/LoPhy/.
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd


def _sample_basename(basename, n_samples, s):
    # Multi-sample files are named {basename}_1, {basename}_2, ... (1-based).
    return basename if n_samples == 1 else f"{basename}_{s + 1}"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _read_variants(sample_basename):
    """Return (loci_df, ref_reads, alt_reads, cell_cols) for one sample.
    loci_df has columns CHR, REGION, NAME; the read arrays have shape (n_loci, n_cells)."""
    df = pd.read_csv(sample_basename + "_variants.csv")
    cell_cols = [c for c in df.columns if c not in ("CHR", "REGION", "NAME")]
    n_loci = len(df)
    n_cells = len(cell_cols)
    ref_reads = np.zeros((n_loci, n_cells), dtype=int)
    alt_reads = np.zeros((n_loci, n_cells), dtype=int)
    for j, c in enumerate(cell_cols):
        parts = df[c].str.split(":", expand=True)
        ref_reads[:, j] = parts[0].astype(int)
        alt_reads[:, j] = parts[1].astype(int)
    loci_df = df[["CHR", "REGION", "NAME"]].copy()
    return loci_df, ref_reads, alt_reads, cell_cols


def _read_regions(sample_basename):
    """Return (region_names, depths) from _regions.csv.
    region_names are the "<chr>_<name>" row labels; depths has shape (n_regions, n_cells)."""
    df = pd.read_csv(sample_basename + "_regions.csv", header=None, index_col=0)
    return list(df.index), df.values.astype(int)


def _region_suffix(region_name):
    """Strip the "<chr>_" prefix from a region name."""
    idx = region_name.find("_")
    return region_name[idx + 1:] if idx != -1 else region_name


def _read_genotypes(sample_basename, n_loci, n_cells):
    """Return the (n_loci, n_cells) array of genotype calls in {0,1,2,3}."""
    return np.loadtxt(sample_basename + "_genotypes.csv", dtype=int).reshape(n_loci, n_cells)


def _character_calls(genotypes):
    """Collapse {0,1,2,3} genotype calls to LoPhy's {0,1,-1} convention."""
    calls = genotypes.copy()
    calls[calls == 2] = 1
    calls[calls == 3] = -1
    return calls


# ---------------------------------------------------------------------------
# COMPASS
# ---------------------------------------------------------------------------

def write_compass_format(sample_basename, out_dir):
    """Copy <basename>_variants.csv and _regions.csv to data_variants.csv and data_regions.csv in out_dir.
    The simulator already writes the COMPASS format (CHR,REGION,NAME,<cells> header, ref:alt cell values)."""
    variants_path = sample_basename + "_variants.csv"
    regions_path = sample_basename + "_regions.csv"
    if not os.path.exists(variants_path):
        raise FileNotFoundError(variants_path)
    if not os.path.exists(regions_path):
        raise FileNotFoundError(regions_path)
    df = pd.read_csv(variants_path)
    for required in ("CHR", "REGION", "NAME"):
        if required not in df.columns:
            raise ValueError(f"{variants_path} missing required column {required!r}")
    region_names, depths = _read_regions(sample_basename)
    n_cells_variants = len(df.columns) - 3
    if depths.shape[1] != n_cells_variants:
        raise ValueError(
            f"{variants_path} has {n_cells_variants} cells but {regions_path} has "
            f"{depths.shape[1]} -- cell counts must match.")

    _ensure_dir(out_dir)
    shutil.copyfile(variants_path, os.path.join(out_dir, "data_variants.csv"))
    shutil.copyfile(regions_path, os.path.join(out_dir, "data_regions.csv"))


# ---------------------------------------------------------------------------
# LoPhy
# ---------------------------------------------------------------------------

def _locus_birth_sample(basename):
    """Return (birth_sample per node, {SNV locus: node that introduces it}).
    Reads _true_model_params.json and _events_ordered.json."""
    with open(basename + "_true_model_params.json") as f:
        true_params = json.load(f)
    birth_sample = true_params["birth_sample"]

    with open(basename + "_events_ordered.json") as f:
        node_events = json.load(f)

    locus_to_node = {}
    for node_str, events in node_events.items():
        for ev in events:
            if ev["type"] == "SNV":
                locus_to_node[ev["locus"]] = int(node_str)
    return birth_sample, locus_to_node


def write_lophy_format(basename, out_dir, n_samples=1):
    """Write LoPhy's character_matrix, variant_reads, total_reads, region_reads, cell_samples and meta files.
    Cells of all samples are concatenated and renamed s<sample>_<cell>; loci and regions are shared."""
    sample_basenames = [_sample_basename(basename, n_samples, s) for s in range(n_samples)]

    all_loci_df = None
    all_ref, all_alt = [], []
    all_region_names = None
    all_depths = []
    all_cell_names = []
    all_cell_samples = []

    for s, sb in enumerate(sample_basenames):
        loci_df, ref_reads, alt_reads, cell_cols = _read_variants(sb)
        region_names, depths = _read_regions(sb)
        if all_loci_df is None:
            all_loci_df = loci_df
            all_region_names = region_names
        all_ref.append(ref_reads)
        all_alt.append(alt_reads)
        all_depths.append(depths)
        renamed = [f"s{s}_{c}" for c in cell_cols]
        all_cell_names.extend(renamed)
        all_cell_samples.extend([s] * len(renamed))

    ref_reads = np.concatenate(all_ref, axis=1)
    alt_reads = np.concatenate(all_alt, axis=1)
    depths = np.concatenate(all_depths, axis=1)
    n_loci, n_cells = ref_reads.shape

    # Genotype calls of every sample, concatenated over cells.
    calls_parts = []
    for s, sb in enumerate(sample_basenames):
        n_cells_s = all_ref[s].shape[1]
        calls_parts.append(_character_calls(_read_genotypes(sb, n_loci, n_cells_s)))
    calls = np.concatenate(calls_parts, axis=1)

    mutation_names = all_loci_df["NAME"].tolist()
    _ensure_dir(out_dir)
    # LoPhy does not strip '\r' from column names, so all files are written with '\n' line endings.
    pd.DataFrame(calls.T, index=all_cell_names, columns=mutation_names).to_csv(
        os.path.join(out_dir, "character_matrix.csv"), lineterminator="\n")
    pd.DataFrame(alt_reads.T, index=all_cell_names, columns=mutation_names).to_csv(
        os.path.join(out_dir, "variant_reads.csv"), lineterminator="\n")
    pd.DataFrame((ref_reads + alt_reads).T, index=all_cell_names, columns=mutation_names).to_csv(
        os.path.join(out_dir, "total_reads.csv"), lineterminator="\n")
    pd.DataFrame(depths.T, index=all_cell_names, columns=all_region_names).to_csv(
        os.path.join(out_dir, "region_reads.csv"), lineterminator="\n")

    with open(os.path.join(out_dir, "cell_samples.txt"), "w", newline="\n") as f:
        for s in all_cell_samples:
            f.write(f"{s}\n")

    # meta.csv: SNP = locus present at the root; SAMPLE = birth sample of the node introducing the locus (0 for SNPs).
    n_alt_root = pd.read_csv(basename + "_n_alt_root.csv", header=None)[0].values
    region_index = {_region_suffix(name): k for k, name in enumerate(all_region_names)}
    birth_sample, locus_to_node = _locus_birth_sample(basename)

    meta_rows = []
    for i in range(n_loci):
        is_snp = bool(n_alt_root[i] > 0)
        node = locus_to_node.get(i)
        sample_of_locus = 0 if (is_snp or node is None) else birth_sample[node]
        meta_rows.append({
            # LoPhy discards the first column of meta.csv as a row label, so CHR comes first.
            "CHR": all_loci_df["CHR"].iloc[i],
            "REGION_INDEX": region_index[all_loci_df["REGION"].iloc[i]],
            "SNP": int(is_snp),
            "SBS": 1,
            "SAMPLE": sample_of_locus,
            "NAME": all_loci_df["NAME"].iloc[i],
        })
    pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "meta.csv"), index=False, lineterminator="\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METHOD_DIRNAMES = {
    "compass": "COMPASS",
    "lophy": "LoPhy",
}


def convert_all(basename, out_root, n_samples=1, methods=("compass", "lophy")):
    """Write the input files of each requested method into <out_root>/<MethodName>/.
    For COMPASS with several samples, each sample goes into COMPASS/sample<s>/."""
    if "compass" in methods:
        sample_basenames = [_sample_basename(basename, n_samples, s) for s in range(n_samples)]
        for s, sb in enumerate(sample_basenames):
            sub = METHOD_DIRNAMES["compass"] if n_samples == 1 else \
                os.path.join(METHOD_DIRNAMES["compass"], f"sample{s}")
            write_compass_format(sb, os.path.join(out_root, sub))

    if "lophy" in methods:
        write_lophy_format(basename, os.path.join(out_root, METHOD_DIRNAMES["lophy"]), n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('basename', type=str,
                        help="Basename of the generated dataset.")
    parser.add_argument('--out_root', type=str, default=None,
                        help="Output root (default: directory of basename).")
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--methods', type=str, nargs='+',
                        default=["compass", "lophy"],
                        choices=["compass", "lophy"])
    args = parser.parse_args()

    out_root = args.out_root or os.path.dirname(args.basename) or "."
    convert_all(args.basename, out_root, n_samples=args.n_samples, methods=args.methods)
    print(f"Converted {args.basename} -> {out_root} [{', '.join(args.methods)}]")
