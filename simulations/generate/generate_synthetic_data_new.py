"""Simulator for single-cell DNA sequencing data of tumours (tree, SNVs, copy-number events, reads).

Usage: python generate_synthetic_data_new.py -o <basename> --seed 1 --nnodes 5 --nsamples 1

Writes <basename>_variants.csv, _regions.csv, .cell_annotation.csv, _cellAssignmentsTRUE.tsv and
_genotypes.csv (per sample, with suffix _<k> if nsamples > 1), and _n_ref_root.csv, _n_alt_root.csv,
_strand_genotypes.json, _events_ordered.json, _true_model_params.json and _treeTRUE.gv (whole tree).
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import betabinom

# Utility helpers

def build_tree(n_nodes, rng):
    """Return parents[], children[], DFT_order[] for a random tree with n_nodes."""
    if n_nodes == 1:
        return [-1], [[]], [0]

    prufer_code = rng.integers(0, n_nodes, size=n_nodes - 2)
    parents = [-1] * n_nodes
    nodes_to_add = set(range(n_nodes - 1))
    for i in range(len(prufer_code)):
        n = 0
        while (n not in nodes_to_add) or (n in prufer_code[i:]):
            n += 1
        parents[n] = int(prufer_code[i])
        nodes_to_add.remove(n)
    parents[list(nodes_to_add)[0]] = n_nodes - 1

    # Swap so root is index 0
    parents[0], parents[n_nodes - 1] = parents[n_nodes - 1], parents[0]
    for i in range(len(parents)):
        if parents[i] == 0:
            parents[i] = n_nodes - 1
        elif parents[i] == n_nodes - 1:
            parents[i] = 0

    children = [[] for _ in range(n_nodes)]
    for i in range(1, n_nodes):
        children[parents[i]].append(i)

    dft = []
    stack = [0]
    while stack:
        top = stack.pop()
        dft.append(top)
        for c in children[top]:
            stack.append(c)

    return parents, children, dft


def contiguous_block(chr_regions, rng):
    """Return a random contiguous sublist of chr_regions."""
    k = len(chr_regions)
    total_blocks = k * (k + 1) // 2
    block_index = int(rng.integers(0, total_blocks))
    start, length, counter = 0, 0, 0
    for l in range(1, k + 1):
        blocks_of_size_l = k - l + 1
        if block_index < counter + blocks_of_size_l:
            length = l
            start = block_index - counter
            break
        counter += blocks_of_size_l
    return chr_regions[start: start + length]


def assign_birth_samples(parents, dft_order, n_samples, rng):
    """Assign each node a birth sample in [0, n_samples-1].

    The root is born at sample 0; every other node is drawn uniformly from
    [parent's birth sample, n_samples-1]. Sample 0 always contains a non-root node."""
    n_nodes = len(parents)
    birth_sample = np.zeros(n_nodes, dtype=int)
    for n in dft_order:
        if n == 0:
            continue
        lo = birth_sample[parents[n]]
        birth_sample[n] = int(rng.integers(lo, n_samples))
    if n_nodes > 1 and not np.any(birth_sample[1:] == 0):
        root_children = [n for n in range(1, n_nodes) if parents[n] == 0]
        birth_sample[root_children[int(rng.integers(0, len(root_children)))]] = 0
    return birth_sample


# Core strand-state helpers

def init_root_strands(n_total_loci, n_regions, root_het_fraction,
                      root_hom_alt_fraction, is_male,
                      region_is_X, region_is_Y, region_to_loci, rng,
                      locus_snv_direction=None, random_phase=False):
    """Build the root's strand arrays and per-locus germline genotype (0 = hom ref, 1 = het, 2 = hom alt).

    Root ploidy is sex-aware, as in input.cpp: autosomes and female chrX have 2 copies,
    male chrX and chrY have 1 copy (strand 0 only), female chrY has none.
    Hemizygous loci cannot be het, so the het fraction is folded into hom ref.

    locus_snv_direction per locus: 0 = ref->alt SNV, 1 = alt->ref SNV, -1 = no SNV,
    -2 = dedicated germline het locus. The germline draw is conditioned on the SNV direction:
    ref->alt loci are hom ref or het, alt->ref loci are het or hom alt.

    random_phase: by default the alt allele of a het locus is on strand 1; if set,
    it is on strand 0 or 1 with probability 1/2.
    """
    root_strand_regions = [np.ones(n_regions, dtype=int),
                           np.ones(n_regions, dtype=int)]
    for k in range(n_regions):
        if region_is_Y[k]:
            root_strand_regions[0][k] = 1 if is_male else 0
            root_strand_regions[1][k] = 0
        elif region_is_X[k] and is_male:
            root_strand_regions[0][k] = 1
            root_strand_regions[1][k] = 0

    hemizygous_locus = np.zeros(n_total_loci, dtype=bool)
    for k in range(n_regions):
        n_present = int(root_strand_regions[0][k] > 0) + int(root_strand_regions[1][k] > 0)
        if n_present == 1:
            for locus in region_to_loci[k]:
                hemizygous_locus[locus] = True

    if locus_snv_direction is None:
        locus_snv_direction = np.full(n_total_loci, -1)
    germline_alt = root_het_fraction + root_hom_alt_fraction
    het_given_alt = root_het_fraction / germline_alt if germline_alt > 0 else 1.0
    locus_root_genotype = np.zeros(n_total_loci, dtype=int)
    for i in range(n_total_loci):
        r = rng.random()
        direction = locus_snv_direction[i]
        if hemizygous_locus[i]:
            if direction == 0:
                locus_root_genotype[i] = 0
            elif direction == 1:
                locus_root_genotype[i] = 2
            else:
                locus_root_genotype[i] = 2 if r < root_hom_alt_fraction else 0
        elif direction == 0:    # ref->alt needs a ref allele
            locus_root_genotype[i] = 1 if r < root_het_fraction else 0
        elif direction == 1:    # alt->ref needs an alt allele
            locus_root_genotype[i] = 1 if r < het_given_alt else 2
        elif direction == -2:   # dedicated germline het locus
            locus_root_genotype[i] = 1
        else:
            if r < root_hom_alt_fraction:
                locus_root_genotype[i] = 2
            elif r < root_hom_alt_fraction + root_het_fraction:
                locus_root_genotype[i] = 1
            # else 0 (hom ref)

    root_strand_ref = [np.zeros(n_total_loci, dtype=int),
                       np.zeros(n_total_loci, dtype=int)]
    root_strand_alt = [np.zeros(n_total_loci, dtype=int),
                       np.zeros(n_total_loci, dtype=int)]

    for i in range(n_total_loci):
        g = locus_root_genotype[i]
        if hemizygous_locus[i]:
            # The present strand is index 0; the absent strand keeps ref = alt = 0.
            if g == 2:
                root_strand_ref[0][i] = 0; root_strand_alt[0][i] = 1
            else:
                root_strand_ref[0][i] = 1; root_strand_alt[0][i] = 0
        elif g == 0:   # hom ref
            root_strand_ref[0][i] = 1; root_strand_alt[0][i] = 0
            root_strand_ref[1][i] = 1; root_strand_alt[1][i] = 0
        elif g == 1: # het
            root_strand_ref[0][i] = 1; root_strand_alt[0][i] = 0
            root_strand_ref[1][i] = 0; root_strand_alt[1][i] = 1
        else:        # hom alt
            root_strand_ref[0][i] = 0; root_strand_alt[0][i] = 1
            root_strand_ref[1][i] = 0; root_strand_alt[1][i] = 1

    if random_phase:
        for i in range(n_total_loci):
            if locus_root_genotype[i] == 1 and not hemizygous_locus[i] and rng.random() < 0.5:
                for arr in (root_strand_ref, root_strand_alt):
                    arr[0][i], arr[1][i] = arr[1][i], arr[0][i]

    return root_strand_regions, root_strand_ref, root_strand_alt, locus_root_genotype


def deep_copy_strands(src_regions, src_ref, src_alt):
    return ([s.copy() for s in src_regions],
            [s.copy() for s in src_ref],
            [s.copy() for s in src_alt])


def propagate_from_parent(n, parents, strands_regions, strands_loci_ref, strands_loci_alt):
    """Copy the parent's strand state into node n."""
    p = parents[n]
    sr, sref, salt = deep_copy_strands(
        strands_regions[p], strands_loci_ref[p], strands_loci_alt[p])
    strands_regions[n]  = sr
    strands_loci_ref[n] = sref
    strands_loci_alt[n] = salt


# Event validity rules (mirror Node.cpp::update_genotype)

MAX_COPY_NUMBER = 10  # input.cpp: parameters.max_copy_number


def cnloh_region_valid(n, region, donor, recip, strands_regions, strands_loci_ref,
                       strands_loci_alt, region_loci):
    """CNLOH rule for one region: equal copy number >= 1 on both strands and at least one locus with different alleles."""
    cn_d, cn_r = strands_regions[n][donor][region], strands_regions[n][recip][region]
    if cn_d != cn_r or cn_d < 1:
        return False
    for locus in region_loci:
        if (strands_loci_ref[n][donor][locus] != strands_loci_ref[n][recip][locus] or
                strands_loci_alt[n][donor][locus] != strands_loci_alt[n][recip][locus]):
            return True
    return False


def _gain_loss_regions(ev):
    """Return the (gained, lost) region sets of one event; FGD and CNLOH count as neither."""
    t = ev['type']
    if t == 'local_gain':
        return set(ev['regions']), set()
    if t == 'aneuploidy_gain':
        return set(ev['chr_regions']), set()
    if t == 'local_loss':
        return set(), set(ev['regions'])
    if t == 'aneuploidy_loss':
        return set(), set(ev['chr_regions'])
    return set(), set()


def conflicts_with_node_events(ev, node_events_n):
    """True if adding ev would make some region both gained and lost in the node."""
    gained, lost = set(), set()
    for past in node_events_n:
        g, l = _gain_loss_regions(past)
        gained |= g; lost |= l
    g, l = _gain_loss_regions(ev)
    return bool(g & lost) or bool(l & gained)


def exceeds_max_copy_number(strands_regions_n):
    return bool(np.any(sum(strands_regions_n) > MAX_COPY_NUMBER)) if strands_regions_n else False


# Event application

def apply_event(n, ev, strands_regions, strands_loci_ref, strands_loci_alt,
                all_locus_to_region, n_total_loci, n_regions):
    """Apply one event (SNV or CNA) to node n's strand arrays in place; raise ValueError if invalid."""
    t = ev['type']

    # An upstream loss can remove a strand that this event refers to.
    n_strands = len(strands_regions[n])
    for key in ('strand', 'donor_strand', 'recipient_strand'):
        if key in ev and ev[key] >= n_strands:
            raise ValueError(f"{t}: strand {ev[key]} no longer exists at node {n}")

    if t == 'SNV':
        locus     = ev['locus']
        direction = ev['direction']
        placed = False
        for s in range(len(strands_loci_ref[n])):
            if direction == 0 and strands_loci_ref[n][s][locus] > 0:
                strands_loci_ref[n][s][locus] -= 1
                strands_loci_alt[n][s][locus] += 1
                placed = True; break
            elif direction == 1 and strands_loci_alt[n][s][locus] > 0:
                strands_loci_alt[n][s][locus] -= 1
                strands_loci_ref[n][s][locus] += 1
                placed = True; break
        if not placed:
            raise ValueError(
                f"SNV locus={locus} direction={'alt->ref' if direction else 'ref->alt'} "
                f"cannot be placed at node {n}: required allele absent on all strands.")

    elif t == 'FGD':
        n_current = len(strands_regions[n])
        for s in range(n_current):
            strands_regions[n].append(strands_regions[n][s].copy())
            strands_loci_ref[n].append(strands_loci_ref[n][s].copy())
            strands_loci_alt[n].append(strands_loci_alt[n][s].copy())

    elif t == 'aneuploidy_gain':
        chr_regs   = ev['chr_regions']
        src_strand = ev['strand']
        chr_regs_set = set(chr_regs)
        if not all(strands_regions[n][src_strand][r] > 0 for r in chr_regs):
            raise ValueError(f"aneuploidy_gain: strand {src_strand} has zero-copy region at node {n}")
        new_sr  = np.zeros(n_regions, dtype=int)
        new_ref = np.zeros(n_total_loci, dtype=int)
        new_alt = np.zeros(n_total_loci, dtype=int)
        for r in chr_regs:
            new_sr[r] = strands_regions[n][src_strand][r]
        for locus in range(n_total_loci):
            if all_locus_to_region[locus] in chr_regs_set:
                new_ref[locus] = strands_loci_ref[n][src_strand][locus]
                new_alt[locus] = strands_loci_alt[n][src_strand][locus]
        strands_regions[n].append(new_sr)
        strands_loci_ref[n].append(new_ref)
        strands_loci_alt[n].append(new_alt)

    elif t == 'aneuploidy_loss':
        chr_regs = set(ev['chr_regions'])
        strand   = ev['strand']
        if not all(strands_regions[n][strand][r] > 0 for r in chr_regs):
            raise ValueError(f"aneuploidy_loss: strand {strand} has zero-copy region at node {n}")
        for r in chr_regs:
            strands_regions[n][strand][r] = 0
        for locus in range(n_total_loci):
            if all_locus_to_region[locus] in chr_regs:
                strands_loci_ref[n][strand][locus] = 0
                strands_loci_alt[n][strand][locus] = 0
        if not np.any(strands_regions[n][strand]):
            strands_regions[n].pop(strand)
            strands_loci_ref[n].pop(strand)
            strands_loci_alt[n].pop(strand)

    elif t in ('local_gain', 'local_loss'):
        strand  = ev['strand']
        delta   = ev['delta']
        regions = ev['regions']
        if not all(strands_regions[n][strand][r] > 0 for r in regions):
            raise ValueError(f"{t}: strand {strand} has zero-copy region at node {n}")

        mult = 0
        if delta > 0:
            mult=2
        for r in regions:
            strands_regions[n][strand][r] = strands_regions[n][strand][r] * mult
        for locus in range(n_total_loci):
            if all_locus_to_region[locus] in set(regions):
                strands_loci_ref[n][strand][locus] = strands_loci_ref[n][strand][locus] * mult
                strands_loci_alt[n][strand][locus] = strands_loci_alt[n][strand][locus] * mult

        if delta < 0 and not np.any(strands_regions[n][strand]):
            strands_regions[n].pop(strand)
            strands_loci_ref[n].pop(strand)
            strands_loci_alt[n].pop(strand)

    elif t == 'CNLOH':
        donor   = ev['donor_strand']
        recip   = ev['recipient_strand']
        regions = ev['regions']
        # Re-check validity, since an upstream change can invalidate the event on replay.
        for r in regions:
            region_loci = [i for i in range(n_total_loci) if all_locus_to_region[i] == r]
            if not cnloh_region_valid(n, r, donor, recip, strands_regions, strands_loci_ref,
                                      strands_loci_alt, region_loci):
                raise ValueError(f"CNLOH: region {r} not copy-neutral/identifiable at node {n}")
        for locus in range(n_total_loci):
            if all_locus_to_region[locus] in set(regions):
                strands_loci_ref[n][recip][locus] = strands_loci_ref[n][donor][locus]
                strands_loci_alt[n][recip][locus] = strands_loci_alt[n][donor][locus]

    else:
        raise ValueError(f"Unknown event type: {t}")



# Event builders (return a dict, or None if infeasible at node n)

def build_snv_event(n, locus, direction, strands_loci_ref, strands_loci_alt):
    for s in range(len(strands_loci_ref[n])):
        if direction == 0 and strands_loci_ref[n][s][locus] > 0:
            return {'type': 'SNV', 'locus': locus, 'direction': direction}
        elif direction == 1 and strands_loci_alt[n][s][locus] > 0:
            return {'type': 'SNV', 'locus': locus, 'direction': direction}
    return None

def build_cna_event(n, cna_type, rng,
                    strands_regions, strands_loci_ref, strands_loci_alt,
                    chromosome_to_regions, n_chromosomes,
                    region_to_loci, node_events_n):
    """Build a random valid CNA event for node n, or return None."""
    ev = _build_cna_event(n, cna_type, rng, strands_regions, strands_loci_ref, strands_loci_alt,
                          chromosome_to_regions, n_chromosomes, region_to_loci)
    if ev is None or conflicts_with_node_events(ev, node_events_n):
        return None
    return ev


def _build_cna_event(n, cna_type, rng,
                     strands_regions, strands_loci_ref, strands_loci_alt,
                     chromosome_to_regions, n_chromosomes,
                     region_to_loci):
    n_strands = len(strands_regions[n])
    if n_strands == 0:
        return None
    strand = int(rng.integers(0, n_strands))

    if cna_type == 'FGD':
        return {'type': 'FGD'}

    elif cna_type == 'aneuploidy_gain':
        chr_idx  = int(rng.integers(0, n_chromosomes))
        chr_regs = chromosome_to_regions[chr_idx]
        if not all(strands_regions[n][strand][r] > 0 for r in chr_regs):
            return None
        return {'type': 'aneuploidy_gain', 'chromosome': chr_idx,
                'chr_regions': chr_regs, 'strand': strand}

    elif cna_type == 'aneuploidy_loss':
        chr_idx  = int(rng.integers(0, n_chromosomes))
        chr_regs = chromosome_to_regions[chr_idx]
        if not all(strands_regions[n][strand][r] > 0 for r in chr_regs):
            return None
        return {'type': 'aneuploidy_loss', 'chromosome': chr_idx,
                'chr_regions': chr_regs, 'strand': strand}

    elif cna_type in ('local_gain', 'local_loss'):
        chr_idx  = int(rng.integers(0, n_chromosomes))
        chr_regs = chromosome_to_regions[chr_idx]
        regions  = contiguous_block(chr_regs, rng)
        delta = 1 if cna_type == 'local_gain' else -1
        if not all(strands_regions[n][strand][r] > 0 for r in regions):
            return None
        return {'type': cna_type, 'strand': strand, 'regions': set(regions),
                'chromosome': chr_idx, 'delta': delta}

    elif cna_type == 'CNLOH':
        if n_strands < 2:
            return None
        chr_idx  = int(rng.integers(0, n_chromosomes))
        chr_regs = chromosome_to_regions[chr_idx]
        regions  = contiguous_block(chr_regs, rng)

        # Every affected region must be valid for the (donor, recipient) pair.
        identifiable_pairs = []
        for d in range(n_strands):
            for r in range(n_strands):
                if d == r:
                    continue
                if all(cnloh_region_valid(n, region, d, r, strands_regions, strands_loci_ref,
                                          strands_loci_alt, region_to_loci[region])
                       for region in regions):
                    identifiable_pairs.append((d, r))

        if identifiable_pairs:
            donor, recip = identifiable_pairs[int(rng.integers(0, len(identifiable_pairs)))]
            return {'type': 'CNLOH', 'strand': donor, 'donor_strand': donor,
                    'recipient_strand': recip, 'regions': set(regions), 'chromosome': chr_idx}

        # Loci are sparse, so a random block is rarely valid. Fall back to a random sub-block
        # of a contiguous run of valid regions around a random valid region.
        candidates = []
        for d in range(n_strands):
            for r in range(n_strands):
                if d == r:
                    continue
                valid = [cnloh_region_valid(n, region, d, r, strands_regions, strands_loci_ref,
                                            strands_loci_alt, region_to_loci[region])
                         for region in chr_regs]
                if any(valid):
                    candidates.append((d, r, valid))
        if not candidates:
            return None
        donor, recip, valid = candidates[int(rng.integers(0, len(candidates)))]
        valid_pos = [i for i, v in enumerate(valid) if v]
        seed_pos = valid_pos[int(rng.integers(0, len(valid_pos)))]
        lo = seed_pos
        while lo > 0 and valid[lo - 1]:
            lo -= 1
        hi = seed_pos
        while hi < len(valid) - 1 and valid[hi + 1]:
            hi += 1
        start = int(rng.integers(lo, seed_pos + 1))
        end = int(rng.integers(seed_pos, hi + 1))
        return {'type': 'CNLOH', 'strand': donor, 'donor_strand': donor,
                'recipient_strand': recip, 'regions': set(chr_regs[start:end + 1]),
                'chromosome': chr_idx}

    return None


# Read-count generative model (mirrors Scores.cpp)

def sample_negbinom(mean, theta, rng):
    """Negative binomial with the given mean and size theta, so variance = mean + mean^2/theta."""
    mean = np.maximum(mean, 1e-6)
    p = theta / (theta + mean)
    return rng.negative_binomial(theta, p)


NOISE_ALPHA_R_FIXED = 0.9  # Scores.cpp::compute_CNA_loglikelihoods_joint
NOISE_WEIGHT = 0.02


def sample_region_counts(cn_cell_matrix, region_weights, region_dispersion,
                         root_cn_regions, dropout_rate, background_rate,
                         total_counts, rng, noise_weight=NOISE_WEIGHT):
    """Draw per-region read counts for all cells, one region at a time.

    Region k of a cell is Beta-Binomial(N=total_counts, alpha_r, alpha_total[k] - alpha_r), with
      k_r            ~ Binomial(cn_cell[k], 1 - dropout)   surviving copies
      alpha_r        = alpha_region[k] * k_r if k_r > 0 else alpha_region[k] * background_rate
      alpha_total[k] = region_dispersion[k] * max(1, root_cn_regions[k]) * sum_w_cn
    where sum_w_cn is the per-cell copy-number-weighted sum scaled by (1 - dropout).
    Regions are independent given N, so counts need not sum to N.

    With probability noise_weight a count is instead drawn from the noise component
    Beta-Binomial(N, NOISE_ALPHA_R_FIXED, NOISE_ALPHA_R_FIXED / mean_frac - NOISE_ALPHA_R_FIXED),
    where mean_frac is the mean of the main model.
    """
    n_regions, n_cells = cn_cell_matrix.shape
    per_copy_weight = region_weights / np.maximum(1, root_cn_regions)
    sum_w_cn = (per_copy_weight[:, None] * cn_cell_matrix).sum(axis=0) * (1.0 - dropout_rate)

    total_counts = np.asarray(total_counts)
    depths = np.zeros((n_regions, n_cells), dtype=int)
    for k in range(n_regions):
        alpha_region_k = region_weights[k] * region_dispersion[k]
        alpha_total_k = region_dispersion[k] * max(1, root_cn_regions[k]) * sum_w_cn
        k_r = rng.binomial(cn_cell_matrix[k].astype(int), 1.0 - dropout_rate)
        alpha_r = np.where(k_r > 0, alpha_region_k * k_r, alpha_region_k * background_rate)
        alpha_r = np.maximum(alpha_r, 1e-9)
        alpha_others = np.maximum(alpha_total_k - alpha_r, 1e-9)
        depths[k] = betabinom.rvs(total_counts, alpha_r, alpha_others, random_state=rng)

        noise = rng.random(n_cells) < noise_weight
        if np.any(noise):
            cn_k = cn_cell_matrix[k, noise]
            mean_frac = alpha_region_k * np.where(cn_k > 0, cn_k, background_rate) / alpha_total_k[noise]
            alpha_total_noise = NOISE_ALPHA_R_FIXED / mean_frac
            depths[k, noise] = betabinom.rvs(
                total_counts[noise], NOISE_ALPHA_R_FIXED,
                np.maximum(alpha_total_noise - NOISE_ALPHA_R_FIXED, 1e-9), random_state=rng)
    return depths


def sample_snv_reads(ref_vec, alt_vec, alpha_region, dropout_rate,
                     allelic_error_rate, background_rate, depth_vec, rng):
    """Sample ref/alt reads at one locus for all cells, as in compute_SNV_loglikelihoods_joint.

    Surviving copies are k ~ Binom(c_ref, 1 - dropout) and l ~ Binom(c_alt, 1 - dropout).
    If k == l == 0, alt reads are BetaBinomial(depth, omega_bg, omega_bg) with
    omega_bg = 0.5 * alpha_region * background_rate. Otherwise they are BetaBinomial with
    alpha_alt = alpha_region * (k+l) * f and alpha_ref = alpha_region * (k+l) * (1-f),
    where f = (l * (1-eps) + k * eps) / (k+l) and eps is the allelic error rate.
    """
    n_cells = len(ref_vec)
    k = rng.binomial(ref_vec.astype(int), 1.0 - dropout_rate)
    l = rng.binomial(alt_vec.astype(int), 1.0 - dropout_rate)

    alt_reads = np.zeros(n_cells, dtype=int)
    both_zero = (k == 0) & (l == 0)

    idx_bg = np.where(both_zero)[0]
    if len(idx_bg) > 0:
        omega_bg = max(0.5 * alpha_region * background_rate, 1e-9)
        alt_reads[idx_bg] = betabinom.rvs(
            depth_vec[idx_bg], omega_bg, omega_bg, random_state=rng)

    idx_norm = np.where(~both_zero)[0]
    if len(idx_norm) > 0:
        eps_a = allelic_error_rate
        k_n = k[idx_norm].astype(float)
        l_n = l[idx_norm].astype(float)
        scale = k_n + l_n
        f = (l_n * (1.0 - eps_a) + k_n * eps_a) / scale
        alpha_alt = np.maximum(alpha_region * scale * f, 1e-9)
        alpha_ref = np.maximum(alpha_region * scale * (1.0 - f), 1e-9)
        alt_reads[idx_norm] = betabinom.rvs(
            depth_vec[idx_norm], alpha_alt, alpha_ref, random_state=rng)

    ref_reads = depth_vec - alt_reads
    return ref_reads, alt_reads


def sample_node_probabilities(alive_mask, healthy_fraction, nodeprobs_concentration, rng):
    """Draw one sample's node probabilities: the root gets healthy_fraction and the rest
    is a Dirichlet draw over the alive non-root nodes (other nodes get 0)."""
    n_nodes = len(alive_mask)
    node_probabilities = np.zeros(n_nodes)
    non_root_alive = [n for n in range(1, n_nodes) if alive_mask[n]]
    if not non_root_alive:
        # No tumour clone is alive, so all cells are healthy.
        node_probabilities[0] = 1.0
        return node_probabilities
    node_probabilities[0] = healthy_fraction
    weights = rng.dirichlet([nodeprobs_concentration] * len(non_root_alive))
    for n, w in zip(non_root_alive, weights):
        node_probabilities[n] = (1.0 - healthy_fraction) * w
    return node_probabilities


# Per-sample cell and read-count generation

def generate_sample_reads(
        sample_basename, n_cells,
        cn_regions_arr, n_ref_allele_arr, n_alt_allele_arr,
        node_probabilities, region_weights, region_dispersion, root_cn_regions,
        n_total_loci, n_SNVs, all_locus_to_region,
        dropout_rate, background_rate, allelic_error_rate,
        mean_total_count, theta_total_count,
        mean_snv_depth, theta_snv_depth,
        doublet_rate, cycling_prob,
        chr_labels_by_region, region_names, region_names_suffix,
        rng, noise_weight=NOISE_WEIGHT, n_het=0):
    """Generate one sample's cells and write its read counts and annotation files.

    The last n_het loci are the dedicated germline het loci (named Het<k>).
    Returns (healthy_cells, node of each cell, is_singlet); is_singlet marks cells
    that are neither doublets nor cycling.
    """
    n_nodes = cn_regions_arr.shape[1]

    node1_of_cell = np.zeros(n_cells, dtype=int)
    node2_of_cell = np.zeros(n_cells, dtype=int)
    is_doublet    = np.zeros(n_cells, dtype=bool)
    is_cycling    = np.zeros(n_cells, dtype=bool)
    healthy_cells = np.zeros(n_cells, dtype=bool)

    for j in range(n_cells):
        if rng.random() < doublet_rate:
            n1 = int(rng.choice(n_nodes, p=node_probabilities))
            n2 = int(rng.choice(n_nodes, p=node_probabilities))
            node1_of_cell[j] = n1
            node2_of_cell[j] = n2
            is_doublet[j] = True
        else:
            n1 = int(rng.choice(n_nodes, p=node_probabilities))
            node1_of_cell[j] = n1
            node2_of_cell[j] = n1
            if rng.random() < cycling_prob[n1]:
                is_cycling[j] = True
            else:
                healthy_cells[j] = (n1 == 0)

    combined = is_doublet | is_cycling
    cn_cell_matrix  = cn_regions_arr[:, node1_of_cell].astype(float)
    ref_cell_matrix = n_ref_allele_arr[:, node1_of_cell].astype(float)
    alt_cell_matrix = n_alt_allele_arr[:, node1_of_cell].astype(float)
    if np.any(combined):
        idx2 = node2_of_cell[combined]
        cn_cell_matrix[:, combined]  += cn_regions_arr[:, idx2]
        ref_cell_matrix[:, combined] += n_ref_allele_arr[:, idx2]
        alt_cell_matrix[:, combined] += n_alt_allele_arr[:, idx2]

    per_copy_weight = region_weights / np.maximum(1, root_cn_regions)
    cn_total_cell = (per_copy_weight[:, None] * cn_cell_matrix).sum(axis=0)
    total_counts = sample_negbinom(mean_total_count * cn_total_cell, theta_total_count, rng)
    total_counts = np.maximum(total_counts, 1)

    depths = sample_region_counts(
        cn_cell_matrix, region_weights, region_dispersion, root_cn_regions,
        dropout_rate, background_rate, total_counts, rng, noise_weight=noise_weight)

    ref_reads = np.zeros((n_total_loci, n_cells), dtype=int)
    alt_reads = np.zeros((n_total_loci, n_cells), dtype=int)
    genotypes = np.zeros((n_total_loci, n_cells), dtype=int)

    for i in range(n_total_loci):
        r = all_locus_to_region[i]
        alpha_region_i = region_weights[r] * region_dispersion[r]

        mean_depth = mean_snv_depth * np.maximum(cn_cell_matrix[r, :], 1e-6) / 2.0
        depth_vec = np.maximum(sample_negbinom(mean_depth, theta_snv_depth, rng), 0)

        ref_vec, alt_vec = sample_snv_reads(
            ref_cell_matrix[i, :], alt_cell_matrix[i, :],
            alpha_region_i, dropout_rate, allelic_error_rate,
            background_rate, depth_vec, rng)
        ref_reads[i, :] = ref_vec
        alt_reads[i, :] = alt_vec

        total_reads_i = ref_vec + alt_vec
        for j in range(n_cells):
            tr = total_reads_i[j]
            if tr < 8:
                genotypes[i, j] = 3
            elif min(ref_vec[j], alt_vec[j]) > 0.17 * tr:
                genotypes[i, j] = 1
            elif alt_vec[j] > 0.88 * tr:
                genotypes[i, j] = 2
            elif ref_vec[j] > 0.88 * tr:
                genotypes[i, j] = 0
            else:
                genotypes[i, j] = 3

    cell_names = [str(j) for j in range(n_cells)]

    d = {
        "CHR":    [chr_labels_by_region[all_locus_to_region[i]] for i in range(n_total_loci)],
        "REGION": [region_names_suffix[all_locus_to_region[i]] for i in range(n_total_loci)],
        "NAME":   [f"SNV{i}" if i < n_SNVs else (f"Het{i - (n_total_loci - n_het)}" if i >= n_total_loci - n_het else f"FalseSNV{i - n_SNVs}")
                   for i in range(n_total_loci)],
    }
    for j in range(n_cells):
        d[cell_names[j]] = [f"{ref_reads[i, j]}:{alt_reads[i, j]}" for i in range(n_total_loci)]
    # Files are written with "\n" line endings because the C++ reader splits on "\n" only.
    pd.DataFrame(d).to_csv(sample_basename + "_variants.csv", index=False, lineterminator="\n")

    pd.DataFrame(depths, index=region_names, columns=cell_names).astype(int).to_csv(
        sample_basename + "_regions.csv", index=True, header=False, lineterminator="\n")

    with open(sample_basename + ".cell_annotation.csv", "w", newline="\n") as f:
        f.write("barcode,cluster,assignment\n")
        for j in range(n_cells):
            assignment = "healthy" if healthy_cells[j] else "tumor"
            f.write(f"{cell_names[j]},0,{assignment}\n")

    pd.DataFrame({
        "cell": cell_names,
        "node": node1_of_cell.tolist(),
        "node2": node2_of_cell.tolist(),
        "is_doublet": is_doublet.tolist(),
        "is_cycling": is_cycling.tolist(),
    }).to_csv(sample_basename + "_cellAssignmentsTRUE.tsv", sep="\t", index=False, lineterminator="\n")

    np.savetxt(sample_basename + "_genotypes.csv", genotypes.astype(int),
               delimiter=" ", fmt='%i')

    return healthy_cells, node1_of_cell.tolist(), ~combined


# Sample colours for the multi-sample pie charts, same as in Tree.cpp's to_dot.
SAMPLE_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]


def pie_bubbles_dot(counts_by_sample, n_nodes, node_colors):
    """Graphviz lines for the multi-sample cell bubbles, as in Tree.cpp's to_dot.

    Bubble n+n_nodes is a pie of node n's cells with one slice per sample. Its area is
    proportional to the node's share of all cells. A legend node maps sample colours."""
    counts_by_sample = np.asarray(counts_by_sample)
    per_node = counts_by_sample.sum(axis=0)
    total = max(int(per_node.sum()), 1)
    lines = []
    for n in range(n_nodes):
        c = node_colors[n % len(node_colors)]
        size = np.sqrt(100.0 * per_node[n] / total) / 3.0 if per_node[n] > 0 else 0.1
        label = f"{per_node[n]} cells\\n{round(100.0 * per_node[n] / total)}%"
        for s_idx in range(counts_by_sample.shape[0]):
            label += f"\\nsample {s_idx}: {counts_by_sample[s_idx, n]}"
        slices = [(SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)], counts_by_sample[s_idx, n] / per_node[n])
                  for s_idx in range(counts_by_sample.shape[0])
                  if per_node[n] > 0 and counts_by_sample[s_idx, n] > 0]
        if len(slices) == 1:
            # The wedged style draws nothing for a single colour.
            style = f'style=filled fillcolor="{slices[0][0]}" color={c}'
        elif slices:
            # The last slice gets no fraction and takes the remainder, so rounding cannot exceed 1.
            fill = ":".join(f"{col};{frac:.4f}" for col, frac in slices[:-1])
            fill = (fill + ":" if fill else "") + slices[-1][0]
            style = f'style=wedged fillcolor="{fill}" color={c}'
        else:
            style = f"style=filled color={c}"
        lines.append(f'{n + n_nodes}[label="{label}" {style} width={size:.3f} height={size:.3f}];\n')
    rows = "".join(
        f'<tr><td bgcolor="{SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)]}" width="24" height="24"></td>'
        f'<td align="left">sample {s_idx}</td></tr>'
        for s_idx in range(counts_by_sample.shape[0]))
    lines.append(f'legend [shape=none margin=0 label=<<table border="0" cellspacing="4">'
                 f'{rows}</table>>];\n')
    return "".join(lines)


# Fraction of all loci that are false (never mutated).
FALSE_SNV_FRACTION = 0.0


def n_false_snvs_for(n_SNVs, false_fraction=FALSE_SNV_FRACTION):
    """Number of false loci so they make up false_fraction of all loci."""
    return int(round(n_SNVs * false_fraction / (1 - false_fraction)))


# Main generator

def generate_data(
        basename,
        n_cells, n_nodes,
        n_SNVs, n_false_SNVs,
        n_local_CNAs, n_aneuploidy, n_FGD, n_CNLOH,
        n_regions,
        chromosome_sizes,
        allow_empty_regions,
        root_het_fraction, root_hom_alt_fraction,
        alt_to_ref_fraction,
        nodeprobs_concentration,
        dropout_rate,
        region_dispersion_min, region_dispersion_max,
        region_weight_sigma,
        background_rate,
        mean_total_count, theta_total_count,
        mean_snv_depth, theta_snv_depth,
        doublet_rate,
        root_cycling_prob, cycling_prob_min, cycling_prob_max,
        cycling_cell_factor,
        allelic_error_rate,
        seed,
        sex="female",
        healthy_fraction=0.15,
        n_samples=1,
        max_ordering_attempts=200,
        noise_weight=NOISE_WEIGHT,
        n_local_gain=None, n_local_loss=None,
        n_aneuploidy_gain=None, n_aneuploidy_loss=None,
        region_calibration=None,
        n_het=0, random_phase=False):
    """Simulate one dataset and write all files with the given basename.

    n_local_gain, n_local_loss, n_aneuploidy_gain, n_aneuploidy_loss: optional per-type counts.
    If None, local events are split half gains and half losses, and n_aneuploidy applies to both.

    region_calibration: optional dict with paired lists "region_weight_relative" and
    "region_dispersion" (see rcc_region_calibration.json). If given, each region's (weight,
    dispersion) is a bootstrap draw of one pair instead of the synthetic draws.

    n_het: number of dedicated germline het loci without a somatic event, placed after the
    SNV and false loci and named Het<k>. random_phase: see init_root_strands."""

    rng = np.random.default_rng(seed)

    if sex not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    is_male = (sex == "male")

    parents, children, dft_order = build_tree(n_nodes, rng)

    # Chromosome-to-region mapping. The last chromosome is chrX and the others are autosomes
    # 1..n-1, with no chrY. With fewer than 2 chromosomes all are autosomal.
    chromosome_to_regions = []
    region_to_chromosome  = np.zeros(n_regions, dtype=int)
    idx = 0
    for ci, sz in enumerate(chromosome_sizes):
        chromosome_to_regions.append(list(range(idx, idx + sz)))
        for r in range(idx, idx + sz):
            region_to_chromosome[r] = ci
        idx += sz
    n_chromosomes = len(chromosome_sizes)

    chr_labels = list(range(1, n_chromosomes)) + (['X'] if n_chromosomes >= 2 else [])
    chr_labels = [str(c) for c in chr_labels]
    while len(chr_labels) < n_chromosomes:
        chr_labels.append(f"Chr{len(chr_labels)}")
    chr_is_X = np.array([lbl == 'X' for lbl in chr_labels])
    chr_is_Y = np.array([lbl == 'Y' for lbl in chr_labels])
    region_is_X = np.array([chr_is_X[region_to_chromosome[k]] for k in range(n_regions)])
    region_is_Y = np.array([chr_is_Y[region_to_chromosome[k]] for k in range(n_regions)])

    region_root_ploidy = np.full(n_regions, 2, dtype=int)
    region_root_ploidy[region_is_Y] = 1 if is_male else 0
    region_root_ploidy[region_is_X & is_male] = 1
    valid_regions = np.where(region_root_ploidy > 0)[0]
    if len(valid_regions) == 0:
        raise ValueError("No regions with nonzero root ploidy -- check chromosome_sizes/sex.")

    chr_labels_by_region = [chr_labels[region_to_chromosome[k]] for k in range(n_regions)]
    region_names = [f"{chr_labels_by_region[k]}_Region{k}" for k in range(n_regions)]
    # input.cpp reads _regions.csv labels as "<chr>_<name>" and matches the name against
    # the REGION column of _variants.csv, which therefore has no chromosome prefix.
    region_names_suffix = [f"Region{k}" for k in range(n_regions)]

    # Assign loci to regions; only regions with nonzero root ploidy can carry a locus.
    n_total_loci = n_SNVs + n_false_SNVs + n_het
    n_valid = len(valid_regions)
    if allow_empty_regions:
        snv_to_region = sorted(rng.choice(valid_regions, size=n_SNVs, replace=True).tolist())
    else:
        if n_SNVs < n_valid:
            raise ValueError(
                f"n_SNVs ({n_SNVs}) must be >= number of nonzero-ploidy regions "
                f"({n_valid}) when allow_empty_regions=False")
        snv_to_region = sorted(
            valid_regions.tolist() +
            rng.choice(valid_regions, size=n_SNVs - n_valid, replace=True).tolist()
        )
    false_snv_to_region  = rng.choice(valid_regions, size=n_false_SNVs, replace=True).tolist()
    het_to_region = rng.choice(valid_regions, size=n_het, replace=True).tolist() if n_het > 0 else []
    all_locus_to_region  = snv_to_region + false_snv_to_region + het_to_region

    region_to_loci = [[] for _ in range(n_regions)]
    for i, r in enumerate(all_locus_to_region):
        region_to_loci[r].append(i)

    # Each SNV is an alt->ref back mutation with probability alt_to_ref_fraction, else ref->alt.
    # The root genotype is drawn conditioned on this (see init_root_strands).
    snv_direction = (rng.random(n_SNVs) < alt_to_ref_fraction).astype(int)
    locus_snv_direction = np.concatenate([snv_direction, np.full(n_false_SNVs, -1), np.full(n_het, -2)])
    germline_het_indices = list(range(n_SNVs + n_false_SNVs, n_total_loci))

    # The root genotype is shared by all ordering attempts.
    (root_strand_regions, root_strand_ref, root_strand_alt,
     locus_root_genotype) = init_root_strands(
        n_total_loci, n_regions, root_het_fraction, root_hom_alt_fraction,
        is_male, region_is_X, region_is_Y, region_to_loci, rng,
        locus_snv_direction=locus_snv_direction, random_phase=random_phase)

    root_cn_regions = (root_strand_regions[0] + root_strand_regions[1]).astype(int)

    candidate_nodes = list(range(1, n_nodes))

    # SNV-to-node assignment is shared by all ordering attempts.
    node_pool = candidate_nodes + list(
        rng.choice(candidate_nodes,
                   size=max(0, n_SNVs - len(candidate_nodes)), replace=True))
    rng.shuffle(node_pool)
    snv_node_assignment = node_pool[:n_SNVs]

    snv_specs = [
        {'kind': 'SNV', 'locus': i, 'node': snv_node_assignment[i],
         'direction': int(snv_direction[i])}
        for i in range(n_SNVs)
    ]
    if n_local_gain is None: n_local_gain = n_local_CNAs // 2
    if n_local_loss is None: n_local_loss = n_local_CNAs - n_local_CNAs // 2
    if n_aneuploidy_gain is None: n_aneuploidy_gain = n_aneuploidy
    if n_aneuploidy_loss is None: n_aneuploidy_loss = n_aneuploidy
    cna_type_list = (
        ['FGD']             * n_FGD +
        ['aneuploidy_gain'] * n_aneuploidy_gain +
        ['aneuploidy_loss'] * n_aneuploidy_loss +
        ['local_gain']      * n_local_gain +
        ['local_loss']      * n_local_loss +
        ['CNLOH']           * n_CNLOH
    )
    cna_specs = [{'kind': 'CNA', 'cna_type': ct} for ct in cna_type_list]

    all_event_specs = snv_specs + cna_specs

    # Try random event orderings until one can be applied to all nodes.
    committed_node_events = None  # node_events[n] = ordered list of applied events
    committed_strands_regions  = None
    committed_strands_loci_ref = None
    committed_strands_loci_alt = None

    for attempt in range(max_ordering_attempts):
        rng.shuffle(all_event_specs)

        strands_regions  = [None] * n_nodes
        strands_loci_ref = [None] * n_nodes
        strands_loci_alt = [None] * n_nodes

        strands_regions[0]  = [s.copy() for s in root_strand_regions]
        strands_loci_ref[0] = [s.copy() for s in root_strand_ref]
        strands_loci_alt[0] = [s.copy() for s in root_strand_alt]

        # Non-root nodes start as copies of their parent.
        for n in dft_order:
            if n == 0:
                continue
            propagate_from_parent(n, parents,
                                  strands_regions, strands_loci_ref, strands_loci_alt)

        node_events    = [[] for _ in range(n_nodes)]

        failed = False
        for spec in all_event_specs:
            if spec['kind'] == 'SNV':
                n     = spec['node']
                locus = spec['locus']
                direction = spec['direction']
                ev = build_snv_event(n, locus, direction,
                                     strands_loci_ref, strands_loci_alt)
                if ev is None:
                    failed = True; break
                apply_event(n, ev, strands_regions, strands_loci_ref, strands_loci_alt,
                            all_locus_to_region, n_total_loci, n_regions)
                node_events[n].append(ev)
                # Propagate to descendants and replay their events; if a replay is
                # invalid, abandon this ordering attempt.
                desc_stack = list(children[n])
                propagation_failed = False
                while desc_stack:
                    d = desc_stack.pop()
                    propagate_from_parent(d, parents,
                                         strands_regions, strands_loci_ref,
                                         strands_loci_alt)
                    try:
                        for past_ev in node_events[d]:
                            apply_event(d, past_ev,
                                        strands_regions, strands_loci_ref, strands_loci_alt,
                                        all_locus_to_region, n_total_loci, n_regions)
                    except ValueError:
                        propagation_failed = True
                        break
                    desc_stack.extend(children[d])
                if propagation_failed:
                    failed = True; break

            else:  # CNA: try each non-root node in random order
                cna_type = spec['cna_type']
                node_order = list(range(1, n_nodes))
                rng.shuffle(node_order)
                placed = False
                propagation_failed = False
                for n in node_order:
                    ev = build_cna_event(
                        n, cna_type, rng,
                        strands_regions, strands_loci_ref, strands_loci_alt,
                        chromosome_to_regions, n_chromosomes,
                        region_to_loci, node_events[n])
                    if ev is None:
                        continue
                    apply_event(n, ev,
                                strands_regions, strands_loci_ref, strands_loci_alt,
                                all_locus_to_region, n_total_loci, n_regions)
                    node_events[n].append(ev)
                    # Propagate and replay as for SNVs; also abandon the attempt if the copy
                    # number exceeds MAX_COPY_NUMBER.
                    desc_stack = list(children[n])
                    propagation_failed_here = exceeds_max_copy_number(strands_regions[n])
                    while desc_stack:
                        d = desc_stack.pop()
                        propagate_from_parent(d, parents,
                                             strands_regions, strands_loci_ref,
                                             strands_loci_alt)
                        try:
                            for past_ev in node_events[d]:
                                apply_event(d, past_ev,
                                            strands_regions, strands_loci_ref,
                                            strands_loci_alt,
                                            all_locus_to_region, n_total_loci, n_regions)
                        except ValueError:
                            propagation_failed_here = True
                            break
                        if exceeds_max_copy_number(strands_regions[d]):
                            propagation_failed_here = True
                            break
                        desc_stack.extend(children[d])
                    if propagation_failed_here:
                        propagation_failed = True
                        break
                    placed = True
                    break

                if propagation_failed:
                    print(f"Attempt {attempt+1}: propagation conflict placing CNA '{cna_type}', retrying ordering.")
                    failed = True; break

                if not placed:
                    print(f"Attempt {attempt+1}: could not place CNA '{cna_type}', retrying ordering.")
                    failed = True; break

        if not failed:
            print(f"Succeeded on attempt {attempt+1}.")
            committed_node_events      = node_events
            committed_strands_regions  = strands_regions
            committed_strands_loci_ref = strands_loci_ref
            committed_strands_loci_alt = strands_loci_alt
            break

    if committed_node_events is None:
        raise RuntimeError(
            f"Could not find a valid event ordering after {max_ordering_attempts} attempts. "
            "Try reducing the number of events or increasing max_ordering_attempts.")

    strands_regions  = committed_strands_regions
    strands_loci_ref = committed_strands_loci_ref
    strands_loci_alt = committed_strands_loci_alt
    node_events      = committed_node_events

    # Per-node SNV and CNA lists for the output files.
    SNVs = [[] for _ in range(n_nodes)]
    CNAs = [[] for _ in range(n_nodes)]
    for n in range(n_nodes):
        for ev in node_events[n]:
            if ev['type'] == 'SNV':
                SNVs[n].append(ev['locus'])
            else:
                CNAs[n].append(ev)

    # Per-node copy-number and allele-count arrays, checked for consistency.
    cn_regions_arr  = np.zeros((n_regions, n_nodes), dtype=int)
    n_ref_allele_arr = np.zeros((n_total_loci, n_nodes), dtype=int)
    n_alt_allele_arr = np.zeros((n_total_loci, n_nodes), dtype=int)

    for n in range(n_nodes):
        cn_regions_arr[:, n] = sum(strands_regions[n])
        for s in range(len(strands_loci_ref[n])):
            n_ref_allele_arr[:, n] += strands_loci_ref[n][s]
            n_alt_allele_arr[:, n] += strands_loci_alt[n][s]

    for n in range(n_nodes):
        for locus in range(n_total_loci):
            r   = all_locus_to_region[locus]
            cn  = cn_regions_arr[r, n]
            ref = n_ref_allele_arr[locus, n]
            alt = n_alt_allele_arr[locus, n]
            assert ref >= 0 and alt >= 0, \
                f"Negative allele count at locus {locus} node {n}"
            assert ref + alt == cn, (
                f"Allele count mismatch at locus {locus} node {n}: "
                f"ref={ref} alt={alt} cn={cn}")

    # region_dispersion is a per-region concentration shared by the SNV and CNA read models
    # (alpha_region = region_dispersion * region_weights), drawn log-uniformly.
    if region_calibration is None:
        region_dispersion = np.exp(rng.uniform(
            np.log(region_dispersion_min), np.log(region_dispersion_max), n_regions))

        # region_weights: relative read share of each region in a diploid cell (sums to 1).
        region_weights = np.clip(np.abs(rng.normal(1.0, region_weight_sigma, n_regions)), 0.2, None)
        region_weights /= region_weights.sum()
    else:
        # Resample the calibrated (weight, dispersion) pairs jointly, with multiplicative jitter.
        cal_w = np.asarray(region_calibration["region_weight_relative"], dtype=float)
        cal_d = np.asarray(region_calibration["region_dispersion"], dtype=float)
        pick = rng.integers(0, len(cal_w), n_regions)
        region_dispersion = cal_d[pick] * np.exp(rng.normal(0.0, 0.1, n_regions))
        region_weights = cal_w[pick] * np.exp(rng.normal(0.0, 0.05, n_regions))
        region_weights /= region_weights.sum()

    # Per-node probability of a cycling (self-doubled) cell: low at the root, random in
    # [cycling_prob_min, cycling_prob_max] for tumour nodes.
    cycling_prob = np.zeros(n_nodes)
    cycling_prob[0] = root_cycling_prob
    if n_nodes > 1:
        cycling_prob[1:] = rng.uniform(cycling_prob_min, cycling_prob_max, n_nodes - 1)

    # Longitudinal setup: a node only contributes cells to samples from its birth sample onward.
    birth_sample = assign_birth_samples(parents, dft_order, n_samples, rng)

    directory = os.path.dirname(basename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    node_probabilities_by_sample = []
    healthy_cells_by_sample = []
    cell_attachments_by_sample = []
    singlet_by_sample = []
    for s in range(n_samples):
        alive_mask = (birth_sample <= s)
        node_probabilities_s = sample_node_probabilities(
            alive_mask, healthy_fraction, nodeprobs_concentration, rng)
        node_probabilities_by_sample.append(node_probabilities_s)

        # Multi-sample files are named {basename}_1, {basename}_2, ... as expected by SCATI.cpp.
        sample_basename = basename if n_samples == 1 else f"{basename}_{s + 1}"
        healthy_cells_s, cell_attachments_s, singlet_s = generate_sample_reads(
            sample_basename, n_cells,
            cn_regions_arr, n_ref_allele_arr, n_alt_allele_arr,
            node_probabilities_s, region_weights, region_dispersion, root_cn_regions,
            n_total_loci, n_SNVs, all_locus_to_region,
            dropout_rate, background_rate, allelic_error_rate,
            mean_total_count, theta_total_count,
            mean_snv_depth, theta_snv_depth,
            doublet_rate, cycling_prob,
            chr_labels_by_region, region_names, region_names_suffix,
            rng, noise_weight=noise_weight, n_het=n_het)
        healthy_cells_by_sample.append(healthy_cells_s)
        cell_attachments_by_sample.append(cell_attachments_s)
        singlet_by_sample.append(singlet_s)

    # Write the whole-tree output files.
    # The root allele counts cover all loci, since false and het loci also carry germline genotypes.
    pd.DataFrame(n_ref_allele_arr[:, 0]).to_csv(
        basename + "_n_ref_root.csv", index=False, header=False, lineterminator="\n")
    pd.DataFrame(n_alt_allele_arr[:, 0]).to_csv(
        basename + "_n_alt_root.csv", index=False, header=False, lineterminator="\n")

    strand_data = {}
    for n in range(n_nodes):
        strand_data[str(n)] = {
            'strands_regions':  [s.tolist() for s in strands_regions[n]],
            'strands_loci_ref': [s.tolist() for s in strands_loci_ref[n]],
            'strands_loci_alt': [s.tolist() for s in strands_loci_alt[n]],
        }
    with open(basename + "_strand_genotypes.json", "w") as f:
        json.dump(strand_data, f)

    def make_serializable(ev):
        out = dict(ev)
        if 'regions' in out:
            out['regions'] = sorted(out['regions'])
        return out

    with open(basename + "_events_ordered.json", "w") as f:
        json.dump({str(n): [make_serializable(ev) for ev in node_events[n]]
                   for n in range(n_nodes)}, f, indent=2)

    # True model parameters, for evaluation against ground truth.
    with open(basename + "_true_model_params.json", "w") as f:
        json.dump({
            "sex": sex,
            "n_samples": n_samples,
            "birth_sample": birth_sample.tolist(),
            "root_cn_regions": root_cn_regions.tolist(),
            "region_is_X": region_is_X.tolist(),
            "region_is_Y": region_is_Y.tolist(),
            "region_dispersion": region_dispersion.tolist(),
            "region_weights": region_weights.tolist(),
            "node_probabilities_by_sample": [np_s.tolist() for np_s in node_probabilities_by_sample],
            "healthy_fraction": healthy_fraction,
            "cycling_prob": cycling_prob.tolist(),
            "dropout_rate": dropout_rate,
            "background_rate": background_rate,
            "allelic_error_rate": allelic_error_rate,
            "doublet_rate": doublet_rate,
            "cycling_cell_factor": cycling_cell_factor,
            "mean_total_count": mean_total_count,
            "theta_total_count": theta_total_count,
            "noise_weight": noise_weight,
            "max_copy_number": MAX_COPY_NUMBER,
            "n_SNVs": n_SNVs,
            "n_false_SNVs": n_false_SNVs,
            "n_het": n_het,
            "random_phase": bool(random_phase),
            "all_locus_to_region": [int(r) for r in all_locus_to_region],
        }, f, indent=2)

    if n_het > 0:
        pd.DataFrame({"locus": germline_het_indices, "region": het_to_region}).to_csv(
            basename + "_germline_het_loci.csv", index=False, lineterminator="\n")

    # True tree as a Graphviz file.
    colors = ["lightcoral", "skyblue3", "sandybrown", "paleturquoise3",
              "thistle", "darkolivegreen3", "lightpink", "mediumpurple",
              "darkseagreen3", "navajowhite", "gold"]

    def region_span(cna):
        chr_regs = chromosome_to_regions[cna['chromosome']]
        ordinals = sorted(chr_regs.index(r) + 1 for r in cna['regions'])
        first, last = ordinals[0], ordinals[-1]
        return f"region {first}" if first == last else f"regions {first}-{last}"

    def cna_label(cna):
        t = cna['type']
        if t == 'FGD':
            return "FGD"
        chr_label = chr_labels[cna['chromosome']]
        if t in ('aneuploidy_gain', 'aneuploidy_loss'):
            kind = "Gain" if t == 'aneuploidy_gain' else "Loss"
            return f"Aneuploidy {kind} chr{chr_label} ({len(cna['chr_regions'])} regions)"
        elif t == 'local_gain':    return f"Gain chr{chr_label} {region_span(cna)}"
        elif t == 'local_loss':    return f"Loss chr{chr_label} {region_span(cna)}"
        elif t == 'CNLOH':         return f"CNLOH chr{chr_label} {region_span(cna)}"
        return t

    # One sample: bubbles show the node probability. Several samples: bubbles are pies of
    # the node's singlet cells per sample, and labels show the birth sample.
    final_node_probabilities = node_probabilities_by_sample[-1]
    counts_by_sample = np.array([
        np.bincount(np.asarray(att)[sing], minlength=n_nodes)
        for att, sing in zip(cell_attachments_by_sample, singlet_by_sample)])
    with open(basename + "_treeTRUE.gv", "w") as f:
        f.write("digraph G{\n")
        f.write("node [color=dimgray fontsize=24 fontcolor=black fontname=Helvetica penwidth=5];\n")
        for n in range(1, n_nodes):
            f.write(f"{parents[n]} -> {n} [color=dimgray penwidth=4 weight=2];\n")
        for n in range(n_nodes):
            label = f'{n}[label=<'
            if n_samples > 1:
                label += f"(sample {birth_sample[n]})<br/>"
            for locus in SNVs[n]:
                direction = "alt-ref" if snv_direction[locus] else "ref-alt"
                label += f"SNV{locus}({direction})<br/>"
            for cna in CNAs[n]:
                label += cna_label(cna) + "<br/>"
            if not SNVs[n] and not CNAs[n]:
                label += " "
            label += ">];\n"
            f.write(label)
        for n in range(n_nodes):
            c = colors[n % len(colors)]
            f.write(f"{n} -> {n + n_nodes} [dir=none style=dashed weight=1 penwidth=5 color={c}];\n")
        if n_samples == 1:
            for n in range(n_nodes):
                size = np.sqrt(100.0 * final_node_probabilities[n]) / 3.0
                f.write(f'{n + n_nodes}[label="{100 * final_node_probabilities[n]:.2f}\\%" '
                        f'style=filled width={size:.3f} height={size:.3f} color={colors[n % len(colors)]}];\n')
        else:
            f.write(pie_bubbles_dot(counts_by_sample, n_nodes, colors))
        f.write("}")

    print(f"Done. Files written to {basename}_*"
          + ("" if n_samples == 1 else f" (and {basename}_1.. {basename}_{n_samples}.. per-sample files)"))
    return {
        'parents': parents, 'children': children,
        'SNVs': SNVs, 'CNAs': CNAs,
        'birth_sample': birth_sample.tolist(),
        'node_probabilities_by_sample': node_probabilities_by_sample,
        'cell_attachments_by_sample': cell_attachments_by_sample,
    }


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',                      type=str,   default="data/test")
    parser.add_argument('--seed',                  type=int,   default=1)
    parser.add_argument('--ncells',                type=int,   default=2000,
                        help="Cells per sample.")
    parser.add_argument('--nnodes',                type=int,   default=5)
    parser.add_argument('--nSNVs',                 type=int,   default=10)
    parser.add_argument('--false_SNV_fraction',    type=float, default=FALSE_SNV_FRACTION,
                        help="Fraction of all loci that are false (never mutated).")
    parser.add_argument('--nLocalCNAs',            type=int,   default=2)
    parser.add_argument('--nAneuploidy',           type=int,   default=1)
    parser.add_argument('--nFGD',                  type=int,   default=1)
    parser.add_argument('--nCNLOH',                type=int,   default=1)
    parser.add_argument('--nregions',              type=int,   default=50)
    parser.add_argument('--nchromosomes',          type=int,   default=10)
    parser.add_argument('--nsamples',               type=int,   default=1,
                        help="Number of longitudinal samples.")
    parser.add_argument('--sex',                    type=str,   default="male",
                        choices=["male", "female"],
                        help="Sex, which sets the root ploidy of chrX and chrY.")
    parser.add_argument('--nodeprob',              type=float, default=10.0)
    parser.add_argument('--healthy_fraction',      type=float, default=0.15,
                        help="Fraction of healthy (root) cells in every sample.")
    parser.add_argument('--root_het_fraction',     type=float, default=0.2)
    parser.add_argument('--root_hom_alt_fraction', type=float, default=0.05)
    parser.add_argument('--alt_to_ref_fraction',   type=float, default=0.1,
                        help="Fraction of real SNVs that are alt->ref back mutations.")
    parser.add_argument('--nhet',                  type=int,   default=0,
                        help="Number of germline het loci without a somatic event.")
    parser.add_argument('--random_phase',          type=int,   default=0,
                        help="1: random phase of germline het loci; 0: alt allele always on strand 1.")
    parser.add_argument('--allow_empty_regions',   type=int,   default=1)
    parser.add_argument('--max_ordering_attempts', type=int,   default=200)

    # Read-count model parameters
    parser.add_argument('--dropout_rate',          type=float, default=0.03,
                        help="Per-copy dropout rate for SNVs and CNAs.")
    parser.add_argument('--allelic_error_rate', type=float, default=0.02,
                        help="Allelic error rate.")
    parser.add_argument('--background_rate',       type=float, default=0.04,
                        help="Background rate for fully dropped-out loci and regions.")
    parser.add_argument('--noise_weight',          type=float, default=NOISE_WEIGHT,
                        help="Weight of the region-count noise component.")
    parser.add_argument('--region_dispersion_min', type=float, default=50.0,
                        help="Lower bound of the log-uniform region dispersion.")
    parser.add_argument('--region_dispersion_max', type=float, default=2000.0)
    parser.add_argument('--region_weight_sigma',   type=float, default=0.3,
                        help="Spread of the region weights.")
    parser.add_argument('--mean_total_count',      type=float, default=10000.0,
                        help="Expected reads per unit copy number.")
    parser.add_argument('--theta_total_count',     type=float, default=10.0,
                        help="Negative-binomial size for total read counts.")
    parser.add_argument('--mean_snv_depth',        type=float, default=100.0,
                        help="Expected SNV depth in a diploid region.")
    parser.add_argument('--theta_snv_depth',       type=float, default=10.0,
                        help="Negative-binomial size for SNV depth.")

    # Doublets and cycling cells
    parser.add_argument('--doubletrate',           type=float, default=0.2,
                        help="Doublet rate.")
    parser.add_argument('--root_cycling_prob',     type=float, default=0.0,
                        help="Probability that a root cell is cycling (self-doubled).")
    parser.add_argument('--cycling_prob_min',      type=float, default=0.0,
                        help="Tumour nodes draw their cycling probability uniformly in [min, max].")
    parser.add_argument('--cycling_prob_max',      type=float, default=0.0)
    parser.add_argument('--cycling_cell_factor',   type=float, default=2.0,
                        help="Only recorded in _true_model_params.json.")
    args = parser.parse_args()

    n_regions = args.nregions
    n_chr     = args.nchromosomes
    base_sz   = n_regions // n_chr
    remainder = n_regions % n_chr
    chromosome_sizes = [base_sz + (1 if i < remainder else 0) for i in range(n_chr)]

    generate_data(
        basename=args.o,
        n_cells=args.ncells,
        n_nodes=args.nnodes,
        n_SNVs=args.nSNVs,
        n_false_SNVs=n_false_snvs_for(args.nSNVs, args.false_SNV_fraction),
        n_local_CNAs=args.nLocalCNAs,
        n_aneuploidy=args.nAneuploidy,
        n_FGD=args.nFGD,
        n_CNLOH=args.nCNLOH,
        n_regions=n_regions,
        chromosome_sizes=chromosome_sizes,
        allow_empty_regions=(args.allow_empty_regions == 1),
        sex=args.sex,
        root_het_fraction=args.root_het_fraction,
        root_hom_alt_fraction=args.root_hom_alt_fraction,
        alt_to_ref_fraction=args.alt_to_ref_fraction,
        nodeprobs_concentration=args.nodeprob,
        healthy_fraction=args.healthy_fraction,
        dropout_rate=args.dropout_rate,
        region_dispersion_min=args.region_dispersion_min,
        region_dispersion_max=args.region_dispersion_max,
        region_weight_sigma=args.region_weight_sigma,
        background_rate=args.background_rate,
        mean_total_count=args.mean_total_count,
        theta_total_count=args.theta_total_count,
        mean_snv_depth=args.mean_snv_depth,
        theta_snv_depth=args.theta_snv_depth,
        doublet_rate=args.doubletrate,
        root_cycling_prob=args.root_cycling_prob,
        cycling_prob_min=args.cycling_prob_min,
        cycling_prob_max=args.cycling_prob_max,
        cycling_cell_factor=args.cycling_cell_factor,
        allelic_error_rate=args.allelic_error_rate,
        seed=args.seed,
        n_samples=args.nsamples,
        max_ordering_attempts=args.max_ordering_attempts,
        noise_weight=args.noise_weight,
        n_het=args.nhet,
        random_phase=(args.random_phase == 1),
    )
