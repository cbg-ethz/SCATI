#include <algorithm>
#include <cmath>
#include <vector>
#include <cfloat>
#include <limits>
#include <stdexcept>

#include "Scores.h"
#include "Node.h"
#include "Structures.h"

extern int n_loci;
extern int n_cells;
extern int n_regions;
extern std::vector<Cell> cells;

// The binomial coefficients of the read counts depend only on the data, so they are computed once.
Scores::Scores(const Params& params) : local_params(params) {

    // One memory budget in bytes for the four CNA caches together. Params::cache_gb is per sample,
    cache_budget_bytes = (size_t)(params.cache_gb * data.n_samples * 1024.0 * 1024.0 * 1024.0);

    log_n_choose_region_count.resize(n_regions, std::vector<double>(n_cells));
    for (int j = 0; j < n_cells; ++j) {
        int N = cells[j].total_counts;
        double lgamma_N_plus_1 = std::lgamma(N + 1);
        for (int r = 0; r < n_regions; ++r) {
            int n = cells[j].region_counts[r];
            log_n_choose_region_count[r][j] = lgamma_N_plus_1
                                              - std::lgamma(n + 1)
                                              - std::lgamma(N - n + 1);
        }
    }

    log_n_choose_snv_count.resize(n_loci, std::vector<double>(n_cells));
    for (int locus = 0; locus < n_loci; ++locus) {
        for (int j = 0; j < n_cells; ++j) {
            int alt = cells[j].alt_counts[locus];
            int ref = cells[j].ref_counts[locus];
            log_n_choose_snv_count[locus][j] = std::lgamma(alt + ref + 1.0)
                                               - std::lgamma(alt + 1.0)
                                               - std::lgamma(ref + 1.0);
        }
    }
}

// SNV: joint likelihood marginalizing over dropout
const std::vector<double>& Scores::compute_SNV_loglikelihoods_joint(
        int c_ref, int c_alt, int locus,
        double omega, double dropout_rate) {

    // All copies absent: the alt count is Beta-Binomial with alpha = beta = omega_bg, i.e. total concentration omega*background_rate,
    // the same concentration that a region of background_rate copies has in the region count model (alpha_k(0) = rho*phi*background_rate).
    double omega_bg = 0.5 * omega * local_params.background_rate;
    int precision = local_params.hash_precision; // for discretization of continuous parameters in the cache

    int discretized_omega_bg = (int)std::round(omega_bg * precision);
    // omega_bg * precision must stay below OMEGA_BG_BASE so that the locus field does not overlap.
    constexpr int64_t OMEGA_BG_BASE = 200000000;
    if (discretized_omega_bg < 0 || discretized_omega_bg >= OMEGA_BG_BASE)
        throw std::overflow_error("SNV background hash overflow");
    int64_t hash_bg = (int64_t)discretized_omega_bg + OMEGA_BG_BASE * (int64_t)locus;

    auto it_bg = cache_dropoutscores_bg.find(hash_bg);
    if (it_bg == cache_dropoutscores_bg.end()) { // Background noise usually leads to few total reads and thus mostly 0/1 VAF
        std::vector<double> bg_scores(n_cells);
        double lgamma_2_omega_bg = 2.0 * std::lgamma(omega_bg);
        double lgamma_2omega_bg  = std::lgamma(2.0 * omega_bg);
        for (int j = 0; j < n_cells; ++j) {
            int alt = cells[j].alt_counts[locus];
            int ref = cells[j].ref_counts[locus];
            bg_scores[j] = std::lgamma(alt + omega_bg)         // Γ(x + α)
                           + std::lgamma(ref + omega_bg)         // Γ(n - x + β)
                           - std::lgamma(alt + ref + 2.0 * omega_bg)  // - Γ(n + α + β)
                           - lgamma_2_omega_bg                  // - Γ(α) - Γ(β)  [α = β = omega_bg]
                           + lgamma_2omega_bg                   // + Γ(α + β)
                           + log_n_choose_snv_count[locus][j];            // + Γ(n + 1) -  Γ(x + 1) - Γ(n - x + 1)
        }
        it_bg = cache_dropoutscores_bg.emplace(hash_bg, std::move(bg_scores)).first;
    }
    const std::vector<double>& bg_scores = it_bg->second;

    if (c_ref == 0 && c_alt == 0) {
        return bg_scores; // in this case we only expect background noise
    }

    int discretized_omega   = (int)std::round(omega*precision/10);
    int discretized_dropout = (int)std::round(dropout_rate * precision);

    constexpr int64_t DROPOUT_BASE = 1500;      // discretized_dropout in [0, precision(1000)]
    constexpr int64_t OMEGA_BASE   = 20000000;  // discretized_omega up to ~1e7
    if (discretized_omega < 0 || discretized_omega >= OMEGA_BASE)
        throw std::overflow_error("SNV joint hash overflow");
    int64_t hash_params = (int64_t)discretized_dropout + DROPOUT_BASE * (int64_t)discretized_omega;
    // Doublets sum two nodes' allele counts uncapped, so c_ref/c_alt can reach up to 2*max_copy_number.
    int max_cn = 2 * local_params.max_copy_number + 1;
    int64_t hash_locus = c_ref + max_cn * c_alt + max_cn * max_cn * locus;

    auto it_params = cache_dropoutrate_cellscores.find(hash_params);
    if (it_params != cache_dropoutrate_cellscores.end()) {
        auto it_locus = it_params->second.find(hash_locus);
        if (it_locus != it_params->second.end())
            return it_locus->second;
    }

    // ── Build per-(k,l) BetaBin scores ───────────────────────────
    double eps  = 1e-10;
    double eps_allelic = local_params.allelic_error_rate;
    double all_dropout  = std::pow(dropout_rate, c_ref + c_alt);
    double log_all_dropout = std::log(all_dropout + eps);

    // result: log-sum-exp over all (k,l) configurations + background
    std::vector<double> result(n_cells);
    for (int j = 0; j < n_cells; ++j)
        result[j] = log_all_dropout + bg_scores[j];  // all-dropout

    for (int k = 0; k <= c_ref; ++k) {
        for (int l = 0; l <= c_alt; ++l) {
            if (k == 0 && l == 0) continue;

            // log probability of this dropout configuration
            double log_p_kl = log_n_choose_k(c_ref, k)
                              + log_n_choose_k(c_alt, l)
                              + k * std::log(1.0 - dropout_rate + eps)
                              + (c_ref - k) * std::log(dropout_rate + eps)
                              + l  * std::log(1.0 - dropout_rate + eps)
                              + (c_alt - l) * std::log(dropout_rate + eps);

            // expected VAF given the surviving copies: each copy contributes an alternative fraction of eps (reference copy) or 1 - eps (alternative copy)
            double f;
            double ea;
            if (data.variant_is_SNV[locus]) {
                ea = eps_allelic;
            } else {
                ea = local_params.allelic_error_rate_indel;
            }

            f = (l * (1.0 - ea) + k * ea) / (k + l);

            // omega scaled by surviving copy count
            double scale = k + l;
            double alpha_alt  = omega * scale * f;
            double alpha_ref  = omega * scale * (1.0 - f);

            int64_t hash_kl = k + max_cn * l + (int64_t)max_cn * max_cn * locus
                              + (int64_t)n_loci * max_cn * max_cn * discretized_omega;

            auto it_kl = cache_dropoutscores.find(hash_kl);
            if (it_kl == cache_dropoutscores.end()) {
                std::vector<double> kl_scores(n_cells);
                double lgamma_alpha_alt      = std::lgamma(alpha_alt);
                double lgamma_alpha_ref      = std::lgamma(alpha_ref);
                double lgamma_alpha_alt_ref  = std::lgamma(alpha_alt + alpha_ref);
                for (int j = 0; j < n_cells; ++j) {
                    int alt = cells[j].alt_counts[locus];
                    int ref = cells[j].ref_counts[locus];
                    int m  = alt + ref;
                    kl_scores[j] = std::lgamma(alt + alpha_alt)                 // Γ(x + α)
                                   + std::lgamma(ref + alpha_ref)                   // Γ(n - x + β)
                                   - std::lgamma(m + alpha_alt + alpha_ref)      // - Γ(n + α + β)
                                   - lgamma_alpha_alt                              // - Γ(α)
                                   - lgamma_alpha_ref                              // - Γ(β)
                                   + lgamma_alpha_alt_ref                          // + Γ(α + β)
                                   + log_n_choose_snv_count[locus][j];               //  + Γ(n + 1) -  Γ(x + 1) - Γ(n - x + 1)

                }
                it_kl = cache_dropoutscores.emplace(hash_kl, std::move(kl_scores)).first;
            }
            const std::vector<double>& kl_scores = it_kl->second;

            for (int j = 0; j < n_cells; ++j)
                result[j] = log_sum_exp(result[j], log_p_kl + kl_scores[j]);
        }
    }

    return cache_dropoutrate_cellscores[hash_params][hash_locus] = std::move(result);
}


// CNA: joint likelihood marginalizing over per-region dropout
// cn_r:  copy number of this region
const std::vector<double>& Scores::compute_CNA_loglikelihoods_joint(
        int region, int cn_r,
        double alpha_per_copy, double alpha_total,
        double dropout_rate, int sample) {

    // Cells scored by this call: all of them (sample == -1) or just one sample's
    const std::vector<int>* subset = (sample >= 0) ? &data.cells_by_sample.at(sample) : nullptr;
    const int n_c = subset ? (int)subset->size() : n_cells;
    auto cell_of = [&](int t) { return subset ? (*subset)[t] : t; };

    double eps = 1e-10;
    double background_rate = local_params.background_rate;
    int precision = local_params.hash_precision / 10;
    // Noise component, mixed in with weight NOISE_WEIGHT: the main model's mean without dropout, but a small fixed
    // alpha_r, so it is wide with a mild peak at n = 0. Far from the mean, two genotypes with the same mean then
    // share the same floor instead of one overtaking the other.
    constexpr double NOISE_ALPHA_R_FIXED = 0.9;
    constexpr double NOISE_WEIGHT = 0.02;
    int discretized_alpha_per_copy = (int)std::round(alpha_per_copy * precision);
    int discretized_alpha_total = (int)std::round(alpha_total * precision / 10);
    int discretized_dropout = (int)std::round(dropout_rate * precision);
    constexpr int64_t ALPHA_PER_COPY_BASE = 100000;   // covers alpha_per_copy up to ~1000 (observed: ~1-5)
    constexpr int64_t ALPHA_TOTAL_BASE = 1000000;  // covers alpha_total up to ~100000 (observed: ~200-2000)
    constexpr int64_t DROPOUT_BASE  = 150;  // dropout_rate in [0,1] discretized at `precision`
    const int64_t CN_R_BASE =  2 * local_params.max_copy_number + 1;       // cn_r up to 2*max_copy_number for doublets

    int64_t hash_full = (int64_t)discretized_alpha_per_copy
                       + ALPHA_PER_COPY_BASE * (int64_t)discretized_alpha_total
                       + ALPHA_PER_COPY_BASE * ALPHA_TOTAL_BASE * (int64_t)discretized_dropout
                       + ALPHA_PER_COPY_BASE * ALPHA_TOTAL_BASE * DROPOUT_BASE * (int64_t)cn_r
                       + ALPHA_PER_COPY_BASE * ALPHA_TOTAL_BASE * DROPOUT_BASE * CN_R_BASE * (int64_t)region;

    if (hash_full < 0)
        throw std::overflow_error("CNA joint cache hash overflow");
    const int64_t hash_stride = ALPHA_PER_COPY_BASE * ALPHA_TOTAL_BASE * DROPOUT_BASE * CN_R_BASE * (int64_t)n_regions;
    if ((int64_t)(sample + 1) > (INT64_MAX / hash_stride) - 1)
        throw std::overflow_error("CNA joint cache hash overflow (too many samples)");
    hash_full += (int64_t)(sample + 1) * hash_stride;

    auto it = cache_cna_joint.find(hash_full);
    ++cna_joint_calls;

    if (it != cache_cna_joint.end())
        return it->second;
    ++cna_joint_misses;
    enforce_cache_budget(); // before any reference into a cache is taken below

    // --- cell counts
    std::vector<int> n_vec(n_c), N_vec(n_c);
    for (int t = 0; t < n_c; ++t) {
        const int j = cell_of(t);
        n_vec[t] = cells[j].region_counts[region];
        N_vec[t] = cells[j].total_counts;
    }

    std::vector<double> result(n_c, -std::numeric_limits<double>::infinity());

    const int discretized_alpha_total_fine = (int)std::round(alpha_total * precision);
    const int64_t key_at = (int64_t)discretized_alpha_total_fine + ((int64_t)(sample + 1) << 32);
    auto it_at = cache_lgamma_N_plus_alpha_total.find(key_at);
    if (it_at == cache_lgamma_N_plus_alpha_total.end()) {
        std::vector<double> vec(n_c);
        for (int t = 0; t < n_c; ++t)
            vec[t] = std::lgamma(N_vec[t] + alpha_total);
        it_at = cache_lgamma_N_plus_alpha_total.emplace(key_at, std::move(vec)).first;
        count_cache_bytes(CACHE_LGAMMA_N_AT, n_c);
        cache_lgamma_alpha_total[key_at] = std::lgamma(alpha_total);
    }
    const std::vector<double>& lgamma_N_plus_alpha_total = it_at->second;
    const double lgamma_alpha_total = cache_lgamma_alpha_total[key_at];

    // Bounds for the sub-cache key below: k ranges over [0, cn_r], and cn_r
    // can reach 2*max_copy_number for doublets.
    const int64_t max_k_digit = 2 * (int64_t)local_params.max_copy_number + 2;
    const int64_t region_digit_base = max_k_digit;
    const int64_t alpha_digit_base  = region_digit_base * (n_regions + 1);

    for (int k = 0; k <= cn_r; ++k) {

        const double log_p_k = log_n_choose_k(cn_r, k)
                               + k * std::log(1.0 - dropout_rate + eps)
                               + (cn_r - k) * std::log(dropout_rate + eps);

        const double alpha_r = (k == 0) ? alpha_per_copy * background_rate
                                        : alpha_per_copy * k;
        const double alpha_others = alpha_total - alpha_r;   // α_total - ω_r·k_r

        const double k_const = lgamma_alpha_total
                               - std::lgamma(alpha_r)
                               - std::lgamma(alpha_others)
                               + log_p_k;

        int64_t hash_alpha_r = k
                               + region_digit_base * region
                               + alpha_digit_base * discretized_alpha_per_copy
                               + (int64_t)(sample + 1) * alpha_digit_base * ALPHA_PER_COPY_BASE;  // per sample
        auto it_ar = cache_cna_betabin_k.find(hash_alpha_r);
        if (it_ar == cache_cna_betabin_k.end()) {
            std::vector<double> lgamma_n_plus_alpha_r(n_c);
            for (int t = 0; t < n_c; ++t)
                lgamma_n_plus_alpha_r[t] = std::lgamma(n_vec[t] + alpha_r);
            it_ar = cache_cna_betabin_k.emplace(hash_alpha_r, std::move(lgamma_n_plus_alpha_r)).first;
            count_cache_bytes(CACHE_BETABIN_K, n_c);
        }
        const std::vector<double>& lgamma_n_plus_alpha_r = it_ar->second;

        for (int t = 0; t < n_c; ++t) {
            const double val = lgamma_n_plus_alpha_r[t]
                               + std::lgamma(N_vec[t] - n_vec[t] + alpha_others)
                               + log_n_choose_region_count[region][cell_of(t)]
                               - lgamma_N_plus_alpha_total[t]
                               + k_const;
            result[t] = log_sum_exp(result[t], val);
        }
    }

    // Noise component
    {
        const double mean_frac = (cn_r == 0)
                                 ? (alpha_per_copy * background_rate) / alpha_total
                                 : (alpha_per_copy * cn_r) / alpha_total;
        const double alpha_total_noise = NOISE_ALPHA_R_FIXED / mean_frac;
        const double alpha_others_noise = alpha_total_noise - NOISE_ALPHA_R_FIXED;
        const double k_const_noise = std::lgamma(alpha_total_noise)
                                     - std::lgamma(NOISE_ALPHA_R_FIXED)
                                     - std::lgamma(alpha_others_noise);

        const int key_noise = region + (sample + 1) * n_regions;   // per sample (sample == -1 -> plain region)
        auto it_noise = cache_noise_lgamma_n_plus_r.find(key_noise);
        if (it_noise == cache_noise_lgamma_n_plus_r.end()) {
            std::vector<double> lgamma_n_plus_r_noise(n_c);
            for (int t = 0; t < n_c; ++t)
                lgamma_n_plus_r_noise[t] = std::lgamma(n_vec[t] + NOISE_ALPHA_R_FIXED);
            it_noise = cache_noise_lgamma_n_plus_r.emplace(key_noise, std::move(lgamma_n_plus_r_noise)).first;
            count_cache_bytes(CACHE_NOISE, n_c);
        }
        const std::vector<double>& lgamma_n_plus_r_noise = it_noise->second;

        for (int t = 0; t < n_c; ++t) {
            const double lp_noise = lgamma_n_plus_r_noise[t]
                                    + std::lgamma(N_vec[t] - n_vec[t] + alpha_others_noise)
                                    + log_n_choose_region_count[region][cell_of(t)]
                                    - std::lgamma(N_vec[t] + alpha_total_noise)
                                    + k_const_noise;
            const double a = std::log(1.0 - NOISE_WEIGHT) + result[t];
            const double b = std::log(NOISE_WEIGHT) + lp_noise;
            result[t] = log_sum_exp(a, b);
        }
    }

    count_cache_bytes(CACHE_CNA_JOINT, n_c);
    return cache_cna_joint.emplace(hash_full, std::move(result)).first->second;
}


// COMPASS betabinomial likelihood model with fixed dispersion.
std::vector<double> Scores::compute_SNV_loglikelihoods(int c_ref, int c_alt, int locus,
                                                       double dropout_rate_ref, double dropout_rate_alt) {

    constexpr double EPS = 1e-7;
    const int64_t max_cn  = 2 * (int64_t)local_params.max_copy_number + 1;
    const int64_t max_cn2 = max_cn * max_cn;

    // Complete homozygous deletion — no alleles present, any observed reads are errors.
    // Treat as complete dropout: use the (k=0, l=0) Beta-Binomial with f=0.5.
    if (c_ref == 0 && c_alt == 0) {
        int64_t hash_dropout = 0 + max_cn * 0 + max_cn2 * locus; // k=0, l=0
        if (cache_dropoutscores.count(hash_dropout))
            return cache_dropoutscores[hash_dropout];

        std::vector<double> scores(n_cells);
        double omega = local_params.omega_het;
        double f     = 0.5;
        double lgamma_omega_f      = std::lgamma(omega * f);
        double lgamma_omega_1mf    = std::lgamma(omega * (1 - f));
        double lgamma_omega        = std::lgamma(omega);
        for (int j = 0; j < n_cells; ++j) {
            int ref_count = cells[j].ref_counts[locus];
            int alt_count = cells[j].alt_counts[locus];
            scores[j] = std::lgamma(alt_count + omega * f)
                        + std::lgamma(ref_count + omega * (1 - f))
                        - std::lgamma(alt_count + ref_count + omega)
                        - lgamma_omega_f
                        - lgamma_omega_1mf
                        + lgamma_omega;
        }
        cache_dropoutscores[hash_dropout] = scores;
        return scores;
    }

    // If homozygous, copy number of the only allele is irrelevant for allelic proportion
    if      (c_ref == 0 && c_alt != 0) c_alt = 1;
    else if (c_alt == 0 && c_ref != 0) c_ref = 1;

    int64_t hash = c_ref + max_cn * c_alt + max_cn2 * locus;

    int discretized_dropout_rate_ref = (int)std::round(dropout_rate_ref * 1000);
    int discretized_dropout_rate_alt = (int)std::round(dropout_rate_alt * 1000);
    int hash_dropoutrates = discretized_dropout_rate_ref + 1000 * discretized_dropout_rate_alt;

    if (cache_dropoutrate_cellscores.count(hash_dropoutrates) &&
        cache_dropoutrate_cellscores[hash_dropoutrates].count(hash))
        return cache_dropoutrate_cellscores[hash_dropoutrates][hash];

    dropout_rate_ref = std::min(1.0 - EPS, std::max(EPS, discretized_dropout_rate_ref / 1000.0));
    dropout_rate_alt = std::min(1.0 - EPS, std::max(EPS, discretized_dropout_rate_alt / 1000.0));

    std::vector<std::vector<double>> likelihood_alleles_cells((c_ref + 1) * (c_alt + 1) - 1);
    std::vector<double> dropoutscores(n_cells);

    int idx = 0;
    for (int k = 0; k <= c_ref; ++k) {
        for (int l = 0; l <= c_alt; ++l) {
            if (k == 0 && l == 0) continue;   // conditioned on at least one surviving copy
            int64_t hash_dropout = k + max_cn * l + max_cn2 * locus;
            if (cache_dropoutscores.count(hash_dropout)) {
                dropoutscores = cache_dropoutscores[hash_dropout];
            } else {
                double omega, eps;
                if (data.variant_is_SNV[locus]) {
                    eps   = local_params.allelic_error_rate;
                    omega = (k == 0 || l == 0) ? local_params.omega_hom : local_params.omega_het;
                } else {
                    eps   = local_params.allelic_error_rate_indel;
                    omega = (k == 0 || l == 0) ? local_params.omega_hom_indel : local_params.omega_het_indel;
                }
                const double f = (1.0 * l * (1 - eps) + 1.0 * k * eps) / (k + l);

                double lgamma_omega_f   = std::lgamma(omega * f);
                double lgamma_omega_1mf = std::lgamma(omega * (1 - f));
                double lgamma_omega     = std::lgamma(omega);
                for (int j = 0; j < n_cells; ++j) {
                    int ref_count = cells[j].ref_counts[locus];
                    int alt_count = cells[j].alt_counts[locus];
                    dropoutscores[j] = std::lgamma(alt_count + omega * f)
                                       + std::lgamma(ref_count + omega * (1 - f))
                                       - std::lgamma(alt_count + ref_count + omega)
                                       - lgamma_omega_f
                                       - lgamma_omega_1mf
                                       + lgamma_omega;
                }
                cache_dropoutscores[hash_dropout] = dropoutscores;
            }

            likelihood_alleles_cells[idx].resize(n_cells);
            double dropout_prob = log_n_choose_k(c_ref, k) + log_n_choose_k(c_alt, l)
                                  + (c_ref - k) * std::log(dropout_rate_ref) + k * std::log(1 - dropout_rate_ref)
                                  + (c_alt - l) * std::log(dropout_rate_alt) + l * std::log(1 - dropout_rate_alt);

            double all_dropout_prob = std::pow(dropout_rate_ref, c_ref) * std::pow(dropout_rate_alt, c_alt);
            dropout_prob -= std::log(1 - all_dropout_prob);

            for (int j = 0; j < n_cells; ++j)
                likelihood_alleles_cells[idx][j] = dropoutscores[j] + dropout_prob;
            ++idx;
        }
    }

    std::vector<double> scores_cells = log_sum_exp_vector(likelihood_alleles_cells);
    cache_dropoutrate_cellscores[hash_dropoutrates][hash] = scores_cells;

    return scores_cells;
}



std::vector<double> Scores::compute_CNA_loglikelihoods_negbinom(int region, double region_proportion) {
    // relative resolution of 0.1%, so that low-weight regions keep distinct copy-number levels
    int64_t discretized_region_proportion = (region_proportion > 0.0)
            ? std::llround(std::log(region_proportion) * 1000) : -999999;
    int64_t hash = region + (int64_t)n_regions * (discretized_region_proportion + 1000000);
    auto it = cache_cnalikelihood_cells.find(hash);
    if (it != cache_cnalikelihood_cells.end()) return it->second;
    enforce_cache_budget();

    double theta = local_params.theta_negbinom;
    std::vector<double> cnv_loglikelihoods(n_cells);

    for (int j = 0; j < n_cells; ++j) {
        double error_rate = local_params.background_rate;
        double mu = cells[j].total_counts * ((1 - error_rate) * region_proportion + error_rate / n_regions); // expected count in the region, with error rate taken into account
        int k = cells[j].region_counts[region];
        cnv_loglikelihoods[j] = std::lgamma(k + theta)
                                - std::lgamma(theta)
                                - std::lgamma(k + 1)
                                + theta * std::log(theta / (theta + mu))
                                + k  * std::log(mu   / (mu   + theta));
    }

    cache_cnalikelihood_cells[hash] = cnv_loglikelihoods;
    count_cache_bytes(CACHE_NEGBINOM, n_cells);
    return cnv_loglikelihoods;
}


void Scores::clear_cache(){
    cache_dropoutrate_cellscores.clear();
    cache_dropoutscores.clear();
    cache_dropoutscores_bg.clear();
    cache_cnalikelihood_cells.clear();
    cache_cna_joint.clear();
    cache_cna_betabin_k.clear();
    cache_lgamma_N_plus_alpha_total.clear();
    cache_lgamma_alpha_total.clear();
    cache_noise_lgamma_n_plus_r.clear();
    for (size_t& b : cache_bytes) b = 0;
}

void Scores::count_cache_bytes(CnaCache which, size_t n_values) {
    cache_bytes[which] += n_values * sizeof(double) + 64; // + rough per-entry overhead (hash node, vector header)
    size_t total = 0;
    for (size_t b : cache_bytes) total += b;
    peak_cache_bytes = std::max(peak_cache_bytes, total);
}

void Scores::enforce_cache_budget() {
    static const char* names[N_CNA_CACHES] = {"cache_cna_joint", "cache_cna_betabin_k",
                                              "cache_lgamma_N_plus_alpha_total", "cache_noise_lgamma_n_plus_r",
                                              "cache_cnalikelihood_cells"};
    for (;;) {
        size_t total = 0;
        int largest = 0;
        for (int c = 0; c < N_CNA_CACHES; ++c) {
            total += cache_bytes[c];
            if (cache_bytes[c] > cache_bytes[largest]) largest = c;
        }
        if (total <= cache_budget_bytes || cache_bytes[largest] == 0) return;
        switch (largest) {
            case CACHE_CNA_JOINT:   cache_cna_joint.clear(); break;
            case CACHE_BETABIN_K:   cache_cna_betabin_k.clear(); break;
            case CACHE_LGAMMA_N_AT: cache_lgamma_N_plus_alpha_total.clear(); break;
            case CACHE_NOISE:       cache_noise_lgamma_n_plus_r.clear(); break;
            case CACHE_NEGBINOM:    cache_cnalikelihood_cells.clear(); break;
        }
        cache_bytes[largest] = 0;
        ++cache_clear_counts[names[largest]];
    }
}


double Scores::log_sum_exp(double a, double b) {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;

    double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

double Scores::log_sum_exp(const std::vector<double>& terms) {
    double max = -DBL_MAX;
    for (double t : terms)
        if (t > max) max = t;
    double sum = 0;
    for (double t : terms)
        sum += std::exp(t - max);
    return std::log(sum) + max;
}

std::vector<double> Scores::log_sum_exp_vector(const std::vector<std::vector<double>>& terms) {
    std::vector<double> maxs(n_cells, -DBL_MAX);
    for (const auto& row : terms)
        for (int j = 0; j < n_cells; ++j)
            if (row[j] > maxs[j]) maxs[j] = row[j];

    std::vector<double> sums(n_cells, 0.0);
    for (const auto& row : terms)
        for (int j = 0; j < n_cells; ++j)
            sums[j] += std::exp(row[j] - maxs[j]);

    for (int j = 0; j < n_cells; ++j)
        sums[j] = std::log(sums[j]) + maxs[j];
    return sums;
}

double Scores::log_n_choose_k(int n, int k) {
    int hash = n + 1000 * k;
    if (cache_n_choose_k.count(hash)) return cache_n_choose_k[hash];
    double res = 0;
    for (int i = 0; i < k; ++i) {
        res += std::log(n - i);
        res -= std::log(i + 1);
    }
    cache_n_choose_k[hash] = res;
    return res;
}






