#ifndef DEF_SCORES
#define DEF_SCORES

#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>
#include <string>

#include "Structures.h"



class Scores{
    private:
        // Caches of per-cell log-likelihood vectors.
        std::unordered_map<int,double> cache_n_choose_k;
        std::unordered_map<int64_t,std::vector<double>> cache_dropoutscores_bg; // [locus, omega_bg]: allelic counts when all copies are absent
        std::unordered_map<int64_t,std::vector<double>> cache_dropoutscores;    // [k, l, locus, omega]: beta-binomial for k ref and l alt surviving copies
        std::unordered_map<int64_t,std::unordered_map<int64_t,std::vector<double>>> cache_dropoutrate_cellscores; // [omega, dropout][genotype, locus]: SNV score, marginalized over dropout
        std::unordered_map<int64_t, std::vector<double>> cache_cnalikelihood_cells; // fixed-dispersion negative binomial (use_joint_likelihood_model = false)
        // The CNA caches below are keyed per sample (sample in the high bits of the key).
        std::unordered_map<int64_t, std::vector<double>> cache_cna_joint;
        std::unordered_map<int64_t, std::vector<double>> cache_cna_betabin_k;
        std::unordered_map<int, std::vector<double>> cache_noise_lgamma_n_plus_r; // [sample, region]: lgamma(n_j + NOISE_ALPHA_R_FIXED)
        std::unordered_map<int64_t, double> cache_lgamma_alpha_total;
        std::unordered_map<int64_t, std::vector<double>> cache_lgamma_N_plus_alpha_total;
        std::vector<std::vector<double>> log_n_choose_region_count; // [region][cell]
        std::vector<std::vector<double>> log_n_choose_snv_count; // [locus][cell]

        const Params& local_params;

        // Memory budget (Params::cache_gb) shared by the region-count caches (cache_cna_joint, cache_cna_betabin_k,
        // cache_lgamma_N_plus_alpha_total, cache_noise_lgamma_n_plus_r, cache_cnalikelihood_cells). When their total
        // exceeds it, the largest one is wiped; this is checked at each cache miss, before any reference into a cache
        // is taken.
        enum CnaCache { CACHE_CNA_JOINT, CACHE_BETABIN_K, CACHE_LGAMMA_N_AT, CACHE_NOISE, CACHE_NEGBINOM, N_CNA_CACHES };
        size_t cache_budget_bytes;
        size_t cache_bytes[N_CNA_CACHES] = {};
        size_t peak_cache_bytes = 0;
        void count_cache_bytes(CnaCache which, size_t n_values);
        void enforce_cache_budget();
public:
        // Run statistics (reported in run_summary.json): how often each cache was wiped to stay within the
        // budget, and the hit rate of the per-region CNA likelihood cache. One Scores object per chain.
        std::map<std::string, size_t> cache_clear_counts;
        size_t cna_joint_calls = 0;
        size_t cna_joint_misses = 0;
        size_t get_cache_budget_bytes() const { return cache_budget_bytes; }
        size_t get_peak_cache_bytes() const { return peak_cache_bytes; }

        Scores(const Params& params);
        double log_n_choose_k(int n, int k);
        static double log_sum_exp(double a, double b);
        static double log_sum_exp(const std::vector<double>& terms); //performs a robust log_sum_exp (ie sum exp(x-max) for better precision)
        static std::vector<double> log_sum_exp_vector(const std::vector<std::vector<double>>& terms);
        std::vector<double> compute_SNV_loglikelihoods(int c_ref,int c_alt, int locus, double dropout_rate_ref, double dropout_rate_alt);
        // The two *_joint functions return a reference into a cache: use it before the next call into Scores.
        const std::vector<double>& compute_SNV_loglikelihoods_joint(int c_ref, int c_alt, int locus, double omega, double dropout_rate);

        // sample >= 0: only that sample's cells are scored; element t is cell data.cells_by_sample[sample][t].
        // sample == -1: all cells, element j is cell j.
        const std::vector<double>& compute_CNA_loglikelihoods_joint(int region, int cn_r, double alpha_per_copy, double alpha_total, double dropout_rate, int sample = -1);
        std::vector<double> compute_CNA_loglikelihoods_negbinom(int region, double region_probability);

        void clear_cache();

};

#endif