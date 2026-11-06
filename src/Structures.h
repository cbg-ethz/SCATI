#ifndef DEF_STRUCTURES
#define DEF_STRUCTURES
#include <vector>
#include <map>
#include <string>


struct Cell{
    std::vector<int> ref_counts; // number of ref reads for each variable locus
    std::vector<int> alt_counts; // number of alt reads for each variable locus
    std::vector<int> region_counts; // number of reads in each region
    std::string name;
    int total_counts = 0; // sum of the read counts in each region
    int sample_id = 0; // sample (or time point) the cell comes from; 0 for single-sample runs
};

struct Data{
    // Dataset characteristics
    std::string sex; // "male" (1 chr X) or "female" (2 chr X)

    // Per-locus (SNV/indel) data
    std::vector<std::string> locus_to_chromosome;
    std::vector<int> locus_to_position;
    std::vector<std::string> locus_to_reference;
    std::vector<std::string> locus_to_alternative;
    std::vector<bool> variant_is_SNV; // true if the variant is a SNV, false otherwise (indel)
    std::vector<std::string> locus_to_name; //name of the variant, which describes the effect on the protein (if any)
    std::vector<int> locus_to_region;

    // Per-region (CNA) data
    std::vector<std::vector<int>> region_to_loci; //list of loci on this amplicon (possibly empty)
    std::vector<std::string> region_to_chromosome;
    std::map<std::string, std::vector<int>> chromosome_to_regions;
    int n_chromosomes;
    std::vector<std::string> chromosome_names;
    std::vector<std::string> region_to_name;

    // Healthy-cell annotation
    std::vector<bool> cell_is_healthy;

    // Root haplotype state
    std::vector<std::vector<int>> root_strands_loci_alt; // number of alt allele copies on each strand for each locus at the root of the tree
    std::vector<std::vector<int>> root_strands_loci_ref; // number of ref allele copies on each strand for each locus at the root of the tree
    std::vector<std::vector<int>> root_strands_regions;

    // Region reference (--region_reference), aligned with the regions kept after matching
    bool has_region_reference = false;
    std::vector<double> reference_rho_per_copy; // region probability per root copy
    std::vector<double> reference_phi;          // region dispersion
    double reference_dropout = -1.0;            // dropout rate (< 0: not given)

    // Multiple samples or time points; a single-sample run has n_samples = 1 and all cells in cells_by_sample[0]
    int n_samples = 1;
    std::vector<std::vector<int>> cells_by_sample; // cells_by_sample[s] = list of cell indices belonging to sample s
};

struct Params{
    // Constant parameters
    int random_seed;
    int max_copy_number;
    double excess_cn_cost;
    double allelic_error_rate; // expected VAF at a homozygous reference locus (1 - this at a homozygous alternative locus): errors from amplification, sequencing and alignment
    double omega_hom; //concentration parameter for beta-binomial in the SNV likelihood, when homozygous
    double omega_het; //concentration parameter for beta-binomial in the SNV likelihood, when heterozygous
    double allelic_error_rate_indel; // more errors for indels than SNVs
    double omega_hom_indel;
    double omega_het_indel;
    double mu_dir_multi;
    double theta_negbinom;
    double background_rate;
    int hash_precision;
    double lambda_rounding_precision; // rounding step of a node's sum_w_cn (Lambda) in the CNA score; doubled for doublets
    double cache_gb; // memory budget (GB, per chain and per sample: multiplied by n_samples) shared by the four CNA likelihood caches (see Scores.h)

    // Priors
    double dropoutrate; // per-copy dropout rate: mean of its Beta prior, starting value of each sample's fit, and fallback
                        // when too few healthy cells are available (the fitted values live in Tree::sample_fits)
    double prior_dropoutrate_omega; // concentration (alpha+beta) of that Beta prior
    bool fit_region_dispersion_mle; // fit region_dispersion via a proper joint-likelihood MLE (Newton) instead of method-of-moments
    bool fit_dropout_mle; // also directly fit dropoutrate via the same joint-likelihood machinery (uses whatever region_dispersion is currently in effect, MOM or MLE)
    double mean_total_count;
    double theta_total_count;

    // Penalty weights
    double node_cost;
    double cn_neutral_node_cost_factor; // node_cost multiplier for copy-number-neutral nodes (only SNVs/CNLOH: same
                                        // cn_regions as the parent), whose SNV evidence is weaker than a CNA's
    // Target number of nodes (-1: none); a tree with d nodes more or fewer pays 10 * node_cost * d^2
    int desired_n_nodes;
    // Doublet rate: the singlet/doublet mixing weight in the cell likelihood, and the mean of a Beta prior
    // that pulls the tree's posterior doublet fraction toward it
    double target_doublet_rate;
    double target_doublet_rate_omega; // concentration; higher = tighter around target_doublet_rate
    double alt_to_ref_cost;
    double ref_to_alt_cost;
    double Loss_cost;
    double Gain_cost;
    double CNLOH_cost;
    double FGD_cost;
    double healthy_cell_attachment_cost; // penalty for healthy cells to attach to non-root nodes.
    double healthy_heterogeneity_credit; // in [0,1]: fraction of a healthy cell's attachment to a CNA-free child of the root
                                         // that counts as attached to the root (healthy SNV heterogeneity, not a clone)
    double cancer_cell_at_root_cost;
    double FGD_count_exponent; // FGDs cost FGD_cost * (n + n^FGD_count_exponent) for n FGDs in the tree

    // Move probabilities
    double new_node_prob;
    double merge_probability;
    double add_SNV_probability;
    double merge_CNA_probability;
    double add_CNA_probability;
    double aneuploidy_probability;
    double fgd_probability;
    double convert_aneuploidy_probability; // in change_CNA_types, probability of doing the aneuploidy<->2-local-CNAs conversion instead of retyping a single CNA
    double merge_aneuploidy_probability; // within that conversion, probability of merging 2 local CNAs into an aneuploidy vs. splitting an aneuploidy into 2 local CNAs

    // Optimization parameters
    int n_chains;
    int chain_length;
    int burn_in;
    double temperature;
    std::string optimizer;
    bool sampling;
    int node_probabilities_em_steps; // number of EM iterations in Tree::update_node_probabilities().
    bool per_sample_node_probabilities; // n_samples > 1: fit node probabilities per sample (true) or pooled over all samples
    int sample_interval_trees_parameters;
    int update_interval_region_weights;
    int min_cells_for_parameter_refit;
    double min_root_attachment_fraction;
    bool use_joint_likelihood_model;
    bool use_total_count_likelihood;
    bool use_constant_dispersion; // CNA scoring: alpha_total kept at its root value instead of scaling with the
                                  // node's own Lambda (sum_w_cn)

    // Running options
    bool use_doublets;
    bool use_healthy_annotation;
    std::string region_reference_path;   // --region_reference: per-region rho/phi/dropout from a reference cohort, used
                                         // instead of healthy cells (only with use_healthy_annotation=false)
    bool region_reference_active = false; // true while the reference parameters are in use: set when a reference is
                                          // loaded, cleared by the single root refit at the end of burn-in
    bool refit_after_burnin;              // --refit_after_burnin (with use_healthy_annotation): once burn-in ends, refit the
                                          // region parameters from the root cells (node 0) instead of the labelled healthy
                                          // cells, and drop the healthy-annotation prior for the rest of the chain
    bool annotation_released = false;     // set by that refit; from then on the labels are no longer used
    bool use_haplotype_annotation; // root genotype read from <input>_haplotypes.tsv and kept fixed
    bool verbose;
};


// Quantities fitted per sample (Tree::sample_fits, one entry per sample)
struct SampleFit {
    double dropoutrate;
    double mean_total_count;
    double theta_total_count;
};

#endif