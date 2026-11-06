#ifndef DEF_TREE
#define DEF_TREE

#include <vector>
#include <set>
#include <string>
#include <map>
#include <tuple>
#include "Scores.h"
#include "Node.h"
#include "Structures.h"

class Tree{
    private:
        // Tree structure
        int n_nodes;
        std::vector<std::unique_ptr<Node>> nodes;
        std::vector<std::unique_ptr<Node>> doublets;
        std::vector<int> parents; //parents[n]: parent of node n. The root is always at index 0.
        std::vector<std::vector<int>> children; // children[n]: vector of the children of node n
        std::vector<int> DFT_order;

        // Parameters
        std::mt19937 rng;
        // Prior attachment probabilities to each node
        std::vector<std::vector<double>> node_probabilities;
        // Indexed [sample][locus_or_region]; single-sample runs have a size-1 outer
        // vector, so every downstream per-sample loop reduces to today's single pass.
        std::vector<std::vector<double>> dropout_rates_ref; // dropout rate of the reference allele
        std::vector<std::vector<double>> dropout_rates_alt; // dropout rate of the alternative allele
        std::vector<std::vector<double>> region_probabilities; // probability for a read to fall on each region, when there are 2 copies of each region
        std::vector<std::vector<int>> root_strand_loci_alt; // number of alt allele copies on each strand for each locus at the root of the tree
        std::vector<std::vector<int>> root_strand_loci_ref; // number of ref allele copies on each strand for each locus at the root of the tree
        std::vector<std::vector<int>> root_strand_regions;
        std::vector<std::vector<double>> region_dispersion;
        // Per-sample fitted dropout/total-count parameters
        std::vector<SampleFit> sample_fits;

        // change_root_genotype(): sum of the healthy cells' SNV log-likelihood, keyed on (locus, c_ref, c_alt, sample,
        // discretized omega, discretized dropout)
        std::map<std::tuple<int,int,int,int,long long,long long>, double> healthy_snv_loglik_cache;
        // Sum of compute_SNV_loglikelihoods_joint(c_ref, c_alt, locus, ...) over
        // healthy cells only, using the cache above. Used by
        // change_root_genotype()'s ambiguity gate.
        double healthy_snv_loglik_sum(int locus, int c_ref, int c_alt);

        // Utils
        std::shared_ptr<Scores> cache_scores;
        std::vector<std::vector<double>> cells_attach_loglik; //index j,k gives the complete (hidden) log likelihood of the data and cell j being attached to node k
        std::vector<double> cells_loglik; // index j gives the observed log likelihood for cell j (marginalizing over attachment points)
        std::vector<std::vector<double>> cells_attach_prob; //index j,k gives the posterior probability that cell j is attached to node k
        std::vector<int> best_attachments;
        std::vector<double> nodes_attachment_counts; //for each node, the expected number of cells attached to it
        // [group][doublet idx]: posterior cell count of each doublet combination
        std::vector<std::vector<double>> doublet_combination_counts;
        std::shared_ptr<Params> local_params;
        struct CNAProposal {
            Node::CNA cna;
            bool valid;
            int  n_strands;
        };

        // Fits region_probabilities/region_dispersion/dropout/mean_total_count/
        // theta_total_count for one sample's cells
        void fit_sample_parameters(int sample_idx, const std::vector<int>& cell_indices,
                                   std::vector<double>& out_region_probabilities,
                                   std::vector<double>& out_region_dispersion,
                                   SampleFit& out_fit,
                                   double& out_effective_root_cells);
        // Burn-in parameters from --region_reference (see apply_region_reference())
        void fit_reference_parameters(const std::vector<int>& cell_indices,
                                      std::vector<double>& out_region_probabilities,
                                      std::vector<double>& out_region_dispersion,
                                      SampleFit& out_fit,
                                      double& out_effective_root_cells);

    public:
        double hastings_ratio; // probability to move from previous tree to this tree / probability to move from this tree to the previous tree
        double log_prior_score{};
        double log_likelihood{};
        double log_score{}; //complete score of the tree, including the prior and the likelihood

        Tree(std::shared_ptr<Scores> cache, std::mt19937& rng_, std::shared_ptr<Params> params); // constructor
        Tree(const Tree& source); // copy constructor (deep copy)
        ~Tree() = default;
        Tree& operator=(const Tree&); // Assignment operator: deep copy

        // *this = source, but keeps this tree's own rng, so a rejected proposal does not replay the same random draws
        void reset_to(const Tree& source);

        // Tree move helper functions
        void attach_subtree(int pruned_node, int attachment_point); // attach a pruned node and it's subtree to a new parent
        void compute_children(); // compute the list of children of each node, from the parent vector. Also compute the DFT order
        int doublet_index(int k, int l) const; // flat index into doublets[] for the pair (k,l) with k<=l

        int get_n_nodes() const { return n_nodes; }
        // Pooled fraction of all cells' soft posterior mass currently attached
        // to the root (node 0).
        double get_root_attachment_fraction() const {
            double sum = 0.0;
            for (const auto& p : cells_attach_prob) sum += p[0];
            return cells_attach_prob.empty() ? 0.0 : sum / static_cast<double>(cells_attach_prob.size());
        }
        // rebuilds doublets[] from nodes[]; needed after changing node genotypes directly
        void rebuild_doublets();
        std::vector<int> root_copy_numbers() const; // copy number of each region at the root

        void add_node (int parent); // Add a node below an existing one, and randomly assign the children of the parent to the new node or their initial parent
        void delete_node(int node); // delete a node, setting the parent of its children to the node's parent, and adapting the node indices
        std::vector<int> affected_nodes_subtrees(int node1, int node2);
        std::vector<int> subtree_nodes(int node) const; // the node and all its descendants, each after its parent
        std::vector<int> subtrees_nodes(const std::vector<int>& roots) const; // every node in their subtrees, once, in DFT order
        static int merge_gap(const Node::CNA &A, const Node::CNA &B); // 0 adjacent, 1 one region apart, -1 not mergeable
        struct MergeCandidate { int node, idx1, idx2; bool has_gap; };
        struct SplitCandidate { int node, event_idx, split_pos; bool is_gap; };
        std::vector<MergeCandidate> merge_candidates(bool whole_chromosome) const; // see Tree.cpp
        std::vector<SplitCandidate> split_candidates(bool aneuploidy) const;       // see Tree.cpp


        // checks that "region present on a strand" agrees with "locus has copies on that strand" for every node;
        // returns the first violation or an empty string
        std::string check_region_locus_consistency() const;

        // Update genotypes and likelihoods
        bool update_genotypes();
        bool update_genotypes_affected_nodes(const std::vector<int> &affected_nodes);
        void compute_likelihood_affected_nodes(const std::vector<int> &affected_nodes);
        void compute_attachment_scores(bool use_doublets_local);
        void compute_attachment_scores_affected_nodes(bool use_doublets_local, const std::vector<int>& affected_nodes);

        void compute_likelihood();
        void compute_prior_score();
        void update_full_score();
        void sync_dropout_arrays(); // refill dropout_rates/_ref/_alt from sample_fits
        void compute_cells_likelihoods(bool use_doublets_local, const std::vector<int>& affected_nodes);
        double compute_CNA_logprior_penalty() const;
        double compute_SNV_logprior_penalty() const;
        double compute_parameter_logprior();
        // n_cells / 5000: the costs' values apply at 5000 cells. Scales the whole-tree terms (node_cost,
        // desired_n_nodes, target_doublet_rate) and the per-event costs; terms that are already sums over cells
        // and the parameter prior are not scaled.
        double cell_scale_factor() const;


    // MCMC tree moves
        void prune_reattach();
        void swap_node_labels();
        void add_remove_SNV();
        void move_event(bool snv); // shared by move_SNV and move_CNA
        void move_SNV();
        void split_merge_node();
        void add_remove_CNA();
        void move_CNA();
        void change_strand();
        void change_root_genotype(bool is_warmup);
        void change_CNA_types();
        void change_CNA_regions();
        void update_region_weights();
        void change_event_order_within_node();
        void merge_split_within_node();
        void swap_CNA();
        CNAProposal propose_CNA(double fgd_probability, double aneuploidy_probability, int node);
        double CNA_proposal_prob(const Node::CNA& cna, int n_strands) const;

        // Create output files
        void to_dot(const std::string& output_dir); // save the tree structure to the dot format (for visualization)
        void sample_parameters(int mcmc_step, const std::string& output_dir); // save the parameters and tree to a file
        void write_tree_json(std::ostream& out, bool one_line) const;
        void write_strand_genotypes_json(std::ostream& out) const;
        void write_parameters_json(std::ostream& out) const;
        void write_cell_assignments(std::ostream& out_assign, std::ostream* out_probs,
                                    const std::string& step, bool header) const;
    double compute_healthy_logprior_penalty();

    void update_node_probabilities(const std::vector<std::vector<double>>& tc_k_const,
                                   const std::vector<std::vector<double>>& tc_log_p,
                                   const std::vector<std::vector<double>>& tc_d_k_const,
                                   const std::vector<std::vector<double>>& tc_d_log_p,
                                   bool use_doublets_local);

    void initialize_root_genotype_from_healthy_cells();
    // Root SNV genotype from the VAF of the given per-cell weights (see initialize_root_genotype_from_healthy_cells)
    void initialize_root_genotype_from_cells(const std::vector<double>& cell_weights);
    // --region_reference runs: root SNV genotype from the cells attached to the root before any tree
    // inference, or from all cells if fewer than min_cells_for_parameter_refit attach
    void initialize_root_genotype_from_root_cells();
    // True while the region parameters are fit from the labelled healthy cells (use_healthy_annotation, until a
    // --refit_after_burnin refit releases the labels); otherwise they are fit from the cells' soft root attachment.
    bool fit_from_labels() const { return local_params->use_healthy_annotation && !local_params->annotation_released; }
    // --refit_after_burnin: prints how far the labelled healthy cells agree with the root (node 0) before the refit
    void report_labels_vs_root() const;
};


#endif

