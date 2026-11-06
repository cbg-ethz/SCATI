#include <random>
#include <cmath>
#include <cfloat>
#include <utility>
#include <vector>
#include <stack>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <stdexcept>
#include <limits>
#include <sstream>
#include <iomanip>
#include <map>
#include <cctype>

#include "Tree.h"
#include "Node.h"
#include "Scores.h"

#include <cassert>


extern int n_cells;
extern int n_loci;
extern int n_regions;
extern std::vector<Cell> cells;


// Constructor for a random tree, used to initialize the MCMC.
Tree::Tree(std::shared_ptr<Scores> cache, std::mt19937& rng_,
           std::shared_ptr<Params> params)
        : rng(rng_),
          local_params(std::move(params)),
          hastings_ratio(-1.0) {

    cache_scores = std::move(cache);

    // Initial tree size (root included).
    std::uniform_int_distribution<int> node_dist(3, 10);
    n_nodes = node_dist(rng);

    nodes_attachment_counts.resize(n_nodes, 0.0);
    region_probabilities.assign(data.n_samples, std::vector<double>(n_regions, 1.0 / n_regions));
    sample_fits.assign(data.n_samples,
                       SampleFit{local_params->dropoutrate, local_params->mean_total_count, local_params->theta_total_count});
    sync_dropout_arrays();

    parents.resize(n_nodes);
    parents[0] = -1;

    for (int i = 1; i < n_nodes; i++) {
        std::uniform_int_distribution<int> parent_dist(0, i - 1);
        parents[i] = parent_dist(rng);
    }

    children.resize(n_nodes);
    compute_children();

    nodes.reserve(n_nodes);
    // Placeholder seed for the node probabilities, which will be updated below
    node_probabilities.assign(1, std::vector<double>(n_nodes, 1.0 / n_nodes));

    for (int i = 0; i < n_nodes; i++) {
        nodes.push_back(std::make_unique<Node>(cache_scores, rng, local_params));
    }

    root_strand_loci_alt = data.root_strands_loci_alt;
    root_strand_loci_ref = data.root_strands_loci_ref;
    root_strand_regions  = data.root_strands_regions;

    if (!local_params->use_haplotype_annotation && local_params->use_healthy_annotation) {
        initialize_root_genotype_from_healthy_cells();
    }
    // else the root starts from data.root_strands_* (the haplotype file, or homozygous reference); without a
    // haplotype file the MCMC can still change it

    // Seed random loci with an SNV on a random non-root node, so the start tree is not collapsed to the root. The
    // allele matches the root genotype on the chosen strand, which must carry the locus's region. Not done without
    // the healthy annotation.
    if (local_params->use_healthy_annotation) {
        std::uniform_int_distribution<int> node_assign_dist(1, n_nodes - 1); // Don't assign SNVs to the root
        std::bernoulli_distribution seed_snv_dist(0.5);
        for (int i = 0; i < n_loci; i++) {
            if (!seed_snv_dist(rng)) continue;
            int locus_region = data.locus_to_region[i];
            std::vector<int> valid_strands;
            for (int s = 0; s < (int)root_strand_regions.size(); ++s)
                if (root_strand_regions[s][locus_region] >= 1) valid_strands.push_back(s);
            if (valid_strands.empty()) continue; // no strand carries this locus's region -- skip
            int strand = valid_strands[std::uniform_int_distribution<int>(0, (int)valid_strands.size() - 1)(rng)];
            int allele = (root_strand_loci_ref[strand][i] > 0) ? 0 : 1; // ref->alt if a ref copy is present, else alt->ref
            Node::SNV snv = {i, strand, allele};
            nodes[node_assign_dist(rng)]->add_mutation(snv);
        }
    }

    // Nodes that ended up without  SNV are pruned.
    for (int i = n_nodes - 1; i >= 1; --i) {
        if (nodes[i]->is_empty()) {
            if (local_params->verbose) std::cout << "Deleting empty node " << i << ". ";
            delete_node(i);
        }
    }

    update_genotypes();

    cells_attach_loglik.resize(n_cells);
    cells_loglik.resize(n_cells);
    cells_attach_prob.resize(n_cells);
    best_attachments.resize(n_cells);
    region_dispersion.assign(data.n_samples, std::vector<double>(n_regions, local_params->mu_dir_multi));

    compute_likelihood();
    compute_prior_score();
    update_full_score();
}

Tree::Tree(const Tree& source)
        : n_nodes(source.n_nodes),
          parents(source.parents),
          children(source.children),
          DFT_order(source.DFT_order),
          rng(source.rng),
          node_probabilities(source.node_probabilities),
          dropout_rates_ref(source.dropout_rates_ref),
          dropout_rates_alt(source.dropout_rates_alt),
          region_probabilities(source.region_probabilities),
          root_strand_loci_alt(source.root_strand_loci_alt),
          root_strand_loci_ref(source.root_strand_loci_ref),
          root_strand_regions(source.root_strand_regions),
          region_dispersion(source.region_dispersion),
          sample_fits(source.sample_fits),
          healthy_snv_loglik_cache(source.healthy_snv_loglik_cache),
          cache_scores(source.cache_scores),   // shared_ptr copy — both trees share ownership
          cells_attach_loglik(source.cells_attach_loglik),
          cells_loglik(source.cells_loglik),
          cells_attach_prob(source.cells_attach_prob),
          best_attachments(source.best_attachments),
          nodes_attachment_counts(source.nodes_attachment_counts),
          doublet_combination_counts(source.doublet_combination_counts),
          local_params(source.local_params),
          hastings_ratio(source.hastings_ratio),
          log_prior_score(source.log_prior_score),
          log_likelihood(source.log_likelihood),
          log_score(source.log_score) {

    nodes.reserve(source.nodes.size());
    for (const auto& node : source.nodes) {
        nodes.push_back(std::make_unique<Node>(*node));
        nodes.back()->rebind_rng(rng); // copy-ctor's Node copy keeps the SOURCE tree's
                                        // rng reference otherwise -- rebind to this
                                        // tree's own (already copied above).
    }

    doublets.reserve(source.doublets.size());
    for (const auto& d : source.doublets) {
        doublets.push_back(std::make_unique<Node>(*d));
        doublets.back()->rebind_rng(rng);
    }
}

Tree& Tree::operator=(const Tree& source) {
    if (this == &source) return *this;

    rng = source.rng;
    cache_scores = source.cache_scores;   // shared_ptr copy
    local_params  = source.local_params;
    n_nodes                 = source.n_nodes;
    parents                 = source.parents;
    children                = source.children;
    DFT_order               = source.DFT_order;
    node_probabilities      = source.node_probabilities;
    dropout_rates_ref       = source.dropout_rates_ref;
    dropout_rates_alt       = source.dropout_rates_alt;
    region_probabilities  = source.region_probabilities;
    nodes_attachment_counts = source.nodes_attachment_counts;
    root_strand_loci_alt    = source.root_strand_loci_alt;
    root_strand_loci_ref    = source.root_strand_loci_ref;
    root_strand_regions     = source.root_strand_regions;
    log_prior_score         = source.log_prior_score;
    log_likelihood          = source.log_likelihood;
    log_score               = source.log_score;
    cells_attach_loglik     = source.cells_attach_loglik;
    cells_loglik            = source.cells_loglik;
    cells_attach_prob       = source.cells_attach_prob;
    best_attachments        = source.best_attachments;
    hastings_ratio          = source.hastings_ratio;
    region_dispersion       = source.region_dispersion;
    sample_fits             = source.sample_fits;
    doublet_combination_counts = source.doublet_combination_counts;
    healthy_snv_loglik_cache   = source.healthy_snv_loglik_cache;

    nodes.clear();
    doublets.clear();

    nodes.reserve(source.nodes.size());
    for (const auto& node : source.nodes) {
        nodes.push_back(std::make_unique<Node>(*node));
        nodes.back()->rebind_rng(rng); // see copy-ctor comment above
    }

    doublets.reserve(source.doublets.size());
    for (const auto& d : source.doublets) {
        doublets.push_back(std::make_unique<Node>(*d));
        doublets.back()->rebind_rng(rng);
    }

    return *this;
}

void Tree::reset_to(const Tree& source) {
    // Preserve this tree's own rng across the reset
    std::mt19937 preserved_rng = std::move(rng);
    *this = source;
    rng = std::move(preserved_rng);
}


// attach a pruned node and it's subtree to a new parent
void Tree::attach_subtree(int pruned_node, int attachment_point) {
    parents[pruned_node] = attachment_point;
    compute_children();
}


void Tree::compute_children(){
    // Compute the list of children of each node, from the parent vector.
    children.resize(n_nodes);
    for (int i=0; i <n_nodes; i++){
        children[i].clear();
    }
    for (int i=1; i < n_nodes; i++){ //start from 1 because the root has no parent
        children[parents[i]].push_back(i);
    }

    std::stack<int> stk;
    DFT_order.clear();
    stk.push(0);
    while (!stk.empty()) {
        int top = stk.top();
        DFT_order.push_back(top);
        stk.pop();
        for (int child: children[top]) {
            stk.push(child);
        }
    }
}

// Flat index into doublets[] for the pair (k,l), k<=l, over the upper-triangular
// n_nodes x n_nodes grid (row-major by k, then offset by l-k within the row).
int Tree::doublet_index(int k, int l) const {
    return n_nodes * k - (k * (k - 1)) / 2 + l - k;
}


// copy numbers of each region at the root, summed over the root strands
std::vector<int> Tree::root_copy_numbers() const {
    std::vector<int> cn(n_regions, 0);
    for (const auto& strand : root_strand_regions)
        for (int k = 0; k < n_regions; ++k)
            cn[k] += strand[k];
    return cn;
}

void Tree::rebuild_doublets() {
    if (!local_params->use_doublets) return;
    doublets.clear();
    for (int k = 0; k < n_nodes; ++k)
        for (int l = k; l < n_nodes; ++l)
            doublets.push_back(std::make_unique<Node>(*nodes[k], *nodes[l], rng));
}

// Update the genotype of the nodes affected by a tree move and return false if the tree move is invalid
bool Tree::update_genotypes_affected_nodes(const std::vector<int>& affected_nodes) {

    bool valid_tree = true;

    for (int top : affected_nodes) {
        bool valid_node;
        if (top != 0)
            valid_node = nodes[top]->update_genotype(nodes[parents[top]].get(), root_strand_loci_ref, root_strand_loci_alt, root_strand_regions);
        else
            valid_node = nodes[top]->update_genotype(nullptr, root_strand_loci_ref, root_strand_loci_alt, root_strand_regions);

        if (!valid_node) { // invalid node
            valid_tree = false;
            break;
        }
    }

    if (valid_tree) rebuild_doublets();

    return valid_tree;
}


// Perform a depth-first traversal and compute the genotype of each node, based on the genotype of its parent.
// Done for the full tree and determines, if the tree is valid.
std::string Tree::check_region_locus_consistency() const {
    for (int n = 0; n < n_nodes; ++n) {
        const Node* node = nodes[n].get();
        int n_strands = (int)node->strands_regions.size();
        for (int s = 0; s < n_strands; ++s) {
            for (int region = 0; region < n_regions; ++region) {
                if (data.region_to_loci[region].empty()) continue;
                bool region_present = node->strands_regions[s][region] >= 1;
                for (int locus : data.region_to_loci[region]) {
                    bool locus_present = (node->strands_loci_ref[s][locus] + node->strands_loci_alt[s][locus]) >= 1;
                    if (region_present != locus_present) {
                        return "node=" + std::to_string(n) + " strand=" + std::to_string(s)
                             + " region=" + std::to_string(region) + " locus=" + std::to_string(locus)
                             + " region_present=" + std::to_string(region_present)
                             + " locus_present=" + std::to_string(locus_present);
                    }
                }
            }
        }
    }
    return "";
}

bool Tree::update_genotypes() {
    return update_genotypes_affected_nodes(subtree_nodes(0)); // all nodes, parents before children
}


// Recompute the attachment scores of the nodes affected by a tree move.
void Tree::compute_attachment_scores_affected_nodes(
        bool use_doublets_local,
        const std::vector<int>& affected_nodes) {

    // sort affected nodes in DFT order so parents are always computed before children
    std::vector<int> ordered_affected;
    for (int i = 0; i < n_nodes; ++i) {
        int node = DFT_order[i];
        if (std::find(affected_nodes.begin(), affected_nodes.end(), node)
            != affected_nodes.end()) {
            ordered_affected.push_back(node);
        }
    }

    for (int top : ordered_affected) {
        if (top == 0) {
            nodes[top]->compute_attachment_scores_root(
                    dropout_rates_ref, dropout_rates_alt,
                    region_probabilities, region_dispersion, sample_fits,
                    nodes[0]->cn_regions);
        } else {
            nodes[top]->compute_attachment_scores_parent(
                    nodes[parents[top]].get(),
                    dropout_rates_ref, dropout_rates_alt,
                    region_probabilities, region_dispersion, sample_fits,
                    nodes[0]->cn_regions);
        }
    }

    if (use_doublets_local) {
        for (int n1 = 0; n1 < n_nodes; ++n1) {
            for (int l = 0; l < n_nodes; ++l) {
                int n2 = DFT_order[l];
                if (n2 < n1) continue;

                int idx = doublet_index(n1, n2);

                if (n1 == 0 && n2 == 0) {
                    doublets[idx]->compute_attachment_scores_root(
                            dropout_rates_ref, dropout_rates_alt,
                            region_probabilities, region_dispersion, sample_fits,
                            nodes[0]->cn_regions);
                }
                else if (n1 == n2) {
                    int p1 = parents[n1];
                    int idx_parent = doublet_index(std::min(p1, n1), std::max(p1, n1));
                    doublets[idx]->compute_attachment_scores_parent(
                            doublets[idx_parent].get(),
                            dropout_rates_ref, dropout_rates_alt,
                            region_probabilities, region_dispersion, sample_fits,
                            nodes[0]->cn_regions);
                }
                else {
                    int p2 = parents[n2];
                    int idx_parent = doublet_index(std::min(n1, p2), std::max(n1, p2));
                    doublets[idx]->compute_attachment_scores_parent(
                            doublets[idx_parent].get(),
                            dropout_rates_ref, dropout_rates_alt,
                            region_probabilities, region_dispersion, sample_fits,
                            nodes[0]->cn_regions);
                }
            }
        }
    }
}

void Tree::compute_attachment_scores(bool use_doublets_local) {
    compute_attachment_scores_affected_nodes(use_doublets_local, DFT_order);
}

void Tree::compute_likelihood_affected_nodes(const std::vector<int>& affected_nodes){

    compute_attachment_scores_affected_nodes(local_params->use_doublets, affected_nodes);

    compute_cells_likelihoods(local_params->use_doublets, affected_nodes);

    log_likelihood = 0;
    for (int j = 0; j < n_cells; ++j)
        log_likelihood += cells_loglik[j];
}

void Tree::compute_likelihood(){
    // Compute the likelihood by marginalizing over the attachment points

    compute_attachment_scores(local_params->use_doublets);

    compute_cells_likelihoods(local_params->use_doublets,{});

    log_likelihood=0;
    for (int j=0;j<n_cells;j++){
        log_likelihood+=cells_loglik[j];
    }
}

// Node probabilities pi: expected fractions of SINGLET cells per node, fitted by EM on the tree's current state.
// Only the singlet part of the posterior votes (counting the members of doublets would let doublets
// inflate the nodes they resemble, which in turn raises the weight of the corresponding doublets).
void Tree::update_node_probabilities(const std::vector<std::vector<double>>& tc_k_const,
                                     const std::vector<std::vector<double>>& tc_log_p,
                                     const std::vector<std::vector<double>>& tc_d_k_const,
                                     const std::vector<std::vector<double>>& tc_d_log_p,
                                     bool use_doublets_local) {
    const double alpha = 1e-10; // Dirichlet prior concentration, alpha=1 is uniform
    const int em_steps = std::max(1, local_params->node_probabilities_em_steps);

    // Per-sample node probabilities (per_sample_node_probabilities, default true)
    // fit each sample's own mixing weights independently
    const bool per_sample = (data.n_samples > 1) && local_params->per_sample_node_probabilities;

    std::vector<std::vector<int>> groups;
    if (per_sample) {
        groups = data.cells_by_sample;
    } else {
        groups.assign(1, std::vector<int>(n_cells));
        for (int j = 0; j < n_cells; ++j) groups[0][j] = j;
    }

    node_probabilities.assign(groups.size(), std::vector<double>(n_nodes));

    const bool use_tc = local_params->use_total_count_likelihood;
    const double log_singlet_factor = use_doublets_local ? std::log(1.0 - local_params->target_doublet_rate) : 0.0;
    const double log_doublet_rate   = use_doublets_local ? std::log(local_params->target_doublet_rate) : 0.0;
    const double log_2 = std::log(2.0);
    const int n_d = use_doublets_local ? n_nodes * (n_nodes + 1) / 2 : 0;
    std::vector<double> a(n_nodes + n_d); // log-weights of one cell's attachment points (scratch)

    for (size_t g = 0; g < groups.size(); ++g) {
        const std::vector<int>& cell_indices = groups[g];
        std::vector<double> w(n_nodes, 1.0 / n_nodes);

        for (int step = 0; step < em_steps; ++step) {
            std::vector<double> log_w(n_nodes);
            for (int k = 0; k < n_nodes; ++k) log_w[k] = std::log(w[k]);

            std::vector<double> new_probs(n_nodes, alpha); // prior pseudocounts
            double singlet_total = 0.0;
            for (int j : cell_indices) {
                const int s = cells[j].sample_id;
                const double Nj = use_tc ? static_cast<double>(cells[j].total_counts) : 0.0;

                double max_a = -std::numeric_limits<double>::infinity();
                for (int k = 0; k < n_nodes; ++k) {
                    double v = nodes[k]->attachment_scores[j] + log_w[k] + log_singlet_factor;
                    if (use_tc) v += tc_k_const[s][k] + Nj * tc_log_p[s][k];
                    a[k] = v;
                    if (v > max_a) max_a = v;
                }
                if (use_doublets_local) {
                    int idx = 0;
                    for (int k = 0; k < n_nodes; ++k) {
                        for (int l = k; l < n_nodes; ++l, ++idx) {
                            double v = doublets[idx]->attachment_scores[j] + log_w[k] + log_w[l] + log_doublet_rate;
                            if (k != l) v += log_2;    // off-diagonal: two orderings (k,l) and (l,k)
                            if (use_tc) v += tc_d_k_const[s][idx] + Nj * tc_d_log_p[s][idx];
                            a[n_nodes + idx] = v;
                            if (v > max_a) max_a = v;
                        }
                    }
                }
                if (max_a == -std::numeric_limits<double>::infinity()) continue; // impossible cell: no vote

                double total = 0.0, singlets = 0.0;
                for (int k = 0; k < n_nodes; ++k) {
                    a[k] = std::exp(a[k] - max_a);
                    singlets += a[k];
                }
                total = singlets;
                for (int idx = 0; idx < n_d; ++idx) total += std::exp(a[n_nodes + idx] - max_a);

                singlet_total += singlets / total;
                for (int k = 0; k < n_nodes; ++k) new_probs[k] += a[k] / total;
            }

            const double denom = singlet_total + alpha * n_nodes;
            for (int k = 0; k < n_nodes; ++k)
                w[k] = new_probs[k] / denom;
        }

        node_probabilities[g] = w;
    }
}

void Tree::compute_cells_likelihoods(bool use_doublets_local, const std::vector<int>& affected_nodes) {

    int n_attachment_points = n_nodes;
    if (use_doublets_local) n_attachment_points = (n_nodes * (n_nodes + 3)) / 2;

    const double log_singlet_factor = use_doublets_local
                                      ? std::log(1.0 - local_params->target_doublet_rate) : 0.0;
    const double log_doublet_rate   = use_doublets_local
                                      ? std::log(local_params->target_doublet_rate) : 0.0;
    const double log_2 = std::log(2.0); // off-diagonal doublet "two orderings" factor,
                                         // hoisted out of the per-cell/per-pair loop below
                                         // (it's a fixed constant, not data-dependent)


    // Precompute per-(sample,node) total-count log-terms (j-independent part).
    // These are additive corrections applied per cell below, keeping attachment_scores
    // clean.
    std::vector<std::vector<double>> tc_k_const;   // [sample][node]
    std::vector<std::vector<double>> tc_log_p;     // [sample][node]
    std::vector<std::vector<double>> tc_d_k_const; // [sample][doublet]
    std::vector<std::vector<double>> tc_d_log_p;   // [sample][doublet]
    const int n_samples_local = (int)sample_fits.size();

    // Total counts: plain negative binomial, mean = mean_total_count * cn_total.
    if (local_params->use_total_count_likelihood) {
        tc_k_const.assign(n_samples_local, {});
        tc_log_p.assign(n_samples_local, {});
        if (use_doublets_local) {
            tc_d_k_const.assign(n_samples_local, {});
            tc_d_log_p.assign(n_samples_local, {});
        }

        for (int s = 0; s < n_samples_local; ++s) {
            const double mean_per_copy = sample_fits[s].mean_total_count;
            const double theta = sample_fits[s].theta_total_count;
            const double lg_theta = std::lgamma(theta);

            // per-copy weight (region_probabilities is the read fraction at the root's copy number), so cn_total(root) = 1
            std::vector<double> cn_total(n_nodes, 0.0);
            for (int k = 0; k < n_nodes; ++k)
                for (int r = 0; r < n_regions; ++r) {
                    double per_copy_weight = region_probabilities[s][r] / std::max(1, nodes[0]->cn_regions[r]);
                    cn_total[k] += per_copy_weight * nodes[k]->cn_regions[r];
                }

            tc_k_const[s].resize(n_nodes);
            tc_log_p[s].resize(n_nodes);
            for (int k = 0; k < n_nodes; ++k) {
                double mu_k  = std::max(mean_per_copy * cn_total[k], 1e-6);
                tc_log_p[s][k]    = std::log(mu_k  / (mu_k + theta));
                tc_k_const[s][k]  = theta * std::log(theta / (mu_k + theta)) - lg_theta;
            }

            if (use_doublets_local) {
                int n_d = n_nodes * (n_nodes + 1) / 2;
                tc_d_k_const[s].resize(n_d);
                tc_d_log_p[s].resize(n_d);
                int idx = 0;
                for (int k = 0; k < n_nodes; ++k) {
                    for (int l = k; l < n_nodes; ++l, ++idx) {
                        double mu_d  = std::max(mean_per_copy * (cn_total[k] + cn_total[l]), 1e-6);
                        tc_d_log_p[s][idx]   = std::log(mu_d  / (mu_d + theta));
                        tc_d_k_const[s][idx] = theta * std::log(theta / (mu_d + theta)) - lg_theta;
                    }
                }
            }
        }
    }

    // Node probabilities: EM on the full posterior (singlets and doublets, total-count terms included) of this tree, see update_node_probabilities().
    update_node_probabilities(tc_k_const, tc_log_p, tc_d_k_const, tc_d_log_p, use_doublets_local);

    std::fill(nodes_attachment_counts.begin(), nodes_attachment_counts.end(), 0.0);

    const int n_prob_groups = (int)node_probabilities.size();
    std::vector<std::vector<double>> log_node_prob(n_prob_groups, std::vector<double>(n_nodes));
    for (int g = 0; g < n_prob_groups; ++g)
        for (int k = 0; k < n_nodes; ++k)
            log_node_prob[g][k] = std::log(node_probabilities[g][k]);

    if (use_doublets_local) {
        int n_d = n_nodes * (n_nodes + 1) / 2;
        doublet_combination_counts.assign(n_prob_groups, std::vector<double>(n_d, 0.0));
    } else {
        doublet_combination_counts.clear();
    }

    for (int j = 0; j < n_cells; ++j) {
        cells_attach_loglik[j].resize(n_attachment_points);

        // j-dependent total-count term and cast, computed once per cell and
        // reused below instead of being recast/recomputed for every k/doublet.
        const int s = cells[j].sample_id;
        const int g = (n_prob_groups > 1) ? s : 0;
        double tc_j_base = 0.0;
        double Nj = 0.0;
        if (local_params->use_total_count_likelihood) {
            Nj = static_cast<double>(cells[j].total_counts);
            tc_j_base = std::lgamma(Nj + sample_fits[s].theta_total_count)
                        - std::lgamma(Nj + 1.0);
        }

        for (int k = 0; k < n_nodes; ++k) {
            double score = nodes[k]->attachment_scores[j]
                           + log_node_prob[g][k]
                           + log_singlet_factor;
            if (local_params->use_total_count_likelihood) {
                score += tc_j_base + tc_k_const[s][k] + Nj * tc_log_p[s][k];
            }
            cells_attach_loglik[j][k] = score;
        }

        if (use_doublets_local) {
            int idx = 0;
            for (int k = 0; k < n_nodes; ++k) {
                for (int l = k; l < n_nodes; ++l, ++idx) {
                    double score = doublets[idx]->attachment_scores[j]
                                   + log_node_prob[g][k] + log_node_prob[g][l]
                                   + log_doublet_rate;
                    if (k != l) score += log_2;    // off-diagonal: two orderings (k,l) and (l,k)
                    if (local_params->use_total_count_likelihood) {
                        score += tc_j_base + tc_d_k_const[s][idx] + Nj * tc_d_log_p[s][idx];
                    }
                    cells_attach_loglik[j][n_nodes + idx] = score;
                }
            }
        }

        auto max_it = std::max_element(cells_attach_loglik[j].begin(), cells_attach_loglik[j].end());
        best_attachments[j] = max_it - cells_attach_loglik[j].begin();
        double max_val = *max_it;
        if (max_val == -std::numeric_limits<double>::infinity()) {
            cells_loglik[j] = -std::numeric_limits<double>::infinity();
        } else {
            double sum_exp = 0.0;
            for (double t : cells_attach_loglik[j]) sum_exp += std::exp(t - max_val);
            cells_loglik[j] = std::log(sum_exp) + max_val;
        }

        // --- Posterior attachment probabilities ---
        cells_attach_prob[j].resize(n_attachment_points);
        double denom = cells_loglik[j];

        for (int k = 0; k < n_nodes; ++k) {
            cells_attach_prob[j][k] = std::exp(cells_attach_loglik[j][k] - denom);
            nodes_attachment_counts[k] += cells_attach_prob[j][k];
        }

        if (use_doublets_local) {
            int idx = 0;
            for (int k = 0; k < n_nodes; ++k) {
                for (int l = k; l < n_nodes; ++l, ++idx) {
                    double p = std::exp(cells_attach_loglik[j][n_nodes + idx] - denom);
                    cells_attach_prob[j][n_nodes + idx] = p;
                    nodes_attachment_counts[k] += p / 2.0;
                    nodes_attachment_counts[l] += p / 2.0;
                    doublet_combination_counts[g][idx] += p;
                }
            }
        }
    }
}


double Tree::cell_scale_factor() const {
    const double REFERENCE_CELLS = 5000.0;
    return (double)n_cells / REFERENCE_CELLS;
}


double Tree::compute_CNA_logprior_penalty() const {
    double penalty = 0.0;

    // Aneuploidy events are cheaper than local CNAs affecting the same regions

    auto aneuploidy_event_cost = [&](double base_event_cost) {
        return 0.8 * base_event_cost;
    };

    // Local CNAs and aneuploidies: a flat cost per event. FGDs: cost per event plus a superlinear term in their count.
    int total_fgd = 0;

    for (int n = 1; n < n_nodes; ++n) {
        const Node* node = nodes[n].get();
        if (node == nullptr) continue;

        // Penalize regions exceeding max copy number
        for (int r = 0; r < n_regions; ++r) {
            int cn = node->cn_regions[r];
            if (cn > 4) {
                penalty += local_params->excess_cn_cost
                           * (cn - 4);
            }
        }

        for (const auto& event : node->Events) {
            if (!std::holds_alternative<Node::CNA>(event)) continue;
            const Node::CNA& cna = std::get<Node::CNA>(event);

            if (cna.fgd) {
                total_fgd += 1;
                continue;
            }

            if (cna.aneuploidy) {
                double base_cost = (cna.CNA_change == 1) ? local_params->Gain_cost : local_params->Loss_cost;
                penalty += aneuploidy_event_cost(base_cost);
                continue;
            }

            if (cna.CNA_change == 1) {
                penalty += local_params->Gain_cost;
            } else if (cna.CNA_change == -1) {
                penalty += local_params->Loss_cost;
            } else { // CNLOH
                penalty += local_params->CNLOH_cost;
            }
        }
    }

    // FGD: linear + superlinear term, so that repeated FGDs along a tree (biologically rare) need strong support
    penalty += local_params->FGD_cost * total_fgd
               + local_params->FGD_cost * std::pow(total_fgd, local_params->FGD_count_exponent);

    return penalty * cell_scale_factor();
}

double Tree::compute_SNV_logprior_penalty() const {
    double penalty = 0.0;
    // iterate through nodes except the root
    for (int n = 1; n < n_nodes; ++n) {
        const Node::SNVStats stats = nodes[n]->get_SNV_stats();

        penalty += static_cast<double>(stats.n_forward_mutations) * local_params->ref_to_alt_cost;
        penalty += static_cast<double>(stats.n_back_mutations) * local_params->alt_to_ref_cost;
    }

    return penalty * cell_scale_factor();
}

double Tree::compute_healthy_logprior_penalty() {
    double penalty = 0.0;

    std::vector<int> healthy_snv_only_children;
    for (int k : children[0]) {
        if (nodes[k]->get_CNA_stats().n_CNA == 0)
            healthy_snv_only_children.push_back(k);
    }

    // Accumulated below, used after the loop for the aggregate convexity term.
    double total_expected_healthy = 0.0;
    double total_root_equivalent = 0.0;

    for (int j = 0; j < n_cells; ++j) {
        double expected_healthy = data.cell_is_healthy[j];  // prior prob of being healthy
        double actual_root = cells_attach_prob[j][0];  // posterior prob of attaching to root

        double log_singlet_total = 0.0;
        if (local_params->use_doublets) {
            log_singlet_total = cells_attach_loglik[j][0];
            for (int k = 1; k < n_nodes; ++k)
                log_singlet_total = Scores::log_sum_exp(
                        log_singlet_total, cells_attach_loglik[j][k]);

            actual_root = std::exp(cells_attach_loglik[j][0] - log_singlet_total);
        }

        // Healthy cells attaching to a CNA-free direct child of the root
        // are credited toward "attached to root" (plausibly just SNV/locus-level
        // heterogeneity within the healthy population rather than a true
        // cancer clone.)
        double root_equivalent = actual_root;
        if (!healthy_snv_only_children.empty()) {
            double credit_mass = 0.0;
            for (int k : healthy_snv_only_children) {
                credit_mass += local_params->use_doublets
                               ? std::exp(cells_attach_loglik[j][k] - log_singlet_total)
                               : cells_attach_prob[j][k];
            }
            root_equivalent += local_params->healthy_heterogeneity_credit * credit_mass;
        }

        total_expected_healthy += expected_healthy;
        total_root_equivalent  += expected_healthy * root_equivalent;

        // Healthy cell not attached to root (or a credited CNA-free child):
        // weighted by expected_healthy * (1 - root_equivalent)
        penalty += local_params->healthy_cell_attachment_cost
                   * expected_healthy * (1.0 - root_equivalent);

        // Cancer cell attached to root: weighted by (1 - expected_healthy) * actual_root
        penalty += local_params->cancer_cell_at_root_cost
                   * (1.0 - expected_healthy) * actual_root;
    }

    // Aggregate convexity term, on top of the per-cell sum above: makes the
    // effective per-cell penalty grow as the overall attached fraction F drops,
    // so a tree where only e.g. 20% of healthy cells attach to root is punished
    // more per bad cell than one where 80% do, not just more in total.
    if (total_expected_healthy > 0.0) {
        double F = total_root_equivalent / total_expected_healthy;
        const double HEALTHY_ROOT_FRACTION_TOLERANCE = 0.50;
        const double HEALTHY_ROOT_FRACTION_POWER = 2.0;
        const double HEALTHY_ROOT_FRACTION_SCALE = 5.0;
        double deficit = (1.0 - HEALTHY_ROOT_FRACTION_TOLERANCE) - F;
        if (deficit > 0.0) {
            penalty += local_params->healthy_cell_attachment_cost * total_expected_healthy
                       * HEALTHY_ROOT_FRACTION_SCALE * std::pow(deficit, HEALTHY_ROOT_FRACTION_POWER);
        }
    }

    return penalty;  // already a sum over cells: no dataset-size factor
}

// log density, without the normalizing constant, of the Beta prior (mean, concentration omega) on a dropout rate;
// 0 when the prior is off (omega <= 0)
static double dropout_log_prior(double rho, double mean, double omega) {
    if (omega <= 0.0) return 0.0;
    return (mean * omega - 1.0) * std::log(rho) + ((1.0 - mean) * omega - 1.0) * std::log(1.0 - rho);
}

double Tree::compute_parameter_logprior() {
    double penalty = 0;
    double mu = local_params->dropoutrate;
    double omega = local_params->prior_dropoutrate_omega;
    if (omega <= 0.0) return 0.0;

    // Log Beta PDF, summed over each sample's own fitted dropout rate
    double log_B = std::lgamma(mu * omega) + std::lgamma((1.0 - mu) * omega) - std::lgamma(mu * omega + (1.0 - mu) * omega);
    for (const SampleFit& fit : sample_fits)
        penalty -= dropout_log_prior(fit.dropoutrate, mu, omega) - log_B;

    return penalty;
}

// digamma (psi) via recurrence to x >= 6 and the asymptotic series
static inline double digamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
    const double r2 = 1.0 / (x * x);
    return result + std::log(x) - 0.5 / x
           - r2 * (1.0/12 - r2 * (1.0/120 - r2 * (1.0/252 - r2 * (1.0/240 - r2 / 132))));
}

// trigamma (psi_1) via recurrence to x >= 6 and the asymptotic series
static inline double trigamma(double x) {
    double result = 0.0;
    while (x < 6.0) { result += 1.0 / (x * x); x += 1.0; }
    const double r = 1.0 / x, r2 = r * r;
    return result + r + 0.5 * r2 + r * r2 * (1.0/6 - r2 * (1.0/30 - r2 * (1.0/42 - r2 / 30)));
}

// Both fits evaluate the root, where alpha_total = region_dispersion * cn_r * (1 - dropout) (see Node::compute_CNA_scores).
static double fit_region_dispersion_newton(
        const std::vector<double>& n_jk, const std::vector<double>& N_j, const std::vector<double>& w_j,
        int cn_r, double region_probability, double dropout,
        double background_rate, double init_dispersion,
        int max_iter = 50, double tol = 1e-4);

// Golden-section search for the dropout rate maximizing the root cells' region log-likelihood plus the dropout
// prior, with region_dispersion and region_probabilities fixed.
static double fit_dropout_golden_section(
        const std::vector<std::vector<double>>& n_jk_by_region,
        const std::vector<double>& N_j, const std::vector<double>& w_j,
        const std::vector<int>& cn_r_by_region,
        const std::vector<double>& region_probabilities,
        const std::vector<double>& region_dispersion,
        double background_rate, double prior_mean, double prior_omega,
        double lo = 0.001, double hi = 0.6, double tol = 1e-4);


// genotype of a diploid root locus from its VAF in root cells: 0 hom-ref, 1 het, 2 hom-alt
static int classify_root_genotype(double vaf) { return vaf < 0.25 ? 0 : (vaf > 0.75 ? 2 : 1); }

// " (sample s)" for messages of multi-sample runs, empty otherwise
static std::string sample_label(int s) { return data.n_samples > 1 ? " (sample " + std::to_string(s) + ")" : ""; }

void Tree::compute_prior_score(){
    log_prior_score = 0.0;

    // Penalize number of nodes.
    double total_node_cost = local_params->node_cost; // root: full cost
    for (int k = 1; k < n_nodes; ++k) {
        bool cn_neutral = (nodes[k]->cn_regions == nodes[parents[k]]->cn_regions);
        total_node_cost += local_params->node_cost * (cn_neutral ? local_params->cn_neutral_node_cost_factor : 1.0);  // Copy neutral clones have a lower cost as usually we have many regions, but few SNVs
    }

    log_prior_score -= total_node_cost * cell_scale_factor();

    // Optional target node/clone count (-1 = disabled): steep extra penalty,
    // quadratic in the mismatch, on top of the per-node cost above.
    if (local_params->desired_n_nodes != -1) {
        int d = std::abs(n_nodes - local_params->desired_n_nodes);
        const double NODE_COUNT_MISMATCH_SCALE = 10.0;
        log_prior_score -= local_params->node_cost * cell_scale_factor()
                            * NODE_COUNT_MISMATCH_SCALE * d * d;
    }

    // Beta prior pulling the tree's posterior doublet fraction toward target_doublet_rate (the same rate the cell
    // likelihood uses)
    if (local_params->use_doublets) {
        double actual_doublet_frac = 0.0;
        for (int j = 0; j < n_cells; ++j) {
            double singlet_mass = 0.0;
            for (int k = 0; k < n_nodes; ++k) singlet_mass += cells_attach_prob[j][k];
            actual_doublet_frac += (1.0 - singlet_mass);
        }
        actual_doublet_frac /= n_cells;

        // Beta(mu*omega, (1-mu)*omega) log-density, shifted so that the cost is 0 at x = mu
        const double eps = 1e-9;
        double mu = local_params->target_doublet_rate;
        double x = std::clamp(actual_doublet_frac, eps, 1.0 - eps);
        double omega = local_params->target_doublet_rate_omega;
        double a = mu * omega, b = (1.0 - mu) * omega;
        double cost = (a - 1.0) * (std::log(mu) - std::log(x))
                    + (b - 1.0) * (std::log(1.0 - mu) - std::log(1.0 - x));

        log_prior_score += -cell_scale_factor() * cost;
    }

    // Per-doublet-combination count check (always on when use_doublets is true):
    // a Poisson log-likelihood comparing each (k,l) pair's actual (soft)
    // cell count  against what random
    // pairing predicts from the tree's own node_probabilities + target_doublet_rate.
    if (local_params->use_doublets && !doublet_combination_counts.empty()) {
        double doublet_combination_penalty = 0.0;
        // Asymmetric around the Poisson mean E
        const double DOUBLET_COMBINATION_SCALE_LOW  = 0.5;   // actual < E (undershoot)
        const double DOUBLET_COMBINATION_SCALE_HIGH = 5.0;   // actual >= E (overshoot) for pair doublets k != l
        const double SELF_DOUBLET_SCALE_HIGH = DOUBLET_COMBINATION_SCALE_HIGH / 2.0;  // k==l self-doublets overshoot E more readily than true pair doublets (cycling/G2M cells), so tolerate it more -- undershoot is still scored at the shared LOW rate above
        for (int g = 0; g < (int)doublet_combination_counts.size(); ++g) {
            double n_cells_g = (doublet_combination_counts.size() > 1)
                ? (double)data.cells_by_sample[g].size() : (double)n_cells;
            int idx = 0;
            for (int k = 0; k < n_nodes; ++k) {
                for (int l = k; l < n_nodes; ++l, ++idx) {
                    double p_k = node_probabilities[g][k];
                    double p_l = node_probabilities[g][l];
                    double E = n_cells_g * local_params->target_doublet_rate
                               * (k == l ? p_k * p_k : 2.0 * p_k * p_l);
                    E = std::max(E, 1e-9); // guard log(0)
                    double actual = doublet_combination_counts[g][idx];

                    double raw_actual = actual * std::log(E) - E - std::lgamma(actual + 1.0);
                    double raw_peak   = E * std::log(E) - E - std::lgamma(E + 1.0);
                    double gap = raw_actual - raw_peak; // <= 0, exactly 0 at actual == E

                    double scale;
                    if (actual < E) scale = DOUBLET_COMBINATION_SCALE_LOW;
                    else            scale = (k == l) ? SELF_DOUBLET_SCALE_HIGH : DOUBLET_COMBINATION_SCALE_HIGH;
                    doublet_combination_penalty += scale * gap;
                }
            }
        }
        log_prior_score += doublet_combination_penalty;
    }

    // Penalize CNA events
    log_prior_score -= compute_CNA_logprior_penalty();

    // Penalize SNV events
    log_prior_score -= compute_SNV_logprior_penalty();

    // Penalize root genotype
    if (!local_params->annotation_released
        && local_params->use_healthy_annotation) {
        log_prior_score -= compute_healthy_logprior_penalty();
    }

    // Parameter priors
    log_prior_score -= compute_parameter_logprior();

}


void Tree::update_full_score(){
    log_score = log_likelihood + log_prior_score;
    if (local_params->verbose){
        std::cout << "Log-likelihood: " << log_likelihood << ", Log-prior: " << log_prior_score << ", Log-score: " << log_score << std::endl;
    }
}

// Golden-section maximization
template <typename F>
static double golden_section_max(F f, double lo, double hi, double tol) {
    const double gr = 0.6180339887498949; // 1/phi
    double a = lo, b = hi;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = f(c), fd = f(d);
    while (std::abs(b - a) > tol) {
        if (fc > fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
        else         { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
    }
    return 0.5 * (a + b);
}


// Region parameters from the --region_reference file instead of a fit: used during the whole burn-in,
// until Inference::mcmc() refits once from the soft root attachment.
void Tree::fit_reference_parameters(const std::vector<int>& cell_indices,
                                    std::vector<double>& out_region_probabilities,
                                    std::vector<double>& out_region_dispersion,
                                    SampleFit& out_fit,
                                    double& out_effective_root_cells) {
    const std::vector<int> copy_numbers_root = root_copy_numbers();

    std::vector<double> region_probabilities(n_regions);
    double sum = 0.0, kept_share = 0.0;
    for (int k = 0; k < n_regions; ++k) {
        kept_share += data.reference_rho_per_copy[k] * copy_numbers_root[k];
        region_probabilities[k] = data.reference_rho_per_copy[k] * copy_numbers_root[k] + 1e-6;
        sum += region_probabilities[k];
    }
    for (int k = 0; k < n_regions; ++k) region_probabilities[k] /= sum;

    // The reference phi is the concentration for a read total over the whole diploid panel; a total over
    // fewer regions (or fewer copies) has its concentration scaled by their read share (Dirichlet aggregation).
    std::vector<double> region_dispersion(n_regions);
    for (int k = 0; k < n_regions; ++k)
        region_dispersion[k] = std::max(0.1, std::min(data.reference_phi[k] * kept_share, 1e5));

    if (data.reference_dropout >= 0.0) out_fit.dropoutrate = data.reference_dropout;

    std::vector<int> totals;
    totals.reserve(cell_indices.size());
    for (int j : cell_indices) totals.push_back(cells[j].total_counts);
    if (!totals.empty()) {
        auto mid = totals.begin() + totals.size() / 2;
        std::nth_element(totals.begin(), mid, totals.end());
        out_fit.mean_total_count = std::max(1.0, static_cast<double>(*mid));
    }

    out_region_probabilities = std::move(region_probabilities);
    out_region_dispersion = std::move(region_dispersion);
    // not a fit from root-attached cells, so update_region_weights() must not fall back to the old values
    out_effective_root_cells = static_cast<double>(cell_indices.size());
}

// Fits region_probabilities/region_dispersion/dropout/mean_total_count/theta_total_count
void Tree::fit_sample_parameters(int sample_idx, const std::vector<int>& cell_indices,
                                  std::vector<double>& out_region_probabilities,
                                  std::vector<double>& out_region_dispersion,
                                  SampleFit& out_fit,
                                  double& out_effective_root_cells) {
    if (data.has_region_reference && local_params->region_reference_active) {
        fit_reference_parameters(cell_indices, out_region_probabilities, out_region_dispersion,
                                 out_fit, out_effective_root_cells);
        return;
    }
    std::vector<double> region_probabilities(n_regions, 0.0);
    std::vector<double> region_dispersion(n_regions, local_params->mu_dir_multi);
    double delta = 1e-6;
    const std::string sample_tag = sample_label(sample_idx);

    std::vector<double> tilde_w(n_regions, 0.0);

    // MOM accumulators: weighted sum of p_mk and p_mk^2 per region
    std::vector<double> sum_w_p(n_regions, 0.0);
    std::vector<double> sum_w_p2(n_regions, 0.0);
    double sum_w_root = 0.0;

    double sum_weight = 0.0;
    double sum_w_N  = 0.0;
    double sum_w_N2 = 0.0;

    // The root cells are the cells attached to node 0 (soft attachment); no other node contributes to the fit.
    auto soft_root = [&](int j) { return cells_attach_prob[j][0]; };

    for (int j : cell_indices) {
        double p_root = fit_from_labels() ? (double)data.cell_is_healthy[j] : soft_root(j);

        double Nj = static_cast<double>(cells[j].total_counts);

        if (Nj < 20.0) continue;

        for (int k = 0; k < n_regions; ++k) {
            if (k >= (int)cells[j].region_counts.size()) continue;

            double n_jk = static_cast<double>(cells[j].region_counts[k]);
            double p_jk = n_jk / Nj;

            tilde_w[k]  += p_root * n_jk;
            sum_w_p[k]  += p_root * p_jk;
            sum_w_p2[k] += p_root * p_jk * p_jk;
        }

        sum_w_root += p_root;

        if (local_params->use_total_count_likelihood) {
            sum_weight += p_root;
            sum_w_N += p_root * Nj;
            sum_w_N2 += p_root * Nj * Nj;
        }
    }

    out_effective_root_cells = sum_w_root;

    // Warn if there aren't enough healthy cells to reliably estimate region weights
    double min_fraction = 0.2;
    double min_count = 50.0;
    if (sum_w_root < min_count
        && sum_w_root < min_fraction * static_cast<double>(cell_indices.size())) {
        std::cerr << "Warning" << sample_tag << ": only " << sum_w_root << " effective healthy cells available "
                  << "out of " << cell_indices.size() << " cells (need at least " << min_count
                  << ", or at least " << (min_fraction * 100.0) << "% of all cells). "
                  << "Region weight and dispersion estimates may be unreliable." << std::endl;
    }

    // Warn about healthy regions with very low observed coverage
    if (sum_w_root > 0.0) {
        double uniform_share = 1.0 / static_cast<double>(n_regions);
        std::vector<std::string> low_coverage_regions;
        for (int k = 0; k < n_regions; ++k) {
            double mu_k = sum_w_p[k] / sum_w_root; // average read fraction in healthy cells
            if (mu_k < 0.05 * uniform_share)
                low_coverage_regions.push_back(data.region_to_name[k]);
        }
        if (!low_coverage_regions.empty()) {
            std::cerr << "Warning" << sample_tag << ": very low coverage in healthy cells for region(s): ";
            for (size_t i = 0; i < low_coverage_regions.size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << low_coverage_regions[i];
            }
            std::cerr << ". Region weight estimates for these regions may be unreliable." << std::endl;
        }
    }

    // ── region_probabilities ────────────────────
    double sum = 0.0;
    for (int k = 0; k < n_regions; ++k) {
        region_probabilities[k] = tilde_w[k] + delta;
        sum += region_probabilities[k];
    }
    if (sum <= 0.0) {
        double uni = 1.0 / static_cast<double>(n_regions);
        for (int k = 0; k < n_regions; ++k) region_probabilities[k] = uni;
    } else {
        for (int k = 0; k < n_regions; ++k) region_probabilities[k] /= sum;
    }


    const std::vector<int> copy_numbers_root = root_copy_numbers();

    double dropout = out_fit.dropoutrate; // seeded by caller

    if (sum_w_root > 10.0) {
        if (local_params->verbose) {
            std::cout << "Fitting region_dispersion via method-of-moments" << sample_tag
                      << " (n_healthy_cells=" << sum_w_root << ")" << std::endl;
        }
        if (dropout > 0.05) {
            std::cerr << "Warning" << sample_tag << ": dropout rate (" << dropout
                      << ") exceeds 0.05 -- the method-of-moments region-dispersion "
                      << "estimate is imprecise at higher dropout and may be unreliable."
                      << std::endl;
        }

        for (int k = 0; k < n_regions; ++k) {
            double mu = sum_w_p[k]  / sum_w_root;
            double eq = sum_w_p2[k] / sum_w_root;
            double v  = std::max(eq - mu*mu, 1e-12);
            double alpha_total = mu*(1.0-mu)/v - 1.0;
            double root_cn = std::max(1, copy_numbers_root[k]);
            region_dispersion[k] = std::max(0.1, std::min(alpha_total / root_cn, 1e5));
        }
    }

    // ── Optional joint-likelihood fits for region_dispersion and/or dropout,
    if (sum_w_root > 50.0 && (local_params->fit_region_dispersion_mle || local_params->fit_dropout_mle)) {

        const int MAX_FIT_CELLS = 4000;
        // cells with weight 0 (unlabelled cells when fitting from the labels) add nothing to the fit
        auto fit_weight = [&](int j) { return fit_from_labels() ? (double)data.cell_is_healthy[j] : soft_root(j); };
        int n_qualifying = 0;
        for (int j : cell_indices)
            if (cells[j].total_counts >= 20.0 && fit_weight(j) > 0.0) ++n_qualifying;
        int stride = (n_qualifying > MAX_FIT_CELLS)
                     ? (n_qualifying + MAX_FIT_CELLS - 1) / MAX_FIT_CELLS
                     : 1;

        std::vector<int> fit_cell_indices;
        fit_cell_indices.reserve(std::min(n_qualifying, MAX_FIT_CELLS) + 1);
        std::vector<double> N_j_vec, w_j_vec;
        N_j_vec.reserve(fit_cell_indices.capacity());
        w_j_vec.reserve(fit_cell_indices.capacity());
        int seen = 0;
        for (int j : cell_indices) {
            if (cells[j].total_counts < 20.0 || !(fit_weight(j) > 0.0)) continue;
            if (seen % stride == 0) {
                fit_cell_indices.push_back(j);
                N_j_vec.push_back(static_cast<double>(cells[j].total_counts));
                w_j_vec.push_back(fit_weight(j));
            }
            ++seen;
        }

        if (local_params->verbose) {
            std::cout << "Starting joint-likelihood fit" << sample_tag << ": region_dispersion="
                      << (local_params->fit_region_dispersion_mle ? "MLE" : "MOM")
                      << ", dropout=" << (local_params->fit_dropout_mle ? "MLE" : "fixed")
                      << ", using " << fit_cell_indices.size() << " cells";
            if (stride > 1) {
                std::cout << " (subsampled with stride " << stride << " from "
                          << n_qualifying << " qualifying cells)";
            }
            std::cout << std::endl;
        }

        std::vector<std::vector<double>> n_jk_by_region(n_regions);
        std::vector<int> cn_r_vec(n_regions);
        for (int k = 0; k < n_regions; ++k) {
            cn_r_vec[k] = std::max(1, copy_numbers_root[k]);
            n_jk_by_region[k].reserve(fit_cell_indices.size());
            for (int j : fit_cell_indices) {
                n_jk_by_region[k].push_back(
                        k < (int)cells[j].region_counts.size()
                        ? static_cast<double>(cells[j].region_counts[k])
                        : 0.0);
            }
        }

        const double OUTER_DISPERSION_RELTOL = 1e-3;
        const double OUTER_DROPOUT_ABSTOL    = 1e-4;

        if (local_params->fit_region_dispersion_mle) {
            int n_outer = local_params->fit_dropout_mle ? 8 : 1;
            for (int outer = 0; outer < n_outer; ++outer) {
                double max_rel_change = 0.0;
                for (int k = 0; k < n_regions; ++k) {
                    double prev = region_dispersion[k];
                    double fitted = fit_region_dispersion_newton(
                            n_jk_by_region[k], N_j_vec, w_j_vec,
                            cn_r_vec[k], region_probabilities[k], dropout,
                            local_params->background_rate, region_dispersion[k]);
                    region_dispersion[k] = std::max(0.1, std::min(fitted, 1e5));
                    max_rel_change = std::max(max_rel_change,
                            std::abs(region_dispersion[k] - prev) / std::max(prev, 1.0));
                }
                if (!local_params->fit_dropout_mle) {
                    if (local_params->verbose) {
                        std::cout << "Fitted region_dispersion via joint-likelihood MLE" << sample_tag
                                  << " (dropout fixed at " << dropout << "), max relative change="
                                  << max_rel_change << std::endl;
                    }
                    break;
                }

                double prev_dropout = dropout;
                // After the first pass we already have a good dropout
                // estimate -- narrow the golden-section bracket around it
                // instead of re-searching the full [0.001,0.6] range.
                double lo = (outer == 0) ? 0.001 : std::max(0.001, dropout - 0.05);
                double hi = (outer == 0) ? 0.6   : std::min(0.6,   dropout + 0.05);
                dropout = fit_dropout_golden_section(
                        n_jk_by_region, N_j_vec, w_j_vec, cn_r_vec,
                        region_probabilities, region_dispersion,
                        local_params->background_rate,
                        local_params->dropoutrate, local_params->prior_dropoutrate_omega,
                        lo, hi);

                if (local_params->verbose) {
                    std::cout << "  round " << (outer + 1) << "/" << n_outer << sample_tag
                              << ": region_dispersion max relative change=" << max_rel_change
                              << ", dropout=" << dropout << " (delta=" << (dropout - prev_dropout) << ")"
                              << std::endl;
                }

                if (max_rel_change < OUTER_DISPERSION_RELTOL
                    && std::abs(dropout - prev_dropout) < OUTER_DROPOUT_ABSTOL) {
                    if (local_params->verbose) {
                        std::cout << "Converged" << sample_tag << " after " << (outer + 1) << " round(s)." << std::endl;
                    }
                    break;
                }
            }
        } else { // fit_dropout_mle only: MOM dispersion doesn't depend on dropout, so one shot suffices
            if (local_params->verbose) {
                std::cout << "Fitting dropout via golden-section search" << sample_tag
                          << " (region_dispersion fixed from method-of-moments)" << std::endl;
            }
            dropout = fit_dropout_golden_section(
                    n_jk_by_region, N_j_vec, w_j_vec, cn_r_vec,
                    region_probabilities, region_dispersion,
                    local_params->background_rate,
                    local_params->dropoutrate, local_params->prior_dropoutrate_omega);
            if (local_params->verbose) {
                std::cout << "Fitted dropout" << sample_tag << "=" << dropout << std::endl;
            }
        }
    }

    out_fit.dropoutrate = dropout;

    if (local_params->use_total_count_likelihood && sum_weight > 1.0) {
        double mean_N = sum_w_N  / sum_weight;
        double eq_N   = sum_w_N2 / sum_weight;
        double var_N  = std::max(eq_N - mean_N * mean_N, 1.0);

        // at the root the expected read fraction sums to 1, so mean_total_count is the mean total count directly
        out_fit.mean_total_count = mean_N;
        out_fit.theta_total_count = std::max(mean_N * mean_N / std::max(var_N - mean_N, 1.0), 0.1);
    }

    if (!local_params->fit_region_dispersion_mle) {
        // Method of moments matches a single beta-binomial's concentration, but the scored model's alpha_total(root)
        // carries an extra (1 - dropout) factor: divide it out. The dropout above was fitted before this correction.
        // The MLE path fits alpha_total directly and needs no correction.
        double keep = std::max(1e-6, 1.0 - dropout);
        for (int k = 0; k < n_regions; ++k)
            region_dispersion[k] = std::max(0.1, std::min(region_dispersion[k] / keep, 1e5));
    }

    out_region_probabilities = std::move(region_probabilities);
    out_region_dispersion = std::move(region_dispersion);
}

// The per-locus dropout arrays read by the non-joint likelihood model (use_joint_likelihood_model=false) and written to
// local_params.tsv: every locus gets its sample's fitted dropout rate. Refilled whenever sample_fits changes.
void Tree::sync_dropout_arrays() {
    dropout_rates_ref.assign(sample_fits.size(), {});
    dropout_rates_alt.assign(sample_fits.size(), {});
    for (size_t s = 0; s < sample_fits.size(); ++s) {
        dropout_rates_ref[s].assign(n_loci, sample_fits[s].dropoutrate);
        dropout_rates_alt[s].assign(n_loci, sample_fits[s].dropoutrate);
    }
}

void Tree::update_region_weights() {
    int n_s = data.n_samples;
    if ((int)sample_fits.size() != n_s)
        sample_fits.assign(n_s, SampleFit{local_params->dropoutrate, local_params->mean_total_count, local_params->theta_total_count});

    // Snapshot the previous per-sample estimates -- when use_healthy_annotation=false,
    // a periodic refit (see Inference::mcmc()) with too few effective soft-root-attached
    // cells keeps these instead of committing a noisy new fit.
    auto old_region_probabilities = region_probabilities;
    auto old_region_dispersion    = region_dispersion;
    old_region_probabilities.resize(n_s);
    old_region_dispersion.resize(n_s);
    for (int s = 0; s < n_s; ++s) {
        if (old_region_probabilities[s].empty())
            old_region_probabilities[s].assign(n_regions, 1.0 / n_regions);
        if (old_region_dispersion[s].empty())
            old_region_dispersion[s].assign(n_regions, local_params->mu_dir_multi);
    }

    region_probabilities.assign(n_s, {});
    region_dispersion.assign(n_s, {});

    bool any_sample_refit = false;
    for (int s = 0; s < n_s; ++s) {
        SampleFit fit = sample_fits[s]; // seeds dropout/mean_total_count/theta_total_count
        std::vector<double> new_region_probabilities, new_region_dispersion;
        double effective_root_cells = 0.0;
        fit_sample_parameters(s, data.cells_by_sample[s], new_region_probabilities, new_region_dispersion,
                              fit, effective_root_cells);

        const std::string sample_tag = sample_label(s);
        bool keep_old = !fit_from_labels()
                        && effective_root_cells < local_params->min_cells_for_parameter_refit;
        if (keep_old) {
            std::cerr << "Sample" << sample_tag << ": only " << effective_root_cells
                      << " effective cells attached to root (< " << local_params->min_cells_for_parameter_refit
                      << ") -- keeping previous region weight/dropout estimate." << std::endl;
            region_probabilities[s] = old_region_probabilities[s];
            region_dispersion[s]    = old_region_dispersion[s];
            // sample_fits[s] left untouched (dropout/mean_total_count/theta_total_count unchanged)
        } else {
            if (data.has_region_reference && local_params->region_reference_active)
                std::cerr << "Sample" << sample_tag << ": region weights/dispersion/dropout from the region reference."
                          << std::endl;
            else if (!fit_from_labels())
                std::cerr << "Sample" << sample_tag << ": refit region weights/dropout from "
                          << effective_root_cells << " effective soft-root-attached cells." << std::endl;
            region_probabilities[s] = std::move(new_region_probabilities);
            region_dispersion[s]    = std::move(new_region_dispersion);
            sample_fits[s] = fit;
            any_sample_refit = true;
        }
    }

    // local_params->dropoutrate is NOT overwritten: it stays the prior mean and fallback of the dropout fit
    sync_dropout_arrays();

    // cache entries are keyed on the replaced parameters, so they can never be used again
    if (any_sample_refit) cache_scores->clear_cache();

    compute_likelihood();
}

void Tree::add_node(int parent) {
    int new_node = n_nodes;
    n_nodes++;
    nodes.push_back(std::make_unique<Node>(cache_scores, rng, local_params));
    parents.push_back(parent);

    nodes_attachment_counts.resize(n_nodes, 0.0);

    // the per-cell posterior belongs to the old tree's attachment points; it is recomputed by the next compute_likelihood()
    for (auto& v : cells_attach_loglik)  v.clear();
    for (auto& v : cells_attach_prob)    v.clear();
    best_attachments.resize(n_cells, 0); // already correct size but safe

    nodes[new_node]->update_genotype(nodes[parent].get(), root_strand_loci_ref,
                                     root_strand_loci_alt, root_strand_regions);

    nodes[new_node]->attachment_scores_CNA = nodes[parent]->attachment_scores_CNA;

    for (auto& group : node_probabilities)
        group.assign(n_nodes, 1.0 / n_nodes);

    std::bernoulli_distribution coin(0.5);
    for (int child : children[parent]) {
        if (coin(rng)) parents[child] = new_node;
    }
    compute_children();
}

void Tree::delete_node(int node) {
    for (int child : children[node])
        parents[child] = parents[node];

    if (node != n_nodes - 1) {
        nodes[node] = std::move(nodes[n_nodes - 1]);
        for (int n = 0; n < n_nodes; n++) {
            if (parents[n] == n_nodes - 1) parents[n] = node;
        }
        parents[node] = parents[n_nodes - 1];
    }

    nodes.pop_back();
    parents.pop_back();
    n_nodes--;

    // nodes_attachment_counts is fully re-accumulated from scratch on every
    // compute_cells_likelihoods() call
    nodes_attachment_counts.resize(n_nodes);

    // the per-cell posterior belongs to the old tree's attachment points; it is recomputed by the next compute_likelihood()
    for (auto& v : cells_attach_loglik) v.clear();
    for (auto& v : cells_attach_prob)   v.clear();

    for (auto& group : node_probabilities)
        group.assign(n_nodes, 1.0 / n_nodes);

    compute_children();
}

// The node and all its descendants, each node listed after its parent.
std::vector<int> Tree::subtree_nodes(int node) const {
    std::vector<int> out;
    std::stack<int> stk;
    stk.push(node);
    while (!stk.empty()) {
        int top = stk.top(); stk.pop();
        out.push_back(top);
        for (int child : children[top]) stk.push(child);
    }
    return out;
}

// Every node in the subtrees of the given nodes, once, in depth-first order (each node after its parent).
std::vector<int> Tree::subtrees_nodes(const std::vector<int>& roots) const {
    std::vector<bool> in(n_nodes, false);
    for (int r : roots) for (int v : subtree_nodes(r)) in[v] = true;
    std::vector<int> out;
    for (int v : DFT_order) if (in[v]) out.push_back(v);
    return out;
}

// Pairs of CNAs in one node that can be merged into one (merge_gap >= 0). whole_chromosome: only gain/loss pairs
// that together span their chromosome (merged into an aneuploidy); otherwise all local CNA pairs, CNLOH included.
std::vector<Tree::MergeCandidate> Tree::merge_candidates(bool whole_chromosome) const {
    std::vector<MergeCandidate> out;
    for (int n = 1; n < n_nodes; ++n) {
        const auto& events = nodes[n]->Events;
        std::vector<int> cna_indices;
        for (int i = 0; i < (int)events.size(); ++i)
            if (std::holds_alternative<Node::CNA>(events[i])) {
                const Node::CNA& c = std::get<Node::CNA>(events[i]);
                if (!c.aneuploidy && !c.fgd && !(whole_chromosome && c.CNA_change == 0))
                    cna_indices.push_back(i);
            }
        for (int i = 0; i < (int)cna_indices.size(); ++i)
            for (int j = i + 1; j < (int)cna_indices.size(); ++j) {
                const Node::CNA& A = std::get<Node::CNA>(events[cna_indices[i]]);
                const Node::CNA& B = std::get<Node::CNA>(events[cna_indices[j]]);
                int gap = merge_gap(A, B);
                if (gap < 0) continue;
                if (whole_chromosome && (int)(A.region_indices.size() + B.region_indices.size()) + gap
                                        != (int)data.chromosome_to_regions[A.chromosome].size())
                    continue;
                out.push_back({n, cna_indices[i], cna_indices[j], gap == 1});
            }
    }
    return out;
}

// Every way to split one CNA into two: between any two neighbouring regions, or by dropping one inner region.
// aneuploidy: split aneuploidy gains/losses into two local CNAs; otherwise split local CNAs, CNLOH included.
std::vector<Tree::SplitCandidate> Tree::split_candidates(bool aneuploidy) const {
    std::vector<SplitCandidate> out;
    for (int n = 1; n < n_nodes; ++n)
        for (int i = 0; i < (int)nodes[n]->Events.size(); ++i) {
            if (!std::holds_alternative<Node::CNA>(nodes[n]->Events[i])) continue;
            const Node::CNA& c = std::get<Node::CNA>(nodes[n]->Events[i]);
            bool eligible = aneuploidy ? (c.aneuploidy && c.CNA_change != 0) : (!c.aneuploidy && !c.fgd);
            if (!eligible) continue;
            int L = (int)c.region_indices.size();
            for (int p = 1; p < L; ++p) out.push_back({n, i, p, false});      // between regions p-1 and p
            for (int p = 1; p < L - 1; ++p) out.push_back({n, i, p, true});   // drop inner region p
        }
    return out;
}

// Determine the nodes in the subtrees of node1 and node 2
std::vector<int> Tree::affected_nodes_subtrees(int node1, int node2) {
    std::vector<int> affected_nodes;
    std::vector<int> affected_nodes1;
    std::vector<int> affected_nodes2;
    bool node2_in_node1_subtree = false;
    bool node1_in_node2_subtree = false;

    // Check if node2 is in node1's subtree
    {
        std::stack<int> stk;
        stk.push(node1);
        while (!stk.empty()) {
            int top = stk.top();
            stk.pop();
            affected_nodes.push_back(top);
            affected_nodes1.push_back(top);
            if (top == node2) {
                node2_in_node1_subtree = true;
            }
            for (int child : children[top]) {
                stk.push(child);
            }
        }
        if (node2_in_node1_subtree) {
            return affected_nodes1;
        }
    }

    // Check if node1 is in node2's subtree
    {
        std::stack<int> stk;
        stk.push(node2);
        while (!stk.empty()) {
            int top = stk.top();
            stk.pop();
            affected_nodes.push_back(top);
            affected_nodes2.push_back(top);
            if (top == node1) {
                node1_in_node2_subtree = true;
            }
            for (int child : children[top]) {
                stk.push(child);
            }
        }
        if (node1_in_node2_subtree) {
            return affected_nodes2;
        }
    }
    // Unrelated case: update both subtrees
    return affected_nodes;
}


// MCMC tree moves (prune_reattach through merge_gap) live in
// Tree_moves.cpp, split out for readability -- declarations are still
// all in Tree.h alongside the rest of the Tree class.

void Tree::initialize_root_genotype_from_healthy_cells() {
    std::vector<double> w(n_cells);
    for (int j = 0; j < n_cells; ++j) w[j] = data.cell_is_healthy[j];
    initialize_root_genotype_from_cells(w);
}

void Tree::initialize_root_genotype_from_root_cells() {
    std::vector<double> w(n_cells);
    double sum_w = 0.0;
    for (int j = 0; j < n_cells; ++j) { w[j] = cells_attach_prob[j][0]; sum_w += w[j]; }
    if (sum_w < local_params->min_cells_for_parameter_refit) {
        std::cout << "Root SNV genotype: only " << sum_w << " cells attached to the root, using the VAF of all cells."
                  << std::endl;
        std::fill(w.begin(), w.end(), 1.0);
    } else {
        std::cout << "Root SNV genotype from the VAF of " << sum_w << " root-attached cells." << std::endl;
    }
    initialize_root_genotype_from_cells(w);
}

void Tree::report_labels_vs_root() const {
    int n_healthy = 0, healthy_off = 0, n_other = 0, other_on = 0;
    for (int j = 0; j < n_cells; ++j) {
        const double w = cells_attach_prob[j][0];
        if (data.cell_is_healthy[j]) { ++n_healthy; if (w < 0.5) ++healthy_off; }
        else                         { ++n_other; if (w >= 0.5) ++other_on; }
    }
    std::cout << "Labels vs root at the end of burn-in: " << healthy_off << " of " << n_healthy
              << " labelled healthy cells attach to non-root nodes. " << other_on << " of " << n_other
              << " other cells attach to the root." << std::endl;
}

void Tree::initialize_root_genotype_from_cells(const std::vector<double>& cell_weights) {
    // For each locus, estimate the germline VAF from the weighted cells and set
    // root_strand_loci_ref/alt accordingly
    const int n_strands = (int)root_strand_regions.size();

    for (int i = 0; i < n_loci; ++i) {
        double weighted_alt = 0.0, weighted_total = 0.0;

        for (int j = 0; j < n_cells; ++j) {
            double w = cell_weights[j];
            if (w < 1e-6) continue;
            weighted_alt   += w * cells[j].alt_counts[i];
            weighted_total += w * (cells[j].alt_counts[i] + cells[j].ref_counts[i]);
        }

        const int region = data.locus_to_region[i];
        std::vector<int> present_strands;
        for (int s = 0; s < n_strands; ++s) {
            if (root_strand_regions[s][region] >= 1) {
                present_strands.push_back(s);
            } else {
                root_strand_loci_ref[s][i] = 0;
                root_strand_loci_alt[s][i] = 0;
            }
        }
        if (present_strands.empty()) continue; // no real copy anywhere -- nothing to set

        if (weighted_total < 1.0) continue; // no coverage — leave default on the present strand(s)

        double vaf = weighted_alt / weighted_total;

        if (present_strands.size() == 1) {
            // Hemizygous locus (e.g. male chrX/chrY): no "het" state, just
            // ref or alt on the one strand that actually carries the region.
            int s = present_strands[0];
            if (vaf > 0.5) {
                root_strand_loci_ref[s][i] = 0;
                root_strand_loci_alt[s][i] = 1;
            } else {
                root_strand_loci_ref[s][i] = 1;
                root_strand_loci_alt[s][i] = 0;
            }
            continue;
        }

        const int genotype = classify_root_genotype(vaf);
        if (genotype == 0) {
            // hom ref: both strands ref
            for (int s : present_strands) {
                root_strand_loci_ref[s][i] = 1;
                root_strand_loci_alt[s][i] = 0;
            }
        } else if (genotype == 2) {
            // hom alt: both strands alt
            for (int s : present_strands) {
                root_strand_loci_ref[s][i] = 0;
                root_strand_loci_alt[s][i] = 1;
            }
        } else {
            // het: first present strand = ref, second = alt
            root_strand_loci_ref[present_strands[0]][i] = 1; root_strand_loci_alt[present_strands[0]][i] = 0;
            root_strand_loci_ref[present_strands[1]][i] = 0; root_strand_loci_alt[present_strands[1]][i] = 1;
        }
    }
}


// Sample the current tree and parameters
// "nodes":[...] with each node's parent and events; one_line drops the line breaks and tabs
void Tree::write_tree_json(std::ostream& out, bool one_line) const {
    const char* nl = one_line ? "" : "\n";
    const char* tab = one_line ? "" : "\t";
    const char* tab2 = one_line ? "" : "\t\t";
    out << "\"nodes\":[" << nl;
    for (int k = 0; k < n_nodes; ++k) {
        out << tab << "{" << nl;
        out << tab2 << "\"name\": \"Node " << k << "\"," << nl;
        out << tab2 << "\"parent\": \"" << (parents[k] >= 0 ? "Node " + std::to_string(parents[k]) : "-") << "\"," << nl;

        out << tab2 << "\"events\": [";
        bool first_event = true;
        for (const auto& ev : nodes[k]->Events) {
            if (!first_event) out << ",";
            first_event = false;
            if (std::holds_alternative<Node::SNV>(ev)) {
                const auto& s = std::get<Node::SNV>(ev);
                out << "{\"type\":\"SNV\""
                    << ",\"locus\":" << s.locus
                    << ",\"affected_strand\":" << s.affected_strand
                    << ",\"affected_allele\":" << s.affected_allele
                    << "}";
            } else {
                const auto& c = std::get<Node::CNA>(ev);
                out << "{\"type\":\"CNA\""
                    << ",\"region_indices\":[";
                for (size_t r = 0; r < c.region_indices.size(); ++r) {
                    if (r > 0) out << ",";
                    out << c.region_indices[r];
                }
                out << "]"
                    << ",\"CNA_change\":" << c.CNA_change
                    << ",\"affected_strand\":" << c.affected_strand
                    << ",\"affected_strand2\":" << c.affected_strand2
                    << ",\"aneuploidy\":" << (c.aneuploidy ? "true" : "false")
                    << ",\"fgd\":" << (c.fgd ? "true" : "false")
                    << ",\"chromosome\":\"" << c.chromosome << "\""
                    << "}";
            }
        }
        out << "]" << nl;

        out << tab << "}" << (k < n_nodes - 1 ? "," : "") << nl;
    }
    out << "]";
}

// {"0":{"strands_regions":[...],"strands_loci_ref":[...],"strands_loci_alt":[...]},...}
void Tree::write_strand_genotypes_json(std::ostream& out) const {
    auto write_strands = [&](const std::vector<std::vector<int>>& strands, int n) {
        for (size_t s = 0; s < strands.size(); ++s) {
            if (s > 0) out << ",";
            out << "[";
            for (int i = 0; i < n; ++i) {
                if (i > 0) out << ",";
                out << strands[s][i];
            }
            out << "]";
        }
    };
    out << "{";
    for (int k = 0; k < n_nodes; ++k) {
        if (k > 0) out << ",";
        out << "\"" << k << "\":{";
        out << "\"strands_regions\":[";
        write_strands(nodes[k]->strands_regions, n_regions);
        out << "],\"strands_loci_ref\":[";
        write_strands(nodes[k]->strands_loci_ref, n_loci);
        out << "],\"strands_loci_alt\":[";
        write_strands(nodes[k]->strands_loci_alt, n_loci);
        out << "]}";
    }
    out << "}";
}

// Fitted parameters of every sample (best_tree_parameters.json)
void Tree::write_parameters_json(std::ostream& out) const {
    auto write_vec = [&](const std::vector<double>& v) {
        out << "[";
        for (size_t i = 0; i < v.size(); ++i) out << (i ? "," : "") << v[i];
        out << "]";
    };
    out << std::setprecision(17);
    out << "{\"annotation_released\":" << (local_params->annotation_released ? "true" : "false")
        << ",\"region_reference_active\":" << (local_params->region_reference_active ? "true" : "false")
        << ",\"samples\":[";
    for (size_t sm = 0; sm < sample_fits.size(); ++sm) {
        out << (sm ? "," : "") << "{\"dropoutrate\":" << sample_fits[sm].dropoutrate
            << ",\"mean_total_count\":" << sample_fits[sm].mean_total_count
            << ",\"theta_total_count\":" << sample_fits[sm].theta_total_count
            << ",\"region_probabilities\":";
        write_vec(sm < region_probabilities.size() ? region_probabilities[sm] : std::vector<double>{});
        out << ",\"region_dispersion\":";
        write_vec(sm < region_dispersion.size() ? region_dispersion[sm] : std::vector<double>{});
        out << "}";
    }
    out << "]}\n";
}

// Best singlet node and doublet flag of every cell, and (if out_probs) its attachment probabilities.
// A non-empty step is written as a first column.
void Tree::write_cell_assignments(std::ostream& out_assign, std::ostream* out_probs,
                                  const std::string& step, bool header) const {
    const std::string step_col = step.empty() ? "" : step + "\t";
    const int n_doublets = local_params->use_doublets ? (n_nodes * (n_nodes + 1)) / 2 : 0;

    if (header) {
        out_assign << (step.empty() ? "" : "step\t") << "cell\tsample\tnode\tdoublet\n";
        if (out_probs) {
            *out_probs << "cell\tsample";
            for (int k = 0; k < n_nodes; ++k)
                *out_probs << "\tNode " << k;
            if (local_params->use_doublets) {
                for (int n1 = 0; n1 < n_nodes; ++n1)
                    for (int n2 = n1; n2 < n_nodes; ++n2)
                        *out_probs << "\tDoublet_" << n1 << "_" << n2;
            }
            *out_probs << "\n";
        }
    }

    for (int j = 0; j < n_cells; ++j) {
        int best_singlet = 0;
        double best_score = -DBL_MAX;
        for (int k = 0; k < n_nodes; ++k) {
            if (cells_attach_loglik[j][k] > best_score) {
                best_score = cells_attach_loglik[j][k];
                best_singlet = k;
            }
        }
        out_assign << step_col << cells[j].name << "\t" << cells[j].sample_id << "\t" << best_singlet
                   << "\t" << (best_attachments[j] >= n_nodes ? "yes" : "no") << "\n";

        if (!out_probs) continue;
        *out_probs << cells[j].name << "\t" << cells[j].sample_id;
        double full_loglik = Scores::log_sum_exp(cells_attach_loglik[j]);
        for (int k = 0; k < n_nodes; ++k)
            *out_probs << "\t" << std::exp(cells_attach_loglik[j][k] - full_loglik);
        for (int idx = 0; idx < n_doublets; ++idx)
            *out_probs << "\t" << std::exp(cells_attach_loglik[j][n_nodes + idx] - full_loglik);
        *out_probs << "\n";
    }
}

void Tree::sample_parameters(int mcmc_step, const std::string &output_dir) {

    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    // the first sample (step 0) starts new files, later samples are appended
    const bool is_first_step = (mcmc_step == 0);
    const auto mode = is_first_step ? std::ios::out : std::ios::app;

    // one line per sample
    std::ofstream out_tree(output_dir + "trees.jsonl", mode);
    out_tree << "{\"step\":" << mcmc_step << ",";
    write_tree_json(out_tree, true);
    out_tree << "}\n";

    std::ofstream out_strands(output_dir + "strand_genotypes.jsonl", mode);
    out_strands << "{\"step\":" << mcmc_step << ",\"nodes\":";
    write_strand_genotypes_json(out_strands);
    out_strands << "}\n";

    std::ofstream out_assign(output_dir + "cellAssignments.tsv", mode);
    write_cell_assignments(out_assign, nullptr, std::to_string(mcmc_step), is_first_step);

    // Local params
    std::ofstream out_params(output_dir + "local_params.tsv", mode);
    if (out_params.is_open()) {
        if (is_first_step) out_params << "step\tparameter\tvalue\n";

        auto write_2d = [&](const std::string &name, const auto &mat) {
            out_params << mcmc_step << "\t" << name << "\t";
            for (size_t i = 0; i < mat.size(); ++i) {
                if (i) out_params << ";";
                for (size_t s = 0; s < mat[i].size(); ++s) {
                    if (s) out_params << ",";
                    out_params << mat[i][s];
                }
            }
            out_params << "\n";
        };

        write_2d("dropout_rates_ref", dropout_rates_ref);
        write_2d("dropout_rates_alt", dropout_rates_alt);
        write_2d("region_probabilities", region_probabilities);
        write_2d("node_probabilities",   node_probabilities);

        write_2d("root_strands_loci_ref",  root_strand_loci_ref);
        write_2d("root_strands_loci_alt",  root_strand_loci_alt);
        write_2d("root_strands_regions",   root_strand_regions);

        out_params.close();
    }
}


// Save the best tree and parameters
void Tree::to_dot(const std::string& output_dir) {
    std::vector<std::string> colors{"lightcoral","skyblue3","sandybrown","paleturquoise3","thistle",
                                    "darkolivegreen3","lightpink","mediumpurple","darkseagreen3","navajowhite","gold"};

    std::string base = output_dir + "/best_tree";

    // --- Graphviz tree ---
    std::ofstream out_gv(base + ".gv");
    out_gv << "digraph G{\n";
    out_gv << "node [color=dimgray fontsize=24 fontcolor=black fontname=Helvetica penwidth=5];\n";

    // Edges
    for (int i = 1; i < n_nodes; ++i)
        out_gv << parents[i] << " -> " << i << " [color=dimgray penwidth=4 weight=2];\n";

    // A local CNA covers a contiguous run of a chromosome's regions; label it by its 1-based span, e.g. "chr3 regions 2-5".
    auto chr_region_span = [&](const Node::CNA& c) -> std::string {
        const std::vector<int>& chr_regions = data.chromosome_to_regions.at(c.chromosome);
        std::vector<int> sorted_regions = c.region_indices;
        std::sort(sorted_regions.begin(), sorted_regions.end());
        auto ordinal = [&](int r) {
            auto it = std::find(chr_regions.begin(), chr_regions.end(), r);
            return (int)(it - chr_regions.begin()) + 1; // 1-based
        };
        int first = ordinal(sorted_regions.front());
        int last  = ordinal(sorted_regions.back());
        return "chr" + c.chromosome + " " + (first == last
            ? ("region " + std::to_string(first))
            : ("regions " + std::to_string(first) + "-" + std::to_string(last)));
    };

    // Node labels with events
    for (int k = 0; k < n_nodes; ++k) {
        std::string label = "Node " + std::to_string(k) + "\\n";

        for (const auto& ev : nodes[k]->Events) {
            if (std::holds_alternative<Node::SNV>(ev)) {
                const auto& s = std::get<Node::SNV>(ev);
                label += data.locus_to_name[s.locus] + "\\n";
            } else if (std::holds_alternative<Node::CNA>(ev)) {
                const auto& c = std::get<Node::CNA>(ev);
                if (c.fgd) {
                    label += "FGD\\n";
                } else if (c.CNA_change == 0) {
                    label += "CNLOH " + chr_region_span(c) + "\\n";
                } else {
                    std::string t = (c.CNA_change > 0) ? "Gain " : "Loss ";
                    if (c.aneuploidy) {
                        // Explicitly labeled "Aneuploidy" so it reads unambiguously
                        // against a local CNA that happens to span many regions on
                        // the same chromosome -- those report an ordinal span below,
                        // unchanged in kind, since they're a different event type,
                        // not just a large version of this one.
                        label += "Aneuploidy " + t + "chr" + c.chromosome
                               + " (" + std::to_string(c.region_indices.size()) + " regions)\\n";
                    } else {
                        label += t + chr_region_span(c) + "\\n";
                    }
                }
            }
        }
        out_gv << k << " [label=\"" << label << "\" color=" << colors[k % colors.size()] << " penwidth=5];\n";
    }

    // Cell count bubbles
    std::vector<int> count_nodes(n_nodes, 0);
    int total = 0;
    // Per-sample breakdown, only appended to the label when data.n_samples>1 --
    // single-sample dot output is otherwise untouched (identical bubble label).
    std::vector<std::vector<int>> count_nodes_by_sample(data.n_samples, std::vector<int>(n_nodes, 0));
    for (int j = 0; j < n_cells; ++j)
        if (best_attachments[j] >= 0 && best_attachments[j] < n_nodes) {
            count_nodes[best_attachments[j]]++;
            total++;
            int s = (j < (int)cells.size()) ? cells[j].sample_id : 0;
            if (s >= 0 && s < (int)count_nodes_by_sample.size())
                count_nodes_by_sample[s][best_attachments[j]]++;
        }

    for (int k = 0; k < n_nodes; ++k) {
        out_gv << k << " -> " << k + n_nodes
               << " [dir=none style=dashed weight=1 penwidth=5 color=" << colors[k % colors.size()] << "];\n";
    }
    // Multi-sample: each bubble is a pie with one slice per
    // sample, sized by that sample's cell count at this node
    const std::vector<std::string> sample_colors{"#4477AA","#EE6677","#228833","#CCBB44",
                                                 "#66CCEE","#AA3377","#BBBBBB"};
    for (int k = 0; k < n_nodes; ++k) {
        double size = (total > 0) ? std::sqrt(100.0 * count_nodes[k] / total) / 3.0 : 0.1;
        out_gv << k + n_nodes << " [label=\"" << count_nodes[k] << " cells\\n"
               << std::round(100.0 * count_nodes[k] / std::max(total, 1)) << "%";
        if (data.n_samples > 1) {
            for (int s = 0; s < data.n_samples; ++s)
                out_gv << "\\nsample " << s << ": " << count_nodes_by_sample[s][k];
        }
        out_gv << "\"";
        if (data.n_samples > 1 && count_nodes[k] > 0) {
            std::vector<int> present;
            for (int s = 0; s < data.n_samples; ++s)
                if (count_nodes_by_sample[s][k] > 0) present.push_back(s);
            // graphviz's wedged style draws nothing for a single colour
            out_gv << (present.size() == 1 ? " style=filled fillcolor=\"" : " style=wedged fillcolor=\"");
            for (size_t i = 0; i < present.size(); ++i) {
                int s = present[i];
                if (i > 0) out_gv << ":";
                out_gv << sample_colors[s % sample_colors.size()];
                // the last slice gets no fraction so it takes the remainder --
                // rounded fractions summing past 1 would make graphviz warn
                if (i + 1 < present.size())
                    out_gv << ";" << std::to_string((double)count_nodes_by_sample[s][k] / count_nodes[k]);
            }
            out_gv << "\"";
        } else {
            out_gv << " style=filled";
        }
        out_gv << " width=" << size << " height=" << size
               << " color=" << colors[k % colors.size()] << "];\n";
    }
    if (data.n_samples > 1) {
        out_gv << "legend [shape=none margin=0 label=<<table border=\"0\" cellspacing=\"4\">";
        for (int s = 0; s < data.n_samples; ++s)
            out_gv << "<tr><td bgcolor=\"" << sample_colors[s % sample_colors.size()]
                   << "\" width=\"24\" height=\"24\"></td><td align=\"left\">sample " << s << "</td></tr>";
        out_gv << "</table>>];\n";
    }
    out_gv << "}\n";
    out_gv.close();

    std::ofstream out_json(base + ".json");
    out_json << "{";
    write_tree_json(out_json, false);
    out_json << "}\n";
    out_json.close();

    std::ofstream out_assign(base + "_cellAssignments.tsv");
    std::ofstream out_probs(base + "_cellAssignmentProbs.tsv");
    write_cell_assignments(out_assign, &out_probs, "", true);
    out_assign.close();
    out_probs.close();

    std::ofstream out_json2(base + "_strand_genotypes.json");
    write_strand_genotypes_json(out_json2);
    out_json2.close();

    std::ofstream out_params(base + "_parameters.json");
    write_parameters_json(out_params);
    out_params.close();

    // --- Verbose logging ---
    if (local_params->verbose) {
        // node_probabilities is indexed [group][node] -- 1 pooled group, or one
        // per sample (see update_node_probabilities()).
        for (size_t g = 0; g < node_probabilities.size(); ++g) {
            std::string group_tag = (node_probabilities.size() > 1) ? (" (group " + std::to_string(g) + ")") : "";
            std::cout << "Node probabilities" << group_tag << ":\n";
            for (int n = 0; n < n_nodes; ++n)
                std::cout << n << ": " << node_probabilities[g][n] << "\n";
        }

        for (int s = 0; s < (int)sample_fits.size(); ++s) {
            const std::string sample_tag = sample_label(s);
            std::cout << "Dropout rate" << sample_tag << ": " << sample_fits[s].dropoutrate << "\n";

            if ((int)region_dispersion[s].size() == n_regions && (int)region_probabilities[s].size() == n_regions) {
                std::cout << "Per-variant region weight w_r = p_r * a_r" << sample_tag << ":\n";
                for (int i = 0; i < n_loci; ++i) {
                    int r = data.locus_to_region[i];
                    double w_r = region_probabilities[s][r] * region_dispersion[s][r];
                    std::cout << i << " (" << data.locus_to_name[i] << ", region " << r << "): " << w_r << "\n";
                }

                const std::vector<int> copy_numbers_root = root_copy_numbers();

                double dropout_s = sample_fits[s].dropoutrate;

                std::cout << "Per-region w_r and alpha_total root" << sample_tag << ":\n";
                for (int r = 0; r < n_regions; ++r) {
                    double w_r = region_probabilities[s][r] * region_dispersion[s][r];
                    // Matches compute_attachment_scores_full/_parent's default-branch
                    // formula evaluated at root: alpha_total(root) =
                    // region_dispersion[r]*root_cn_regions[r]*(1-dropout).
                    double alpha_total = region_dispersion[s][r] * std::max(1, copy_numbers_root[r]) * (1.0 - dropout_s);
                    std::cout << r << " (" << data.region_to_name[r] << "): w_r=" << w_r
                              << ", alpha_total=" << alpha_total << "\n";
                }
            } else {
                std::cout << "Region weights not yet computed" << sample_tag
                          << " (region_dispersion/region_probabilities empty) -- skipping w_r/alpha_total logging.\n";
            }
        }
    }
}

// Newton fit of region_dispersion d in log space, with dropout fixed, at the root (alpha_total = d * cn_r * (1 - dropout),
// alpha_per_copy = d * region_probability)
static double fit_region_dispersion_newton(
        const std::vector<double>& n_jk,
        const std::vector<double>& N_j,
        const std::vector<double>& w_j,
        int  cn_r,
        double region_probability,
        double dropout,
        double background_rate,
        double init_dispersion,
        int  max_iter,
        double tol) {

    int M = (int)n_jk.size();
    if (M == 0) return init_dispersion;

    double c_total = cn_r * (1.0 - dropout); // alpha_total = d * c_total, evaluated at root
    double x = std::log(std::max(init_dispersion, 0.1));

    std::vector<double> logw(cn_r + 1), scales(cn_r + 1);
    for (int m = 0; m <= cn_r; ++m) {
        logw[m] = std::lgamma(cn_r+1) - std::lgamma(m+1) - std::lgamma(cn_r-m+1)
                  + m*(std::log(1.0-dropout+1e-12))
                  + (cn_r-m)*(std::log(dropout+1e-12));
        scales[m] = (m == 0) ? background_rate : (double)m;
    }

    std::vector<double> log_binC(M);
    for (int j = 0; j < M; ++j)
        log_binC[j] = std::lgamma(N_j[j]+1) - std::lgamma(n_jk[j]+1)
                      - std::lgamma(N_j[j]-n_jk[j]+1);

    std::vector<double> log_mix(cn_r+1), resp(cn_r+1), alpha_r_vec(cn_r+1), alpha_o_vec(cn_r+1);

    for (int iter = 0; iter < max_iter; ++iter) {
        double d = std::exp(x);
        double alpha_total = d * c_total;
        double dig_at  = digamma(alpha_total);
        double trig_at = trigamma(alpha_total);

        double score = 0.0, hess = 0.0;

        for (int j = 0; j < M; ++j) {
            double nj = n_jk[j], Nj = N_j[j];

            // E-step: mixture over m = copies surviving dropout at this region
            double dig_Nat  = digamma(Nj + alpha_total);
            double trig_Nat = trigamma(Nj + alpha_total);
            double lgamma_at  = std::lgamma(alpha_total);
            double lgamma_Nat = std::lgamma(Nj + alpha_total);

            double lmax = -std::numeric_limits<double>::infinity();
            for (int m = 0; m <= cn_r; ++m) {
                double alpha_r = region_probability * d * scales[m];
                double alpha_o = alpha_total - alpha_r;
                alpha_r_vec[m] = alpha_r;
                alpha_o_vec[m] = alpha_o;
                log_mix[m] = log_binC[j]
                             + std::lgamma(nj      + alpha_r)
                             + std::lgamma(Nj - nj + alpha_o)
                             - lgamma_Nat
                             + lgamma_at
                             - std::lgamma(alpha_r)
                             - std::lgamma(alpha_o)
                             + logw[m];
                lmax = std::max(lmax, log_mix[m]);
            }
            double lse = 0.0;
            for (int m = 0; m <= cn_r; ++m) { resp[m] = std::exp(log_mix[m] - lmax); lse += resp[m]; }
            for (int m = 0; m <= cn_r; ++m) resp[m] /= lse;

            for (int m = 0; m <= cn_r; ++m) {
                double alpha_r = alpha_r_vec[m], alpha_o = alpha_o_vec[m];
                double dig_ar   = digamma(alpha_r),        trig_ar   = trigamma(alpha_r);
                double dig_nar  = digamma(nj + alpha_r),   trig_nar  = trigamma(nj + alpha_r);
                double dig_ao   = digamma(alpha_o),        trig_ao   = trigamma(alpha_o);
                double dig_Nnao = digamma(Nj - nj + alpha_o), trig_Nnao = trigamma(Nj - nj + alpha_o);

                // d(alpha_r)/dx = alpha_r, d(alpha_o)/dx = alpha_o, d(alpha_total)/dx = alpha_total
                // (everything is linear in d = exp(x))
                double g_m = alpha_r   * (dig_nar  - dig_ar)
                           + alpha_o   * (dig_Nnao - dig_ao)
                           - alpha_total * (dig_Nat - dig_at);
                double h_m = alpha_r   * (dig_nar  - dig_ar)  + alpha_r*alpha_r   * (trig_nar  - trig_ar)
                           + alpha_o   * (dig_Nnao - dig_ao)  + alpha_o*alpha_o   * (trig_Nnao - trig_ao)
                           - alpha_total * (dig_Nat - dig_at) - alpha_total*alpha_total * (trig_Nat - trig_at);

                double wj_resp = w_j[j] * resp[m];
                score += wj_resp * g_m;
                hess  += wj_resp * h_m;
            }
        }

        double step = (hess < 0.0 && std::isfinite(hess))
                      ? std::max(-2.0, std::min(2.0, -score / hess))
                      : -0.01 * ((score > 0.0) ? 1.0 : -1.0);
        double new_x = x + step;
        bool converged = std::abs(new_x - x) < tol;
        x = new_x;
        if (converged) break;
    }

    return std::exp(x);
}


// Marginal log-likelihood of one cell's region-k read count, mixing over
// m = copies of region k surviving dropout (0..cn_r).
static double region_cell_mixture_loglik(
        double n_jk, double N_j, int cn_r,
        double alpha_per_copy, double alpha_total,
        double dropout, double background_rate) {
    double eps = 1e-12;
    double lgamma_at  = std::lgamma(alpha_total);
    double lgamma_Nat = std::lgamma(N_j + alpha_total);
    double log_binC   = std::lgamma(N_j + 1) - std::lgamma(n_jk + 1) - std::lgamma(N_j - n_jk + 1);

    std::vector<double> log_mix(cn_r + 1);
    double lmax = -std::numeric_limits<double>::infinity();
    for (int m = 0; m <= cn_r; ++m) {
        double log_p_m = std::lgamma(cn_r+1) - std::lgamma(m+1) - std::lgamma(cn_r-m+1)
                         + m*std::log(1.0-dropout+eps) + (cn_r-m)*std::log(dropout+eps);
        double scale = (m == 0) ? background_rate : (double)m;
        double alpha_r = alpha_per_copy * scale;
        double alpha_o = alpha_total - alpha_r;
        log_mix[m] = log_binC + std::lgamma(n_jk + alpha_r) + std::lgamma(N_j - n_jk + alpha_o)
                     - lgamma_Nat + lgamma_at - std::lgamma(alpha_r) - std::lgamma(alpha_o) + log_p_m;
        lmax = std::max(lmax, log_mix[m]);
    }
    double s = 0.0;
    for (int m = 0; m <= cn_r; ++m) s += std::exp(log_mix[m] - lmax);
    return lmax + std::log(s);
}

// Total log-likelihood across all regions and healthy cells, as a function
// of dropout alone, given fixed region_dispersion/region_probabilities.
static double total_loglik_given_dropout(
        double dropout,
        const std::vector<std::vector<double>>& n_jk_by_region,
        const std::vector<double>& N_j, const std::vector<double>& w_j,
        const std::vector<int>& cn_r_by_region,
        const std::vector<double>& region_probabilities,
        const std::vector<double>& region_dispersion,
        double background_rate) {

    double total = 0.0;
    int n_regions_local = (int)region_dispersion.size();

    for (int k = 0; k < n_regions_local; ++k) {
        int cn_r = cn_r_by_region[k];
        if (cn_r <= 0) continue;
        double alpha_per_copy = region_probabilities[k] * region_dispersion[k];
        double alpha_total    = region_dispersion[k] * cn_r * (1.0 - dropout);
        const std::vector<double>& n_jk = n_jk_by_region[k];

        for (int j = 0; j < (int)n_jk.size(); ++j) {
            double ll = region_cell_mixture_loglik(
                    n_jk[j], N_j[j], cn_r, alpha_per_copy, alpha_total, dropout, background_rate);
            total += w_j[j] * ll;
        }
    }
    return total;
}

static double fit_dropout_golden_section(
        const std::vector<std::vector<double>>& n_jk_by_region,
        const std::vector<double>& N_j, const std::vector<double>& w_j,
        const std::vector<int>& cn_r_by_region,
        const std::vector<double>& region_probabilities,
        const std::vector<double>& region_dispersion,
        double background_rate, double prior_mean, double prior_omega,
        double lo, double hi, double tol) {

    // Beta prior on dropout
    auto f = [&](double rho) {
        return total_loglik_given_dropout(rho, n_jk_by_region, N_j, w_j, cn_r_by_region,
                                          region_probabilities, region_dispersion, background_rate)
               + dropout_log_prior(rho, prior_mean, prior_omega);
    };
    return golden_section_max(f, lo, hi, tol);
}
