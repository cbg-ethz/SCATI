#include <random>
#include <cfloat>
#include <cmath>
#include <cassert>
#include <iostream>
#include <string>
#include <chrono>

#include <fstream>
#include <iomanip>
#include <filesystem>

#include "Inference.h"
#include "Structures.h"

#include <utility>

extern int n_loci;
extern int n_regions;


std::vector<double> Inference::default_move_weights() const {

    std::vector<double> weights = {
            1.0,   // 0 Prune and reattach
            1.0,   // 1 Swap node labels
            1.0,   // 2 Add/remove SNV
            1.0,   // 3 Move SNV
            1.0,   // 4 Split/merge node
            1.0,   // 5 Change root genotype
            1.0,   // 6 Change strand affected by a CNA/SNV
            1.0,   // 7 Change event order within a node
            1.0,   // 8 Add/remove CNA
            1.0,   // 9 Move CNA
            1.0,   // 10 Exchange CNA types
            1.0,   // 11 Change number of regions affected by CNA
            1.0,   // 12 Split/Merge CNAs within a node
            1.0,   // 13 Swap CNA (remove one, propose a new one on the same node)
    };

    if (local_params->use_haplotype_annotation) weights[5] = 0.0; // the annotated root genotype is kept fixed
    return weights;
}

// one generator per (seed, chain) pair, so different seeds share no chains
static std::mt19937 chain_seed(int seed, int chain) {
    std::seed_seq seq{seed, chain};
    return std::mt19937(seq);
}

Inference::Inference(int idx, Params params)
        : local_params(std::make_shared<Params>(std::move(params))),
          cache_scores(std::make_shared<Scores>(*local_params)),
          rng_{chain_seed(local_params->random_seed, idx)},
          t(cache_scores, rng_, local_params),
          t_prime(cache_scores, rng_, local_params),
          best_tree(cache_scores, rng_, local_params),
          max_temperature(local_params->temperature),
          move_weights(default_move_weights()),
          index(idx)
{
    precompute_move_weights();
}

Inference::~Inference() = default;

void Inference::precompute_move_weights() {
    cumulative_weights.resize(move_weights.size());
    std::partial_sum(move_weights.begin(), move_weights.end(), cumulative_weights.begin());
    total_move_weight = cumulative_weights.empty() ? 0.0 : cumulative_weights.back();
}

// Draws a move from [min_move_index, max_move_index), proportionally to those moves' weights
int Inference::select_move(int min_move_index, int max_move_index) {
    double lo = (min_move_index > 0) ? cumulative_weights[min_move_index - 1] : 0.0;
    double hi = cumulative_weights[max_move_index - 1];
    std::uniform_real_distribution<double> uniform(lo, hi);
    double r = uniform(rng_);
    for (int k = min_move_index; k < max_move_index - 1; ++k)
        if (r <= cumulative_weights[k]) return k;
    return max_move_index - 1;
}

Tree Inference::find_best_tree(int n_steps, int burn_in_steps, const std::string& output_path){
    std::cout<<"Chain "<<std::to_string(index)<< ": Start tree inference."<<std::endl;
    t = best_tree;
    t_prime = t;
    mcmc(n_steps, burn_in_steps, output_path);

    return best_tree;
}


void Inference::mcmc(int n_steps, int n_burnin_steps, const std::string& output_path){
    best_tree = t;
    double best_log_score = -DBL_MAX;
    int move_id;
    precompute_move_weights();

    // Fit region weights once, before any tree move is proposed.
    t.update_region_weights();
    // With a region reference (so without the healthy annotation) the root SNV genotype would start hom-ref;
    // set it from the cells attached to the root now (else germline SNPs pull the healthy cells into a
    // CNA-free child of the root).
    if (data.has_region_reference && !local_params->use_healthy_annotation && !local_params->use_haplotype_annotation) {
        t.initialize_root_genotype_from_root_cells();
        if (!t.update_genotypes())
            std::cerr << "Warning: the initial tree is invalid under the root SNV genotype from the root cells." << std::endl;
        t.compute_likelihood();
    }
    t.compute_prior_score();
    t.update_full_score();
    t_prime.reset_to(t);
    std::ofstream trace_out(output_path + "mcmc_trace.txt", std::ios::out);
    trace_out << std::setprecision(10);
    if (trace_out.is_open())
        trace_out << "step\tlog_likelihood\tlog_prior\tlog_score\n";

    for (int step=0;step<n_steps;step++){
        if (local_params->verbose) std::cout<<"MCMC step " <<step<<"  ----------------------------------------"<<std::endl;
        bool is_warmup = (step < n_burnin_steps);

        // --region_reference: the reference parameters are used for the whole burn-in; once it ends,
        // refit them a single time from the cells' soft root attachment (update_region_weights keeps
        // the reference values if fewer than min_cells_for_parameter_refit cells attach to the root).
        // --refit_after_burnin: same single refit, but from the labelled healthy cells to the root cells (node 0); the
        // labels (fit and prior) are not used after it. No further refits follow, so the root cannot drift.
        if (step == n_burnin_steps && local_params->use_healthy_annotation && local_params->refit_after_burnin
            && !local_params->annotation_released) {
            t.report_labels_vs_root();
            local_params->annotation_released = true;
            std::cout << "Step " << step << ": end of burn-in, refitting the region parameters from the "
                         "root cells instead of the healthy-cell labels." << std::endl;
            t.update_region_weights();
            t.compute_prior_score();
            t.update_full_score();
            t_prime.reset_to(t);
        }

        if (step == n_burnin_steps && local_params->region_reference_active) {
            local_params->region_reference_active = false;
            std::cout << "Step " << step << ": end of burn-in, refitting the region reference parameters "
                         "from the root-attached cells." << std::endl;
            t.update_region_weights();
            t.compute_prior_score();
            t.update_full_score();
            t_prime.reset_to(t);
        }

        move_id = select_move(0, static_cast<int>(move_weights.size()));

        switch (move_id) {
                // -------- SNV moves --------

            case 0:
                if (local_params->verbose) std::cout << "Selected prune and reattach" << std::endl;
                t_prime.prune_reattach();
                break;

            case 1:
                if (local_params->verbose) std::cout << "Selected swap node labels" << std::endl;
                t_prime.swap_node_labels();
                break;

            case 2:
                if (local_params->verbose) std::cout << "Selected add/remove SNV" << std::endl;
                t_prime.add_remove_SNV();
                break;

            case 3:
                if (local_params->verbose) std::cout << "Selected move SNV" << std::endl;
                t_prime.move_SNV();
                break;

            case 4:
                if (local_params->verbose) std::cout << "Selected split/merge node" << std::endl;
                t_prime.split_merge_node();
                break;

            case 5:
                if (local_params->verbose) std::cout << "Selected change root genotype" << std::endl;
                t_prime.change_root_genotype(is_warmup);
                break;

                // ------- SNV/CNA moves --------

            case 6:
                if (local_params->verbose) std::cout << "Selected change strand affected by a CNA/SNV" << std::endl;
                t_prime.change_strand();
                break;

            case 7:
                if (local_params->verbose) std::cout << "Selected change event order within a node" << std::endl;
                t_prime.change_event_order_within_node();
                break;


                // -------- CNA moves --------

            case 8:
                if (local_params->verbose) std::cout << "Selected add/remove CNA" << std::endl;
                t_prime.add_remove_CNA();
                break;

            case 9:
                if (local_params->verbose) std::cout << "Selected move CNA" << std::endl;
                t_prime.move_CNA();
                break;

            case 10:
                if (local_params->verbose) std::cout << "Selected exchange CNA types" << std::endl;
                t_prime.change_CNA_types();
                break;

            case 11:
                if (local_params->verbose) std::cout << "Selected change number of regions affected by CNA" << std::endl;
                t_prime.change_CNA_regions();
                break;

            case 12:
                if (local_params->verbose) std::cout << "Selected Split/Merge CNAs within a node" << std::endl;
                t_prime.merge_split_within_node();
                break;

            case 13:
                if (local_params->verbose) std::cout << "Selected swap CNA" << std::endl;
                t_prime.swap_CNA();
                break;

            default:
                if (local_params->verbose) std::cout << "Unknown move_id: " << move_id << std::endl;
                break;
        }

        double acceptance_ratio = 0.0;

        // Only compute acceptance ratio if we have a valid proposal.
        if (t_prime.hastings_ratio > 0.0 && std::isfinite(t_prime.hastings_ratio)) {


            if (t_prime.hastings_ratio > 10000000) {
                std::cout << "Potentially invalid Hastings ratio: " << t_prime.hastings_ratio << "\n";
            }

            t_prime.compute_prior_score();
            t_prime.update_full_score();

            // Without a healthy-cell annotation, nothing else anchors the root
            // to a realistic (near-diploid) genotype -- reject
            // outright (acceptance_ratio stays 0.0) any proposal that would push
            // the pooled soft root-attachment fraction below the configured
            // floor. Applied for the whole run, not only burn-in: once the parameters are
            // frozen the likelihood can prefer a tree that moves the healthy cells into a
            // child node and empties the root (an accepted add/remove-SNV move right after
            // burn-in, +245 nats on a toy dataset). After burn-in the rule is a hard
            // constraint (the posterior restricted to trees at or above the floor), so it
            // keeps the sampler valid as long as burn-in ended above the floor.
            //
            // The floor only ever blocks a move that pushes the fraction below the floor AND
            // below the current tree's own fraction. If the current tree already violates the
            // floor (e.g. the initial root-only tree, where every cell can be explained as a
            // root+root doublet), an absolute test would reject every proposal for the entire
            // burn-in -- the chain could never move. Moves that keep or raise the fraction
            // must stay possible so the chain can climb out of the violating state.
            bool check_root_floor = !local_params->use_healthy_annotation
                && local_params->min_root_attachment_fraction > 0.0;
            double root_attachment_fraction = check_root_floor ? t_prime.get_root_attachment_fraction() : 1.0;
            bool violates_root_floor = check_root_floor
                && root_attachment_fraction < local_params->min_root_attachment_fraction
                && root_attachment_fraction < t.get_root_attachment_fraction();

            if (violates_root_floor) {
                if (local_params->verbose)
                    std::cout << "Proposal rejected: root attachment fraction "
                              << root_attachment_fraction << " below floor "
                              << local_params->min_root_attachment_fraction << std::endl;
            } else if (local_params->optimizer == "MCMC") {
                acceptance_ratio = std::exp((t_prime.log_score - t.log_score) + std::log(t_prime.hastings_ratio));
            } else if (local_params->optimizer == "simulated_annealing") {

                double temperature = max_temperature - (max_temperature - 1.0) * step / n_steps;
                if (t_prime.log_score >= t.log_score) {
                    acceptance_ratio = 1.0; // always accept improvement
                } else {
                    acceptance_ratio = std::exp((t_prime.log_score - t.log_score) / temperature
                                                + std::log(t_prime.hastings_ratio));
                }
            } else {
                throw std::runtime_error("Unknown optimizer type: " + local_params->optimizer);
            }
        }

        if (local_params->verbose) {
            std::cout << "t_prime score: " << t_prime.log_score
                      << ", t score: " << t.log_score
                      << ", hastings ratio: " << t_prime.hastings_ratio
                      << ", acceptance ratio: " << acceptance_ratio
                      << std::endl;
        }

        static thread_local std::uniform_real_distribution<double> uni01(0.0, 1.0);
        double rd = uni01(rng_);

        if (rd <= acceptance_ratio) {
            // accept t_prime
            if (local_params->verbose) std::cout << "Accepted move" << std::endl;
            t = t_prime;
            std::string violation = t.check_region_locus_consistency();
            if (!violation.empty()) {
                std::cerr << "WARNING: CONSISTENCY VIOLATION at step " << step << " after move_id=" << move_id
                          << " -- " << violation << std::endl;
            }
        } else {
            // reject t_prime
            if (local_params->verbose) std::cout << "Rejected move" << std::endl;
            t_prime.reset_to(t); // keep t_prime's own rng advancing so the next proposal (even the same move type) draws fresh randomness
        }

        // state after this step
        if (trace_out.is_open()) {
            trace_out << step << "\t" << t.log_likelihood << "\t" << t.log_prior_score << "\t" << t.log_score << "\n";
            if (step % 100 == 0) trace_out.flush();
        }

        if (step >= n_burnin_steps) { // sample best trees only after region weights and total count mean have been optimized during burn-in so that we can properly sample from the posterior distribution
            // compare score with best score
            if (t.log_score > best_log_score) {
                best_log_score = t.log_score;
                best_tree = t;
            }
            if (local_params->verbose) std::cout << "Current best score " << best_log_score << std::endl;
        }

        // When there's no healthy-cell annotation, region weights/dispersion/dropout
        // were only seeded once at the start of the chain. During burn-in, periodically refit them from the cells' soft root attachment.
        if (!local_params->use_healthy_annotation && is_warmup
            && local_params->update_interval_region_weights > 0
            && step > 0 && step % local_params->update_interval_region_weights == 0) {
            t.update_region_weights();
            t.compute_prior_score();
            t.update_full_score();
            t_prime.reset_to(t);
            if (local_params->region_reference_active)
                std::cout << "Step " << step << ": region probabilities updated from the reference for the current root." << std::endl;
            else
                std::cout << "Step " << step << ": periodic soft-root-attachment parameter refit." << std::endl;
        }

        if (local_params->sampling && (step % local_params->sample_interval_trees_parameters == 0)) {
            t.sample_parameters(step, output_path); // save current tree and current parameter samples
        }

        if (!(t.log_score == t_prime.log_score && t.log_likelihood == t_prime.log_likelihood)) {
            std::cerr << "Assertion failed in `Inference.cpp`: t and t_prime scores/likelihoods mismatch. "
                      << "t.log_score=" << t.log_score << ", t_prime.log_score=" << t_prime.log_score
                      << ", t.log_likelihood=" << t.log_likelihood << ", t_prime.log_likelihood=" << t_prime.log_likelihood
                      << std::endl;
            assert(t.log_score == t_prime.log_score && t.log_likelihood == t_prime.log_likelihood);
        }
    }
}
