#include <string>
#include <utility>
#include <vector>
#include <set>
#include <random>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>

#include "Node.h"
#include "Structures.h"
#include "Scores.h"

extern int n_cells;
extern int n_loci;
extern int n_regions;
extern std::vector<Cell> cells;

Node::Node(std::shared_ptr<Scores> cache, std::mt19937& rng_, std::shared_ptr<Params> params)
        : cache_scores(std::move(cache)),
          local_params(std::move(params)),
          rng(rng_) {
    init_structures();
}

// Doublet
Node::Node(const Node& source1, const Node& source2, std::mt19937& rng_)
        : cache_scores(source1.cache_scores),   // shared_ptr copy
          local_params(source1.local_params),
          is_doublet(true),
          rng(rng_) {
    init_structures();

    for (int i = 0; i < n_loci; ++i) {
        n_ref_allele[i] = source1.n_ref_allele[i] + source2.n_ref_allele[i];
        n_alt_allele[i] = source1.n_alt_allele[i] + source2.n_alt_allele[i];
    }

    // During doublet tree traversal, node1 is fixed and node2 iterates
    // only node2's events are added, so affected loci/regions come from node2
    affected_loci    = source2.affected_loci;
    affected_regions = source2.affected_regions;

    for (int i = 0; i < n_regions; ++i)
        cn_regions[i] = source1.cn_regions[i] + source2.cn_regions[i];
}


void Node::init_structures() {
    n_ref_allele.resize(n_loci);
    n_alt_allele.resize(n_loci);
    cn_regions.resize(n_regions);
    attachment_scores.resize(n_cells);
    attachment_scores_SNV.resize(n_cells);
    attachment_scores_CNA.resize(n_cells);
}

void Node::recompute_derived_from_strands() {
    affected_loci.clear();
    affected_regions.clear();
    for (int i = 0; i < n_loci; ++i) affected_loci.insert(i);
    for (int k = 0; k < n_regions; ++k) affected_regions.insert(k);

    count_copies_from_strands();   // loaded strands are kept even above max_copy_number
}


bool Node::update_genotype(Node* parent,
                           const std::vector<std::vector<int>>& root_strand_loci_ref,
                           const std::vector<std::vector<int>>& root_strand_loci_alt,
                           const std::vector<std::vector<int>>& root_strand_regions) {

    // Validate: no region should be both gained and lost within one node
    std::set<int> gained_regions, lost_regions;

    for (const auto& event : Events) {
        if (!std::holds_alternative<CNA>(event)) continue;
        const CNA& cna = std::get<CNA>(event);
        if (cna.fgd || cna.CNA_change == 0) continue;

        for (int region : cna.region_indices) {
            if (cna.CNA_change == 1)  gained_regions.insert(region);
            if (cna.CNA_change == -1) lost_regions.insert(region);
        }
    }

    // If any region appears in both gained and lost sets, reject regardless of strand —> net effect is unidentifiable or should be modeled as CNLOH
    for (int r : gained_regions) {
        if (lost_regions.count(r)) return false;
    }

    // 1. Initialize from parent
    if (parent == nullptr) { // root node
        strands_loci_ref = root_strand_loci_ref;
        strands_loci_alt = root_strand_loci_alt;
        strands_regions = root_strand_regions;
    } else {
        strands_loci_ref = parent->strands_loci_ref;
        strands_loci_alt = parent->strands_loci_alt;
        strands_regions = parent->strands_regions;
    }

    affected_loci.clear();
    affected_regions.clear();

    // 2. Apply events in order
    for (const auto& event : Events) {

        // ---------------- SNV ----------------
        if (std::holds_alternative<SNV>(event)) {
            const SNV& snv = std::get<SNV>(event);
            int locus  = snv.locus;
            int strand = snv.affected_strand;

            if (strand >= (int)strands_loci_ref.size()) return false; // can happen if an event is moved to a new node for example

            if (snv.affected_allele == 0) { // ref -> alt
                if (strands_loci_ref[strand][locus] < 1) return false;
                strands_loci_ref[strand][locus] -= 1;
                strands_loci_alt[strand][locus] += 1;
            } else { // alt -> ref
                if (strands_loci_alt[strand][locus] < 1) return false;
                strands_loci_alt[strand][locus] -= 1;
                strands_loci_ref[strand][locus] += 1;
            }
            affected_loci.insert(locus);
        }

        // ---------------- CNA ----------------
        else if (std::holds_alternative<CNA>(event)) {
            const CNA& cna = std::get<CNA>(event);

            // ---------------- FGD ----------------
            if (cna.fgd) {
                int n_strands_current = (int)strands_regions.size();
                for (int s = 0; s < n_strands_current; ++s) {
                    strands_regions.push_back(strands_regions[s]);
                    strands_loci_ref.push_back(strands_loci_ref[s]);
                    strands_loci_alt.push_back(strands_loci_alt[s]);
                }
                for (int k = 0; k < n_regions; ++k) affected_regions.insert(k); // all regions affected
            for (int i = 0; i < n_loci; ++i) affected_loci.insert(i); // all loci affected
            }

            else {
                int strand = cna.affected_strand;
                if (strand >= (int)strands_regions.size()) return false;
                int delta  = cna.CNA_change;

                if (delta == -1) { // LOSS
                    for (int region : cna.region_indices) {
                        if (strands_regions[strand][region] < 1) return false;
                        strands_regions[strand][region] = 0;
                        affected_regions.insert(region);
                        for (int locus : data.region_to_loci[region]) {
                            strands_loci_ref[strand][locus] = 0;
                            strands_loci_alt[strand][locus] = 0;
                            affected_loci.insert(locus);
                        }
                    }
                    // Remove strand if fully deleted
                    bool all_zero = true;
                    for (int k = 0; k < n_regions && all_zero; ++k)
                        if (strands_regions[strand][k] != 0) all_zero = false;
                    if (all_zero) {
                        strands_regions.erase(strands_regions.begin() + strand);
                        strands_loci_ref.erase(strands_loci_ref.begin() + strand);
                        strands_loci_alt.erase(strands_loci_alt.begin() + strand);
                    }

                } else if (delta == 1) { // GAIN
                    if (cna.aneuploidy) { // create new strand as copy
                        std::vector<int> new_regions(n_regions, 0);
                        std::vector<int> new_loci_ref(n_loci, -1);
                        std::vector<int> new_loci_alt(n_loci, -1);
                        for (int region : cna.region_indices) {
                            if (strands_regions[strand][region] < 1) return false;
                            new_regions[region] = strands_regions[strand][region];
                            affected_regions.insert(region);
                        }
                        for (int i = 0; i < n_loci; ++i) {
                            // >= 1, not == 1: a region already doubled by a local gain has CN 2 on
                            // this strand, and its alleles must be copied along with it
                            if (new_regions[data.locus_to_region[i]] >= 1) {
                                new_loci_ref[i] = strands_loci_ref[strand][i];
                                new_loci_alt[i] = strands_loci_alt[strand][i];
                                affected_loci.insert(i);
                            }
                        }
                        strands_regions.push_back(std::move(new_regions));
                        strands_loci_ref.push_back(std::move(new_loci_ref));
                        strands_loci_alt.push_back(std::move(new_loci_alt));

                    } else { // local gain: duplicate all existing copies on this strand within the gained regions
                        for (int region : cna.region_indices) {
                            if (strands_regions[strand][region] < 1) return false;
                            strands_regions[strand][region] *= 2;
                            affected_regions.insert(region);
                            for (int locus : data.region_to_loci[region]) {
                                strands_loci_ref[strand][locus] *= 2;
                                strands_loci_alt[strand][locus] *= 2;
                                affected_loci.insert(locus);
                            }
                        }
                    }

                } else if (delta == 0) { // CNLOH: strand2 is donor, strand receives copy of strand2
                    int strand2 = cna.affected_strand2;
                    if (strand2 < 0 || strand2 >= (int)strands_regions.size()) return false;

                    for (int region : cna.region_indices) {
                        // A CNLOH must be copy-neutral: the recipient takes over the donor's copies and alleles, so
                        // both strands need the same copy number >= 1 (otherwise it is a gain or loss priced as CNLOH).
                        if (strands_regions[strand][region] != strands_regions[strand2][region]) return false;
                        if (strands_regions[strand][region] < 1) return false;

                        // Every affected region must be identifiable: at least one locus
                        // in it where the two strands carry different alleles. A region
                        // without such a locus leaves the event unobservable there.
                        bool allele_difference = false;
                        for (int locus : data.region_to_loci[region]) {
                            if (strands_loci_ref[strand][locus] != strands_loci_ref[strand2][locus] ||
                                strands_loci_alt[strand][locus] != strands_loci_alt[strand2][locus]) {
                                allele_difference = true;
                                break;
                            }
                        }
                        if (!allele_difference) return false;

                        affected_regions.insert(region);
                        for (int locus : data.region_to_loci[region]) {
                            strands_loci_ref[strand][locus] = strands_loci_ref[strand2][locus];
                            strands_loci_alt[strand][locus] = strands_loci_alt[strand2][locus];
                            affected_loci.insert(locus);
                        }
                    }
                } else {
                    return false; // invalid delta
                }
            }
        }
    }

    // 3. and 4. Recompute allele counts and region copy numbers
    return count_copies_from_strands();
}

// Sums the allele and region copies over the strands. False if a copy number exceeds max_copy_number.
bool Node::count_copies_from_strands() {
    n_ref_allele.assign(n_loci, 0);
    n_alt_allele.assign(n_loci, 0);
    for (const auto& strand : strands_loci_ref)
        for (int i = 0; i < n_loci; ++i)
            if (strand[i] >= 0) n_ref_allele[i] += strand[i];
    for (const auto& strand : strands_loci_alt)
        for (int i = 0; i < n_loci; ++i)
            if (strand[i] >= 0) n_alt_allele[i] += strand[i];

    bool valid = true;
    cn_regions.assign(n_regions, 0);
    for (const auto& strand : strands_regions)
        for (int k = 0; k < n_regions; ++k) {
            cn_regions[k] += strand[k];
            if (cn_regions[k] > local_params->max_copy_number) valid = false;
        }
    return valid;
}

// Adds sign times the allelic log-likelihood of genotype (c_ref, c_alt) at locus i to out, for the cells of each sample.
void Node::add_SNV_scores(int i, int c_ref, int c_alt, double sign,
                          const std::vector<std::vector<double>>& dropout_rates_ref,
                          const std::vector<std::vector<double>>& dropout_rates_alt,
                          const std::vector<std::vector<double>>& region_probabilities,
                          const std::vector<std::vector<double>>& region_dispersion,
                          const std::vector<SampleFit>& sample_fits, std::vector<double>& out) const {
    const int region_i = data.locus_to_region[i];
    for (int s = 0; s < data.n_samples; ++s) {
        if (local_params->use_joint_likelihood_model) {
            // the allelic concentration is the per-copy concentration of the locus's region
            const double omega = region_dispersion[s][region_i] * region_probabilities[s][region_i];
            const std::vector<double>& scores = cache_scores->compute_SNV_loglikelihoods_joint(
                    c_ref, c_alt, i, omega, sample_fits[s].dropoutrate);
            for (int j : data.cells_by_sample[s]) out[j] += sign * scores[j];
        } else {
            std::vector<double> scores = cache_scores->compute_SNV_loglikelihoods(
                    c_ref, c_alt, i, dropout_rates_ref[s][i], dropout_rates_alt[s][i]);
            for (int j : data.cells_by_sample[s]) out[j] += sign * scores[j];
        }
    }
}

// Region-count log-likelihood of every cell at this node's copy numbers.
void Node::compute_CNA_scores(const std::vector<std::vector<double>>& region_probabilities,
                              const std::vector<std::vector<double>>& region_dispersion,
                              const std::vector<SampleFit>& sample_fits,
                              const std::vector<int>& root_cn_regions, std::vector<double>& out) const {
    out.assign(n_cells, 0.0);
    // Lambda is rounded so that nodes of similar ploidy share cache entries; doublets, whose Lambda is about twice as
    // large, use twice the step. A finer step is more exact but gets fewer cache hits.
    const double rounding_precision = local_params->lambda_rounding_precision * (is_doublet ? 2.0 : 1.0);
    for (int s = 0; s < data.n_samples; ++s) {
        const std::vector<int>& cells_s = data.cells_by_sample[s];
        if (!local_params->use_joint_likelihood_model) {
            // fixed-dispersion negative binomial around the expected read fraction of region k
            double normalization_factor = 0.0;
            for (int k = 0; k < n_regions; k++)
                normalization_factor += (region_probabilities[s][k] / std::max(1, root_cn_regions[k])) * cn_regions[k];
            for (int k = 0; k < n_regions; k++) {
                std::vector<double> scores = cache_scores->compute_CNA_loglikelihoods_negbinom(
                        k, (region_probabilities[s][k] / std::max(1, root_cn_regions[k])) * cn_regions[k] / normalization_factor);
                for (int j : cells_s) out[j] += scores[j];
            }
            continue;
        }
        const double dropout = sample_fits[s].dropoutrate;
        // Lambda with per-copy weights: region_probabilities is the read fraction at the root's copy number (1 on male
        // chrX/chrY), so a copy change counts the same on haploid and diploid regions. At the root Lambda = 1 - dropout.
        double sum_w_cn = 0;
        for (int k = 0; k < n_regions; ++k)
            sum_w_cn += region_probabilities[s][k] / std::max(1, root_cn_regions[k]) * cn_regions[k] * (1.0 - dropout);
        sum_w_cn = std::round(sum_w_cn / rounding_precision) * rounding_precision;
        if (sum_w_cn <= 0.0) sum_w_cn = rounding_precision;
        for (int k = 0; k < n_regions; ++k) {
            const double alpha_region = region_probabilities[s][k] * region_dispersion[s][k];
            // region_dispersion is fitted per root copy, so at the root alpha_total = region_dispersion * root_cn *
            // (1 - dropout), which makes the expected read fraction region_probabilities[k]. Other nodes scale it with
            // their own Lambda, unless use_constant_dispersion keeps it at the root value.
            const double alpha_total = region_dispersion[s][k] * std::max(1, root_cn_regions[k])
                                       * (local_params->use_constant_dispersion ? (1.0 - dropout) : sum_w_cn);
            const std::vector<double>& scores = cache_scores->compute_CNA_loglikelihoods_joint(
                    k, cn_regions[k], alpha_region, alpha_total, dropout, s);
            for (size_t t = 0; t < cells_s.size(); ++t)   // scores are compact: element t is cell cells_s[t]
                out[cells_s[t]] += scores[t];
        }
    }
}

void Node::combine_attachment_scores() {
    for (int j = 0; j < n_cells; ++j)
        attachment_scores[j] = attachment_scores_SNV[j] + attachment_scores_CNA[j];
}

void Node::compute_attachment_scores_root(const std::vector<std::vector<double>>& dropout_rates_ref,
                                          const std::vector<std::vector<double>>& dropout_rates_alt,
                                          const std::vector<std::vector<double>>& region_probabilities,
                                          const std::vector<std::vector<double>>& region_dispersion,
                                          const std::vector<SampleFit>& sample_fits,
                                          const std::vector<int>& root_cn_regions) {
    attachment_scores_SNV.assign(n_cells, 0.0);
    for (int i = 0; i < n_loci; ++i)
        add_SNV_scores(i, n_ref_allele[i], n_alt_allele[i], 1.0, dropout_rates_ref, dropout_rates_alt,
                       region_probabilities, region_dispersion, sample_fits, attachment_scores_SNV);
    compute_CNA_scores(region_probabilities, region_dispersion, sample_fits, root_cn_regions, attachment_scores_CNA);
    combine_attachment_scores();
}

// Starts from the parent's scores and recomputes only what this node's events change: the loci whose allele counts
// differ, and every region if any region's copy number differs (Lambda depends on all regions).
void Node::compute_attachment_scores_parent(Node* parent,
                                            const std::vector<std::vector<double>>& dropout_rates_ref,
                                            const std::vector<std::vector<double>>& dropout_rates_alt,
                                            const std::vector<std::vector<double>>& region_probabilities,
                                            const std::vector<std::vector<double>>& region_dispersion,
                                            const std::vector<SampleFit>& sample_fits,
                                              const std::vector<int>& root_cn_regions) {
    attachment_scores_SNV = parent->attachment_scores_SNV;
    for (int i : affected_loci) {
        add_SNV_scores(i, parent->n_ref_allele[i], parent->n_alt_allele[i], -1.0, dropout_rates_ref, dropout_rates_alt,
                       region_probabilities, region_dispersion, sample_fits, attachment_scores_SNV);
        add_SNV_scores(i, n_ref_allele[i], n_alt_allele[i], 1.0, dropout_rates_ref, dropout_rates_alt,
                       region_probabilities, region_dispersion, sample_fits, attachment_scores_SNV);
    }
    if (affected_regions.empty()) attachment_scores_CNA = parent->attachment_scores_CNA;
    else compute_CNA_scores(region_probabilities, region_dispersion, sample_fits, root_cn_regions, attachment_scores_CNA);
    combine_attachment_scores();
}


Node::SNV Node::remove_random_mutation(){
    // Randomly remove one of the mutations and return it
    // This method should only be called if the node has at least one somatic mutation.
    std::vector<int> snv_indices;

    for (int i = 0; i < Events.size(); ++i) {
        if (std::holds_alternative<SNV>(Events[i])) {
            snv_indices.push_back(i);
        }
    }

    if (snv_indices.empty())
        throw std::runtime_error("No SNV to remove");

    std::uniform_int_distribution<int> dist(0, (int)snv_indices.size() - 1);
    int selected_position = dist(rng.get());
    int idx = snv_indices[selected_position];

    SNV removed_event = std::get<SNV>(Events[idx]);
    Events.erase(Events.begin() + idx);
    return removed_event;
}

void Node::add_mutation(SNV new_mutation){
    // Add a mutation to the node at a random position in the event list
    std::uniform_int_distribution<int> dist(0, (int)Events.size());
    int selected_position = dist(rng.get());
    Events.insert(Events.begin() + selected_position, new_mutation);
}

void Node::add_event(const std::variant<SNV, CNA>& new_event){
    Events.push_back(new_event);
}

void Node::add_CNA(CNA new_CNA){
    // Add a CNA event at a random position in the event list
    std::uniform_int_distribution<int> dist(0, (int)Events.size());
    int selected_position = dist(rng.get());
    Events.insert(Events.begin() + selected_position, new_CNA);
}

std::variant<Node::SNV, Node::CNA> Node::remove_last_event(){
    // Remove the last event and return it
    if (Events.empty())
        throw std::runtime_error("No event to remove");
    std::variant<SNV, CNA> removed_event = Events.back();
    Events.pop_back();
    return removed_event;
}

void Node::remove_event(int index) {
    if (index < 0 || index >= Events.size())
        throw std::runtime_error("Index out of bounds in remove_event");
    Events.erase(Events.begin() + index);
}


std::vector<std::variant<Node::SNV, Node::CNA>> Node::remove_events_by_indices(const std::vector<int>& indices_to_remove)
{
    std::vector<std::variant<SNV, CNA>> removed_events;

    // Copy and sort indices in descending order to avoid invalidating indices when removing events
    std::vector<int> sorted = indices_to_remove;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());

    for (int idx : sorted) {
        if (idx < 0 || idx >= static_cast<int>(Events.size()))
            throw std::runtime_error("Index out of bounds in remove_events_by_indices");

        removed_events.push_back(Events[idx]);
        Events.erase(Events.begin() + idx);
    }

    // back to the original order, so that appending the events to another node keeps their order
    std::reverse(removed_events.begin(), removed_events.end());
    return removed_events;
}

// Eligibility rule shared by change_strand_random_event() and
// count_strand_change_eligible()
static bool is_eligible_for_strand_change(const std::variant<Node::SNV, Node::CNA>& ev,
                                           size_t n_loci_strands, size_t n_region_strands) {
    if (std::holds_alternative<Node::SNV>(ev))
        return n_loci_strands > 1;
    const Node::CNA& cna = std::get<Node::CNA>(ev);
    return !cna.fgd && n_region_strands > 1;
}

int Node::count_strand_change_eligible() const {
    int count = 0;
    for (const auto& ev : Events)
        if (is_eligible_for_strand_change(ev, strands_loci_ref.size(), strands_regions.size()))
            ++count;
    return count;
}

Node::StrandChangeResult Node::change_strand_random_event() {
    // Collect indices of events eligible for strand change:
    // - SNVs with at least 2 strand choices
    // - CNAs that are not FGD, with at least 2 strands available
    // - CNLOH: swap affected_strand and affected_strand2
    std::vector<int> eligible;

    for (int i = 0; i < (int)Events.size(); ++i)
        if (is_eligible_for_strand_change(Events[i], strands_loci_ref.size(), strands_regions.size()))
            eligible.push_back(i);

    StrandChangeResult result;
    if (eligible.empty()) return result; // valid=false

    int selected_position = eligible[
            std::uniform_int_distribution<int>(0, (int)eligible.size() - 1)(rng.get())
    ];

    if (std::holds_alternative<SNV>(Events[selected_position])) {
        SNV& snv = std::get<SNV>(Events[selected_position]);
        int n_strands = (int)strands_loci_ref.size();

        int new_strand;
        do {
            new_strand = std::uniform_int_distribution<int>(0, n_strands - 1)(rng.get());
        } while (new_strand == snv.affected_strand);

        snv.affected_strand = new_strand;
        result.touches_regions = false;

    } else {
        CNA& cna = std::get<CNA>(Events[selected_position]);

        if (cna.CNA_change == 0) { // CNLOH: swap donor and recipient strands
            std::swap(cna.affected_strand, cna.affected_strand2);
            result.is_cnloh_swap = true;
        } else {
            int n_strands = (int)strands_regions.size();
            int new_strand;
            do {
                new_strand = std::uniform_int_distribution<int>(0, n_strands - 1)(rng.get());
            } while (new_strand == cna.affected_strand);
            cna.affected_strand = new_strand;
        }
        result.touches_regions = true;
    }

    result.valid = true;
    return result;
}

void Node::add_events(const std::vector<std::variant<SNV, CNA>>& new_events){
    for (const auto& event : new_events) {
        Events.push_back(event);
    }
}

Node::CNA Node::remove_random_CNA() {

    std::vector<int> cna_indices;

    // Collect indices of CNA events
    for (int i = 0; i < Events.size(); ++i) {
        if (std::holds_alternative<CNA>(Events[i]))
            cna_indices.push_back(i);
    }

    if (cna_indices.empty())
        throw std::runtime_error("remove_random_CNA called but no CNA present");

    // Pick random CNA index
    std::uniform_int_distribution<size_t> dist(0, cna_indices.size() - 1);
    int chosen = cna_indices[dist(rng.get())];

    CNA removed = std::get<CNA>(Events[chosen]);
    Events.erase(Events.begin() + chosen);

    return removed;
}

// The index-th CNA of this node, skipping WGDs (and aneuploidies if local_only).
Node::CNA& Node::nth_CNA(int index, bool local_only) {
    for (auto& event : Events) {
        if (!std::holds_alternative<CNA>(event)) continue;
        CNA& cna = std::get<CNA>(event);
        if (cna.fgd || (local_only && cna.aneuploidy)) continue;
        if (index-- == 0) return cna;
    }
    throw std::runtime_error("CNA index out of bounds");
}

const Node::CNA& Node::get_non_fgd_CNA_by_index(int index) const {
    return const_cast<Node*>(this)->nth_CNA(index, false);
}

void Node::change_CNA_type(int index, int new_CNA_change, int new_strand2) {
    CNA& cna = nth_CNA(index, false);
    cna.CNA_change = new_CNA_change;
    cna.affected_strand2 = new_strand2;
}


Node::CNA Node::get_local_CNA_by_index(int index) { return nth_CNA(index, true); }


Node::CNAStats Node::get_CNA_stats() const {
    CNAStats stats;
    for (const auto& event : Events) {
        if (std::holds_alternative<CNA>(event)) {
            stats.n_CNA += 1;
            const CNA& cna = std::get<CNA>(event);

            switch (cna.CNA_change) {
                case -1:
                    if (cna.aneuploidy) {
                        stats.n_aneuploidy += 1; // If the CNA deletes all regions of a chromosome on a strand
                    } else {
                        stats.n_loss += 1;
                        stats.n_local_CNA += 1;
                    }
                    break;

                case 0:
                    stats.n_cnloh += 1;
                    stats.n_local_CNA += 1;
                    break;

                case 1:
                    if (cna.aneuploidy) {
                        stats.n_aneuploidy += 1; // If the CNA creates a new strand by gaining an additional chromosome copy.
                    } else if (cna.fgd) {
                        stats.n_fgd += 1; // If the CNA is a full genome duplication
                    } else {
                        stats.n_gain += 1;
                        stats.n_local_CNA += 1;
                    }
                    break;

                default:
                    throw std::runtime_error("Invalid CNA change value");
            }
        }

    }
    return stats;
}


Node::SNVStats Node::get_SNV_stats() const
{
    SNVStats stats;
    for (const auto& event : Events)
    {
        if (std::holds_alternative<SNV>(event))
        {
            stats.n_mutations += 1;
            // affected_allele: 0 = ref->alt (forward), 1 = alt->ref (back)
            if (std::get<SNV>(event).affected_allele == 0)
                stats.n_forward_mutations += 1;
            else
                stats.n_back_mutations += 1;
        }
    }
    return stats;
}


bool Node::events_interact(int idx_a, int idx_b) {
    // Two events interact if swapping them can change the genotype: an SNV and a CNA covering its region, or two CNAs with a
    // common region. A WGD affects all regions. A WGD or an aneuploidy also adds or removes whole strands, so exchanging it with
    // any other event can change which strand that event refers to: it interacts with every other event of the node.
    const auto& a = Events[idx_a];
    const auto& b = Events[idx_b];

    // SNV vs CNA
    if ((std::holds_alternative<SNV>(a) && std::holds_alternative<CNA>(b)) ||
        (std::holds_alternative<CNA>(a) && std::holds_alternative<SNV>(b))) {
        const SNV &s = std::holds_alternative<SNV>(a) ? std::get<SNV>(a) : std::get<SNV>(b);
        const CNA &c = std::holds_alternative<CNA>(a) ? std::get<CNA>(a) : std::get<CNA>(b);
        if (c.fgd || c.aneuploidy) return true;   // adds or removes strands (a WGD duplicates every strand), so the SNV's position relative to it matters
        int region = data.locus_to_region[s.locus];
        for (int r : c.region_indices) if (r == region) return true;
        return false;
    }


    // SNV vs SNV: SNVs always affect different loci, so they don't interact
    if (std::holds_alternative<SNV>(a) && std::holds_alternative<SNV>(b)) {
        return false;
    }

    // CNA vs CNA
    if (std::holds_alternative<CNA>(a) && std::holds_alternative<CNA>(b)) {
        const CNA &ca = std::get<CNA>(a);
        const CNA &cb = std::get<CNA>(b);

        // If either is full-genome duplication they interact
        if (ca.fgd || cb.fgd) return true;

        // An aneuploidy adds or removes a whole strand, which changes the strand numbers that the other events refer to, so it is
        // treated as interacting with every other event. A CNLOH only copies the alleles of one existing strand onto another within
        // its own regions, so (like a local gain or loss) it interacts only through shared regions.
        if (ca.aneuploidy || cb.aneuploidy) return true;

        // Overlap in regions
        for (int ra : ca.region_indices) {
            for (int rb : cb.region_indices) {
                if (ra == rb) return true;
            }
        }
        return false;
    }
    return false;
}

void Node::change_event_order(int idx1, int idx2) {
    std::swap(Events[idx1], Events[idx2]);
}

void Node::update_local_CNA_regions(int cna_index, const std::vector<int>& new_regions) {
    nth_CNA(cna_index, true).region_indices = new_regions;
}

