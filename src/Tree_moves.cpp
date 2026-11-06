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

#include "Tree.h"
#include "Node.h"
#include "Scores.h"

#include <cassert>

extern int n_cells;
extern int n_loci;
extern int n_regions;
extern std::vector<Cell> cells;

// Prune a subtree, and reattach it somewhere else in the tree
void Tree::prune_reattach(){

    if (n_nodes<=2){ // Only for at least 3 nodes pruning and reattaching makes sense.
        hastings_ratio=0.0;
        return;
    }

    // Randomly select one node (apart from the root) to prune and reattach
    std::uniform_int_distribution<int> node_dist(1, n_nodes - 1);
    int pruned_node_id = node_dist(rng);

    // Find all nodes that are not a descendant of the pruned node
    // Perform a DFT from the pruned node to identify all its descendants
    std::vector<int> affected_nodes;
    std::vector<bool> is_descendant(n_nodes,false);
    std::stack<int> stk;
    stk.push(pruned_node_id);
    while (!stk.empty()) {
        int top = stk.top();
        stk.pop();
        affected_nodes.push_back(top);
        for (int child:children[top]) {
            stk.push(child);
        }
        is_descendant[top] = true;
    }

    int old_parent_id = parents[pruned_node_id];

    std::vector<int> candidate_attachment_points;
    for (int i=0;i<n_nodes;i++){
        // Exclude descendants of the pruned node, and the pruned node's current
        // parent (reattaching there would recreate the exact same tree).
        if (!is_descendant[i] && i!=old_parent_id)
            candidate_attachment_points.push_back(i);
    }

    if (candidate_attachment_points.empty()) {
        hastings_ratio = 0.0;
        return;
    }

    // Sample one attachment point among all nodes that are not descendant of the pruned node
    std::uniform_int_distribution<std::size_t> attach_dist(0, candidate_attachment_points.size() - 1);
    int attachment_point_id = candidate_attachment_points[attach_dist(rng)];

    // Update the parent child relationships
    attach_subtree(pruned_node_id, attachment_point_id);

    // Update the genotypes and likelihoods, but only of the pruned node and its descendants
    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);

    if (local_params->verbose){
        std::cout << "Prune node " << pruned_node_id << " and reattach to node " << attachment_point_id << ". ";
    }

    if (!valid_tree){
        // The tree is not valid and is rejected
        hastings_ratio=0.0;
        return;
    }

    compute_likelihood_affected_nodes(affected_nodes);
    hastings_ratio=1.0;
}

// Select the 2 nodes and swap their content (excluding the root node, which does not carry events)
void Tree::swap_node_labels(){
    if (n_nodes<=2){
        hastings_ratio=0.0;
        return;
    }

    std::uniform_int_distribution<int> dist1(1, n_nodes - 1);
    std::uniform_int_distribution<int> dist2(1, n_nodes - 2);
    int node1 = dist1(rng);
    int node2 = dist2(rng);
    if (node2==node1) node2 = n_nodes-1; // node2 is uniform over the non-root nodes other than node1

    // Swap the nodes
    std::swap(nodes[node1], nodes[node2]);
    hastings_ratio=1.0;

    if (local_params->verbose){
        std::cout << "Swap node " << node1 << " and node " << node2 << ". ";
    }

    std::vector<int> affected_nodes = affected_nodes_subtrees(node1, node2);

    // Update the genotypes and likelihoods of the affected nodes
    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree){
        hastings_ratio=0.0;
        return;
    }
    compute_likelihood_affected_nodes(affected_nodes);
}

// Add / remove a somatic SNV at a random locus.
void Tree::add_remove_SNV() {
    const double add_SNV_probability = local_params->add_SNV_probability;
    std::bernoulli_distribution add_dist(add_SNV_probability);
    bool add = add_dist(rng);

    int node = -1;
    std::vector<int> affected_nodes;
    bool node_deleted = false;

    // ======================= ADD ==========================
    if (add) {
        // Loci that currently have no SNV anywhere in the tree
        std::vector<bool> locus_has_snv(n_loci, false);
        for (int i = 1; i < n_nodes; ++i)
            for (const auto& ev : nodes[i]->Events)
                if (std::holds_alternative<Node::SNV>(ev))
                    locus_has_snv[std::get<Node::SNV>(ev).locus] = true;

        std::vector<int> free_loci;
        for (int l = 0; l < n_loci; ++l)
            if (!locus_has_snv[l]) free_loci.push_back(l);

        if (free_loci.empty()) {
            hastings_ratio = 0.0;
            if (local_params->verbose)
                std::cout << "No free locus to add an SNV to. Rejecting move.";
            return;
        }

        const int n_free_loci_before = (int)free_loci.size();
        int locus = free_loci[std::uniform_int_distribution<int>(0, n_free_loci_before - 1)(rng)];

        const int total_target_nodes_before = 2 * n_nodes - 1; // existing non-root nodes, or a new node attached to any node
        std::uniform_int_distribution<int> target_dist(0, total_target_nodes_before - 1);
        int draw = target_dist(rng);

        int new_node_children_before = 0;
        if (draw < n_nodes - 1) {
            node = draw + 1; // not added to the root node
        } else {
            int parent = draw - (n_nodes - 1);
            new_node_children_before = (int)children[parent].size();
            add_node(parent); // new node added at the end of nodes
            node = n_nodes - 1;
        }

        const int n_strands = (int)nodes[node]->strands_loci_ref.size();
        int strand = std::uniform_int_distribution<int>(0, n_strands - 1)(rng);
        int allele = std::uniform_int_distribution<int>(0, 1)(rng);

        const int n_events_before_insert = nodes[node]->get_num_events();
        Node::SNV new_snv = {locus, strand, allele};
        nodes[node]->add_mutation(new_snv);

        // Forward: add (p_add), one of the free loci, one of 2*n_nodes-1 targets (an existing non-root node, or a new
        // child of any node that takes each of that node's children with probability 1/2), strand, allele (1/2), and
        // one of the n_events+1 positions in the node's event list (the factor (n_events_before_insert + 1) below).
        // Reverse: remove (1 - p_add) and pick this SNV among all SNVs, one per locus; a node left empty is deleted.
        double q_forward = add_SNV_probability
                           * (1.0 / n_free_loci_before)
                           * (1.0 / total_target_nodes_before)
                           * (1.0 / std::pow(2.0, new_node_children_before))
                           * (1.0 / n_strands) * 0.5;

        const int n_loci_with_SNV_after = n_loci - n_free_loci_before + 1;
        double q_reverse = (1.0 - add_SNV_probability) * (1.0 / n_loci_with_SNV_after);

        hastings_ratio = q_reverse / q_forward * (n_events_before_insert + 1);
    }

    // ====================== REMOVE =======================
    else {
        std::vector<std::pair<int,int>> snv_positions; // (node, index within node's Events)
        for (int i = 1; i < n_nodes; ++i)
            for (int idx = 0; idx < (int)nodes[i]->Events.size(); ++idx)
                if (std::holds_alternative<Node::SNV>(nodes[i]->Events[idx]))
                    snv_positions.emplace_back(i, idx);

        if (snv_positions.empty()) {
            hastings_ratio = 0.0;
            if (local_params->verbose)
                std::cout << "No SNV to remove. Rejecting move.";
            return;
        }

        const int n_loci_with_SNV_before = (int)snv_positions.size();
        const auto& selected = snv_positions[std::uniform_int_distribution<int>(0, n_loci_with_SNV_before - 1)(rng)];
        node = selected.first;
        int index = selected.second;

        const int n_strands_before = (int)nodes[node]->strands_loci_ref.size();
        const int n_events_before_remove = nodes[node]->get_num_events();

        nodes[node]->remove_event(index);

        double q_reverse_children_factor = 1.0;

        if (nodes[node]->is_empty()) {
            node_deleted = true;
            int original_parent = parents[node];
            int n_nodes_before = n_nodes;
            std::stack<int> stk;
            for (int child : children[node]) stk.push(child);
            while (!stk.empty()) {
                int top = stk.top(); stk.pop();
                affected_nodes.push_back(top);
                for (int child : children[top]) stk.push(child);
            }
            delete_node(node);
            for (int& n : affected_nodes)
                if (n == n_nodes_before - 1) n = node;
            if (original_parent == n_nodes_before - 1) original_parent = node;
            q_reverse_children_factor = 1.0 / std::pow(2.0, (double)children[original_parent].size());
        }

        const int total_target_nodes_before = 2 * n_nodes - 1;

        // Forward: remove (1 - p_add) and pick one of the SNVs. Reverse: the ADD branch from the tree after removal
        // (and after deleting the node if it became empty), putting the SNV back at its old position (1 / n_events).
        double q_forward = (1.0 - add_SNV_probability) * (1.0 / n_loci_with_SNV_before);

        const int n_free_loci_after = n_loci - n_loci_with_SNV_before + 1;
        double q_reverse = q_reverse_children_factor * add_SNV_probability
                           * (1.0 / n_free_loci_after)
                           * (1.0 / total_target_nodes_before)
                           * (1.0 / n_strands_before) * 0.5;

        hastings_ratio = q_reverse / q_forward / n_events_before_remove;
    }

    // Update genotypes and likelihoods of affected nodes
    if (!node_deleted) affected_nodes = subtree_nodes(node);

    if (local_params->verbose)
        std::cout << (add ? "Add" : "Remove") << " SNV in node " << node << ". ";

    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree) { hastings_ratio = 0.0; return; }
    compute_likelihood_affected_nodes(affected_nodes);
}

// Move one SNV (snv=true) or CNA to another node, or to a new node.
void Tree::move_event(bool snv) {
    auto n_of_type = [&](int v) { return snv ? nodes[v]->get_SNV_stats().n_mutations : nodes[v]->get_CNA_stats().n_CNA; };
    auto count_holders = [&]() { int c = 0; for (int v = 1; v < n_nodes; ++v) c += (n_of_type(v) > 0); return c; };

    std::vector<int> holders;  // non-root nodes with an event of this type
    for (int v = 1; v < n_nodes; ++v) if (n_of_type(v) > 0) holders.push_back(v);
    if (holders.empty()) { hastings_ratio = 0.0; return; }

    // 1. Source and one of its events of this type; remove it
    int source = holders[std::uniform_int_distribution<int>(0, (int)holders.size() - 1)(rng)];
    const int n_holders_before = (int)holders.size();
    const int n_type_source = n_of_type(source);
    const int n_events_source = nodes[source]->get_num_events();
    std::variant<Node::SNV, Node::CNA> event;
    if (snv) event = nodes[source]->remove_random_mutation();
    else event = nodes[source]->remove_random_CNA();

    // 2. Delete the source if it is now empty (its children go to its parent). delete_node() gives the last node the
    //    deleted node's index, so indices that pointed to the last node are updated.
    std::vector<int> changed = {source};                        // roots of the subtrees whose genotype changes
    int source_parent = parents[source];
    const bool source_deleted = nodes[source]->is_empty();
    int n_children_source_parent = 0;                           // after the deletion
    if (source_deleted) {
        changed = children[source];
        const int last = n_nodes - 1;
        delete_node(source);
        if (source_parent == last) source_parent = source;
        for (int& v : changed) if (v == last) v = source;
        n_children_source_parent = (int)children[source_parent].size();
        source = -1;
    }

    // 3. Destination in the tree without the emptied source: a new node under any node, taking each of that node's
    //    children with probability 1/2, or an existing non-root node other than the source.
    const int n0 = n_nodes;
    const double p_new = (n0 <= 2) ? 1.0 : local_params->new_node_prob;
    const int n_existing = n0 - 1 - (source_deleted ? 0 : 1);
    int destination;
    double q_forward_dest;
    const bool to_new_node = std::bernoulli_distribution(p_new)(rng);
    if (to_new_node) {
        int parent = std::uniform_int_distribution<int>(0, n0 - 1)(rng);
        q_forward_dest = p_new / n0 / std::pow(2.0, (double)children[parent].size());
        add_node(parent);
        destination = n_nodes - 1;
    } else {
        int pick = std::uniform_int_distribution<int>(1, n_existing)(rng);   // skips the source
        destination = (!source_deleted && pick >= source) ? pick + 1 : pick;
        q_forward_dest = (1.0 - p_new) / n_existing;
    }
    const int n_events_dest = nodes[destination]->get_num_events();
    if (snv) nodes[destination]->add_mutation(std::get<Node::SNV>(event));
    else nodes[destination]->add_CNA(std::get<Node::CNA>(event));

    // 4. Reverse: pick the destination among the nodes with an event of this type and this event in it, remove it
    //    (a new destination is then deleted, giving back the same intermediate tree), and recreate the source: a new
    //    node under its parent taking exactly its former children, or the existing source. Event positions: one of
    //    n_events_dest+1 forward, the old one of n_events_source back.
    const double q_reverse_dest = source_deleted
            ? p_new / n0 / std::pow(2.0, (double)n_children_source_parent)
            : (1.0 - p_new) / (n0 - 1 - (to_new_node ? 0 : 1));
    hastings_ratio = (double)n_holders_before * n_type_source * q_reverse_dest * (n_events_dest + 1)
                   / ((double)count_holders() * n_of_type(destination) * q_forward_dest * n_events_source);

    changed.push_back(destination);
    std::vector<int> affected_nodes = subtrees_nodes(changed);
    if (local_params->verbose)
        std::cout << "Move " << (snv ? "SNV" : "CNA") << " to node " << destination << ". ";
    if (!update_genotypes_affected_nodes(affected_nodes)) { hastings_ratio = 0.0; return; }
    compute_likelihood_affected_nodes(affected_nodes);
}

void Tree::move_SNV() { move_event(true); }
void Tree::move_CNA() { move_event(false); }


// Split a node into two or merge a node with its parent
void Tree::split_merge_node() {
    double merge_prob = local_params->merge_probability;
    const double fixed_merge_prob = merge_prob;
    const int n_nodes_before = n_nodes;

    if (n_nodes_before <= 2) // need at least two non-root nodes that could be merged.
        merge_prob = 0.0;

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::bernoulli_distribution coin(0.5);

    if (unif(rng) <= merge_prob) { // MERGE
        std::vector<int> valid_nodes;

        for (int i = 1; i < n_nodes_before; ++i) {
            if (parents[i] != 0) {
                valid_nodes.push_back(i); // Root cannot be merged
            }
        }

        if (valid_nodes.empty()) {
            hastings_ratio = 0.0;
            return;
        }

        int num_valid_nodes = (int)valid_nodes.size();
        std::uniform_int_distribution<int> dist(0, num_valid_nodes - 1);
        int node1 = valid_nodes[dist(rng)];
        int node2 = parents[node1];

        const int n_events   = nodes[node1]->get_num_events() + nodes[node2]->get_num_events();
        const int n_children = (int)children[node1].size() + (int)children[node2].size() - 1; // exclude node1 itself

        // Move node1's events onto node2
        {
            std::vector<std::variant<Node::SNV, Node::CNA>> node1_events;
            while (!nodes[node1]->is_empty())
                node1_events.push_back(nodes[node1]->remove_last_event());
            std::reverse(node1_events.begin(), node1_events.end());
            nodes[node2]->add_events(node1_events);
        }

        delete_node(node1);

        // Fix node2 index if it was the last node and got swapped during deletion
        if (node2 == n_nodes_before - 1) node2 = node1;

        double q_forward = merge_prob * (1.0 / (num_valid_nodes));

        double reverse_split_choice_prob = (n_nodes <= 2) ? 1.0 : (1.0 - fixed_merge_prob);

        double q_reverse = reverse_split_choice_prob // Split
                           * (1.0 / (n_nodes-1)) // Randomly choose non-root node
                           * (1.0 / (n_events - 1)) // Choose the one cut point that recreates node1's original suffix
                           * (1.0 / std::pow(2.0, n_children)); // distribute the children

        hastings_ratio = q_reverse / q_forward;

        std::vector<int> affected_nodes = subtree_nodes(node2);

        bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
        if (!valid_tree) { hastings_ratio = 0.0; return; }
        compute_likelihood_affected_nodes(affected_nodes);

    }
    else { // SPLIT
        if (n_nodes_before < 2) { // no non-root node exists to split (tree is root-only)
            hastings_ratio = 0.0;
            return;
        }
        std::uniform_int_distribution<int> node_dist(1, n_nodes_before - 1);
        int node = node_dist(rng);

        const int n_events   = nodes[node]->get_num_events();
        const int n_children = (int)children[node].size();

        if (n_events <= 1) {
            hastings_ratio = 0.0;
            return;
        }

        add_node(node); // new node is child of node
        int new_node = n_nodes - 1;

        // One cut point: events [0, cut) stay on `node`, events [cut, n_events) move to the new child, in their order.
        // This is exactly the reverse of MERGE, which appends the child's events after the parent's.
        std::uniform_int_distribution<int> cut_dist(1, n_events - 1);
        int cut = cut_dist(rng);
        std::vector<int> indices_to_move;
        for (int i = cut; i < n_events; ++i) indices_to_move.push_back(i);
        nodes[new_node]->add_events(nodes[node]->remove_events_by_indices(indices_to_move));

        double q_forward = (1.0 - merge_prob)
                           * (1.0 / (n_nodes_before-1))
                           * (1.0 / (n_events - 1))
                           * (1.0 / std::pow(2.0, n_children));


        std::vector<int> valid_nodes;

        for (int i = 1; i < n_nodes; ++i) {
            if (parents[i] != 0) {
                valid_nodes.push_back(i);
            }
        }

        int num_valid_nodes = (int)valid_nodes.size();

        double q_reverse = fixed_merge_prob
                           * (1.0 / (num_valid_nodes));

        hastings_ratio = q_reverse / q_forward;

        // Full subtree of node, which includes new_node and all reassigned children
        std::vector<int> affected_nodes = subtree_nodes(node);

        if (local_params->verbose){
            std::cout << "Split node " << node << " into new nodes " << new_node << ". ";
        }

        bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
        if (!valid_tree) { hastings_ratio = 0.0; return; }
        compute_likelihood_affected_nodes(affected_nodes);
    }
}

// Select a random node except root and change the strand affected by a random event in this node
void Tree::change_strand() {
    if (n_nodes <= 1) { // not applicable to root
        hastings_ratio = 0.0;
        return;
    }

    std::uniform_int_distribution<int> node_dist(1, n_nodes - 1);
    int node = node_dist(rng);

    const int n_elig_before = nodes[node]->count_strand_change_eligible();
    const int S_loci_before = (int)nodes[node]->strands_loci_ref.size();
    const int S_regions_before = (int)nodes[node]->strands_regions.size();

    Node::StrandChangeResult result = nodes[node]->change_strand_random_event();
    if (!result.valid) {
        hastings_ratio = 0.0;
        if (local_params->verbose){
            std::cout << "Change strand of a random event in node " << node << " failed because it would lead to an invalid tree. ";
        }
        return;
    }

    std::vector<int> affected_nodes = subtree_nodes(node);

    if (local_params->verbose){
        std::cout << "Change strand of a random event in node " << node << ". ";
    }

    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree){
        hastings_ratio=0.0;
        return;
    }

    const int n_elig_after = nodes[node]->count_strand_change_eligible();
    if (result.is_cnloh_swap) {
        hastings_ratio = (double)n_elig_before / (double)n_elig_after;
    } else {
        const int S_before = result.touches_regions ? S_regions_before : S_loci_before;
        const int S_after = result.touches_regions ? (int)nodes[node]->strands_regions.size()
                                                    : (int)nodes[node]->strands_loci_ref.size();
        if (S_after < 2) { hastings_ratio = 0.0; return; }
        hastings_ratio = (double)(n_elig_before * (S_before - 1))
                        / (double)(n_elig_after  * (S_after  - 1));
    }

    compute_likelihood_affected_nodes(affected_nodes);
}

// Sum, over healthy cells only, of the SNV log-likelihood at `locus` under a
// root genotype with `c_ref`/`c_alt` copies (summed across strands). Used by
// change_root_genotype()'s to disallow root genotypes that are clearly
// contradicted by the healthy cells' own reads at that locus.
double Tree::healthy_snv_loglik_sum(int locus, int c_ref, int c_alt) {
    double total = 0.0;
    int region_i = data.locus_to_region[locus];
    int precision = local_params->hash_precision;
    for (int s = 0; s < data.n_samples; ++s) {
        double omega = region_dispersion[s][region_i] * region_probabilities[s][region_i];
        double dropout = sample_fits[s].dropoutrate;
        long long discretized_omega = std::llround(omega * precision);
        long long discretized_dropout = std::llround(dropout * precision);

        auto key = std::make_tuple(locus, c_ref, c_alt, s, discretized_omega, discretized_dropout);
        auto it = healthy_snv_loglik_cache.find(key);
        if (it != healthy_snv_loglik_cache.end()) {
            total += it->second;
            continue;
        }

        const std::vector<double>& scores = cache_scores->compute_SNV_loglikelihoods_joint(c_ref, c_alt, locus, omega, dropout);
        double sample_sum = 0.0;
        for (int j : data.cells_by_sample[s])
            if (data.cell_is_healthy[j]) sample_sum += scores[j];

        healthy_snv_loglik_cache.emplace(key, sample_sum);
        total += sample_sum;
    }
    return total;
}

void Tree::change_root_genotype(bool is_warmup)
{
    std::uniform_int_distribution<int> locus_dist(0, n_loci - 1);
    int locus = locus_dist(rng);

    const int n_strands = (int)root_strand_loci_ref.size();

    // Capture root's current (pre-flip) genotype at this locus, summed across
    // strands, so it can be compared against the proposed one below.
    int n_ref_before = 0, n_alt_before = 0;
    for (int s = 0; s < n_strands; ++s) {
        n_ref_before += root_strand_loci_ref[s][locus];
        n_alt_before += root_strand_loci_alt[s][locus];
    }

    // A strand with zero copies of this locus's region (e.g. the deleted
    // second copy of a male chrX/chrY region) can never carry the locus
    const int locus_region = data.locus_to_region[locus];
    std::bernoulli_distribution coin(0.5);
    for (int s = 0; s < n_strands; ++s) {
        if (root_strand_regions[s][locus_region] < 1) {
            root_strand_loci_ref[s][locus] = 0;
            root_strand_loci_alt[s][locus] = 0;
            continue;
        }
        bool is_alt = coin(rng);
        root_strand_loci_ref[s][locus] = is_alt ? 0 : 1;
        root_strand_loci_alt[s][locus] = is_alt ? 1 : 0;
    }

    // Burn-in gate: with annotated healthy cells, reject a root genotype change that their own reads at this locus
    // clearly contradict (worse by more than ROOT_GENOTYPE_AMBIGUITY_THRESHOLD nats). Otherwise the root can drift to
    // a state no healthy cell supports, only because it lets other nodes fit more cells. Burn-in only.
    if (is_warmup && local_params->use_joint_likelihood_model && local_params->use_healthy_annotation) {
        int n_ref_after = 0, n_alt_after = 0;
        for (int s = 0; s < n_strands; ++s) {
            n_ref_after += root_strand_loci_ref[s][locus];
            n_alt_after += root_strand_loci_alt[s][locus];
        }

        double healthy_loglik_before = healthy_snv_loglik_sum(locus, n_ref_before, n_alt_before);
        double healthy_loglik_after = healthy_snv_loglik_sum(locus, n_ref_after, n_alt_after);

        const double ROOT_GENOTYPE_AMBIGUITY_THRESHOLD = 50.0;
        if (healthy_loglik_before - healthy_loglik_after > ROOT_GENOTYPE_AMBIGUITY_THRESHOLD) {
            if (local_params->verbose)
                std::cout << "Root genotype change at locus " << locus
                          << " clearly contradicted by healthy cells' own data ("
                          << (healthy_loglik_before - healthy_loglik_after)
                          << " nats worse) -- rejecting.";
            hastings_ratio = 0.0;
            return;
        }
    }

    // Find and remove the SNV at this locus, if one currently exists anywhere
    // in the tree. Otherwise, after changing the root, this move would generally be rejected
    int snv_node = -1, snv_index = -1;
    for (int k = 1; k < n_nodes; ++k) {
        const auto& events = nodes[k]->Events;
        for (int i = 0; i < (int)events.size(); ++i) {
            if (!std::holds_alternative<Node::SNV>(events[i])) continue;
            if (std::get<Node::SNV>(events[i]).locus == locus) {
                snv_node = k; snv_index = i;
                break;
            }
        }
        if (snv_node != -1) break;
    }
    const bool had_snv_before = (snv_node != -1);
    const int n_strands_before = had_snv_before ? (int)nodes[snv_node]->strands_loci_ref.size() : -1;
    const int n_events_snv_node_before_remove = had_snv_before ? nodes[snv_node]->get_num_events() : -1;

    if (had_snv_before) nodes[snv_node]->remove_event(snv_index);
    const bool would_empty_snv_node = had_snv_before && nodes[snv_node]->is_empty();

    // Randomly decide whether the locus ends up with a somatic SNV afterwards.
    // this lets the move introduce a brand-new SNV where none existed, or
    // remove an existing one outright, not just relocate it.
    bool has_snv_after = (n_nodes > 1) &&
            std::bernoulli_distribution(local_params->add_SNV_probability)(rng);

    int n_strands_after = -1;
    int dest = -1;
    int n_events_dest_before_insert = -1;
    if (has_snv_after) {
        std::uniform_int_distribution<int> node_dist(1, n_nodes - 1);
        dest = node_dist(rng);
        n_strands_after = (int)nodes[dest]->strands_loci_ref.size();
        std::uniform_int_distribution<int> strand_dist2(0, n_strands_after - 1);
        std::uniform_int_distribution<int> allele_dist(0, 1);

        Node::SNV new_snv = {locus, strand_dist2(rng), allele_dist(rng)};
        n_events_dest_before_insert = nodes[dest]->get_num_events();
        nodes[dest]->add_mutation(new_snv);
    }

    // Never empty a node: the reverse move would have to create one, which this move cannot do.
    if (would_empty_snv_node && !(has_snv_after && dest == snv_node)) {
        hastings_ratio = 0.0;
        return;
    }

    const double p_add = local_params->add_SNV_probability;

    double q_forward_snv;
    if (has_snv_after) {
        // Forward move had to: choose to add (p_add), then land on this exact
        // destination node, strand, and allele.
        q_forward_snv = p_add * (1.0 / (n_nodes - 1)) * (1.0 / n_strands_after) * 0.5;
    } else {
        // Forward move just had to choose not to add.
        q_forward_snv = 1.0 - p_add;
    }

    double q_reverse_snv;
    if (had_snv_before) {
        // To undo this move, the reverse move would have to: choose to add
        // (p_add), then land back on the original node/strand/allele.
        q_reverse_snv = p_add * (1.0 / (n_nodes - 1)) * (1.0 / n_strands_before) * 0.5;
    } else {
        // To undo this move, the reverse move would just have to choose not
        // to add.
        q_reverse_snv = 1.0 - p_add;
    }

    hastings_ratio = q_reverse_snv / q_forward_snv;
    if (had_snv_before) hastings_ratio /= n_events_snv_node_before_remove;
    if (has_snv_after) hastings_ratio *= (n_events_dest_before_insert + 1);

    if (local_params->verbose)
        std::cout << "Change root genotype at locus " << locus << ". ";

    bool valid_tree = update_genotypes();
    if (!valid_tree) { hastings_ratio = 0.0; return; }
    // The root genotype is inherited by every node, so every node is updated. Currently, the copy-number scores
    // can't be skipped although this move only changes SNVs as updating the genotypes recreates all doublets
    compute_likelihood();
}


// Add / remove local CNA events, aneuploidies or FGD.
void Tree::add_remove_CNA() {
    const double add_CNA_probability = local_params->add_CNA_probability;
    const double aneuploidy_probability = local_params->aneuploidy_probability;
    const double fgd_probability = local_params->fgd_probability;

    std::bernoulli_distribution add_dist(add_CNA_probability);
    bool add = add_dist(rng);

    int node = -1;
    std::vector<int> affected_nodes;
    bool node_deleted = false;

    // ======================= ADD ==========================
    if (add) {
        const int total_target_nodes_before = 2 * n_nodes - 1; // n_nodes nodes can be added attached to existing nodes or added to all but the root node
        std::uniform_int_distribution<int> target_dist(0, total_target_nodes_before - 1);
        int draw = target_dist(rng);

        int new_node_children_before = 0;
        if (draw < n_nodes - 1) {
            node = draw + 1; // not added to the root node
        } else {
            int parent = draw - (n_nodes - 1);
            new_node_children_before = (int)children[parent].size();
            add_node(parent); // New node added at the end of nodes
            node = n_nodes - 1;
        }

        auto proposal = propose_CNA(fgd_probability, aneuploidy_probability, node);
        if (!proposal.valid) { hastings_ratio = 0.0; return; }

        const int n_events_before_insert = nodes[node]->get_num_events();
        nodes[node]->add_CNA(proposal.cna);

        // Forward: add (p_add), one of 2*n_nodes-1 targets as in add_remove_SNV, the CNA drawn by propose_CNA()
        // (CNA_proposal_prob), and one of the n_events+1 positions in the node's event list.
        // Reverse: remove (1 - p_add), pick one of the nodes with a CNA, then one of its CNAs.
        double q_forward = add_CNA_probability
                           * (1.0 / total_target_nodes_before)
                           * (1.0 / std::pow(2.0, new_node_children_before))
                           * CNA_proposal_prob(proposal.cna, proposal.n_strands);

        int n_nodes_with_CNA_after = 0;
        for (int i = 1; i < n_nodes; ++i)
            if (nodes[i]->get_CNA_stats().n_CNA > 0)
                n_nodes_with_CNA_after++;

        double q_reverse = (1.0 - add_CNA_probability)
                           * (1.0 / n_nodes_with_CNA_after)
                           * (1.0 / nodes[node]->get_CNA_stats().n_CNA);

        hastings_ratio = q_reverse / q_forward * (n_events_before_insert + 1);
    }

    // ====================== REMOVE =======================
    else {
        std::vector<int> nodes_with_CNA;
        for (int i = 1; i < n_nodes; ++i)
            if (nodes[i]->get_CNA_stats().n_CNA > 0)
                nodes_with_CNA.push_back(i);

        if (nodes_with_CNA.empty()) {
            hastings_ratio = 0.0;
            if (local_params->verbose)
                std::cout << "No CNA to remove. Rejecting move.";
            return;
        }

        std::uniform_int_distribution<int> node_dist(0, (int)nodes_with_CNA.size() - 1);
        node = nodes_with_CNA[node_dist(rng)];

        const int n_nodes_with_CNA_before = (int)nodes_with_CNA.size();
        const int n_CNA_before = nodes[node]->get_CNA_stats().n_CNA;

        const int n_events_before_remove = nodes[node]->get_num_events();

        Node::CNA removed = nodes[node]->remove_random_CNA();

        // n_strands for the reverse (hypothetical re-add) proposal must reflect
        // the node's strand count as it exists after his removal
        int n_strands_reverse;
        double q_reverse_children_factor = 1.0;

        if (nodes[node]->is_empty()) {
            node_deleted = true;
            int original_parent = parents[node];
            int n_nodes_before = n_nodes;
            std::stack<int> stk;
            for (int child : children[node]) stk.push(child);
            while (!stk.empty()) {
                int top = stk.top(); stk.pop();
                affected_nodes.push_back(top);
                for (int child : children[top]) stk.push(child);
            }
            delete_node(node);
            for (int& n : affected_nodes)
                if (n == n_nodes_before - 1) n = node;
            // The reverse move must recreate this node from scratch via
            // add_node, which seeds a fresh node with its parent's current
            // genotype (no events yet). Relabel the parent index too, in case delete_node's
            // compaction just moved it into the deleted slot.
            if (original_parent == n_nodes_before - 1) original_parent = node;
            n_strands_reverse = (int)nodes[original_parent]->strands_regions.size();
            q_reverse_children_factor = 1.0 / std::pow(2.0, (double)children[original_parent].size());
        } else {
            // Refresh this node's genotype so its strand count reflects the removal
            nodes[node]->update_genotype(nodes[parents[node]].get(),
                                         root_strand_loci_ref, root_strand_loci_alt, root_strand_regions);
            n_strands_reverse = (int)nodes[node]->strands_regions.size();
        }

        const int total_target_nodes_before = 2 * n_nodes - 1;

        // Forward: remove (1 - p_add), pick one of the nodes with a CNA, then one of its CNAs. Reverse: the ADD branch
        // from the tree after removal, with the strand count after removal, putting the CNA back at its old position.
        double q_forward = (1.0 - add_CNA_probability)
                           * (1.0 / n_nodes_with_CNA_before)
                           * (1.0 / n_CNA_before);

        double q_reverse = add_CNA_probability
                           * (1.0 / total_target_nodes_before)
                           * q_reverse_children_factor
                           * CNA_proposal_prob(removed, n_strands_reverse);

        hastings_ratio = q_reverse / q_forward / n_events_before_remove;
    }

    // Update genotypes and likelihoods of affected nodes
    if (!node_deleted) affected_nodes = subtree_nodes(node);

    if (local_params->verbose)
        std::cout << (add ? "Add" : "Remove") << " CNA in node " << node << ". ";

    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree) { hastings_ratio = 0.0; return; }
    compute_likelihood_affected_nodes(affected_nodes);
}

// Propose a new CNA to add
Tree::CNAProposal Tree::propose_CNA(double fgd_probability, double aneuploidy_probability, int node) {
    CNAProposal result{};
    result.valid = false;
    result.n_strands = (int)nodes[node]->strands_regions.size();
    bool full_genome = false;
    bool aneuploidy  = false;
    int type = 0;
    int total_blocks = 1;

    full_genome = std::bernoulli_distribution(fgd_probability)(rng);

    if (full_genome) {
        result.cna   = {{}, 1, -1, -1, false, true, ""};
        result.valid = true;
        return result;
    }

    type = std::uniform_int_distribution<int>(-1, 1)(rng);
    std::uniform_int_distribution<int> strand_dist(0, result.n_strands - 1);
    int strand        = strand_dist(rng);
    int second_strand = -1;

    if (type == 0) {
        if (result.n_strands < 2) return result; // valid=false
        do { second_strand = strand_dist(rng); } while (second_strand == strand);
    } else { // no cnloh for aneuploidy
        aneuploidy = std::bernoulli_distribution(aneuploidy_probability)(rng);
    }

    std::uniform_int_distribution<int> chr_dist(0, data.n_chromosomes - 1);
    std::string chr = data.chromosome_names[chr_dist(rng)];
    const std::vector<int>& chr_regions = data.chromosome_to_regions[chr];
    int k = (int)chr_regions.size();
    total_blocks = k * (k + 1) / 2;

    std::vector<int> regions;
    if (aneuploidy) {
        regions = chr_regions;
    } else {
        std::uniform_int_distribution<int> block_dist(0, total_blocks - 1);
        int block_index = block_dist(rng);
        int start = 0, length = 0, counter = 0;
        for (int l = 1; l <= k; ++l) {
            int blocks_of_size_l = k - l + 1;
            if (block_index < counter + blocks_of_size_l) {
                length = l; start = block_index - counter; break;
            }
            counter += blocks_of_size_l;
        }
        for (int i = 0; i < length; ++i)
            regions.push_back(chr_regions[start + i]);
    }

    if (type == 0)
        result.cna = {regions, type, strand, second_strand, false, false, chr};
    else
        result.cna = {regions, type, strand, second_strand, aneuploidy, false, chr};

    result.valid = true;
    return result;
}

double Tree::CNA_proposal_prob(const Node::CNA& cna, int n_strands) const {
    if (cna.fgd) return local_params->fgd_probability;

    // A CNA that propose_CNA() cannot draw with this many strands (none, or a CNLOH with only one strand) has
    // probability 0.
    if (n_strands <= 0) return 0.0;
    if (cna.CNA_change == 0 && n_strands <= 1) return 0.0;

    std::string chr  = cna.chromosome;
    int k = (int)data.chromosome_to_regions[chr].size();
    int total_blocks = k * (k + 1) / 2;

    double p = (1.0 - local_params->fgd_probability)
               * (1.0 / n_strands) // to choose one affected strand
               * (1.0 / data.n_chromosomes)
               * (1.0 / 3.0); // loss, gain, cnloh

    if (cna.CNA_change == 0) {
        p *= (1.0 / (n_strands - 1)) * (1.0 / total_blocks); // second strand, and the region-block, for cnloh
    } else {
        p *= cna.aneuploidy
             ? local_params->aneuploidy_probability
             : (1.0 - local_params->aneuploidy_probability) * (1.0 / total_blocks);
    }
    return p;
}

// Change the type of a random local CNA or aneuploidy (i.e. gain to loss, or loss to CNLOH)
void Tree::change_CNA_types() {
    // ============ Aneuploidy <-> two local CNAs conversion ============
    // With probability convert_aneuploidy_probability, either split an
    // aneuploidy event into two local CNAs of the same type (covering its
    // chromosome with 0 or 1 gap), or merge two same-type local CNAs that
    // jointly span a whole chromosome (with 0 or 1 gap) into one aneuploidy
    // event.
    if (std::bernoulli_distribution(local_params->convert_aneuploidy_probability)(rng)) {
        const double p_merge_an = local_params->merge_aneuploidy_probability;
        bool do_merge = std::bernoulli_distribution(p_merge_an)(rng);
        int node = -1;

        if (do_merge) {
            auto candidates = merge_candidates(true);
            const int M = (int)candidates.size();
            if (M == 0) { hastings_ratio = 0.0; return; }

            auto chosen = candidates[std::uniform_int_distribution<int>(0, M - 1)(rng)];
            node = chosen.node;
            const Node::CNA& A = std::get<Node::CNA>(nodes[node]->Events[chosen.idx1]);

            Node::CNA merged;
            merged.CNA_change       = A.CNA_change;
            merged.affected_strand  = A.affected_strand;
            merged.affected_strand2 = -1;
            merged.aneuploidy = true;
            merged.fgd = false;
            merged.chromosome = A.chromosome;
            merged.region_indices = data.chromosome_to_regions[A.chromosome]; // aneuploidy always covers the whole chromosome

            const int n_events_before_merge = nodes[node]->get_num_events();

            nodes[node]->remove_event(std::max(chosen.idx1, chosen.idx2));
            nodes[node]->remove_event(std::min(chosen.idx1, chosen.idx2));
            nodes[node]->add_CNA(merged);

            const int S_after = (int)split_candidates(true).size();

            double q_forward = local_params->convert_aneuploidy_probability * p_merge_an * (1.0 / M);
            double q_reverse = local_params->convert_aneuploidy_probability * (1.0 - p_merge_an) * (1.0 / S_after);
            hastings_ratio = q_reverse / q_forward / n_events_before_merge;

            if (local_params->verbose)
                std::cout << "Merged 2 local CNAs into an aneuploidy in node " << node << ". ";

        } else {
            auto candidates = split_candidates(true);
            const int S = (int)candidates.size();
            if (S == 0) { hastings_ratio = 0.0; return; }

            auto chosen = candidates[std::uniform_int_distribution<int>(0, S - 1)(rng)];
            node = chosen.node;

            Node::CNA original = std::get<Node::CNA>(nodes[node]->Events[chosen.event_idx]);
            const std::vector<int>& chr_regions = data.chromosome_to_regions[original.chromosome];

            Node::CNA left, right;
            left.CNA_change = right.CNA_change = original.CNA_change;
            left.affected_strand = right.affected_strand = original.affected_strand;
            left.affected_strand2 = right.affected_strand2 = -1;
            left.aneuploidy = right.aneuploidy = false;
            left.fgd = right.fgd = false;
            left.chromosome = right.chromosome = original.chromosome;

            if (!chosen.is_gap) {
                left.region_indices.assign(chr_regions.begin(), chr_regions.begin() + chosen.split_pos);
                right.region_indices.assign(chr_regions.begin() + chosen.split_pos, chr_regions.end());
            } else {
                left.region_indices.assign(chr_regions.begin(), chr_regions.begin() + chosen.split_pos);
                right.region_indices.assign(chr_regions.begin() + chosen.split_pos + 1, chr_regions.end());
            }

            const int n_events_before_split = nodes[node]->get_num_events();

            nodes[node]->remove_event(chosen.event_idx);
            nodes[node]->add_CNA(left);
            nodes[node]->add_CNA(right);

            const int M_after = (int)merge_candidates(true).size();

            double q_forward = local_params->convert_aneuploidy_probability * (1.0 - p_merge_an) * (1.0 / S);
            double q_reverse = local_params->convert_aneuploidy_probability * p_merge_an * (1.0 / M_after);
            hastings_ratio = q_reverse / q_forward * (n_events_before_split + 1);

            if (local_params->verbose)
                std::cout << "Split an aneuploidy into 2 local CNAs in node " << node << ". ";
        }

        std::vector<int> affected_nodes = subtree_nodes(node);

        bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
        if (!valid_tree) { hastings_ratio = 0.0; return; }
        compute_likelihood_affected_nodes(affected_nodes);
        return;
    }

    // ============ Retype a single CNA (existing behavior) ============
    // --- Collect non-root nodes with changeable CNAs ---
    std::vector<int> nodes_with_event;
    for (int n = 1; n < n_nodes; ++n) {
        Node::CNAStats cna_stats = nodes[n]->get_CNA_stats();
        if (cna_stats.n_local_CNA + cna_stats.n_aneuploidy > 0) // exclude full genome duplications
            nodes_with_event.push_back(n);
    }

    if (nodes_with_event.empty()) {
        hastings_ratio = 0.0;

        if (local_params->verbose) {
            std::cout << "No CNA available to change type. Rejecting move.";
        }
        return;
    }

    // Sample node and CNA uniformly
    int node = nodes_with_event[
            std::uniform_int_distribution<int>(0, (int)nodes_with_event.size() - 1)(rng)
    ];

    // Sample a non-FGD CNA uniformly
    int n_cna = nodes[node]->get_CNA_stats().n_local_CNA + nodes[node]->get_CNA_stats().n_aneuploidy;
    int cna_index = std::uniform_int_distribution<int>(0, n_cna - 1)(rng);

    // Get the current CNA to determine the transition
    const Node::CNA& cna = nodes[node]->get_non_fgd_CNA_by_index(cna_index);
    const int old_CNA_change = cna.CNA_change;
    const int S_before = (int)nodes[node]->strands_regions.size();
    // |other_types(type, S)|: how many of the 2 non-`type` CNA types are
    // offered given a strand count S (CNLOH is excluded when S<2).
    auto other_types_size = [](int type, int S) { return (type != 0 && S < 2) ? 1 : 2; };

    int new_CNA_change;
    int new_strand2 = -1;

    if (cna.aneuploidy) {
        // Aneuploidy: toggle between gain and loss
        new_CNA_change = cna.CNA_change * -1;
    } else {
        // Local CNA: pick uniformly among the 2 other types
        std::vector<int> other_types;
        for (int t : {-1, 0, 1}) {
            if (t != cna.CNA_change) {
                if (t == 0 && S_before < 2) continue; // CNLOH needs two strands (identifiability is checked by Node::update_genotype)
                other_types.push_back(t);
            }
        }
        new_CNA_change = other_types[
                std::uniform_int_distribution<int>(0, (int)other_types.size() - 1)(rng)
        ];
        if (new_CNA_change == 0) {
            // Switching to CNLOH: pick random second strand
            std::uniform_int_distribution<int> dist(0, S_before - 2);
            int tmp = dist(rng);
            new_strand2 = (tmp >= cna.affected_strand) ? tmp + 1 : tmp;
        }
    }

    nodes[node]->change_CNA_type(cna_index, new_CNA_change, new_strand2);

    // Update genotypes and likelihoods of affected subtree
    std::vector<int> affected_nodes = subtree_nodes(node);

    if (local_params->verbose){
        std::cout << "Change CNA type of a CNA in node " << node << ". ";
    }

    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree) { hastings_ratio = 0.0; return; }

    if (!cna.aneuploidy) {
        const int S_after = (int)nodes[node]->strands_regions.size();
        if (old_CNA_change == 0 && S_after < 2) { hastings_ratio = 0.0; return; }
        double q_forward = 1.0 / other_types_size(old_CNA_change, S_before)
                          * (new_CNA_change == 0 ? 1.0 / (S_before - 1) : 1.0);
        double q_reverse = 1.0 / other_types_size(new_CNA_change, S_after)
                          * (old_CNA_change == 0 ? 1.0 / (S_after - 1) : 1.0);
        hastings_ratio = q_reverse / q_forward;
    } else {
        hastings_ratio = 1.0;
    }

    compute_likelihood_affected_nodes(affected_nodes);
}


// Select a random node and change the order of two random events in this node
void Tree::change_event_order_within_node() {
    // Only consider nodes where reordering actually changes the genotype:
    // - two CNAs with overlapping affected regions, or
    // - a CNA and an SNV where the SNV lies on a region affected by the CNA
    std::vector<std::tuple<int,int,int>> candidates; // (node, event_index1, event_index2)

    for (int n = 0; n < n_nodes; ++n) {
        int n_events = (int)nodes[n]->Events.size();
        for (int i = 0; i < n_events; ++i) {
            for (int j = i + 1; j < n_events; ++j) {
                if (nodes[n]->events_interact(i, j))
                    candidates.emplace_back(n, i, j);
            }
        }
    }

    if (candidates.empty()) {
        hastings_ratio = 0.0;
        if (local_params->verbose) {
            std::cout << "No pair of events found where changing the order would change the genotype. Rejecting move.";
        }
        return;
    }

    auto [node, idx1, idx2] = candidates[
            std::uniform_int_distribution<int>(0, (int)candidates.size() - 1)(rng)
    ];

    nodes[node]->change_event_order(idx1, idx2);

    // Hastings ratio = 1.0: candidate pairs are the same before and after the swap
    hastings_ratio = 1.0;

    // --- Update genotypes and likelihoods of affected subtree ---
    std::vector<int> affected_nodes = subtree_nodes(node);

    if (local_params->verbose){
        std::cout << "Change event order of two events in node " << node << ". ";
    }

    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree) { hastings_ratio = 0.0; return; }
    compute_likelihood_affected_nodes(affected_nodes);
}

// Select a random node and a random local CNA event in this node, and change the regions affected by this CNA.
void Tree::change_CNA_regions()  {
    // --- Select node with at least one local CNA ---
    std::vector<int> nodes_with_event;
    for (int n = 1; n < n_nodes; ++n)
        if (nodes[n]->get_CNA_stats().n_local_CNA > 0)
            nodes_with_event.push_back(n);

    if (nodes_with_event.empty()) { hastings_ratio = 0.0; return; }

    int node = nodes_with_event[std::uniform_int_distribution<int>(0, (int)nodes_with_event.size() - 1)(rng)];

    // --- Select a local CNA uniformly ---
    int n_local_cna = nodes[node]->get_CNA_stats().n_local_CNA; // changing region size only makes sense for local CNAs
    int cna_index = std::uniform_int_distribution<int>(0, n_local_cna - 1)(rng);

    Node::CNA cna = nodes[node]->get_local_CNA_by_index(cna_index);

    std::vector<int> regions = cna.region_indices;

    const std::vector<int>& chr_regions = data.chromosome_to_regions[cna.chromosome];
    int left_pos  = (int)(std::find(chr_regions.begin(), chr_regions.end(), regions.front()) - chr_regions.begin());
    int right_pos = (int)(std::find(chr_regions.begin(), chr_regions.end(), regions.back())  - chr_regions.begin());

    // --- Proposal: pick end (left/right), direction (add/remove), amount (1-10) ---

    static const double AMOUNT_DECAY = 0.7;
    static const std::vector<double> amount_weights = [] {
        std::vector<double> w(10);
        for (int k = 1; k <= 10; ++k) w[k - 1] = std::pow(AMOUNT_DECAY, k - 1);
        return w;
    }();
    static std::discrete_distribution<int> amount_dist(amount_weights.begin(), amount_weights.end());

    std::bernoulli_distribution coin(0.5);
    bool modify_left = coin(rng);
    bool add = coin(rng);
    int  amount = amount_dist(rng) + 1; // discrete_distribution is 0-indexed

    // Proposal is symmetric in all valid cases — asymmetric cases are rejected with HR=0.
    hastings_ratio = 1.0;

    if (add) {
        for (int i = 0; i < amount; ++i) {
            if (modify_left) {
                if (left_pos == 0) { hastings_ratio = 0.0; return; } // chromosome boundary
                int candidate = chr_regions[left_pos - 1];
                regions.insert(regions.begin(), candidate);
                left_pos--;
            } else {
                if (right_pos == (int)chr_regions.size() - 1) { hastings_ratio = 0.0; return; } // chromosome boundary
                int candidate = chr_regions[right_pos + 1];
                regions.push_back(candidate);
                right_pos++;
            }
        }
    } else {
        if ((int)regions.size() <= amount) { hastings_ratio = 0.0; return; } // would remove CNA entirely
        for (int i = 0; i < amount; ++i) {
            if (modify_left) regions.erase(regions.begin());
            else regions.pop_back();
        }
    }

    // --- Write modified regions back to the node ---
    nodes[node]->update_local_CNA_regions(cna_index, regions);

    // --- Update genotypes and likelihoods of affected subtree ---
    std::vector<int> affected_nodes = subtree_nodes(node);

    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree) { hastings_ratio = 0.0; return; }
    compute_likelihood_affected_nodes(affected_nodes);
}


void Tree::merge_split_within_node() {
    const double p_merge = local_params->merge_CNA_probability;
    const bool merge = std::bernoulli_distribution(p_merge)(rng);
    int node = -1;

    if (n_nodes < 2) {
        hastings_ratio = 0.0;
        return;
    }

    // ======================= MERGE =======================
    if (merge) {
        std::vector<MergeCandidate> candidates = merge_candidates(false);

        const int M = (int)candidates.size();
        if (M == 0) { hastings_ratio = 0.0; return; }

        auto chosen = candidates[std::uniform_int_distribution<int>(0, M - 1)(rng)];
        node = chosen.node;
        auto& events = nodes[node]->Events;

        Node::CNA left  = std::get<Node::CNA>(events[chosen.idx1]);
        Node::CNA right = std::get<Node::CNA>(events[chosen.idx2]);
        if (left.region_indices.front() > right.region_indices.front())
            std::swap(left, right);

        // Build merged region list, inserting the gap region if present
        Node::CNA merged = left;
        if (chosen.has_gap) {
            // find the single region sitting between left.back() and right.front()
            // on the same chromosome
            const std::vector<int>& chr_regions =
                    data.chromosome_to_regions[left.chromosome];
            int left_back_pos  = (int)(std::find(chr_regions.begin(), chr_regions.end(),
                                                 left.region_indices.back())  - chr_regions.begin());
            merged.region_indices.push_back(chr_regions[left_back_pos + 1]); // the gap region
        }
        merged.region_indices.insert(merged.region_indices.end(),
                                     right.region_indices.begin(),
                                     right.region_indices.end());

        const int n_events_before_merge = nodes[node]->get_num_events();

        nodes[node]->remove_event(std::max(chosen.idx1, chosen.idx2));
        nodes[node]->remove_event(std::min(chosen.idx1, chosen.idx2));
        nodes[node]->add_CNA(merged);

        const int possible_splits_after = (int)split_candidates(false).size();

        // forward: one of M merge candidates; reverse: one of possible_splits_after split candidates
        // (each (CNA, split position, gap or not) is one candidate, as in the SPLIT branch below)
        double q_forward = p_merge * (1.0 / M);
        double q_reverse = (1.0 - p_merge)
                           * (1.0 / possible_splits_after);
        hastings_ratio = q_reverse / q_forward / n_events_before_merge;
    }

        // ======================= SPLIT =======================
    else {
        std::vector<SplitCandidate> candidates = split_candidates(false);

        const int S = (int)candidates.size();
        if (S == 0) { hastings_ratio = 0.0; return; }

        auto chosen = candidates[std::uniform_int_distribution<int>(0, S - 1)(rng)];
        node = chosen.node;

        Node::CNA original = std::get<Node::CNA>(nodes[node]->Events[chosen.event_idx]);

        Node::CNA left  = original;
        Node::CNA right = original;

        if (!chosen.is_gap) {
            // clean split: left gets [0, split_pos), right gets [split_pos, L)
            left.region_indices.assign(original.region_indices.begin(),
                                       original.region_indices.begin() + chosen.split_pos);
            right.region_indices.assign(original.region_indices.begin() + chosen.split_pos,
                                        original.region_indices.end());
        } else {
            // gap split: left gets [0, split_pos), gap region discarded,
            // right gets [split_pos+1, L)
            left.region_indices.assign(original.region_indices.begin(),
                                       original.region_indices.begin() + chosen.split_pos);
            right.region_indices.assign(original.region_indices.begin() + chosen.split_pos + 1,
                                        original.region_indices.end());
        }

        const int n_events_before_split = nodes[node]->get_num_events();

        nodes[node]->remove_event(chosen.event_idx);
        nodes[node]->add_CNA(left);
        nodes[node]->add_CNA(right);

        const int possible_merges_after = (int)merge_candidates(false).size();

        double q_forward = (1.0 - p_merge) * (1.0 / S);
        double q_reverse = p_merge * (1.0 / possible_merges_after);
        hastings_ratio = q_reverse / q_forward * (n_events_before_split + 1);
    }

    std::vector<int> affected_nodes = subtree_nodes(node);

    bool valid_tree = update_genotypes_affected_nodes(affected_nodes);
    if (!valid_tree) { hastings_ratio = 0.0; return; }
    compute_likelihood_affected_nodes(affected_nodes);
}


// Removes an existing gain/loss CNA event and immediately proposes a new one on the same node
void Tree::swap_CNA() {
    // 1. Collect every existing gain/loss CNA event in the tree. CNLOH
    // (CNA_change==0) and FGD are excluded.
    std::vector<std::pair<int,int>> candidates; // (node, event_index)
    for (int n = 1; n < n_nodes; ++n) {
        auto& events = nodes[n]->Events;
        for (int i = 0; i < (int)events.size(); ++i) {
            if (!std::holds_alternative<Node::CNA>(events[i])) continue;
            const Node::CNA& c = std::get<Node::CNA>(events[i]);
            if (!c.fgd && c.CNA_change != 0) candidates.emplace_back(n, i);
        }
    }
    if (candidates.empty()) { hastings_ratio = 0.0; return; }

    // 2. Pick E_old uniformly, remove it.
    auto [node, event_idx] = candidates[std::uniform_int_distribution<int>(0, (int)candidates.size() - 1)(rng)];
    Node::CNA E_old = std::get<Node::CNA>(nodes[node]->Events[event_idx]);
    nodes[node]->remove_event(event_idx);

    std::vector<int> affected_nodes = subtree_nodes(node);
    if (!update_genotypes_affected_nodes(affected_nodes)) { hastings_ratio = 0.0; return; }

    CNAProposal proposal = propose_CNA(local_params->fgd_probability,
                                        local_params->aneuploidy_probability, node);
    if (!proposal.valid || proposal.cna.fgd || proposal.cna.CNA_change == 0) {
        hastings_ratio = 0.0; return;
    }
    const int n_strands_forward = proposal.n_strands; // captured post-E_old-removal
    nodes[node]->add_CNA(proposal.cna);

    // Recompute genotype/likelihood for the final (post-swap) state.
    if (!update_genotypes_affected_nodes(affected_nodes)) { hastings_ratio = 0.0; return; }
    compute_likelihood_affected_nodes(affected_nodes);

    hastings_ratio = CNA_proposal_prob(E_old, n_strands_forward)
                    / CNA_proposal_prob(proposal.cna, n_strands_forward);
}


// Gap between two CNAs of the same type, strand(s) and chromosome: 0 if adjacent, 1 if one region apart,
// -1 if they cannot be merged.
int Tree::merge_gap(const Node::CNA& A, const Node::CNA& B) {
    if (A.CNA_change != B.CNA_change || A.affected_strand != B.affected_strand
        || (A.CNA_change == 0 && A.affected_strand2 != B.affected_strand2) || A.chromosome != B.chromosome)
        return -1;
    const std::vector<int>& chr_regions = data.chromosome_to_regions[A.chromosome];
    auto pos = [&](int region) { return (int)(std::find(chr_regions.begin(), chr_regions.end(), region) - chr_regions.begin()); };
    int A_start = pos(A.region_indices.front()), A_end = pos(A.region_indices.back());
    int B_start = pos(B.region_indices.front()), B_end = pos(B.region_indices.back());
    for (int gap : {0, 1})
        if (A_end + 1 + gap == B_start || B_end + 1 + gap == A_start) return gap;
    return -1;
}
