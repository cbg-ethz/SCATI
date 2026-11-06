#ifndef DEF_NODE
#define DEF_NODE

#include <vector>
#include <set>
#include <string>
#include <variant>
#include <memory>
#include <random>
#include <functional>
#include "Structures.h"
#include "Scores.h"

extern Data data;

class Node {
    private:
        std::set<int> affected_loci; // loci for which the copy number of the ref or alt alleles is changed in the node.
        std::set<int> affected_regions; // regions affected by a CNA event in the node

        std::shared_ptr<Scores> cache_scores;
        std::shared_ptr<Params> local_params;

    public:
        std::vector<int> cn_regions; // copy number of each region
        std::vector<int> n_ref_allele;  // number of copies of the reference allele
        std::vector<int> n_alt_allele;  // number of copies of the alternate allele
        std::vector<std::vector<int>> strands_loci_alt; // strands_loci[i] is a vector of size n_loci, where strands_loci[i][j] is the number of alt alleles at locus j.
        std::vector<std::vector<int>> strands_loci_ref; // strands_loci[i] is a vector of size n_loci, where strands_loci[i][j] is the number of ref alleles at locus j.
        std::vector<std::vector<int>> strands_regions; // strands_regions[i] is a vector of size n_regions, where strands_regions[i][k] indicates whether region k is in strand i


        struct SNV {
            int locus;
            int affected_strand; // strand on which the SNV occurs
            int affected_allele; // 0 for ref->alt, 1 for alt->ref
        };

        struct CNA {
            std::vector<int> region_indices;
            int CNA_change; // -1 for loss, +1 for gain, 0 for CNLOH
            int affected_strand;
            int affected_strand2; // used only for cnloh
            bool aneuploidy; // duplicate the whole chromosome and create a new strand or delete all regions within a chromosome.
            bool fgd; // whether the whole genome is duplicated
            std::string chromosome; // chromosome on which the CNA occurs
        };
        std::vector<std::variant<SNV, CNA>> Events; // list of the SNV and CNA events in the node.

        std::vector<double> attachment_scores_SNV;
        std::vector<double> attachment_scores_CNA;
        std::vector<double> attachment_scores;

        // true for doublets, which round Lambda with twice the step (see compute_CNA_scores)
        bool is_doublet = false;

        Node(std::shared_ptr<Scores> cache, std::mt19937& rng_, std::shared_ptr<Params> params); //constructor
        Node(const Node& source) = default;
        Node(const Node& source1, const Node& source2, std::mt19937& rng_); // create doublet from 2 nodes
        void init_structures();

        // Recomputes cn_regions and the allele counts from the strand arrays, for strands loaded directly (for example
        // from best_tree_strand_genotypes.json) instead of replayed from Events. Marks every locus and region as affected.
        void recompute_derived_from_strands();

        // A copied node keeps the source's rng; a node copied into another Tree must be rebound to that tree's rng.
        void rebind_rng(std::mt19937& new_rng) { rng = std::ref(new_rng); }

        std::reference_wrapper<std::mt19937> rng;

        struct CNAStats {
            int n_CNA=0;
            int n_local_CNA=0;
            int n_loss=0;
            int n_gain=0;
            int n_cnloh=0;
            int n_aneuploidy=0;
            int n_fgd=0;
        };

        struct SNVStats {
            int n_mutations=0;
            int n_forward_mutations=0;
            int n_back_mutations=0;
        };

        // Reports what change_strand_random_event() actually did: whether the
        // selected event's own strand count could change as a side effect
        // (touches_regions selects which of strands_loci_ref/strands_regions
        // to recount), and whether it was a CNLOH donor/recipient swap (no
        // "pick a new strand" draw, so no (S-1) factor).
        struct StrandChangeResult {
            bool valid = false;
            bool is_cnloh_swap = false;
            bool touches_regions = false; // true = CNA (strands_regions), false = SNV (strands_loci_ref)
        };

        CNAStats get_CNA_stats() const;
        SNVStats get_SNV_stats() const;

        // Compute genotype of the node, depending on the genotype of its parent
        bool update_genotype(Node* parent, const std::vector<std::vector<int>>& root_strand_loci_ref,
                           const std::vector<std::vector<int>>& root_strand_loci_alt,
                           const std::vector<std::vector<int>>& root_strand_regions);
        // dropout_rates_ref/alt, region_probabilities and region_dispersion are indexed [sample][locus or region];
        // root_cn_regions is the root's copy number per region, the ploidy at which region_dispersion is fitted.
        void compute_attachment_scores_root(const std::vector<std::vector<double>>& dropout_rates_ref,
                                            const std::vector<std::vector<double>>& dropout_rates_alt,
                                            const std::vector<std::vector<double>>& region_probabilities,
                                            const std::vector<std::vector<double>>& region_dispersion,
                                            const std::vector<SampleFit>& sample_fits,
                                            const std::vector<int>& root_cn_regions);
        void compute_attachment_scores_parent(Node* parent, const std::vector<std::vector<double>>& dropout_rates_ref,
                                        const std::vector<std::vector<double>>& dropout_rates_alt,
                                        const std::vector<std::vector<double>>& region_probabilities,
                                              const std::vector<std::vector<double>>& region_dispersion,
                                              const std::vector<SampleFit>& sample_fits,
                                              const std::vector<int>& root_cn_regions);

    private:
        void add_SNV_scores(int i, int c_ref, int c_alt, double sign,
                            const std::vector<std::vector<double>>& dropout_rates_ref,
                            const std::vector<std::vector<double>>& dropout_rates_alt,
                            const std::vector<std::vector<double>>& region_probabilities,
                            const std::vector<std::vector<double>>& region_dispersion,
                            const std::vector<SampleFit>& sample_fits, std::vector<double>& out) const;
        void compute_CNA_scores(const std::vector<std::vector<double>>& region_probabilities,
                                const std::vector<std::vector<double>>& region_dispersion,
                                const std::vector<SampleFit>& sample_fits,
                                const std::vector<int>& root_cn_regions, std::vector<double>& out) const;
        void combine_attachment_scores();
        bool count_copies_from_strands(); // allele and region copies summed over the strands
        CNA& nth_CNA(int index, bool local_only); // the index-th CNA, skipping WGDs (and aneuploidies if local_only)

    public:

        // Helper functions for MCMC moves for the nodes
        SNV remove_random_mutation(); // removes a random mutation, and return the index of the mutation
        CNA remove_random_CNA();
        std::vector<std::variant<Node::SNV, Node::CNA>> remove_events_by_indices(const std::vector<int> &indices_to_remove);
        void add_events(const std::vector<std::variant<SNV, CNA>> &new_events);
        StrandChangeResult change_strand_random_event();
        int count_strand_change_eligible() const;
        void add_CNA(CNA);
        void change_CNA_type(int index, int new_CNA_change, int new_strand2);
        CNA get_local_CNA_by_index(int index);
        void remove_event(int index);
        void add_mutation(SNV new_mutation);
        void add_event(const std::variant<SNV, CNA>& new_event);
        std::variant<SNV, CNA> remove_last_event();
        int get_num_events() const { return static_cast<int>(Events.size());}
        bool events_interact(int a, int b);
        void change_event_order(int idx1, int idx2);
        void update_local_CNA_regions(int cna_index, const std::vector<int> &new_regions);
        bool is_empty() const{ return (Events.empty());}
        const CNA &get_non_fgd_CNA_by_index(int index) const;

};

#endif
