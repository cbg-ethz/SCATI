#ifndef DEF_INFERENCE
#define DEF_INFERENCE

#include <vector>
#include <string>
#include "Tree.h"

class Inference{
    private:
        std::shared_ptr<Params> local_params;

        std::shared_ptr<Scores> cache_scores;

        std::mt19937 rng_;

        Tree t; // current tree
        Tree t_prime; // proposed tree
        Tree best_tree; // best tree seen so far
        double max_temperature; // start MCMC at this temperature, and decreases to 1
        std::vector<double> move_weights;
        std::vector<double> cumulative_weights; // cumulative weights for weighted sampling
        double total_move_weight = 0.0; // sum of all weights
        int index; // to identify the different chains that are run in parallel


public:
        Inference(int idx, Params params); // takes own copy
        ~Inference();
        std::vector<double> default_move_weights() const;
        Tree find_best_tree(int n_steps, int burn_in_steps, const std::string& output_path);
        std::shared_ptr<Scores> get_cache_scores() const { return cache_scores; } // for run statistics
        void mcmc(int n_steps, int n_burnin_steps, const std::string& output_path);
        int select_move(int min_move_index, int max_move_index);

    void precompute_move_weights();
};

#endif