#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <omp.h>
#include <optional>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <map>
#include <sstream>
#if defined(__linux__) || defined(__CYGWIN__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "Inference.h"
#include "Tree.h"
#include "Scores.h"
#include "input.h"

int n_cells;
int n_loci;
int n_regions;
std::vector<Cell> cells;
Data data;
Params parameters;

// Per-chain run statistics, written to <chain dir>/chain_summary.json as soon as a chain finishes and
// collected into <output>/run_summary.json at the end.
struct ChainStats {
    double seconds = 0.0;
    double log_score = 0.0;
    int n_nodes = 0;
    size_t cna_calls = 0, cna_misses = 0;
    std::map<std::string, size_t> cache_clears;
    size_t cache_budget_bytes = 0, peak_cache_bytes = 0;
};

static std::string json_escape(const std::string& in) {
    std::string out;
    for (char c : in) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

// JSON has no inf or nan: those are written as null
static std::string json_num(double x) {
    if (!std::isfinite(x)) return "null";
    std::ostringstream o;
    o << std::setprecision(10) << x;
    return o.str();
}

static std::string chain_stats_json(int chain, const ChainStats& c, const std::string& indent) {
    std::ostringstream o;
    o << std::setprecision(10);
    o << indent << "{\"chain\": " << chain
      << ", \"seconds\": " << json_num(c.seconds)
      << ", \"mcmc_steps\": " << parameters.chain_length
      << ", \"steps_per_second\": " << json_num(c.seconds > 0 ? parameters.chain_length / c.seconds : 0.0)
      << ", \"log_score\": " << json_num(c.log_score)
      << ", \"n_nodes\": " << c.n_nodes
      << ", \"cna_likelihood_calls\": " << c.cna_calls
      << ", \"cna_likelihood_cache_misses\": " << c.cna_misses
      << ", \"cna_cache_hit_rate\": " << json_num(c.cna_calls > 0 ? 1.0 - (double)c.cna_misses / (double)c.cna_calls : 0.0)
      << ", \"cache_budget_gb\": " << json_num(c.cache_budget_bytes / 1073741824.0)
      << ", \"peak_cache_gb\": " << json_num(c.peak_cache_bytes / 1073741824.0)
      << ", \"cache_clears\": {";
    bool first = true;
    for (const auto& kv : c.cache_clears) {
        o << (first ? "" : ", ") << "\"" << kv.first << "\": " << kv.second;
        first = false;
    }
    o << "}}";
    return o.str();
}

static void print_usage() {
    std::cout <<
        "Usage: SCATI -i <input prefix> -o <output dir> [options]\n"
        "  -i <prefix>      reads <prefix>_variants.csv, <prefix>_regions.csv and <prefix>.cell_annotation.csv\n"
        "                   (the _filtered.csv versions when present)\n"
        "  -o <dir>         output directory (default: the input prefix)\n"
        "  --n_samples <n>  joint analysis of <prefix>_1 ... <prefix>_n\n"
        "  common options:  --n_chains, --chain_length, --burn_in, --random_seed,\n"
        "                   --use_healthy_annotation, --use_haplotype_annotation, --sex male|female\n"
        "See README.md for all options.\n";
}

int main(int argc, char* argv[]) {
  const auto program_start = std::chrono::steady_clock::now();
  try {
      init_params();

      std::string input_file{};
      std::string output{};
      int n_samples_cli = 1; // --n_samples: number of longitudinal/multi-sample timepoints to load jointly

      for (int i = 1; i < argc; i++) {
          std::string argument{argv[i]};
          if (argument == "-h" || argument == "--help") { print_usage(); return 0; }
          if (argument.empty() || argument[0] != '-') {
              if (argc == 2) { input_file = argument; continue; } // SCATI <input prefix>
              throw std::invalid_argument("Unexpected argument: " + argument);
          }
          if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + argument);
          bool known = true;
          try {
              if (!parse_common_param_flag(argument, i, argc, argv, parameters, data)) {
                  // flags of this binary only
                  if      (argument == "-i")          input_file = argv[++i];
                  else if (argument == "-o")          output = argv[++i];
                  else if (argument == "--n_samples") n_samples_cli = std::stoi(argv[++i]);
                  else known = false;
              }
          } catch (const std::logic_error& e) { // stoi/stod and the value checks
              throw std::invalid_argument("Invalid value for " + argument + ": " + e.what());
          }
          if (!known) throw std::invalid_argument("Unrecognized argument: " + argument);
      }

      if (input_file.empty()) throw std::invalid_argument("No input provided. Use -i option (or --help).");

      if (output.empty()) {
          output = input_file;
          std::cout << "No output name provided. Using input basename." << std::endl;
      }

      if (parameters.burn_in == -1) {
          parameters.burn_in = parameters.chain_length / 2;
          std::cout << "No burn-in provided. Using half chain length." << std::endl;
      }
      if (parameters.n_chains < 1) throw std::invalid_argument("--n_chains must be >= 1");
      if (parameters.chain_length < 1) throw std::invalid_argument("--chain_length must be >= 1");
      if (parameters.burn_in < 0 || parameters.burn_in >= parameters.chain_length)
          throw std::invalid_argument("--burn_in must be >= 0 and < --chain_length");
      if (parameters.sample_interval_trees_parameters < 1 || parameters.update_interval_region_weights < 0)
          throw std::invalid_argument("--sample_interval_trees must be >= 1 and --update_interval_region_weights >= 0");

      if (n_samples_cli < 1) {
          throw std::invalid_argument("--n_samples must be >= 1");
      }
      // The region reference replaces the healthy-cell fit of the region parameters, so it is only used
      // without the healthy-cell annotation.
      if (!parameters.region_reference_path.empty() && parameters.use_healthy_annotation) {
          std::cout << "WARNING: the healthy-cell annotation takes precedence over --region_reference, so the "
                       "region reference is ignored. To use the reference, set --use_healthy_annotation 0."
                    << std::endl;
          parameters.region_reference_path.clear();
      }

      if (n_samples_cli == 1) {
          load_CSV(input_file);
      } else {
          // File naming convention: {input_file}_{k}_variants_filtered.csv /
          // {input_file}_{k}_regions_filtered.csv / {input_file}_{k}.cell_annotation.csv
          // for k=1..n_samples_cli. The root haplotype file (if used) stays shared and
          // un-suffixed: {input_file}_haplotypes.tsv (patient/germline-level, not
          // sample-specific) -- passed as load_CSV_multi_sample's shared_haplotype_base_name.
          std::vector<std::string> base_names(n_samples_cli);
          for (int s = 0; s < n_samples_cli; ++s)
              base_names[s] = input_file + "_" + std::to_string(s + 1);
          load_CSV_multi_sample(base_names, input_file);
      }
      if (!parameters.region_reference_path.empty()) {
          apply_region_reference(parameters.region_reference_path);
          parameters.region_reference_active = true; // cleared by the root refit at the end of burn-in
      }

      // Created after loading, so a failed load leaves no output directory
      if (!std::filesystem::exists(output)) {
          if (!std::filesystem::create_directories(output))
              throw std::runtime_error("Failed to create output directory: " + output);
          std::cout << "Directory created: " << output << std::endl;
      } else {
          std::cout << "Output directory already exists: " << output << std::endl;
      }

      const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - program_start).count();

      std::vector<double> results(parameters.n_chains);
      std::vector<std::optional<Tree>> best_trees(parameters.n_chains);
      std::vector<ChainStats> chain_stats(parameters.n_chains);

      // Each chain gets its own subdirectory for the in-progress diagnostic
      // writes (mcmc_trace.txt, and sample_parameters' periodic dumps of
      // local_params/cellAssignments/strand_genotypes)
      std::vector<std::string> chain_output_paths(parameters.n_chains);
      for (int i = 0; i < parameters.n_chains; i++) {
          chain_output_paths[i] = output + "/chain_" + std::to_string(i) + "/";
          std::filesystem::create_directories(chain_output_paths[i]);
      }

      omp_set_num_threads(std::min(parameters.n_chains, omp_get_num_procs()));
      std::cout << "Starting " << parameters.n_chains << " MCMC chains in parallel" << std::endl;

      // An exception cannot leave an OpenMP loop (std::terminate), so each chain keeps its error for after the loop
      std::vector<std::string> chain_errors(parameters.n_chains);
    #pragma omp parallel for default(none) shared(parameters, best_trees, results, chain_output_paths, chain_stats, chain_errors, std::cout)
      for (int i = 0; i < parameters.n_chains; i++) {
        try {
            Inference infer{i, parameters};
            auto t1 = std::chrono::high_resolution_clock::now();
            best_trees[i] = infer.find_best_tree(parameters.chain_length,
                                                 parameters.burn_in, chain_output_paths[i]);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::string msg = "Chain " + std::to_string(i) + " completed in "
                              + std::to_string(std::chrono::duration<double>(t2 - t1).count())
                              + " seconds.\n";
          #pragma omp critical
            { std::cout << msg; }

            results[i] = best_trees[i].value().log_score;

            // Run statistics of this chain (own file, so a job killed later still keeps finished chains)
            ChainStats& cs = chain_stats[i];
            cs.seconds = std::chrono::duration<double>(t2 - t1).count();
            cs.log_score = results[i];
            cs.n_nodes = best_trees[i].value().get_n_nodes();
            if (auto sc = infer.get_cache_scores()) {
                cs.cna_calls = sc->cna_joint_calls;
                cs.cna_misses = sc->cna_joint_misses;
                cs.cache_clears = sc->cache_clear_counts;
                cs.cache_budget_bytes = sc->get_cache_budget_bytes();
                cs.peak_cache_bytes = sc->get_peak_cache_bytes();
            }
            std::ofstream chain_file(chain_output_paths[i] + "chain_summary.json");
            chain_file << chain_stats_json(i, cs, "") << "\n";
        } catch (const std::exception& e) {
          chain_errors[i] = e.what();
        }
      }
      for (int i = 0; i < parameters.n_chains; i++)
          if (!chain_errors[i].empty())
              throw std::runtime_error("Chain " + std::to_string(i) + ": " + chain_errors[i]);

      int best_score_index = (int)(std::max_element(results.begin(), results.end()) - results.begin());

      best_trees[best_score_index].value().to_dot(output);


      // ---- run_summary.json: runtime and MCMC steps of this run ----
      {
          const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - program_start).count();
          double slowest = 0.0, sum_sps = 0.0;
          for (const ChainStats& cs : chain_stats) {
              slowest = std::max(slowest, cs.seconds);
              sum_sps += cs.seconds > 0 ? parameters.chain_length / cs.seconds : 0.0;
          }
          std::string command_line;
          for (int a = 0; a < argc; a++) command_line += (a ? " " : "") + std::string(argv[a]);
          std::time_t now_t = std::time(nullptr);
          std::ostringstream stamp;
          stamp << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S");

          // Peak resident memory of the whole process (all chains, caches, data), -1 if unavailable.
          // getrusage's ru_maxrss is in KiB on Linux/Cygwin and in bytes on macOS.
          double peak_rss_gb = -1.0;
#if defined(__linux__) || defined(__CYGWIN__)
          { struct rusage ru; if (getrusage(RUSAGE_SELF, &ru) == 0) peak_rss_gb = ru.ru_maxrss / (1024.0 * 1024.0); }
#elif defined(__APPLE__)
          { struct rusage ru; if (getrusage(RUSAGE_SELF, &ru) == 0) peak_rss_gb = ru.ru_maxrss / (1024.0 * 1024.0 * 1024.0); }
#endif

          std::ofstream summary(output + "/run_summary.json");
          summary << std::setprecision(10);
          summary << "{\n"
                  << "  \"finished_at\": \"" << stamp.str() << "\",\n"
                  << "  \"command_line\": \"" << json_escape(command_line) << "\",\n"
                  << "  \"total_wall_seconds\": " << json_num(total_seconds) << ",\n"
                  << "  \"data_loading_seconds\": " << json_num(load_seconds) << ",\n"
                  << "  \"slowest_chain_seconds\": " << json_num(slowest) << ",\n"
                  << "  \"n_chains\": " << parameters.n_chains << ",\n"
                  << "  \"mcmc_steps_per_chain\": " << parameters.chain_length << ",\n"
                  << "  \"burn_in_steps\": " << parameters.burn_in << ",\n"
                  << "  \"total_mcmc_steps\": " << (long long)parameters.n_chains * parameters.chain_length << ",\n"
                  << "  \"mean_steps_per_second_per_chain\": " << json_num(sum_sps / parameters.n_chains) << ",\n"
                  << "  \"omp_threads\": " << std::min(parameters.n_chains, omp_get_num_procs()) << ",\n"
                  << "  \"peak_rss_gb\": " << json_num(peak_rss_gb) << ",\n"
                  << "  \"n_samples\": " << data.n_samples << ",\n"
                  << "  \"n_cells\": " << n_cells << ",\n"
                  << "  \"n_loci\": " << n_loci << ",\n"
                  << "  \"n_regions\": " << n_regions << ",\n"
                  << "  \"use_doublets\": " << (parameters.use_doublets ? "true" : "false") << ",\n"
                  << "  \"use_healthy_annotation\": " << (parameters.use_healthy_annotation ? "true" : "false") << ",\n"
                  << "  \"use_haplotype_annotation\": " << (parameters.use_haplotype_annotation ? "true" : "false") << ",\n"
                  << "  \"refit_after_burnin\": " << (parameters.refit_after_burnin ? "true" : "false") << ",\n"
                  << "  \"fit_region_dispersion_mle\": " << (parameters.fit_region_dispersion_mle ? "true" : "false") << ",\n"
                  << "  \"random_seed\": " << parameters.random_seed << ",\n"
                  << "  \"sex\": \"" << data.sex << "\",\n"
                  << "  \"node_cost\": " << json_num(parameters.node_cost) << ",\n"
                  << "  \"target_doublet_rate\": " << json_num(parameters.target_doublet_rate) << ",\n"
                  << "  \"lambda_rounding_precision\": " << json_num(parameters.lambda_rounding_precision) << ",\n"
                  << "  \"hash_precision\": " << parameters.hash_precision << ",\n"
                  << "  \"cache_gb\": " << json_num(parameters.cache_gb) << ",\n"
                  << "  \"best_chain\": " << best_score_index << ",\n"
                  << "  \"best_log_score\": " << json_num(results[best_score_index]) << ",\n"
                  << "  \"chains\": [\n";
          for (int c = 0; c < parameters.n_chains; c++)
              summary << chain_stats_json(c, chain_stats[c], "    ") << (c + 1 < parameters.n_chains ? ",\n" : "\n");
          summary << "  ]\n}\n";
      }

      std::cout << "Completed! Output written to " << output << std::endl;
      // All files are written and closed. Exit without destroying the trees and their likelihood caches (several GB): freeing them takes minutes.
      std::_Exit(0);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
