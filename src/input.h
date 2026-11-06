#ifndef DEF_INPUT
#define DEF_INPUT

#include <string>
#include <vector>
#include "Structures.h"

void load_CSV(std::string& base_name,
              const std::string& haplotype_base_name_override = "");
void load_CSV_multi_sample(const std::vector<std::string>& base_names,
                           const std::string& shared_haplotype_base_name = "");
void filter_regions();
void apply_region_reference(const std::string& path);
void init_params();
bool parse_common_param_flag(const std::string& argument, int& i, int argc, char* argv[],
                              Params& parameters, Data& data);

#endif