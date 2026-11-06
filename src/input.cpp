#include<cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <cctype>
#include <stdexcept>


#include "Structures.h"
#include "input.h"

//global variables
extern int n_cells;
extern int n_loci;
extern int n_regions;
extern std::vector<Cell> cells;
extern Data data;
extern Params parameters;

// std::getline that also drops a trailing '\r' (files saved with Windows line endings)
static bool read_line(std::istream& in, std::string& line) {
    if (!std::getline(in, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

void load_CSV(std::string& base_name,
              const std::string& haplotype_base_name_override){
    // Prefer "*_filtered.csv" versions (e.g. produced by the region/variant
    // coverage filtering notebook in Experiments/preprocessing/)
    // falling back to the plain files otherwise.
    std::string variants_path = base_name + "_variants_filtered.csv";
    std::ifstream file_variants(variants_path);
    if (!file_variants.is_open()) {
        variants_path = base_name + "_variants.csv";
        file_variants.open(variants_path);
    } else {
        std::cout << "Using filtered variants file: " << variants_path << std::endl;
    }
    if(!file_variants.is_open()) throw std::runtime_error("Could not open the variants file " + base_name + "_variants.csv (or _variants_filtered.csv); relative paths start at the working directory " + std::filesystem::current_path().string());

    // Read region counts
    std::vector<std::vector<int>> region_counts{};
    data.region_to_name.clear();
    data.region_to_chromosome.clear();
    data.chromosome_names.clear();
    data.chromosome_to_regions.clear(); // else each sample of a multi-sample load appends its regions again
    data.n_chromosomes=0;

    std::string line, val;
    {
        std::string region_path = base_name + "_regions_filtered.csv";
        std::ifstream file_region(region_path);
        if (!file_region.is_open()) {
            region_path = base_name + "_regions.csv";
            file_region.open(region_path);
        } else {
            std::cout << "Using filtered regions file: " << region_path << std::endl;
        }
        if (!file_region.is_open()) throw std::runtime_error("Could not open the regions file " + base_name + "_regions.csv (or _regions_filtered.csv)");
        int region_index=0;
        bool first_line = true;
        while(read_line(file_region, line)){
            if (first_line && !line.empty() && line[0]==',') { first_line = false; continue; }
            first_line = false;
            std::stringstream ss(line);
            //index: chromosome and name of the region
            std::getline(ss, val, ',');
            int idx_split=0;
            std::vector<int> counts{};
            while (idx_split<val.size() && val[idx_split]!='_') idx_split++;
            if (idx_split<val.size()){
                // First column is chr_regionName
                data.region_to_name.push_back(val.substr(idx_split + 1,val.size()-idx_split));
                data.region_to_chromosome.push_back(val.substr(0,idx_split));
            }
            else{
                // no chromosome and region_name were given: first column corresponds to the first cell
                data.region_to_name.push_back(std::to_string(region_index));
                data.region_to_chromosome.push_back(std::to_string(region_index));
                counts.push_back(stoi(val));
            }
            while (std::getline(ss, val, ',')){
                counts.push_back(stoi(val));
            }
            region_counts.push_back(counts);
            region_index++;
        }
        n_regions = (int)region_counts.size();
        file_region.close();

        for (const std::string& chr : data.region_to_chromosome) {
            if (std::find(data.chromosome_names.begin(), data.chromosome_names.end(), chr) == data.chromosome_names.end()) {
                data.chromosome_names.push_back(chr);
            }
        }
        data.n_chromosomes = (int)data.chromosome_names.size();
        for (int i = 0; i < data.region_to_chromosome.size(); ++i) {
            const std::string& chr = data.region_to_chromosome[i];
            data.chromosome_to_regions[chr].push_back(i);
        }
    }

    // Read variants
    data.locus_to_chromosome.clear();
    data.locus_to_position.clear();
    data.locus_to_reference.clear();
    data.locus_to_alternative.clear();
    data.variant_is_SNV.clear();
    data.locus_to_region.clear();
    data.region_to_loci.clear();
    data.locus_to_name.clear();

    std::vector<std::vector<int>> ref_counts{};
    std::vector<std::vector<int>> alt_counts{};
    std::vector<std::string> cell_names{};
    std::vector<int> ref_counts_variant{};
    std::vector<int> alt_counts_variant{};
    
    // First line: header
    read_line(file_variants, line);
    std::stringstream header(line);
    std::vector<std::string> columns{};
    while (std::getline(header, val, ',')){
        columns.push_back(val);
        if (val != "CHR" && val!="POS" && val!="REF" && val!="ALT" && val!="REGION" && val!="NAME" && val!="FREQ"){
            cell_names.push_back(val);
        }
    }


    // content
    while(read_line(file_variants, line)){
        ref_counts_variant.clear();
        alt_counts_variant.clear();
        std::stringstream ss(line);
        int column_count=0;
        while (std::getline(ss, val, ',')){
            
            if (columns[column_count]=="CHR"){
                data.locus_to_chromosome.push_back(val);
            }
            else if (columns[column_count]=="POS"){
                data.locus_to_position.push_back(stoi(val));
            }
            else if (columns[column_count]=="REF"){
                data.locus_to_reference.push_back(val);
            }
            else if (columns[column_count]=="ALT"){
                data.locus_to_alternative.push_back(val);
            }
            else if (columns[column_count]=="REGION"){
                int region_index = -1;
                for (int i=0;i<data.region_to_name.size();i++){
                    if (val==data.region_to_name[i]) region_index=i;
                }
                if (region_index==-1){
                    throw std::runtime_error("One region was referenced in the variants file, but is missing in the regions file: "+val );
                }
                data.locus_to_region.push_back(region_index);
            }
            else if (columns[column_count]=="NAME"){
                data.locus_to_name.push_back(val);
            }
            else if (columns[column_count]=="FREQ"){
                // population frequency of the variant: accepted in the input but not used
            }
            else{
                // for each cell, contains RO:AD or RO:AD:GT (the genotype GT is not used)
                size_t pos = val.find(':');
                if (pos == std::string::npos)
                    throw std::runtime_error("Variants file: cell entry \"" + val + "\" in column " + columns[column_count]
                                             + " is not RO:AD");
                ref_counts_variant.push_back(stoi(val.substr(0,pos)));
                alt_counts_variant.push_back(stoi(val.substr(pos+1))); // stoi stops at ":GT"
            }
            column_count++;
        }
        ref_counts.push_back(ref_counts_variant);
        alt_counts.push_back(alt_counts_variant);
    }
    file_variants.close();
    if (ref_counts.empty()) throw std::runtime_error("Variants file " + variants_path + " has no variants");
    n_cells = (int)ref_counts[0].size();
    n_loci = (int)ref_counts.size();
    for (int i=0;i<n_loci;i++){
        if (ref_counts[i].size()!=n_cells){
            throw std::runtime_error("Not all loci had the same number of cells in the variants file: the first locus has "+
                std::to_string(n_loci)+" cells, and locus "+std::to_string(i)+" has counts for "+
                std::to_string(ref_counts[i].size()) +" cells.");
        }
    }
    for (int k=0;k<n_regions;k++){
        if (region_counts[k].size()!=n_cells){
            throw std::runtime_error("Region "+ std::to_string(k)+" has counts for "+std::to_string(region_counts[k].size())+" cells, "+
            " but the variants file had counts for "+std::to_string(n_cells)+" cells.");
        }
    }

    if ((int)data.locus_to_reference.size()==0){
        data.variant_is_SNV = std::vector<bool>(n_loci,true);
    }
    else{
        for (int i=0;i<n_loci;i++){
            data.variant_is_SNV.push_back((data.locus_to_reference[i]=="A" || data.locus_to_reference[i]=="C" 
                                            || data.locus_to_reference[i]=="G" || data.locus_to_reference[i]=="T") 
                                &&(data.locus_to_alternative[i]=="A" || data.locus_to_alternative[i]=="C" 
                                            || data.locus_to_alternative[i]=="G" || data.locus_to_alternative[i]=="T"));
        }
    }

    if ((int)data.locus_to_name.size()==0){
        if (data.locus_to_region.size()==n_loci){
            for (int i=0;i<n_loci;i++){
                data.locus_to_name.push_back(data.region_to_name[data.locus_to_region[i]]);
            }
        }
        else{
            for (int i=0;i<n_loci;i++) data.locus_to_name.push_back(std::to_string(i));
        }
    }

    if ((int)data.region_to_name.size()==0){
        for (int k=0;k<n_regions;k++) data.region_to_name.push_back(std::to_string(k));
    }

    // The variants file must say to which region (amplicon or gene) each variant belongs
    if ((int)data.locus_to_region.size()==0){
        throw std::invalid_argument("Missing region information for variants: the variants file must contain a column indicating to which region (amplicon or gene) each variant belongs.");
    }
    // Map region index to loci index
    data.region_to_loci.resize(n_regions);
    for (int i=0;i<n_loci;i++){
        data.region_to_loci[data.locus_to_region[i]].push_back(i);
    }

    // store by cell
    cells.clear();
    cells.reserve(n_cells);
    for (int j=0;j<n_cells;j++){
        cells.push_back(Cell{});
        cells[j].ref_counts.reserve(n_loci);
        cells[j].alt_counts.reserve(n_loci);
        for (int i=0;i<n_loci;i++){
            cells[j].ref_counts.push_back(ref_counts[i][j]);
            cells[j].alt_counts.push_back(alt_counts[i][j]);
        }
        cells[j].name = cell_names[j];
        int total_count=0;
        for (int i=0;i<n_regions;i++){
            cells[j].region_counts.push_back(region_counts[i][j]);
            total_count+=region_counts[i][j];
        }
        cells[j].total_counts=total_count;
    }
    if (parameters.use_healthy_annotation) {
        std::string lines;
        data.cell_is_healthy.clear();

        std::string annotation_path = base_name + ".cell_annotation.csv";
        std::ifstream file_healthy(annotation_path);
        if (!file_healthy.is_open()) {
            std::filesystem::path p(base_name);
            std::filesystem::path sibling_path = p.parent_path().parent_path()
                                                 / "cell_annotation"
                                                 / (p.filename().string() + ".cell_annotation.csv");
            file_healthy.open(sibling_path);
            if (file_healthy.is_open()) annotation_path = sibling_path.string();
        }
        if (!file_healthy.is_open())
            throw std::runtime_error("Could not open healthy annotation file (tried " + base_name
                + ".cell_annotation.csv and the sibling cell_annotation/ folder).");


        std::unordered_map<std::string, bool> barcode_to_health;

        read_line(file_healthy, lines); // skip header

        while (read_line(file_healthy, lines)) {
            if (!lines.empty()) {
                std::stringstream ss(lines);
                std::string barcode, cluster, assignment;

                std::getline(ss, barcode, ',');
                std::getline(ss, cluster, ',');
                std::getline(ss, assignment, ',');

                barcode_to_health[barcode] = (assignment == "healthy");
            }
        }
        file_healthy.close();

        data.cell_is_healthy.clear();

        int n_missing_barcodes = 0;
        for (const auto& name : cell_names) {
            auto it = barcode_to_health.find(name);

            if (it != barcode_to_health.end()) {
                data.cell_is_healthy.push_back(it->second);
            } else {
                data.cell_is_healthy.push_back(false); // default: treat unmatched barcodes as not healthy
                n_missing_barcodes++;
            }
        }

        if (n_missing_barcodes > 0) {
            std::cerr << "Warning: " << n_missing_barcodes << " out of " << cell_names.size()
                      << " cell barcodes were not found in the healthy annotation file "
                      << "(treated as not healthy) -- check that barcode formats match between files."
                      << std::endl;
        }

        bool any_healthy = false;
        for (bool v : data.cell_is_healthy) {
            if (v) {
                any_healthy = true;
                break;
            }
        }

        if (!any_healthy) {
            std::cerr << "Warning: No healthy cells found in annotation file or wrong file format. "
                       << "Region weight and dropout estimation will fall back to uninformed defaults."
                       << std::endl;
        }
    }

    if (parameters.use_haplotype_annotation) {
        const std::string& haplotype_base = haplotype_base_name_override.empty() ? base_name : haplotype_base_name_override;
        std::string haplotype_filename = haplotype_base + "_haplotypes.tsv";
        std::ifstream file_haplotypes(haplotype_filename);
        if (!file_haplotypes.is_open()) {
            throw std::runtime_error("use_haplotype_annotation is set but the haplotype file was not found: "
                                      + haplotype_filename);
        }

        auto parse_2d = [](const std::string& value) {
            std::vector<std::vector<int>> result;
            std::stringstream strand_ss(value);
            std::string strand_str;
            while (std::getline(strand_ss, strand_str, ';')) {
                std::vector<int> values;
                std::stringstream val_ss(strand_str);
                std::string val_str;
                while (std::getline(val_ss, val_str, ',')) {
                    values.push_back(std::stoi(val_str));
                }
                result.push_back(values);
            }
            return result;
        };

        std::unordered_map<std::string, std::vector<std::vector<int>>> haplotype_data;
        std::string hap_line;
        while (read_line(file_haplotypes, hap_line)) {
            if (hap_line.empty()) continue;
            size_t tab_pos = hap_line.find('\t');
            if (tab_pos == std::string::npos) {
                throw std::runtime_error("Malformed line in haplotype file (missing tab): " + hap_line);
            }
            haplotype_data[hap_line.substr(0, tab_pos)] = parse_2d(hap_line.substr(tab_pos + 1));
        }
        file_haplotypes.close();

        for (const std::string& key : {"root_strands_regions", "root_strands_loci_ref", "root_strands_loci_alt"}) {
            if (haplotype_data.find(key) == haplotype_data.end()) {
                throw std::runtime_error("Haplotype file " + haplotype_filename + " is missing required row: " + key);
            }
        }

        data.root_strands_regions  = haplotype_data["root_strands_regions"];
        data.root_strands_loci_ref = haplotype_data["root_strands_loci_ref"];
        data.root_strands_loci_alt = haplotype_data["root_strands_loci_alt"];

        int n_strands = (int)data.root_strands_regions.size();
        if (n_strands == 0) throw std::runtime_error("Haplotype file " + haplotype_filename + " defines zero strands.");
        if ((int)data.root_strands_loci_ref.size() != n_strands || (int)data.root_strands_loci_alt.size() != n_strands) {
            throw std::runtime_error("Haplotype file " + haplotype_filename
                + ": root_strands_regions, root_strands_loci_ref and root_strands_loci_alt must all define the same number of strands.");
        }
        for (int s = 0; s < n_strands; ++s) {
            if ((int)data.root_strands_regions[s].size() != n_regions) {
                throw std::runtime_error("Haplotype file " + haplotype_filename + ": strand " + std::to_string(s)
                    + " of root_strands_regions has " + std::to_string(data.root_strands_regions[s].size())
                    + " values, expected n_regions=" + std::to_string(n_regions));
            }
            if ((int)data.root_strands_loci_ref[s].size() != n_loci) {
                throw std::runtime_error("Haplotype file " + haplotype_filename + ": strand " + std::to_string(s)
                    + " of root_strands_loci_ref has " + std::to_string(data.root_strands_loci_ref[s].size())
                    + " values, expected n_loci=" + std::to_string(n_loci));
            }
            if ((int)data.root_strands_loci_alt[s].size() != n_loci) {
                throw std::runtime_error("Haplotype file " + haplotype_filename + ": strand " + std::to_string(s)
                    + " of root_strands_loci_alt has " + std::to_string(data.root_strands_loci_alt[s].size())
                    + " values, expected n_loci=" + std::to_string(n_loci));
            }
            for (int v : data.root_strands_regions[s])
                if (v < 0) throw std::runtime_error("Haplotype file " + haplotype_filename
                    + ": negative copy number in root_strands_regions, strand " + std::to_string(s));
            for (int v : data.root_strands_loci_ref[s])
                if (v < 0) throw std::runtime_error("Haplotype file " + haplotype_filename
                    + ": negative value in root_strands_loci_ref, strand " + std::to_string(s));
            for (int v : data.root_strands_loci_alt[s])
                if (v < 0) throw std::runtime_error("Haplotype file " + haplotype_filename
                    + ": negative value in root_strands_loci_alt, strand " + std::to_string(s));
        }

        bool is_male = (data.sex == "male");
        for (int k = 0; k < n_regions; ++k) {
            const std::string& chr = data.region_to_chromosome[k];
            bool is_X = (chr == "X" || chr == "chrX");
            bool is_Y = (chr == "Y" || chr == "chrY");
            if (!is_X && !is_Y) continue;
            int total_copies = 0;
            for (int s = 0; s < n_strands; ++s) total_copies += data.root_strands_regions[s][k];
            int expected = is_Y ? (is_male ? 1 : 0) : (is_male ? 1 : 2); // X
            if (total_copies != expected) {
                std::cerr << "Warning: region " << data.region_to_name[k] << " on chromosome " << chr
                          << " has " << total_copies << " copies in the haplotype file, but " << expected
                          << " would be expected for sex=\"" << data.sex << "\". Using the file's value."
                          << std::endl;
            }

            for (int s = 0; s < n_strands; ++s) {
                if (data.root_strands_regions[s][k] != 0) continue;
                for (int locus : data.region_to_loci[k]) {
                    if (data.root_strands_loci_ref[s][locus] != 0 || data.root_strands_loci_alt[s][locus] != 0) {
                        std::cerr << "Warning: locus " << locus << " in region " << data.region_to_name[k]
                                  << " has nonzero ref/alt on strand " << s << ", which has zero copies of "
                                  << "this region in the haplotype file. Using the file's value."
                                  << std::endl;
                    }
                }
            }
        }
    }
    else {
        data.root_strands_loci_alt = std::vector<std::vector<int>>(2, std::vector<int>(n_loci, 0)); // default two strands with no mutations
        data.root_strands_loci_ref = std::vector<std::vector<int>>(2, std::vector<int>(n_loci, 1)); // default two strands with 1 ref allele each
        data.root_strands_regions = std::vector<std::vector<int>>(2, std::vector<int>(n_regions, 1)); // default two strands with all regions present once per strand

        bool is_male = (data.sex == "male");
        // Zeroing out a strand's region copy (below) must also zero that same
        // strand's own loci on that region -- otherwise a locus keeps its
        // default ref=1 on a strand that no longer exists, so n_ref_allele
        // sums to 2 for a hemizygous locus instead of the correct 1.
        auto zero_strand_loci = [&](int strand, int region) {
            for (int locus : data.region_to_loci[region]) {
                data.root_strands_loci_ref[strand][locus] = 0;
                data.root_strands_loci_alt[strand][locus] = 0;
            }
        };
        for (int k = 0; k < n_regions; ++k) {
            const std::string& chr = data.region_to_chromosome[k];
            bool is_X = (chr == "X" || chr == "chrX");
            bool is_Y = (chr == "Y" || chr == "chrY");
            if (is_Y) {
                data.root_strands_regions[0][k] = is_male ? 1 : 0;
                data.root_strands_regions[1][k] = 0;
                zero_strand_loci(1, k);
                if (!is_male) zero_strand_loci(0, k); // no Y chromosome in females
            } else if (is_X && is_male) {
                data.root_strands_regions[0][k] = 1;
                data.root_strands_regions[1][k] = 0;
                zero_strand_loci(1, k);
            }
            // female X and all autosomes keep the default 1/1 (2 copies)
        }
    }

    data.n_samples = 1;
    data.cells_by_sample.assign(1, {});
    data.cells_by_sample[0].resize(n_cells);
    for (int j = 0; j < n_cells; ++j) data.cells_by_sample[0][j] = j;
}


void load_CSV_multi_sample(const std::vector<std::string>& base_names,
                            const std::string& shared_haplotype_base_name) {
    int n_samples = (int)base_names.size();
    if (n_samples < 1) throw std::runtime_error("load_CSV_multi_sample requires at least one sample");

    std::vector<std::vector<Cell>> cells_per_sample(n_samples);
    std::vector<std::vector<bool>> healthy_per_sample(n_samples);

    // Reference panel snapshot (from sample 1), used to validate every other sample's
    // panel matches exactly. Multi-sample support requires an identical
    // locus/region panel across all samples of a patient.
    int ref_n_loci = -1, ref_n_regions = -1;
    std::vector<std::string> ref_locus_chr, ref_locus_ref, ref_locus_alt;
    std::vector<int> ref_locus_pos;
    std::vector<std::string> ref_region_names, ref_region_chr;

    for (int s = 0; s < n_samples; ++s) {
        std::string base_name_s = base_names[s]; // load_CSV takes a mutable reference
        load_CSV(base_name_s, shared_haplotype_base_name);

        if (s == 0) {
            ref_n_loci    = n_loci;
            ref_n_regions = n_regions;
            ref_locus_chr = data.locus_to_chromosome;
            ref_locus_pos = data.locus_to_position;
            ref_locus_ref = data.locus_to_reference;
            ref_locus_alt = data.locus_to_alternative;
            ref_region_names = data.region_to_name;
            ref_region_chr   = data.region_to_chromosome;
        } else {
            if (n_loci != ref_n_loci || n_regions != ref_n_regions) {
                throw std::runtime_error("Sample " + std::to_string(s + 1) + " (" + base_names[s]
                    + ") has a different panel size than sample 1: " + std::to_string(n_loci) + " loci / "
                    + std::to_string(n_regions) + " regions, vs " + std::to_string(ref_n_loci) + " / "
                    + std::to_string(ref_n_regions) + " for sample 1. All samples must use the identical panel.");
            }
            // POS/REF/ALT are optional variants.csv columns (load_CSV leaves their
            // vectors empty when absent, e.g. simulated data): compare only the
            // columns present, and require every sample to have the same ones.
            if (data.locus_to_chromosome.size() != ref_locus_chr.size() || data.locus_to_position.size() != ref_locus_pos.size()
                || data.locus_to_reference.size() != ref_locus_ref.size() || data.locus_to_alternative.size() != ref_locus_alt.size()) {
                throw std::runtime_error("Sample " + std::to_string(s + 1) + " (" + base_names[s] + ") has different "
                    "CHR/POS/REF/ALT columns than sample 1. All samples must use the identical variant panel.");
            }
            auto differs = [](const auto& a, const auto& b, int i) {
                return i < (int)a.size() && a[i] != b[i];  // sizes already checked equal
            };
            for (int i = 0; i < n_loci; ++i) {
                if (differs(data.locus_to_chromosome, ref_locus_chr, i) || differs(data.locus_to_position, ref_locus_pos, i)
                    || differs(data.locus_to_reference, ref_locus_ref, i) || differs(data.locus_to_alternative, ref_locus_alt, i)) {
                    throw std::runtime_error("Sample " + std::to_string(s + 1) + " (" + base_names[s] + ") locus "
                        + std::to_string(i) + " (CHR/POS/REF/ALT) does not match sample 1's panel. All samples must "
                        "use the identical variant panel, in the same order.");
                }
            }
            for (int k = 0; k < n_regions; ++k) {
                if (data.region_to_name[k] != ref_region_names[k] || data.region_to_chromosome[k] != ref_region_chr[k]) {
                    throw std::runtime_error("Sample " + std::to_string(s + 1) + " (" + base_names[s] + ") region "
                        + std::to_string(k) + " does not match sample 1's panel. All samples must use the identical "
                        "region panel, in the same order.");
                }
            }
        }

        cells_per_sample[s] = cells;
        healthy_per_sample[s] = data.cell_is_healthy;

        int n_healthy = 0;
        for (bool h : healthy_per_sample[s]) if (h) n_healthy++;
        if (parameters.use_healthy_annotation && n_healthy == 0) {
            std::cerr << "Warning: sample " << (s + 1) << " (" << base_names[s]
                      << ") has no healthy cells -- its own contribution to region-weight/dropout "
                      << "fitting will fall back to uninformed defaults for this sample." << std::endl;
        }
    }

    cells.clear();
    data.cell_is_healthy.clear();
    data.cells_by_sample.assign(n_samples, {});
    int running_index = 0;
    for (int s = 0; s < n_samples; ++s) {
        for (Cell& c : cells_per_sample[s]) {
            c.sample_id = s;
            data.cells_by_sample[s].push_back(running_index++);
            cells.push_back(std::move(c));
        }
        data.cell_is_healthy.insert(data.cell_is_healthy.end(),
                                     healthy_per_sample[s].begin(), healthy_per_sample[s].end());
    }
    n_cells = (int)cells.size();
    data.n_samples = n_samples;

    std::cout << "Loaded " << n_samples << " samples: " << n_cells << " cells total ("
               << n_loci << " loci, " << n_regions << " regions)." << std::endl;
}


// --region_reference: reads a per-amplicon reference (built from a cohort's healthy cells by
// Experiments/preprocessing/build_region_reference.py)
void apply_region_reference(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Could not open region reference file: " + path);
    std::unordered_map<std::string, std::pair<double, double>> ref; // amplicon -> (rho_per_copy, phi)
    double ref_dropout = -1.0;
    std::string line;
    std::vector<std::string> header;
    while (read_line(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            std::stringstream ss(line.substr(1));
            std::string key, val;
            ss >> key >> val;
            if (key == "dropout" && !val.empty()) ref_dropout = std::stod(val);
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, '\t')) fields.push_back(tok);
        if (header.empty()) { header = fields; continue; }
        auto col = [&](const std::string& name) {
            auto it = std::find(header.begin(), header.end(), name);
            if (it == header.end()) throw std::runtime_error("Region reference " + path + " lacks column " + name);
            return (size_t)(it - header.begin());
        };
        size_t ca = col("amplicon"), cr = col("rho_per_copy"), cp = col("phi");
        ref[fields.at(ca)] = {std::stod(fields.at(cr)), std::stod(fields.at(cp))};
    }
    if (ref.empty()) throw std::runtime_error("Region reference " + path + " has no regions");

    // Regions to keep, and the old -> new index maps for regions and loci
    std::vector<int> new_region(n_regions, -1);
    std::vector<int> kept_regions;
    for (int k = 0; k < n_regions; ++k) {
        const std::string& name = data.region_to_name[k];
        if (ref.count(name.substr(0, name.find('_')))) {
            new_region[k] = (int)kept_regions.size();
            kept_regions.push_back(k);
        }
    }
    if (kept_regions.empty()) throw std::runtime_error("No input region matches the region reference " + path);
    std::vector<int> kept_loci;
    for (int i = 0; i < n_loci; ++i)
        if (new_region[data.locus_to_region[i]] >= 0) kept_loci.push_back(i);

    auto keep = [](auto& v, const std::vector<int>& idx, int n_old) {
        if ((int)v.size() != n_old) return; // optional per-locus fields may be empty
        std::remove_reference_t<decltype(v)> out;
        out.reserve(idx.size());
        for (int i : idx) out.push_back(v[i]);
        v = std::move(out);
    };
    const int n_regions_old = n_regions, n_loci_old = n_loci;

    for (Cell& c : cells) {
        keep(c.region_counts, kept_regions, n_regions_old);
        keep(c.ref_counts, kept_loci, n_loci_old);
        keep(c.alt_counts, kept_loci, n_loci_old);
        c.total_counts = 0;
        for (int v : c.region_counts) c.total_counts += v;
    }
    keep(data.region_to_name, kept_regions, n_regions_old);
    keep(data.region_to_chromosome, kept_regions, n_regions_old);
    for (auto& s : data.root_strands_regions) keep(s, kept_regions, n_regions_old);
    for (auto& s : data.root_strands_loci_ref) keep(s, kept_loci, n_loci_old);
    for (auto& s : data.root_strands_loci_alt) keep(s, kept_loci, n_loci_old);
    keep(data.locus_to_chromosome, kept_loci, n_loci_old);
    keep(data.locus_to_position, kept_loci, n_loci_old);
    keep(data.locus_to_reference, kept_loci, n_loci_old);
    keep(data.locus_to_alternative, kept_loci, n_loci_old);
    keep(data.variant_is_SNV, kept_loci, n_loci_old);
    keep(data.locus_to_name, kept_loci, n_loci_old);
    keep(data.locus_to_region, kept_loci, n_loci_old);
    for (int& r : data.locus_to_region) r = new_region[r];
    n_regions = (int)kept_regions.size();
    n_loci = (int)kept_loci.size();

    data.region_to_loci.assign(n_regions, {});
    for (int i = 0; i < n_loci; ++i) data.region_to_loci[data.locus_to_region[i]].push_back(i);
    data.chromosome_names.clear();
    data.chromosome_to_regions.clear();
    for (int k = 0; k < n_regions; ++k) {
        const std::string& chr = data.region_to_chromosome[k];
        if (!data.chromosome_to_regions.count(chr)) data.chromosome_names.push_back(chr);
        data.chromosome_to_regions[chr].push_back(k);
    }
    data.n_chromosomes = (int)data.chromosome_names.size();

    data.reference_rho_per_copy.assign(n_regions, 0.0);
    data.reference_phi.assign(n_regions, 0.0);
    for (int k = 0; k < n_regions; ++k) {
        const std::string& name = data.region_to_name[k];
        const auto& e = ref.at(name.substr(0, name.find('_')));
        data.reference_rho_per_copy[k] = e.first;
        data.reference_phi[k] = e.second;
    }
    data.reference_dropout = ref_dropout;
    data.has_region_reference = true;
    std::cout << "Region reference " << path << ": kept " << n_regions << " of " << n_regions_old
              << " regions (" << (n_regions_old - n_regions) << " not in the reference dropped) and "
              << n_loci << " of " << n_loci_old << " loci (" << (n_loci_old - n_loci)
              << " on dropped regions dropped)." << std::endl;
}


void init_params() {
    // Constant parameters
    parameters.random_seed = 42;
    parameters.max_copy_number = 10; // maximum copy number allowed in the tree
    parameters.excess_cn_cost  = 1.0;  // penalty per extra copy above copy number 4
    parameters.allelic_error_rate=0.02; // eps: VAF at a homozygous reference locus (1 - eps at homozygous alternative).
    parameters.allelic_error_rate_indel=0.02; // indels: allelic calls are less reliable
    parameters.mu_dir_multi = 500.0;  // starting region_dispersion (per root copy) before it is fitted
    parameters.theta_negbinom = 5.0; // negative binomial size of the region counts (use_joint_likelihood_model = false)
    parameters.background_rate = 0.04; // MLE fit on real CN 0 regions of 18 samples (0.0375)
    parameters.hash_precision = 1000; // discretization of likelihood cache keys: higher is more exact but uses more memory
    parameters.region_reference_path = ""; // --region_reference <file>: see apply_region_reference()
    parameters.lambda_rounding_precision = 0.01; // rounding step of a node's Lambda (sum_w_cn), doubled for doublets:
                                                 // smaller is more exact but gets fewer cache hits
    parameters.cache_gb = 4.0; // memory budget of the CNA likelihood caches per chain and sample (total ~ n_chains * n_samples * cache_gb)

    // Not used in the current likelihood model, but kept used if the joint likelihood model is disabled (use_joint_likelihood_model = false)
    parameters.omega_hom = 20.0; // beta-binomial concentration of the VAF at homozygous loci
    parameters.omega_het = 8.0;  // beta-binomial concentration of the VAF at heterozygous loci
    parameters.omega_hom_indel = 15.0;
    parameters.omega_het_indel = 4.0;

    // Priors
    parameters.dropoutrate = 0.03; // per-copy dropout rate: Beta prior mean, start of the fit, and fallback with too few healthy cells
    parameters.prior_dropoutrate_omega = 1000; // concentration of that Beta prior
    parameters.fit_region_dispersion_mle = false; // true: region_dispersion by MLE; false: by method of moments
    parameters.fit_dropout_mle = true; // fit the dropout rate by MLE whenever the region parameters are fitted (before MCMC and at refits during burn-in)
    parameters.mean_total_count = 25000; // starting mean total read count of a root cell; fitted per sample with the region parameters
    parameters.theta_total_count = 7; // starting negative binomial size of the total read count; fitted with mean_total_count

    // Penalty weights
    parameters.node_cost = 500.0; // higher: fewer nodes
    parameters.cn_neutral_node_cost_factor = 1.0; // node_cost factor for nodes without a CNA (SNVs/CNLOH only); 1: no discount, since without
                                                  // a healthy-cell annotation a discounted SNV-only node lets the root swap with a founder clone
    parameters.desired_n_nodes = -1; // -1: no target number of nodes
    parameters.target_doublet_rate = 0.08; // doublet probability in the cell likelihood and mean of the doublet-fraction prior
    parameters.target_doublet_rate_omega = 10000.0; // concentration of the doublet-fraction Beta prior
    parameters.alt_to_ref_cost = 500.0; // per SNV from alt to ref
    parameters.ref_to_alt_cost = 50.0; // per SNV from ref to alt
    parameters.Loss_cost = 500.0; // per local loss (an aneuploidy costs 0.8x)
    parameters.Gain_cost = 500.0; // per local gain (an aneuploidy costs 0.8x)
    parameters.CNLOH_cost = 500.0; // per CNLOH
    parameters.FGD_cost = 2500.0; // per whole genome duplication
    parameters.healthy_cell_attachment_cost = 5.0; // for healthy cells attached to non-root nodes
    parameters.healthy_heterogeneity_credit = 0.8; // fraction of a healthy cell's attachment to a CNA-free child of the root
                                                   // that counts as attached to the root
    parameters.cancer_cell_at_root_cost = 3.0; // for cancer cells attached to the root
    parameters.FGD_count_exponent = 1.5; // n FGDs cost FGD_cost * (n + n^FGD_count_exponent); other events cost a flat amount

    // Move probabilities
    parameters.new_node_prob = 0.2; // moving an SNV or CNA: probability of moving it to a new node
    parameters.merge_probability = 0.4; // split/merge node: probability of merging
    parameters.add_SNV_probability = 0.5; // add/remove SNV: probability of adding
    parameters.merge_CNA_probability = 0.9; // split/merge CNAs: probability of merging two adjacent CNAs
    parameters.add_CNA_probability = 0.7; // add/remove CNA: probability of adding
    parameters.aneuploidy_probability = 0.1; // probability that a new CNA is a chromosome gain/loss
    parameters.fgd_probability = 0.01; // probability that a new CNA is a whole genome duplication
    parameters.convert_aneuploidy_probability = 0.3; // change CNA types: probability of converting aneuploidy <-> 2 local CNAs
    parameters.merge_aneuploidy_probability = 0.5; // within that conversion, probability of merging

    // Optimization parameters
    parameters.n_chains = 4;
    parameters.chain_length = 100000;
    parameters.burn_in = -1;
    parameters.temperature = 10; // only used for simulated annealing
    parameters.optimizer = "MCMC"; // other option: "simulated_annealing"
    parameters.sampling = false; // also write the current tree and parameters every sample_interval_trees_parameters steps
    parameters.node_probabilities_em_steps = 1; // number of EM iterations for node probabilities
    parameters.per_sample_node_probabilities = true; // n_samples > 1: fit node probabilities per sample
    parameters.sample_interval_trees_parameters = 500;
    parameters.update_interval_region_weights = 1000; // steps between the soft-root-attachment refits during burn-in (use_healthy_annotation = false)
    parameters.min_cells_for_parameter_refit = 100; // such a refit is skipped (previous values kept) with fewer effective root cells
    parameters.min_root_attachment_fraction = 0.05; // use_healthy_annotation = false: reject a tree whose mean
                                                    // root-attachment probability falls below this
    parameters.use_joint_likelihood_model = true; // true: SNV and region likelihoods share the fitted per-region dispersion and
                                                  // dropout; false: fixed-dispersion negative binomial for regions and per-locus SNV dropout
    parameters.use_total_count_likelihood = true; // likelihood of a cell's total read count given its total copy number
    parameters.use_constant_dispersion = false; // true: alpha_total kept at its root value instead of scaling with the node's Lambda (see Node.cpp)

    // Running options
    parameters.use_doublets = true;
    parameters.refit_after_burnin = true; // with use_healthy_annotation: refit the region parameters once from the root cells at the
                                           // end of burn-in and stop using the labels (corrects partly wrong labels)
    parameters.use_healthy_annotation = true; // use the healthy-cell labels (<input>.cell_annotation.csv) for the root genotype, the
                                              // region parameter fit and the healthy-attachment penalty; false: the root starts
                                              // homozygous reference and the region parameters are fit from the cells' soft root
                                              // attachment, refitted every update_interval_region_weights steps during burn-in
    parameters.use_haplotype_annotation = false; // root genotype read from <input>_haplotypes.tsv and kept fixed
    parameters.verbose = false; // print the step and the chosen move at every MCMC step

    // Dataset characteristics
    data.sex = "female"; // chrX and chrY copy numbers of the root
}

bool parse_common_param_flag(const std::string& argument, int& i, int argc, char* argv[],
                              Params& parameters, Data& data) {
    auto parse_bool = [&](int& idx) -> bool {
        if (idx + 1 >= argc) throw std::invalid_argument("Missing value for argument");
        std::string val = argv[++idx];
        std::transform(val.begin(), val.end(), val.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (val == "1" || val == "true")  return true;
        if (val == "0" || val == "false") return false;
        throw std::invalid_argument("Invalid boolean value \"" + val + "\" -- expected one of: 0, 1, true, false");
    };

    // --- Constant parameters ---
    if      (argument == "--random_seed")            parameters.random_seed = std::stoi(argv[++i]);
    else if (argument == "--max_copy_number")        parameters.max_copy_number = std::stoi(argv[++i]);
    else if (argument == "--lambda_rounding_precision") {
        parameters.lambda_rounding_precision = std::stod(argv[++i]);
        if (!(parameters.lambda_rounding_precision > 0.0))
            throw std::invalid_argument("--lambda_rounding_precision must be > 0");
    }
    else if (argument == "--cache_gb") {
        parameters.cache_gb = std::stod(argv[++i]);
        if (!(parameters.cache_gb > 0.0))
            throw std::invalid_argument("--cache_gb must be > 0");
    }
    else if (argument == "--allelic_error_rate")     parameters.allelic_error_rate = std::stod(argv[++i]);
    else if (argument == "--allelic_error_rate_indel") parameters.allelic_error_rate_indel = std::stod(argv[++i]);
    else if (argument == "--omega_hom")              parameters.omega_hom = std::stod(argv[++i]);
    else if (argument == "--omega_het")              parameters.omega_het = std::stod(argv[++i]);
    else if (argument == "--omega_hom_indel")        parameters.omega_hom_indel = std::stod(argv[++i]);
    else if (argument == "--omega_het_indel")        parameters.omega_het_indel = std::stod(argv[++i]);
    else if (argument == "--mu_dir_multi")           parameters.mu_dir_multi = std::stod(argv[++i]);
    else if (argument == "--sex") {
        data.sex = std::string(argv[++i]);
        std::transform(data.sex.begin(), data.sex.end(), data.sex.begin(), ::tolower);
        if (data.sex != "male" && data.sex != "female")
            throw std::runtime_error("--sex must be male or female, got " + std::string(argv[i]));
    }

        // --- Priors ---
    else if (argument == "--dropoutrate")            parameters.dropoutrate = std::stod(argv[++i]);
    else if (argument == "--prior_dropoutrate_omega")parameters.prior_dropoutrate_omega = std::stod(argv[++i]);

        // --- Penalty weights ---
    else if (argument == "--node_cost")              parameters.node_cost = std::stod(argv[++i]);
    else if (argument == "--cn_neutral_node_cost_factor") {
        parameters.cn_neutral_node_cost_factor = std::stod(argv[++i]);
        if (!(parameters.cn_neutral_node_cost_factor >= 0.0))
            throw std::invalid_argument("--cn_neutral_node_cost_factor must be >= 0");
    }
    else if (argument == "--desired_n_nodes")        parameters.desired_n_nodes = std::stoi(argv[++i]);
    else if (argument == "--target_doublet_rate")    parameters.target_doublet_rate = std::stod(argv[++i]);
    else if (argument == "--target_doublet_rate_omega") parameters.target_doublet_rate_omega = std::stod(argv[++i]);
    else if (argument == "--alt_to_ref_cost")        parameters.alt_to_ref_cost = std::stod(argv[++i]);
    else if (argument == "--ref_to_alt_cost")        parameters.ref_to_alt_cost = std::stod(argv[++i]);
    else if (argument == "--CNLOH_cost")             parameters.CNLOH_cost = std::stod(argv[++i]);
    else if (argument == "--Loss_cost")              parameters.Loss_cost = std::stod(argv[++i]);
    else if (argument == "--Gain_cost")              parameters.Gain_cost = std::stod(argv[++i]);
    else if (argument == "--FGD_cost")               parameters.FGD_cost = std::stod(argv[++i]);

        // --- Optimization ---
    else if (argument == "--theta_negbinom")         parameters.theta_negbinom = std::stod(argv[++i]);
    else if (argument == "--n_chains")               parameters.n_chains = std::stoi(argv[++i]);
    else if (argument == "--chain_length")           parameters.chain_length = std::stoi(argv[++i]);
    else if (argument == "--burn_in")                parameters.burn_in = std::stoi(argv[++i]);
    else if (argument == "--temperature")            parameters.temperature = std::stod(argv[++i]);
    else if (argument == "--optimizer")              parameters.optimizer = std::string(argv[++i]);
    else if (argument == "--sample_interval_trees")  parameters.sample_interval_trees_parameters = std::stoi(argv[++i]);
    else if (argument == "--update_interval_region_weights") parameters.update_interval_region_weights = std::stoi(argv[++i]);
    else if (argument == "--min_cells_for_parameter_refit") parameters.min_cells_for_parameter_refit = std::stoi(argv[++i]);
    else if (argument == "--min_root_attachment_fraction") parameters.min_root_attachment_fraction = std::stod(argv[++i]);
    else if (argument == "--node_probabilities_em_steps") parameters.node_probabilities_em_steps = std::stoi(argv[++i]);

        // --- Boolean flags ---
    else if (argument == "--sampling")                   parameters.sampling = parse_bool(i);
    else if (argument == "--refit_after_burnin")         parameters.refit_after_burnin = parse_bool(i);
    else if (argument == "--use_doublets")               parameters.use_doublets = parse_bool(i);
    else if (argument == "--use_constant_dispersion")    parameters.use_constant_dispersion = parse_bool(i);
    else if (argument == "--use_joint_likelihood_model") parameters.use_joint_likelihood_model = parse_bool(i);
    else if (argument == "--use_healthy_annotation")     parameters.use_healthy_annotation = parse_bool(i);
    else if (argument == "--use_haplotype_annotation")   parameters.use_haplotype_annotation = parse_bool(i);
    else if (argument == "--region_reference")           parameters.region_reference_path = argv[++i];
    else if (argument == "--verbose")                    parameters.verbose = parse_bool(i);
    else if (argument == "--fit_region_dispersion_mle")  parameters.fit_region_dispersion_mle = parse_bool(i);
    else if (argument == "--fit_dropout_mle")            parameters.fit_dropout_mle = parse_bool(i);
    else if (argument == "--per_sample_node_probabilities") parameters.per_sample_node_probabilities = parse_bool(i);

    else return false;
    return true;
}

