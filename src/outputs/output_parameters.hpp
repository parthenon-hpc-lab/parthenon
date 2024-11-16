//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#ifndef OUTPUTS_OUTPUT_PARAMETERS_HPP_
#define OUTPUTS_OUTPUT_PARAMETERS_HPP_

#include <map>
#include <set>
#include <string>
#include <vector>

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \struct OutputParameters
//  \brief  container for parameters read from <output> block in the input file
struct OutputParameters {
  int block_number;
  std::string block_name;
  std::string file_basename;
  int file_number_width;
  bool file_label_final;
  bool analysis_flag; // write this output for analysis/postprocessing restarts
  std::string file_id;
  std::vector<std::string> variables;
  std::vector<std::string> component_labels;
  std::map<std::string, std::set<std::string>> swarms;
  std::vector<std::string> swarm_vars;
  std::string file_type;
  std::string data_format;
  std::vector<std::string> packages;
  Real next_time, dt;
  int next_n, dn;
  int file_number;
  bool include_ghost_zones, cartesian_vector;
  bool single_precision_output;
  bool sparse_seed_nans;
  int hdf5_compression_level;
  bool write_xdmf;
  bool write_swarm_xdmf;
  // TODO(felker): some of the parameters in this class are not initialized in constructor
  OutputParameters()
      : block_number(0), next_time(0.0), dt(-1.0), next_n(0), dn(-1), file_number(0),
        include_ghost_zones(false), cartesian_vector(false),
        single_precision_output(false), sparse_seed_nans(false),
        hdf5_compression_level(5), write_xdmf(false), write_swarm_xdmf(false) {}
};

} // namespace parthenon

#endif // OUTPUTS_OUTPUT_PARAMETERS_HPP_
