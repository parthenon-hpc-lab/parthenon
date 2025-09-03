//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

// JMM: I designed this for HDF5 but in pinciple this switching could
// also work for other output types... Any output type that is capable
// of outputting a full dump can do this.
enum class DumpOutputMode { DUMP, RESTART, CORE };

//----------------------------------------------------------------------------------------
//! \struct OutputParameters
//  \brief  container for parameters read from <output> block in the input file
struct OutputParameters {
  OutputParameters() = default;

  int block_number = 0;
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
  std::string meshdata_name;
  std::vector<std::string> packages;
  Real dt = -1.0;
  int dn = -1;
  bool include_ghost_zones = false;
  bool cartesian_vector = false;
  bool single_precision_output = false;
  bool sparse_seed_nans = false;
  int hdf5_compression_level = 5;
  bool write_xdmf = false;
  bool write_swarm_xdmf = false;

  // These change after initialization, the other parameters do not.
  Real last_time;
  Real next_time = 0.0;
  int last_n;
  int next_n = 0;
  int file_number = 0;
};

} // namespace parthenon

#endif // OUTPUTS_OUTPUT_PARAMETERS_HPP_
