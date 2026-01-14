//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2026 The Parthenon collaboration
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
#ifndef INTERFACE_OUTPUTS_HPP_
#define INTERFACE_OUTPUTS_HPP_
//! \file outputs.hpp
//  \brief provides base classes to handle ALL types of data output

#include <map>
#include <set>
#include <string>
#include <vector>

#include "basic_types.hpp"

namespace parthenon {

// forward declarations
class Mesh;
class ParameterInput;

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
  bool include_in_final;
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

//----------------------------------------------------------------------------------------
//  \brief abstract base class for different output types (modes/formats). Each OutputType
//  is designed to be a node in a singly linked list created & stored in the Outputs class

class OutputType {
 public:
  // mark single parameter constructors as "explicit" to prevent them from acting as
  // implicit conversion functions: for f(OutputType arg), prevent f(anOutputParameters)
  explicit OutputType(OutputParameters oparams);

  // rule of five:
  virtual ~OutputType() = default;
  // copied)
  OutputType(const OutputType &copy_other) = default;
  OutputType &operator=(const OutputType &copy_other) = default;
  // move constructor and assignment operator
  OutputType(OutputType &&) = default;
  OutputType &operator=(OutputType &&) = default;

  // data
  OutputParameters output_params; // control data read from <output> block

  // following pure virtual function must be implemented in all derived classes
  virtual void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                               const SignalHandler::OutputSignal signal) = 0;
  virtual void WriteContainer(SimTime &tm, Mesh *pm, ParameterInput *pin, bool flag) {
    return;
  }

 protected:
  int num_vars_; // number of variables in output

  // Update book-keeping such as next output time to next output
  void UpdateNextOutput_(Mesh *pm, SimTime *tm);
};

} // namespace parthenon
#endif // INTERFACE_OUTPUTS_HPP_