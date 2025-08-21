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
#ifndef OUTPUTS_OUTPUTS_HPP_
#define OUTPUTS_OUTPUTS_HPP_
//! \file outputs.hpp
//  \brief provides classes to handle ALL types of data output

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Kokkos_ScatterView.hpp"

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "io_wrapper.hpp"
#include "kokkos_abstraction.hpp"
#include "outputs/output_parameters.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// forward declarations
class Mesh;
class ParameterInput;

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

//----------------------------------------------------------------------------------------
// Helper definitions to enroll user output variables

// Function signature for currently supported user output functions
using HstFun_t = std::function<Real(MeshData<Real> *md)>;
using HstVecFun_t = std::function<std::vector<Real>(MeshData<Real> *md)>;

// Container
struct HistoryOutputVar {
  UserHistoryOperation hst_op; // Reduction operation
  HstFun_t hst_fun;            // Function to be called
  std::string label;           // column label in hst output file
  HistoryOutputVar(const UserHistoryOperation &hst_op_, const HstFun_t &hst_fun_,
                   const std::string &label_)
      : hst_op(hst_op_), hst_fun(hst_fun_), label(label_) {}
};

struct HistoryOutputVec {
  UserHistoryOperation hst_op;
  HstVecFun_t hst_vec_fun;
  std::string label;
  HistoryOutputVec(const UserHistoryOperation &hst_op_, const HstVecFun_t &hst_vec_fun_,
                   const std::string &label_)
      : hst_op(hst_op_), hst_vec_fun(hst_vec_fun_), label(label_) {}
};

using HstVar_list = std::vector<HistoryOutputVar>;
using HstVec_list = std::vector<HistoryOutputVec>;
// Hardcoded global entry to be used by each package to enroll user output functions
const char hist_param_key[] = "HistoryFunctions";
const char hist_vec_param_key[] = "HistoryVectorFunctions";

//----------------------------------------------------------------------------------------
//! \class HistoryFile
//  \brief derived OutputType class for history dumps

class HistoryOutput : public OutputType {
 public:
  explicit HistoryOutput(const OutputParameters &oparams) : OutputType(oparams) {}
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                       const SignalHandler::OutputSignal signal) override;
};

//----------------------------------------------------------------------------------------
//! \class AscentOutput
//  \brief derived OutputType class for Ascent in situ situ visualization and analysis

class AscentOutput : public OutputType {
 public:
  explicit AscentOutput(const OutputParameters &oparams) : OutputType(oparams) {}
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                       const SignalHandler::OutputSignal signal) override;

 private:
  //  Ghost mask currently (Ascent 0.9) needs to be of float type on device as the
  //  automated conversion between int and float does not work
  ParArray1D<Real> ghost_mask_;
};

#ifdef ENABLE_HDF5
//----------------------------------------------------------------------------------------
//! \class PHDF5Output
//  \brief derived OutputType class for Athena HDF5 files or restart dumps

class PHDF5Output : public OutputType {
 public:
  // Function declarations
  PHDF5Output(const OutputParameters &oparams, DumpOutputMode mode)
      : OutputType(oparams), mode_(mode) {}
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                       const SignalHandler::OutputSignal signal) override;
  template <bool WRITE_SINGLE_PRECISION>
  void WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, SimTime *tm,
                           const SignalHandler::OutputSignal signal);

 private:
  std::string GenerateFilename_(ParameterInput *pin, SimTime *tm,
                                const SignalHandler::OutputSignal signal);
  void WriteBlocksMetadata_(Mesh *pm, hid_t file, const HDF5::H5P &pl, hsize_t offset,
                            hsize_t max_blocks_global) const;
  void WriteCoordinates_(Mesh *pm, const IndexDomain &domain, hid_t file,
                         const HDF5::H5P &pl, hsize_t offset,
                         hsize_t max_blocks_global) const;
  void WriteLevelsAndLocs_(Mesh *pm, hid_t file, const HDF5::H5P &pl, hsize_t offset,
                           hsize_t max_blocks_global) const;
  void WriteSparseInfo_(Mesh *pm, hbool_t *sparse_allocated,
                        const std::vector<int> &dealloc_count,
                        const std::vector<std::string> &sparse_names, hsize_t num_sparse,
                        hid_t file, const HDF5::H5P &pl, size_t offset,
                        hsize_t max_blocks_global) const;
  std::string FilePostfix_() const {
    if (mode_ == DumpOutputMode::DUMP) {
      return ".phdf";
    } else if (mode_ == DumpOutputMode::RESTART) {
      return ".rhdf";
    } else if (mode_ == DumpOutputMode::CORE) {
      return ".chdf";
    } else {
      PARTHENON_FAIL("Unknown dump output mode");
      return "";
    }
  }
  const DumpOutputMode mode_;
};

//----------------------------------------------------------------------------------------
//! \class HistogramOutput
//  \brief derived OutputType class for histograms

namespace HistUtil {

enum class VarType { X1, X2, X3, R, Var, Unused };
enum class EdgeType { Lin, Log, List, Undefined };

struct Histogram {
  std::string name_;                      // name (id) of histogram
  int ndim_;                              // 1D or 2D histogram
  std::string x_var_name_, y_var_name_;   // variable(s) for bins
  VarType x_var_type_, y_var_type_;       // type, e.g., coord related or actual field
  int x_var_component_, y_var_component_; // components of bin variables (vector)
  ParArray1D<Real> x_edges_, y_edges_;
  EdgeType x_edges_type_, y_edges_type_;
  // Lowest edge and difference between edges.
  // Internally used to speed up lookup for log (and lin) bins as otherwise
  // two more log10 calls would be required per index.
  Real x_edge_min_, x_edge_dbin_, y_edge_min_, y_edge_dbin_;
  bool accumulate_;             // accumulate data outside binning range in outermost bins
  std::string binned_var_name_; // variable name of variable to be binned
  // component of variable to be binned. If -1 means no variable is binned but the
  // histgram is a sample count.
  int binned_var_component_;
  bool weight_by_vol_;          // use volume weighting
  std::string weight_var_name_; // variable name of variable used as weight
  // component of variable to be used as weight. If -1 means no weighting
  int weight_var_component_;
  ParArray2D<Real> result_; // resulting histogram

  // temp view for histogram reduction for better performance (switches
  // between atomics and data duplication depending on the platform)
  Kokkos::Experimental::ScatterView<Real **, LayoutWrapper> scatter_result;
  Histogram(ParameterInput *pin, const std::string &block_name, const std::string &name);
  void CalcHist(Mesh *pm);
};

} // namespace HistUtil

class HistogramOutput : public OutputType {
 public:
  HistogramOutput(const OutputParameters &oparams, ParameterInput *pin);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                       const SignalHandler::OutputSignal signal) override;

 private:
  std::string GenerateFilename_(ParameterInput *pin, SimTime *tm,
                                const SignalHandler::OutputSignal signal);
  std::vector<std::string> hist_names_; // names (used as id) for different histograms
  std::vector<HistUtil::Histogram> histograms_;
};
#endif // ifdef ENABLE_HDF5

//----------------------------------------------------------------------------------------
//! \class UserOutput
//  \brief derived OutputType class for User enrolled outputs

class UserOutput : public OutputType {
 public:
  explicit UserOutput(const OutputParameters &oparams) : OutputType(oparams) {}
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                       const SignalHandler::OutputSignal signal) override;

 private:
  std::string GenerateFilename_(ParameterInput *pin, SimTime *tm,
                                const SignalHandler::OutputSignal signal);
};

//----------------------------------------------------------------------------------------
//! \class Outputs

//  \brief root class for all Athena++ outputs. Provides a singly linked list of
//  OutputTypes, with each node representing one mode/format of output to be made.

class Outputs {
 public:
  Outputs(Mesh *pm, ParameterInput *pin, SimTime *tm = nullptr);

  void
  MakeOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm = nullptr,
              SignalHandler::OutputSignal signal = SignalHandler::OutputSignal::none);

 private:
  std::vector<std::shared_ptr<OutputType>> output_types_;
};

} // namespace parthenon

#endif // OUTPUTS_OUTPUTS_HPP_
