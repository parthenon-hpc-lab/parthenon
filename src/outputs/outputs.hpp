//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// forward declarations
class Mesh;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \struct OutputParameters
//  \brief  container for parameters read from <output> block in the input file

struct OutputParameters {
  int block_number;
  std::string block_name;
  std::string file_basename;
  int file_number_width;
  bool file_label_final;
  std::string file_id;
  std::vector<std::string> variables;
  std::vector<std::string> component_labels;
  std::map<std::string, std::set<std::string>> swarms;
  std::vector<std::string> swarm_vars;
  std::string file_type;
  std::string data_format;
  Real next_time, dt;
  int file_number;
  bool include_ghost_zones, cartesian_vector;
  bool single_precision_output;
  bool sparse_seed_nans;
  int hdf5_compression_level;
  bool write_xdmf;
  // TODO(felker): some of the parameters in this class are not initialized in constructor
  OutputParameters()
      : block_number(0), next_time(0.0), dt(-1.0), file_number(0),
        include_ghost_zones(false), cartesian_vector(false),
        single_precision_output(false), sparse_seed_nans(false),
        hdf5_compression_level(5), write_xdmf(true) {}
};

//----------------------------------------------------------------------------------------
//! \struct OutputData
//  \brief container for output data and metadata; node in nested doubly linked list

struct OutputData {
  std::string type; // one of (SCALARS,VECTORS) used for vtk outputs
  std::string name;
  ParArrayND<Real> data; // array containing data (usually shallow copy/slice)
  // ptrs to previous and next nodes in doubly linked list:
  OutputData *pnext, *pprev;

  OutputData() : pnext(nullptr), pprev(nullptr) {}
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
  // copy constructor and assignment operator (pnext_type, pfirst_data, etc. are shallow
  // copied)
  OutputType(const OutputType &copy_other) = default;
  OutputType &operator=(const OutputType &copy_other) = default;
  // move constructor and assignment operator
  OutputType(OutputType &&) = default;
  OutputType &operator=(OutputType &&) = default;

  // data
  int out_is, out_ie, out_js, out_je, out_ks, out_ke; // OutputData array start/end index
  OutputParameters output_params; // control data read from <output> block
  OutputType *pnext_type;         // ptr to next node in singly linked list of OutputTypes

  // functions
  void LoadOutputData(MeshBlock *pmb);
  void AppendOutputDataNode(OutputData *pdata);
  void ReplaceOutputDataNode(OutputData *pold, OutputData *pnew);
  void ClearOutputData();
  // following pure virtual function must be implemented in all derived classes
  virtual void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                               const SignalHandler::OutputSignal signal) = 0;
  virtual void WriteContainer(SimTime &tm, Mesh *pm, ParameterInput *pin, bool flag) {
    return;
  }

 protected:
  int num_vars_; // number of variables in output
  // nested doubly linked list of OutputData nodes (of the same OutputType):
  OutputData *pfirst_data_; // ptr to head OutputData node in doubly linked list
  OutputData *plast_data_;  // ptr to tail OutputData node in doubly linked list
};

//----------------------------------------------------------------------------------------
// Helper definitions to enroll user output variables

// Function signature for currently supported user output functions
using HstFun_t = std::function<Real(MeshData<Real> *md)>;

// Container
struct HistoryOutputVar {
  UserHistoryOperation hst_op; // Reduction operation
  HstFun_t hst_fun;            // Function to be called
  std::string label;           // column label in hst output file
  HistoryOutputVar(const UserHistoryOperation &hst_op_, const HstFun_t &hst_fun_,
                   const std::string &label_)
      : hst_op(hst_op_), hst_fun(hst_fun_), label(label_) {}
};

using HstVar_list = std::vector<HistoryOutputVar>;
// Hardcoded global entry to be used by each package to enroll user output functions
const char hist_param_key[] = "HistoryFunctions";

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
//! \class VTKOutput
//  \brief derived OutputType class for vtk dumps

class VTKOutput : public OutputType {
 public:
  explicit VTKOutput(const OutputParameters &oparams) : OutputType(oparams) {}
  void WriteContainer(SimTime &tm, Mesh *pm, ParameterInput *pin, bool flag) override;
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
  PHDF5Output(const OutputParameters &oparams, bool restart)
      : OutputType(oparams), restart_(restart) {}
  void WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                       const SignalHandler::OutputSignal signal) override;
  template <bool WRITE_SINGLE_PRECISION>
  void WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, SimTime *tm,
                           const SignalHandler::OutputSignal signal);

 private:
  std::string GenerateFilename_(ParameterInput *pin, SimTime *tm,
                                const SignalHandler::OutputSignal signal);
  const bool restart_; // true if we write a restart file, false for regular output files
  // TODO(JMM): these methods might want to live in the base class or in output_utils.hpp
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
//! \class Outputs

//  \brief root class for all Athena++ outputs. Provides a singly linked list of
//  OutputTypes, with each node representing one mode/format of output to be made.

class Outputs {
 public:
  Outputs(Mesh *pm, ParameterInput *pin, SimTime *tm = nullptr);
  ~Outputs();

  void
  MakeOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm = nullptr,
              SignalHandler::OutputSignal signal = SignalHandler::OutputSignal::none);

 private:
  OutputType *pfirst_type_; // ptr to head OutputType node in singly linked list
  // (not storing a reference to the tail node)
};

} // namespace parthenon

#endif // OUTPUTS_OUTPUTS_HPP_
