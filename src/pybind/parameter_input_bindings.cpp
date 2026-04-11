//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "globals.hpp"
#include "parameter_input.hpp"

namespace py = pybind11;

// Functions that query MPI directly rather than using Globals
// Note: We can't use Globals::my_rank here because the Python shared library (.so)
// gets its own copy of the static variables when linking against libparthenon.a,
// separate from the executable's copy. When MPI_Init sets Globals::my_rank in the
// executable, the Python module's copy remains at 0. Calling MPI functions directly
// avoids this issue.
int GetMyRank() {
#ifdef MPI_PARALLEL
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

int GetNRanks() {
#ifdef MPI_PARALLEL
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  return nranks;
#else
  return 1;
#endif
}

PYBIND11_MODULE(parthenon, m) {
  m.doc() = "Parthenon Python bindings for parameter input";

  // Expose MPI rank info as functions that query MPI directly
  m.def("my_rank", &GetMyRank, "Get the current MPI rank");
  m.def("nranks", &GetNRanks, "Get the total number of MPI ranks");

  py::class_<parthenon::ParameterInput>(m, "ParameterInput")
      .def(py::init<>())

      // Parser interface - add parameters without creating QueryRecords
      .def(
          "add_unresolved",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name, const std::string &value) {
            self.AddParsedParameter(block, name,
                                    parthenon::ParameterInput::UnresolvedString(value));
          },
          "Add a parameter as unresolved string (from file)")

      .def(
          "add_int",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name,
             int value) { self.AddParsedParameter(block, name, value); },
          "Add an integer parameter")

      .def(
          "add_real",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name,
             parthenon::Real value) { self.AddParsedParameter(block, name, value); },
          "Add a real parameter")

      .def(
          "add_bool",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name,
             bool value) { self.AddParsedParameter(block, name, value); },
          "Add a boolean parameter")

      .def(
          "add_string",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name,
             const std::string &value) { self.AddParsedParameter(block, name, value); },
          "Add a string parameter")

      // Vector add methods
      .def(
          "add_int_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name, const std::vector<int> &value) {
            self.AddParsedParameter(block, name, value);
          },
          "Add an integer vector parameter")

      .def(
          "add_real_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name, const std::vector<parthenon::Real> &value) {
            self.AddParsedParameter(block, name, value);
          },
          "Add a real vector parameter")

      .def(
          "add_bool_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name, const std::vector<bool> &value) {
            self.AddParsedParameter(block, name, value);
          },
          "Add a boolean vector parameter")

      .def(
          "add_string_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name, const std::vector<std::string> &value) {
            self.AddParsedParameter(block, name, value);
          },
          "Add a string vector parameter")

      // Query methods (const, safe to call during parsing)
      .def("does_parameter_exist", &parthenon::ParameterInput::DoesParameterExist,
           "Check if a parameter exists")

      .def("does_block_exist", &parthenon::ParameterInput::DoesBlockExist,
           "Check if a block exists")

      .def("get_parameter_names", &parthenon::ParameterInput::GetParameterNames,
           "Get all parameter names in a block")

      .def("get_blocks_with_prefix", &parthenon::ParameterInput::GetBlockNamesWithPrefix,
           "Get all blocks with a given prefix")

      // Get methods (for field initialization)
      // WARNING: These trigger FinalizeParsing(), do NOT use in
      // parthenon_init_parameters()
      .def(
          "get_int",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) { return self.GetInteger(block, name); },
          "Get integer parameter (triggers finalization)")

      .def(
          "get_real",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) { return self.GetReal(block, name); },
          "Get real parameter (triggers finalization)")

      .def(
          "get_bool",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) { return self.GetBoolean(block, name); },
          "Get boolean parameter (triggers finalization)")

      .def(
          "get_string",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) { return self.GetString(block, name); },
          "Get string parameter (triggers finalization)")

      .def(
          "get_int_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) { return self.GetVector<int>(block, name); },
          "Get integer vector parameter (triggers finalization)")

      .def(
          "get_real_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) {
            return self.GetVector<parthenon::Real>(block, name);
          },
          "Get real vector parameter (triggers finalization)")

      .def(
          "get_bool_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) { return self.GetVector<bool>(block, name); },
          "Get boolean vector parameter (triggers finalization)")

      .def(
          "get_string_vector",
          [](parthenon::ParameterInput &self, const std::string &block,
             const std::string &name) { return self.GetVector<std::string>(block, name); },
          "Get string vector parameter (triggers finalization)");

  // NOTE: get_parameter_input() removed in favor of explicit parameter passing
  // Python input files should define: def parthenon_init_parameters(pin):
}
