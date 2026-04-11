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

PYBIND11_MODULE(parthenon, m) {
  m.doc() = "Parthenon Python bindings for parameter input";

  // Expose MPI rank info so scripts can do rank-specific operations
  m.attr("my_rank") = parthenon::Globals::my_rank;
  m.attr("nranks") = parthenon::Globals::nranks;

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
      // Note: Get methods are intentionally NOT exposed to prevent premature
      // finalization. Python input scripts should only ADD parameters, not query their
      // values.
      .def("does_parameter_exist", &parthenon::ParameterInput::DoesParameterExist,
           "Check if a parameter exists")

      .def("does_block_exist", &parthenon::ParameterInput::DoesBlockExist,
           "Check if a block exists")

      .def("get_parameter_names", &parthenon::ParameterInput::GetParameterNames,
           "Get all parameter names in a block")

      .def("get_blocks_with_prefix", &parthenon::ParameterInput::GetBlocksWithPrefix,
           "Get all blocks with a given prefix");

  // NOTE: get_parameter_input() removed in favor of explicit parameter passing
  // Python input files should define: def parthenon_init_parameters(pin):
}
