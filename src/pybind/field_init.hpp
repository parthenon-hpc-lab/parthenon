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

#ifndef PYBIND_FIELD_INIT_HPP_
#define PYBIND_FIELD_INIT_HPP_

#include <string>
#include <vector>

#include "basic_types.hpp"
#include "config.hpp"

namespace parthenon {

// Forward declarations
class MeshBlock;
class ParameterInput;

#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS

// Initialize a field from a Python function
//
// Args:
//   pmb: MeshBlock containing the field
//   var_name: Name of the variable to initialize
//   block: Input file block name containing initialization parameters
//   func_param: Parameter name for the Python function name
//   file_param: Parameter name for the Python file path
//   component: Component indices for multi-component fields (e.g., {0}, {1,2})
//              Empty vector {} means scalar or single component
//
// The Python function should have signature:
//   def init_function(x, y, z, component, data):
//       # x, y, z: 1D numpy arrays of coordinates (flattened, zero-copy views)
//       # component: tuple of component indices (e.g., (), (0,), (1,2))
//       # data: 1D numpy array to write to (flattened, zero-copy view, same length as
//       x/y/z)
//
// IMPORTANT: NumPy is required for Python field initialization
void InitializeFieldFromPython(MeshBlock *pmb, const std::string &var_name,
                                ParameterInput *pin, const std::string &block,
                                const std::string &func_param,
                                const std::string &file_param,
                                const std::vector<int> &component = {});

#endif // PARTHENON_ENABLE_PYTHON_BINDINGS

} // namespace parthenon

#endif // PYBIND_FIELD_INIT_HPP_
