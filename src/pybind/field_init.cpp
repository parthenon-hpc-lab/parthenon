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

#include "field_init.hpp"

#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::vector conversion

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/meshblock_data.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"

namespace py = pybind11;

namespace parthenon {

void InitializeFieldFromPython(MeshBlock *pmb, const std::string &var_name,
                                ParameterInput *pin, const std::string &block,
                                const std::string &func_param,
                                const std::string &file_param,
                                const std::vector<int> &component) {
  // Keep interpreter alive for all initializations (one per rank)
  static std::unique_ptr<py::scoped_interpreter> guard = nullptr;
  if (!guard) {
    guard = std::make_unique<py::scoped_interpreter>();
  }

  try {
    // Get function name and file path from input
    std::string func_name = pin->GetString(block, func_param);
    std::string file_path = pin->GetString(block, file_param);

    // Get the init function from the Python file
    // Try to reuse already-loaded module (e.g., if it's the input file)
    py::module_ sys = py::module_::import("sys");
    py::dict modules = sys.attr("modules");

    py::object init_func;
    bool found = false;

    // Check if module is already loaded by looking for function in existing modules
    for (auto item : modules) {
      try {
        py::module_ mod = item.second.cast<py::module_>();
        if (py::hasattr(mod, func_name.c_str())) {
          // Check if this is the right module by comparing __file__ attribute
          if (py::hasattr(mod, "__file__")) {
            std::string mod_file = mod.attr("__file__").cast<std::string>();
            if (mod_file == file_path) {
              init_func = mod.attr(func_name.c_str());
              found = true;
              break;
            }
          }
        }
      } catch (...) {
        // Skip modules that can't be cast or don't have the attribute
        continue;
      }
    }

    // If not found in loaded modules, execute the file
    if (!found) {
      py::dict globals = py::globals();
      py::eval_file(file_path, globals);
      init_func = globals[func_name.c_str()];
    }

    // Get the variable
    auto &var = pmb->meshblock_data.Get()->Get(var_name);

    // Get coordinate system
    auto &coords = pmb->coords;

    // Get index ranges for cell-centered data
    // TODO: Support other topological elements (face, edge, node)
    auto cellbounds = pmb->cellbounds;
    int is = cellbounds.is(IndexDomain::interior);
    int ie = cellbounds.ie(IndexDomain::interior);
    int js = cellbounds.js(IndexDomain::interior);
    int je = cellbounds.je(IndexDomain::interior);
    int ks = cellbounds.ks(IndexDomain::interior);
    int ke = cellbounds.ke(IndexDomain::interior);

    int ncells = (ie - is + 1) * (je - js + 1) * (ke - ks + 1);

    // Allocate coordinate arrays
    std::vector<Real> x_coords(ncells);
    std::vector<Real> y_coords(ncells);
    std::vector<Real> z_coords(ncells);
    std::vector<Real> data(ncells);

    // Extract coordinates (cell centers)
    int idx = 0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          x_coords[idx] = coords.Xc<1>(i);
          y_coords[idx] = coords.Xc<2>(j);
          z_coords[idx] = coords.Xc<3>(k);
          idx++;
        }
      }
    }

    // Convert component vector to Python tuple
    py::tuple comp_tuple = py::cast(component);

    // Try to create numpy arrays for performance
    // These are views into C++ memory (no copy)
    try {
      // Create numpy arrays that directly reference C++ vector memory
      // The arrays reference our C++ memory but we manage the lifetime
      auto x_array = py::array_t<Real>(ncells, x_coords.data());
      auto y_array = py::array_t<Real>(ncells, y_coords.data());
      auto z_array = py::array_t<Real>(ncells, z_coords.data());
      auto data_array = py::array_t<Real>(ncells, data.data());

      // Call Python function with numpy arrays
      // Python will write directly to our C++ vectors
      init_func(x_array, y_array, z_array, comp_tuple, data_array);

      // No copy back needed - Python wrote directly to C++ memory

    } catch (py::error_already_set &e) {
      // If numpy isn't available or the function expects lists, fall back
      py::list x_list = py::cast(x_coords);
      py::list y_list = py::cast(y_coords);
      py::list z_list = py::cast(z_coords);
      py::list data_list = py::cast(data);

      // Clear the error and try with lists
      e.restore();
      PyErr_Clear();

      init_func(x_list, y_list, z_list, comp_tuple, data_list);

      // Copy data back from Python list
      for (int i = 0; i < ncells; i++) {
        data[i] = data_list[i].cast<Real>();
      }
    }

    // Copy data back to field
    // Get host mirror for device compatibility
    auto var_host = var.data.GetHostMirrorAndCopy();

    // Determine indices for the variable based on component
    // For now, assume cell-centered scalar or first component of vector
    // TODO: Properly handle multi-component fields with component tuple
    int t = 0; // topological element (for cell-centered, just 0)

    // Map component vector to variable indices
    // Parthenon supports up to 3 non-spatial indices
    int u = (component.size() > 0) ? component[0] : 0;
    int v = (component.size() > 1) ? component[1] : 0;
    int w = (component.size() > 2) ? component[2] : 0;

    // Copy data from flattened array back to field
    idx = 0;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          var_host(t, u, v, w, k, j, i) = data[idx];
          idx++;
        }
      }
    }

    // Copy back to device
    var.data.DeepCopy(var_host);

  } catch (py::error_already_set &e) {
    std::stringstream msg;
    msg << "### FATAL ERROR in InitializeFieldFromPython" << std::endl
        << "Variable: " << var_name << std::endl
        << "Block: " << block << std::endl
        << "Python error: " << e.what() << std::endl;
    PARTHENON_FAIL(msg);
  } catch (std::exception &e) {
    std::stringstream msg;
    msg << "### FATAL ERROR in InitializeFieldFromPython" << std::endl
        << "Variable: " << var_name << std::endl
        << "Block: " << block << std::endl
        << "Error: " << e.what() << std::endl;
    PARTHENON_FAIL(msg);
  }
}

} // namespace parthenon

#endif // PARTHENON_ENABLE_PYTHON_BINDINGS
