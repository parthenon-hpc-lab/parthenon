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

#include "python_parser.hpp"

#ifdef PARTHENON_ENABLE_PYTHON_BINDINGS

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <sstream>
#include <string>

#include "utils/error_checking.hpp"

namespace py = pybind11;

namespace parthenon {

std::unique_ptr<ParameterInput> LoadParameterInputFromPython(const char *python_filename,
                                                             int argc, char *argv[]) {
  // Create ParameterInput in C++ - we own this
  auto pinput = std::make_unique<ParameterInput>();

  // Start Python interpreter
  py::scoped_interpreter guard{};

  try {
    // Import the parthenon module to make ParameterInput bindings available
    // The parthenon.so module must be in PYTHONPATH
    py::module_::import("parthenon");

    // Build sys.argv for the Python script
    // Include the script name and all command line arguments after "-i script.py"
    // This allows Python scripts to use argparse to parse their own arguments
    py::list py_argv;
    py_argv.append(python_filename);

    // Find where the input file appears in argv and include everything after it
    bool found_input_file = false;
    for (int i = 1; i < argc; i++) {
      if (found_input_file) {
        py_argv.append(argv[i]);
      } else if (std::string(argv[i]) == "-i" && i + 1 < argc) {
        // Skip -i and the filename, start collecting args after
        i++; // Skip the filename
        found_input_file = true;
      }
    }

    // Set sys.argv for the Python script
    py::module_::import("sys").attr("argv") = py_argv;

    // Inject the ParameterInput into Python's global namespace
    // The Python script can retrieve it via parthenon.get_parameter_input()
    py::globals()["__parthenon_pi__"] =
        py::cast(pinput.get(), py::return_value_policy::reference);

    // Execute the Python script
    // The script can now:
    //   1. Use argparse.parse_known_args() to parse Python-style flags (e.g., --nx=32)
    //   2. Retrieve ParameterInput via parthenon.get_parameter_input()
    //   3. Configure parameters programmatically
    // After this returns, C++ will apply Parthenon-style overrides (block/param=value)
    // via ModifyFromCmdline(), which ignores Python-style flags
    py::eval_file(python_filename, py::globals());
  } catch (py::error_already_set &e) {
    std::stringstream msg;
    msg << "### FATAL ERROR loading Python input file: " << python_filename << std::endl
        << e.what() << std::endl;
    PARTHENON_FAIL(msg);
  }

  return pinput;
}

} // namespace parthenon

#endif // PARTHENON_ENABLE_PYTHON_BINDINGS
