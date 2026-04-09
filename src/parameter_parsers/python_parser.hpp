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

#ifndef PARAMETER_PARSERS_PYTHON_PARSER_HPP_
#define PARAMETER_PARSERS_PYTHON_PARSER_HPP_

#include <memory>

#include "parameter_input.hpp"

namespace parthenon {

// Load ParameterInput from a Python script
// The script is executed in an embedded Python interpreter and can use
// parthenon.get_parameter_input() to populate parameters programmatically.
std::unique_ptr<ParameterInput> LoadParameterInputFromPython(const char *python_filename,
                                                              int argc, char *argv[]);

} // namespace parthenon

#endif // PARAMETER_PARSERS_PYTHON_PARSER_HPP_
