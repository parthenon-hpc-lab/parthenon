//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#ifndef INTERFACE_SET_GRAPHICS_HPP_
#define INTERFACE_SET_GRAPHICS_HPP_

#include <memory>

#include "StateDescriptor.hpp"
#include "parameter_input.hpp"

namespace parthenon {

void SetGraphics(std::unique_ptr<ParameterInput> &pin, Packages_t &pkg);

} // namespace parthenon

#endif // INTERFACE_SET_GRAPHICS_HPP_
