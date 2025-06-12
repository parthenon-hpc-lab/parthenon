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

#include "defs.hpp"
#include "parameter_input.hpp"

namespace parthenon {

RegionSize::RegionSize(ParameterInput *pin)
    : RegionSize({pin->GetReal("parthenon/mesh", "x1min"),
                  pin->GetReal("parthenon/mesh", "x2min"),
                  pin->GetReal("parthenon/mesh", "x3min")},
                 {pin->GetReal("parthenon/mesh", "x1max"),
                  pin->GetReal("parthenon/mesh", "x2max"),
                  pin->GetReal("parthenon/mesh", "x3max")},
                 {pin->GetOrAddReal("parthenon/mesh", "x1rat", 1.0),
                  pin->GetOrAddReal("parthenon/mesh", "x2rat", 1.0),
                  pin->GetOrAddReal("parthenon/mesh", "x3rat", 1.0)},
                 {pin->GetInteger("parthenon/mesh", "nx1"),
                  pin->GetInteger("parthenon/mesh", "nx2"),
                  pin->GetInteger("parthenon/mesh", "nx3")},
                 {false, pin->GetInteger("parthenon/mesh", "nx2") == 1,
                  pin->GetInteger("parthenon/mesh", "nx3") == 1}) {}

} // namespace parthenon
