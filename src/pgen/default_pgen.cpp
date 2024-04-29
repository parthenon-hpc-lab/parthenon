//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
//! \file default_pgen.cpp
//  \brief Provides default versions of user callbacks that loop over per-package
//  functions.

#include "defs.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

void Mesh::PreStepUserDiagnosticsInLoopDefault(Mesh *pmesh, ParameterInput *,
                                               SimTime const &simtime) {
  for (auto &package : pmesh->packages.AllPackages()) {
    package.second->PreStepDiagnostics(simtime, pmesh->mesh_data.Get().get());
  }
}

void Mesh::PostStepUserDiagnosticsInLoopDefault(Mesh *pmesh, ParameterInput *,
                                                SimTime const &simtime) {
  for (auto &package : pmesh->packages.AllPackages()) {
    package.second->PostStepDiagnostics(simtime, pmesh->mesh_data.Get().get());
  }
}

} // namespace parthenon
