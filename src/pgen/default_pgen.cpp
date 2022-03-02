//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
//  \brief Provides default (empty) versions of all functions in problem generator files
//  This means user does not have to implement these functions if they are not needed.
//
// By default, function pointers are set to these functions. Users can override these
// defaults by setting the relevant functions in ParthenonManager prior to calling
// ParthenonInit.

#include "defs.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

// 3x members of Mesh class:

namespace parthenon {

//========================================================================================
//! \fn void Mesh::InitUserMeshDataDefault(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshDataDefault(Mesh *, ParameterInput *) {
  // do nothing
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoopDefault()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoopDefault(Mesh *, ParameterInput *, SimTime const &) {
  // do nothing
  return;
}

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

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoopDefault(ParameterInput *pin, SimTime &tm)
//  \brief Function called after main loop is finished for user-defined work.
//========================================================================================

void Mesh::UserWorkAfterLoopDefault(Mesh *mesh, ParameterInput *pin, SimTime &tm) {
  // do nothing
  return;
}

// 5x members of MeshBlock class:

//========================================================================================
//! \fn std::unique_ptr<MeshBlockApplicationData>
//! MeshBlock::InitApplicationMeshBlockDataDefault(ParameterInput *pin)
//  \brief Function to initialize application-specific data in MeshBlock class.  Can also
//  be used to initialize variables which are global to other functions in this file.
//  Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================

std::unique_ptr<MeshBlockApplicationData>
MeshBlock::InitApplicationMeshBlockDataDefault(MeshBlock * /*pmb*/,
                                               ParameterInput * /*pin*/) {
  // do nothing
  return nullptr;
}

//========================================================================================
//! \fn void MeshBlock::InitMeshBlockUserDataDefault(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//  used to initialize variables which are global to other functions in this file.
//  Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================

void MeshBlock::InitMeshBlockUserDataDefault(MeshBlock *pmb, ParameterInput *pin) {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGeneratorDefault(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGeneratorDefault(MeshBlock *pmb, ParameterInput *pin) {
  // In practice, this function should *always* be replaced by a version
  // that sets the initial conditions for the problem of interest.
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoopDefault()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void MeshBlock::UserWorkInLoopDefault() {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutputDefault(ParameterInput *pin)
//  \brief Function called before generating output files
//========================================================================================

void MeshBlock::UserWorkBeforeOutputDefault(ParameterInput *pin) {
  // do nothing
  return;
}

} // namespace parthenon
