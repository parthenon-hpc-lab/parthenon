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
//! \file mesh.cpp
//  \brief implementation of functions in MeshBlock class

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/meshblock_data_iterator.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_tree.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// MeshBlock constructor: constructs coordinate, boundary condition, field
//                        and mesh refinement objects.
MeshBlock::MeshBlock(const int n_side, const int ndim)
    : exec_space(DevExecSpace()), pmy_mesh(nullptr), cost_(1.0) {
  // initialize grid indices
  if (ndim == 1) {
    InitializeIndexShapes(n_side, 0, 0);
  } else if (ndim == 2) {
    InitializeIndexShapes(n_side, n_side, 0);
  } else {
    InitializeIndexShapes(n_side, n_side, n_side);
  }
}

// Factory method deals with initialization for you
std::shared_ptr<MeshBlock> MeshBlock::Make(int igid, int ilid, LogicalLocation iloc,
                                           RegionSize input_block,
                                           BoundaryFlag *input_bcs, Mesh *pm,
                                           ParameterInput *pin, ApplicationInput *app_in,
                                           Properties_t &properties, Packages_t &packages,
                                           int igflag, double icost) {
  auto pmb = std::make_shared<MeshBlock>();
  pmb->Initialize(igid, ilid, iloc, input_block, input_bcs, pm, pin, app_in, properties,
                  packages, igflag, icost);
  return pmb;
}

void MeshBlock::Initialize(int igid, int ilid, LogicalLocation iloc,
                           RegionSize input_block, BoundaryFlag *input_bcs, Mesh *pm,
                           ParameterInput *pin, ApplicationInput *app_in,
                           Properties_t &properties, Packages_t &packages, int igflag,
                           double icost) {
  exec_space = DevExecSpace();
  pmy_mesh = pm;
  loc = iloc;
  block_size = input_block;
  gid = igid;
  lid = ilid;
  gflag = igflag;
  this->properties = properties;
  this->packages = packages;
  cost_ = icost;

  // initialize grid indices
  if (pmy_mesh->ndim >= 3) {
    InitializeIndexShapes(block_size.nx1, block_size.nx2, block_size.nx3);
  } else if (pmy_mesh->ndim >= 2) {
    InitializeIndexShapes(block_size.nx1, block_size.nx2, 0);
  } else {
    InitializeIndexShapes(block_size.nx1, 0, 0);
  }

  // Allow for user overrides to default Parthenon functions
  if (app_in->InitApplicationMeshBlockData != nullptr) {
    InitApplicationMeshBlockData = app_in->InitApplicationMeshBlockData;
  }
  if (app_in->InitUserMeshBlockData != nullptr) {
    InitUserMeshBlockData = app_in->InitUserMeshBlockData;
  }
  if (app_in->ProblemGenerator != nullptr) {
    ProblemGenerator = app_in->ProblemGenerator;
  }
  if (app_in->MeshBlockUserWorkInLoop != nullptr) {
    UserWorkInLoop = app_in->MeshBlockUserWorkInLoop;
  }
  if (app_in->UserWorkBeforeOutput != nullptr) {
    UserWorkBeforeOutput = app_in->UserWorkBeforeOutput;
  }

  auto &real_container = meshblock_data.Get();
  auto &swarm_container = swarm_data.Get();
  // Set the block pointer for the containers
  real_container->SetBlockPointer(shared_from_this());
  swarm_container->SetBlockPointer(shared_from_this());

  // (probably don't need to preallocate space for references in these vectors)
  vars_cc_.reserve(3);
  vars_fc_.reserve(3);

  // construct objects stored in MeshBlock class.  Note in particular that the initial
  // conditions for the simulation are set in problem generator called from main

  coords = Coordinates_t(block_size, pin);

  // mesh-related objects
  // Boundary
  pbval = std::make_unique<BoundaryValues>(shared_from_this(), input_bcs, pin);
  pbval->SetBoundaryFlags(boundary_flag);

  // Add field properties data
  // TOOD(JMM): Should packages be resolved for state descriptors in
  // properties?
  for (int i = 0; i < properties.size(); i++) {
    StateDescriptor &state = properties[i]->State();
    for (auto const &q : state.AllFields()) {
      real_container->Add(q.first, q.second);
    }
    for (auto const &q : state.AllSparseFields()) {
      for (auto const &p : q.second) {
        real_container->Add(q.first, p.second);
      }
    }
  }
  // Add physics data, including dense, sparse, and swarm variables.
  // Resolve issues.
  resolved_packages = ResolvePackages(packages);
  auto &pkg = resolved_packages;
  for (auto const &q : pkg->AllFields()) {
    real_container->Add(q.first, q.second);
  }
  for (auto const &q : pkg->AllSparseFields()) {
    for (auto const &p : q.second) {
      real_container->Add(q.first, p.second);
    }
  }
  for (auto const &q : pkg->AllSwarms()) {
    swarm_container->Add(q.first, q.second);
    // Populate swarm values
    auto &swarm = swarm_container->Get(q.first);
    for (auto const &m : pkg->AllSwarmValues(q.first)) {
      swarm->Add(m.first, m.second);
    }
  }

  // TODO(jdolence): Should these loops be moved to Variable creation
  MeshBlockDataIterator<Real> ci(real_container, {Metadata::Independent});
  int nindependent = ci.vars.size();
  for (int n = 0; n < nindependent; n++) {
    RegisterMeshBlockData(ci.vars[n]);
  }

  if (pm->multilevel) {
    pmr = std::make_unique<MeshRefinement>(shared_from_this(), pin);
    // This is very redundant, I think, but necessary for now
    for (int n = 0; n < nindependent; n++) {
      pmr->AddToRefinement(ci.vars[n]->data, ci.vars[n]->coarse_s);
    }
  }

  // Create user mesh data
  // InitUserMeshBlockData(pin);
  app = InitApplicationMeshBlockData(this, pin);
  return;
}

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlock::~MeshBlock() = default;

void MeshBlock::InitializeIndexShapes(const int nx1, const int nx2, const int nx3) {
  cellbounds = IndexShape(nx3, nx2, nx1, Globals::nghost);

  if (pmy_mesh != nullptr) {
    if (pmy_mesh->multilevel) {
      cnghost = (Globals::nghost + 1) / 2 + 1;
      c_cellbounds = IndexShape(nx3 / 2, nx2 / 2, nx1 / 2, Globals::nghost);
    } else {
      c_cellbounds = IndexShape(nx3 / 2, nx2 / 2, nx1 / 2, 0);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::SetCostForLoadBalancing(double cost)
//  \brief stop time measurement and accumulate it in the MeshBlock cost

void MeshBlock::SetCostForLoadBalancing(double cost) {
  if (pmy_mesh->lb_manual_) {
    cost_ = std::min(cost, TINY_NUMBER);
    pmy_mesh->lb_flag_ = true;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ResetTimeMeasurement()
//  \brief reset the MeshBlock cost for automatic load balancing

void MeshBlock::ResetTimeMeasurement() {
  if (pmy_mesh->lb_automatic_) cost_ = TINY_NUMBER;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::StartTimeMeasurement()
//  \brief start time measurement for automatic load balancing

void MeshBlock::StartTimeMeasurement() {
  if (pmy_mesh->lb_automatic_) {
    lb_timer.reset();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::StartTimeMeasurement()
//  \brief stop time measurement and accumulate it in the MeshBlock cost

void MeshBlock::StopTimeMeasurement() {
  if (pmy_mesh->lb_automatic_) {
    cost_ += lb_timer.seconds();
  }
}

void MeshBlock::RegisterMeshBlockData(std::shared_ptr<CellVariable<Real>> pvar_cc) {
  vars_cc_.push_back(pvar_cc);
  return;
}

void MeshBlock::RegisterMeshBlockData(std::shared_ptr<FaceField> pvar_fc) {
  vars_fc_.push_back(pvar_fc);
  return;
}

} // namespace parthenon
