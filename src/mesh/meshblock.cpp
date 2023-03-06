//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
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
MeshBlock::MeshBlock(const int n_side, const int ndim, bool init_coarse, bool multilevel)
    : exec_space(DevExecSpace()), pmy_mesh(nullptr), cost_(1.0) {
  // initialize grid indices
  if (ndim == 1) {
    InitializeIndexShapesImpl(n_side, 0, 0, init_coarse, multilevel);
  } else if (ndim == 2) {
    InitializeIndexShapesImpl(n_side, n_side, 0, init_coarse, multilevel);
  } else {
    InitializeIndexShapesImpl(n_side, n_side, n_side, init_coarse, multilevel);
  }
}

// Factory method deals with initialization for you
std::shared_ptr<MeshBlock>
MeshBlock::Make(int igid, int ilid, LogicalLocation iloc, RegionSize input_block,
                BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin,
                ApplicationInput *app_in, Packages_t &packages,
                std::shared_ptr<StateDescriptor> resolved_packages, int igflag,
                double icost) {
  auto pmb = std::make_shared<MeshBlock>();
  pmb->Initialize(igid, ilid, iloc, input_block, input_bcs, pm, pin, app_in, packages,
                  resolved_packages, igflag, icost);
  return pmb;
}

void MeshBlock::Initialize(int igid, int ilid, LogicalLocation iloc,
                           RegionSize input_block, BoundaryFlag *input_bcs, Mesh *pm,
                           ParameterInput *pin, ApplicationInput *app_in,
                           Packages_t &packages,
                           std::shared_ptr<StateDescriptor> resolved_packages, int igflag,
                           double icost) {
  exec_space = DevExecSpace();
  pmy_mesh = pm;
  loc = iloc;
  block_size = input_block;
  gid = igid;
  lid = ilid;
  gflag = igflag;
  physics_init_complete = false;
  this->packages = packages;
  this->resolved_packages = resolved_packages;
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
  if (app_in->InitMeshBlockUserData != nullptr) {
    InitMeshBlockUserData = app_in->InitMeshBlockUserData;
  }
  if (app_in->ProblemGenerator != nullptr) {
    ProblemGenerator = app_in->ProblemGenerator;
    // Only set default block pgen when no mesh pgen is set
  } else if (app_in->MeshProblemGenerator == nullptr) {
    ProblemGenerator = &ProblemGeneratorDefault;
  }
  if (app_in->MeshBlockUserWorkBeforeOutput != nullptr) {
    UserWorkBeforeOutput = app_in->MeshBlockUserWorkBeforeOutput;
  }

  // (probably don't need to preallocate space for references in these vectors)
  vars_cc_.reserve(3);
  vars_fc_.reserve(3);

  // construct objects stored in MeshBlock class.  Note in particular that the initial
  // conditions for the simulation are set in problem generator called from main

  // Coords has host and device objects
  coords = Coordinates_t(block_size, pin);
  coords_device = ParArray0D<Coordinates_t>("coords on device");
  auto coords_host_mirror = Kokkos::create_mirror_view(coords_device);
  coords_host_mirror() = coords;
  Kokkos::deep_copy(coords_device, coords_host_mirror);

  // mesh-related objects
  // Boundary
  pbval = std::make_unique<BoundaryValues>(shared_from_this(), input_bcs, pin);
  pbval->SetBoundaryFlags(boundary_flag);
  pbswarm = std::make_unique<BoundarySwarms>(shared_from_this(), input_bcs, pin);
  pbswarm->SetBoundaryFlags(boundary_flag);

  // Add physics data, including dense, sparse, and swarm variables.
  // Resolve issues.

  auto &real_container = meshblock_data.Get();
  auto &swarm_container = swarm_data.Get();

  real_container->Initialize(resolved_packages, shared_from_this());

  swarm_container->SetBlockPointer(shared_from_this());
  for (auto const &q : resolved_packages->AllSwarms()) {
    swarm_container->Add(q.first, q.second);
    // Populate swarm values
    auto &swarm = swarm_container->Get(q.first);
    for (auto const &m : resolved_packages->AllSwarmValues(q.first)) {
      swarm->Add(m.first, m.second);
    }
  }

  swarm_container->AllocateBoundaries();

  // TODO(jdolence): Should these loops be moved to Variable creation
  // TODO(JMM): What variables should be in vars_cc_? They are used
  // for counting load-balance cost. Should it be different than the
  // variables used for refinement?
  // Should we even have both of these arrays? Are they both necessary?

  // TODO(JMM): In principal this should be `Metadata::Independent`
  // only. However, I am making it `Metadata::Independent` OR
  // `Metadata::FillGhost` to work around the old Athena++
  // `bvals_refine` machinery. When this machinery is completely
  // removed, which can happen after dense-on-block for sparse
  // variables is in place and after we write "prolongate-in-one,"
  // this should be only for `Metadata::Independent`.

  // TODO(LFR): vars_cc_ sets what variables are communicated across
  // ranks during remeshing, so we want to be able to explicitly flag
  // variables that need to be communicated using `Metadata::RemeshComm`.
  // In the future, this needs to be cleaned up since `vars_cc_` is
  // potentially used in the load balancing calculation, but not all
  // variables that we may want to communicate are necessarily relevant
  // to the cost per meshblock.
  CellVariableVector<Real> vars = GetAnyVariables(real_container->GetCellVariableVector(), 
    {Metadata::Independent, Metadata::FillGhost, Metadata::RemeshComm});
  for (int n = 0; n < vars.size(); ++n) {
    //std::cout << "vars_cc_: " << vars[n]->label() << std::endl;
    RegisterMeshBlockData(vars[n]);
  }

  if (pm->multilevel) {
    CellVariableVector<Real> refine_vars = GetAnyVariables(
      real_container->GetCellVariableVector(), 
      {Metadata::Independent, Metadata::FillGhost, Metadata::RemeshComm});       
    pmr = std::make_unique<MeshRefinement>(shared_from_this(), pin);
    // This is very redundant, I think, but necessary for now
    for (int n = 0; n < refine_vars.size(); n++) {
      // These are used for doing refinement
      //std::cout << "pvars_cc_: " << refine_vars[n]->label() << std::endl;
      pmr->AddToRefinement(refine_vars[n]);
    }
  }

  // Create user mesh data
  // InitMeshBlockUserData(pin);
  app = InitApplicationMeshBlockData(this, pin);
}

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlock::~MeshBlock() = default;

void MeshBlock::InitializeIndexShapesImpl(const int nx1, const int nx2, const int nx3,
                                          bool init_coarse, bool multilevel) {
  cellbounds = IndexShape(nx3, nx2, nx1, Globals::nghost);

  if (init_coarse) {
    if (multilevel) {
      cnghost = (Globals::nghost + 1) / 2 + 1;
      c_cellbounds = IndexShape(nx3 / 2, nx2 / 2, nx1 / 2, Globals::nghost);
    } else {
      c_cellbounds = IndexShape(nx3 / 2, nx2 / 2, nx1 / 2, 0);
    }
  }
}

void MeshBlock::InitializeIndexShapes(const int nx1, const int nx2, const int nx3) {
  const bool init_coarse = (pmy_mesh != nullptr);
  const bool multilevel = (init_coarse && pmy_mesh->multilevel);
  InitializeIndexShapesImpl(nx1, nx2, nx3, init_coarse, multilevel);
}


void MeshBlock::UserSetCost(double cost) {
  cost_ = cost;
  pmy_mesh->lb_flag_ = true;
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

void MeshBlock::AllocateSparse(std::string const &label, bool only_control, bool flag_uninitialized) {
  auto &mbd = meshblock_data;
  auto AllocateVar = [flag_uninitialized, &mbd, this](const std::string &l) {
    // first allocate variable in base stage
    auto base_var = mbd.Get()->AllocateSparse(l, flag_uninitialized);

    // now allocate in all other stages
    for (auto stage : mbd.Stages()) {
      if (stage.first == "base") {
        // we've already done this
        continue;
      }

      auto v = stage.second->GetCellVarPtr(l);

      if (v->IsSet(Metadata::OneCopy)) {
        // nothing to do, we already allocated variable on base stage, and all other
        // stages share that variable
        continue;
      }

      if (!v->IsAllocated()) {
        // allocate data of target variable
        v->AllocateData(this, flag_uninitialized);

        // copy fluxes and boundary variable from variable on base stage
        v->CopyFluxesAndBdryVar(base_var.get());
      }
    }
  };

  bool cont_set = false;
  if ((pmy_mesh != nullptr) && pmy_mesh->resolved_packages) {
    cont_set = pmy_mesh->resolved_packages->ControlVariablesSet();
  }

  if (cont_set && meshblock_data.Get()->GetCellVarPtr(label)->IsSparse()) {
    auto clabel = label; 
    if (!only_control) clabel = pmy_mesh->resolved_packages->GetFieldController(label);
    const auto &var_labels = pmy_mesh->resolved_packages->GetControlledVariables(clabel);
    for (const auto &l : var_labels)
      AllocateVar(l);
  } else {
    AllocateVar(label);
  }
}

void MeshBlock::DeallocateSparse(std::string const &label) {
  auto &mbd = meshblock_data;
  auto DeallocateVar = [&mbd](const std::string &l) {
    for (auto stage : mbd.Stages()) {
      stage.second->DeallocateSparse(l);
    }
  };

  bool cont_set = false;
  if ((pmy_mesh != nullptr) && pmy_mesh->resolved_packages) {
    cont_set = pmy_mesh->resolved_packages->ControlVariablesSet();
  }

  if (cont_set && meshblock_data.Get()->GetCellVarPtr(label)->IsSparse()) {
    const auto &var_labels = pmy_mesh->resolved_packages->GetControlledVariables(label);
    for (const auto &l : var_labels)
      DeallocateVar(l);
  } else {
    DeallocateVar(label);
  }
}

} // namespace parthenon
