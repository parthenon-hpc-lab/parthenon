//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "interface/container_iterator.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_tree.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// MeshBlock constructor: constructs coordinate, boundary condition, field
//                        and mesh refinement objects.
MeshBlock::MeshBlock(int igid, int ilid, LogicalLocation iloc, RegionSize input_block,
                     BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin,
                     Properties_t &properties, Packages_t &packages, int igflag,
                     bool ref_flag)
    : exec_space(DevExecSpace()), pmy_mesh(pm), loc(iloc), block_size(input_block),
      gid(igid), lid(ilid), gflag(igflag), properties(properties), packages(packages),
      prev(nullptr), next(nullptr), new_block_dt_{}, new_block_dt_hyperbolic_{},
      new_block_dt_parabolic_{}, new_block_dt_user_{}, cost_(1.0) {
  // initialize grid indices
  InitializeIndexShapes();

  Container<Real> &real_container = real_containers.Get();
  // Set the block pointer for the containers
  real_container.setBlock(this);

  // (probably don't need to preallocate space for references in these vectors)
  vars_cc_.reserve(3);
  vars_fc_.reserve(3);

  // construct objects stored in MeshBlock class.  Note in particular that the initial
  // conditions for the simulation are set in problem generator called from main

  // mesh-related objects
  // Boundary
  pbval = std::make_unique<BoundaryValues>(this, input_bcs, pin);
  pbval->SetBoundaryFlags(boundary_flag);

  // Coordinates
  pcoord = std::make_unique<Cartesian>(this, pin, false);

  // Set the block for containers
  real_container.setBlock(this);

  // Reconstruction: constructor may implicitly depend on Coordinates, and PPM variable
  // floors depend on EOS, but EOS isn't needed in Reconstruction constructor-> this is ok
  precon = std::make_unique<Reconstruction>(this, pin);

  // Add field properties data
  for (int i = 0; i < properties.size(); i++) {
    StateDescriptor &state = properties[i]->State();
    for (auto const &q : state.AllFields()) {
      real_container.Add(q.first, q.second);
    }
  }
  // Add physics data
  for (auto const &pkg : packages) {
    for (auto const &q : pkg.second->AllFields()) {
      real_container.Add(q.first, q.second);
    }
    for (auto const &q : pkg.second->AllSparseFields()) {
      for (auto const &m : q.second) {
        real_container.Add(q.first, m);
      }
    }
  }

  // TODO(jdolence): Should these loops be moved to Variable creation
  ContainerIterator<Real> ci(real_container, {Metadata::Independent});
  int nindependent = ci.vars.size();
  for (int n = 0; n < nindependent; n++) {
    RegisterMeshBlockData(ci.vars[n]);
  }

  if (pm->multilevel) {
    pmr = std::make_unique<MeshRefinement>(this, pin);
    // This is very redundant, I think, but necessary for now
    for (int n = 0; n < nindependent; n++) {
      pmr->AddToRefinement(ci.vars[n]->data, ci.vars[n]->coarse_s);
    }
  }

  // Create user mesh data
  // InitUserMeshBlockData(pin);
  app = InitApplicationMeshBlockData(pin);
}


//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts
#if 0
MeshBlock::MeshBlock(int igid, int ilid, Mesh *pm, ParameterInput *pin,
                     Properties_t &properties, Packages_t &packages, LogicalLocation iloc,
                     RegionSize input_block, BoundaryFlag *input_bcs, double icost,
                     char *mbdata, int igflag)
    : pmy_mesh(pm), loc(iloc), block_size(input_block), gid(igid), lid(ilid),
      gflag(igflag), nuser_out_var(), properties(properties), packages(packages),
      prev(nullptr), next(nullptr), new_block_dt_{}, new_block_dt_hyperbolic_{},
      new_block_dt_parabolic_{}, new_block_dt_user_{}, nreal_user_meshblock_data_(),
      nint_user_meshblock_data_(), cost_(icost), exec_space(DevExecSpace()) {
  // initialize grid indices

  // std::cerr << "WHY AM I HERE???" << std::endl;

  // initialize grid indices
  InitializeIndexShapes();
    
  // Set the block pointer for the containers
  real_containers.Get().setBlock(this);

  // (re-)create mesh-related objects in MeshBlock

  // Boundary
  pbval = std::make_unique<BoundaryValues>(this, input_bcs, pin);

  // Coordinates
  pcoord = std::make_unique<Cartesian>(this, pin, false);

  // Reconstruction (constructor may implicitly depend on Coordinates)
  precon = std::make_unique<Reconstruction>(this, pin);

  if (pm->multilevel) pmr = std::make_unique<MeshRefinement>(this, pin);

  app = InitApplicationMeshBlockData(pin);
  InitUserMeshBlockData(pin);

  std::size_t os = 0;

  // load user MeshBlock data
  for (int n = 0; n < nint_user_meshblock_data_; n++) {
    std::memcpy(iuser_meshblock_data[n].data(), &(mbdata[os]),
                iuser_meshblock_data[n].GetSizeInBytes());
    os += iuser_meshblock_data[n].GetSizeInBytes();
  }
  for (int n = 0; n < nreal_user_meshblock_data_; n++) {
    std::memcpy(ruser_meshblock_data[n].data(), &(mbdata[os]),
                ruser_meshblock_data[n].GetSizeInBytes());
    os += ruser_meshblock_data[n].GetSizeInBytes();
  }

  return;
}
#endif

//----------------------------------------------------------------------------------------
// MeshBlock destructor

MeshBlock::~MeshBlock() {
  if (prev != nullptr) prev->next = next;
  if (next != nullptr) next->prev = prev;
}

void MeshBlock::InitializeIndexShapes() {

  cellbounds = IndexShape(block_size.nx1,block_size.nx2,block_size.nx3,NGHOST);

  const IndexDomain interior = IndexDomain::interior;
  if (pmy_mesh->multilevel) {

    cnghost = (NGHOST + 1)/2 + 1;
    c_cellbounds = IndexShape(
      cellbounds.ncellsi(interior)/2,
      cellbounds.ncellsj(interior)/2,
      cellbounds.ncellsk(interior)/2,
      NGHOST);

  }else{

    c_cellbounds = IndexShape(
      cellbounds.ncellsi(interior)/2,
      cellbounds.ncellsj(interior)/2,
      cellbounds.ncellsk(interior)/2,
      0);
  }

}

//----------------------------------------------------------------------------------------
//! \fn std::size_t MeshBlock::GetBlockSizeInBytes()
//  \brief Calculate the block data size required for restart.

std::size_t MeshBlock::GetBlockSizeInBytes() {
  throw std::runtime_error("MeshBlock::GetBlockSizeInBytes not yet implemented.");
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
#ifdef OPENMP_PARALLEL
    lb_time_ = omp_get_wtime();
#else
    lb_time_ = static_cast<double>(clock());
#endif
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::StartTimeMeasurement()
//  \brief stop time measurement and accumulate it in the MeshBlock cost

void MeshBlock::StopTimeMeasurement() {
  if (pmy_mesh->lb_automatic_) {
#ifdef OPENMP_PARALLEL
    lb_time_ = omp_get_wtime() - lb_time_;
#else
    lb_time_ = static_cast<double>(clock()) - lb_time_;
#endif
    cost_ += lb_time_;
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
