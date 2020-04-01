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

// C headers

// C++ headers
#include <algorithm>  // sort()
#include <cstdlib>
#include <cstring>    // memcpy()
#include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <iterator>

// Athena++ headers
#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "meshblock_tree.hpp"
#include "interface/Metadata.hpp"
#include "interface/Variable.hpp"
#include "interface/ContainerIterator.hpp"

namespace parthenon {
//----------------------------------------------------------------------------------------
// MeshBlock constructor: constructs coordinate, boundary condition, field
//                        and mesh refinement objects.
MeshBlock::MeshBlock(int igid, int ilid, LogicalLocation iloc, RegionSize input_block,
                     BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin,
                     Properties_t& properties,
                     Packages_t& packages,
                     int igflag, bool ref_flag) :
    exec_space(DevSpace()),
    pmy_mesh(pm), loc(iloc), block_size(input_block),
    gid(igid), lid(ilid), gflag(igflag), nuser_out_var(), 
    properties(properties), packages(packages),
    prev(nullptr), next(nullptr),
    new_block_dt_{}, new_block_dt_hyperbolic_{}, new_block_dt_parabolic_{},
    new_block_dt_user_{},
    nreal_user_meshblock_data_(), nint_user_meshblock_data_(),
    cost_(1.0) {
  // initialize grid indices
  is = NGHOST;
  ie = is + block_size.nx1 - 1;

  ncells1 = block_size.nx1 + 2*NGHOST;
  ncc1 = block_size.nx1/2 + 2*NGHOST;
  if (pmy_mesh->ndim >= 2) {
    js = NGHOST;
    je = js + block_size.nx2 - 1;
    ncells2 = block_size.nx2 + 2*NGHOST;
    ncc2 = block_size.nx2/2 + 2*NGHOST;
  } else {
    js = je = 0;
    ncells2 = 1;
    ncc2 = 1;
  }

  if (pmy_mesh->ndim >= 3) {
    ks = NGHOST;
    ke = ks + block_size.nx3 - 1;
    ncells3 = block_size.nx3 + 2*NGHOST;
    ncc3 = block_size.nx3/2 + 2*NGHOST;
  } else {
    ks = ke = 0;
    ncells3 = 1;
    ncc3 = 1;
  }

  Container<Real>& real_container = real_containers.Get();

  // Set the block pointer for the containers
  real_container.setBlock(this);

  if (pm->multilevel) {
    cnghost = (NGHOST + 1)/2 + 1;
    cis = NGHOST; cie = cis + block_size.nx1/2 - 1;
    cjs = cje = cks = cke = 0;
    if (pmy_mesh->ndim >= 2) // 2D or 3D
      cjs = NGHOST, cje = cjs + block_size.nx2/2 - 1;
    if (pmy_mesh->ndim >= 3) // 3D
      cks = NGHOST, cke = cks + block_size.nx3/2 - 1;
  }

  // construct objects stored in MeshBlock class.  Note in particular that the initial
  // conditions for the simulation are set in problem generator called from main

  // mesh-related objects
  // Boundary
  pbval  = std::make_unique<BoundaryValues>(this, input_bcs, pin);
  pbval->SetBoundaryFlags(boundary_flag);

  // Coordinates
  pcoord = std::make_unique<Cartesian>(this, pin, false);

  // Set the block for containers
  real_container.setBlock(this);

  // Reconstruction: constructor may implicitly depend on Coordinates, and PPM variable
  // floors depend on EOS, but EOS isn't needed in Reconstruction constructor-> this is ok
  precon = std::make_unique<Reconstruction>(this, pin);

  // physics-related, per-MeshBlock objects: may depend on Coordinates for diffusion
  // terms, and may enroll quantities in AMR and BoundaryVariable objs. in BoundaryValues
  //  if (Globals::my_rank == 0) { real_container.print(); }

  // KGF: suboptimal solution, since developer must copy/paste BoundaryVariable derived
  // class type that is used in each PassiveScalars, Field, ... etc. class
  // in order to correctly advance the BoundaryValues::bvars_next_phys_id_ local counter.

  // TODO(felker): check that local counter pbval->bvars_next_phys_id_ agrees with shared
  // Mesh::next_phys_id_ counter (including non-BoundaryVariable / per-MeshBlock reserved
  // values). Compare both private member variables via BoundaryValues::CheckCounterPhysID

  // adding a dummy variable to container to test comms
  /*Metadata m;
  m = Metadata({Metadata::Cell, Metadata::Advected, Metadata::FillGhost});
  real_container.Add(std::string("TestGhost"),m);
  Variable<Real> &styx = real_container.Get("TestGhost");
  Real *data = styx.data();
  for (int k=0; k<styx.GetSize(); k++) data[k] = pcoord->x1f(0,0,0);
  */
  // end dummy variable

  // Add field properties data
  //std::cerr << "Adding " << properties.size() << " properties to block" << std::endl;
  for (int i = 0; i < properties.size(); i++) {
    StateDescriptor& state = properties[i]->State();
    for (auto const & q : state.AllFields()) {
      real_container.Add(q.first, q.second);
    }
  }
  // Add physics data
  for (auto const & pkg : packages) {
    //std::cerr << "  Physics: " << ph.first << std::endl;
    for (auto const & q : pkg.second->AllFields()) {
      //std::cerr << "    Adding " << q.first << std::endl;
      std::cerr << "adding " << q.first << " " << q.second.IsSet(Metadata::FillGhost) << std::endl;
      real_container.Add(q.first, q.second);
    }
  }

  // TODO: Should these loops be moved to Variable creation
  ContainerIterator<Real> ci(real_container, {Metadata::Independent});
  int nindependent = ci.vars.size();
  for (int n=0; n<nindependent; n++) {
    RegisterMeshBlockData(ci.vars[n]);
  }

  if (pm->multilevel) {
    pmr = std::make_unique<MeshRefinement>(this, pin);
    // This is very redundant, I think, but necessary for now
    for (int n=0; n<nindependent; n++) {
      pmr->AddToRefinement(&(ci.vars[n]->data), ci.vars[n]->coarse_s.get());
    }
  }

  // Create user mesh data
  //InitUserMeshBlockData(pin);
  app = InitApplicationMeshBlockData(pin);
}

//----------------------------------------------------------------------------------------
// MeshBlock constructor for restarts
#if 0
MeshBlock::MeshBlock(int igid, int ilid, Mesh *pm, ParameterInput *pin,
                     Properties_t& properties, Packages_t& packages,
                     LogicalLocation iloc, RegionSize input_block,
                     BoundaryFlag *input_bcs,
                     double icost, char *mbdata, int igflag) :
    pmy_mesh(pm), loc(iloc), block_size(input_block),
    gid(igid), lid(ilid), gflag(igflag), nuser_out_var(), 
    properties(properties), packages(packages),
    prev(nullptr), next(nullptr),
    new_block_dt_{}, new_block_dt_hyperbolic_{}, new_block_dt_parabolic_{},
    new_block_dt_user_{},
    nreal_user_meshblock_data_(), nint_user_meshblock_data_(), cost_(icost),
    exec_space(DevSpace()) {
  // initialize grid indices

  //std::cerr << "WHY AM I HERE???" << std::endl;

  is = NGHOST;
  ie = is + block_size.nx1 - 1;

  ncells1 = block_size.nx1 + 2*NGHOST;
  ncc1 = block_size.nx1/2 + 2*NGHOST;
  if (pmy_mesh->ndim >= 2) {
    js = NGHOST;
    je = js + block_size.nx2 - 1;
    ncells2 = block_size.nx2 + 2*NGHOST;
    ncc2 = block_size.nx2/2 + 2*NGHOST;
  } else {
    js = je = 0;
    ncells2 = 1;
    ncc2 = 1;
  }

 if (pmy_mesh->ndim >= 3) {
    ks = NGHOST;
    ke = ks + block_size.nx3 - 1;
    ncells3 = block_size.nx3 + 2*NGHOST;
    ncc3 = block_size.nx3/2 + 2*NGHOST;
  } else {
    ks = ke = 0;
    ncells3 = 1;
    ncc3 = 1;
  }

  // Set the block pointer for the containers
  real_containers.Get().setBlock(this);

  if (pm->multilevel) {
    cnghost = (NGHOST + 1)/2 + 1;
    cis = NGHOST; cie = cis + block_size.nx1/2 - 1;
    cjs = cje = cks = cke = 0;
    if (pmy_mesh->ndim >= 2) // 2D or 3D
      cjs = NGHOST, cje = cjs + block_size.nx2/2 - 1;
    if (pmy_mesh->ndim >= 3) // 3D
      cks = NGHOST, cke = cks + block_size.nx3/2 - 1;
  }

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
  for (int n=0; n<nint_user_meshblock_data_; n++) {
    std::memcpy(iuser_meshblock_data[n].data(), &(mbdata[os]),
                iuser_meshblock_data[n].GetSizeInBytes());
    os += iuser_meshblock_data[n].GetSizeInBytes();
  }
  for (int n=0; n<nreal_user_meshblock_data_; n++) {
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

  // delete user output variables array
#if 0
  if (nuser_out_var > 0) {
    delete [] user_out_var_names_;
  }
  // delete user MeshBlock data
  if (nreal_user_meshblock_data_ > 0) delete [] ruser_meshblock_data;
  if (nint_user_meshblock_data_ > 0) delete [] iuser_meshblock_data;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::AllocateRealUserMeshBlockDataField(int n)
//  \brief Allocate Real ParArrayNDs for user-defned data in MeshBlock

void MeshBlock::AllocateRealUserMeshBlockDataField(int n) {
  if (nreal_user_meshblock_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::AllocateRealUserMeshBlockDataField"
        << std::endl << "User MeshBlock data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nreal_user_meshblock_data_ = n;
  ruser_meshblock_data = nullptr;//new ParArrayND<Real>[n];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::AllocateIntUserMeshBlockDataField(int n)
//  \brief Allocate integer ParArrayNDs for user-defned data in MeshBlock

void MeshBlock::AllocateIntUserMeshBlockDataField(int n) {
  if (nint_user_meshblock_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::AllocateIntusermeshblockDataField"
        << std::endl << "User MeshBlock data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  nint_user_meshblock_data_=n;
  iuser_meshblock_data = nullptr;//new ParArrayND<int>[n];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::AllocateUserOutputVariables(int n)
//  \brief Allocate user-defined output variables

void MeshBlock::AllocateUserOutputVariables(int n) {
  if (n <= 0) return;
  else throw std::runtime_error("Not yet implemented in parthenon");
  if (nuser_out_var != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::AllocateUserOutputVariables"
        << std::endl << "User output variables are already allocated." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  nuser_out_var = n;
  user_out_var.NewParArrayND(nuser_out_var, ncells3, ncells2, ncells1);
  user_out_var_names_ = new std::string[n];
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::SetUserOutputVariableName(int n, const char *name)
//  \brief set the user-defined output variable name

void MeshBlock::SetUserOutputVariableName(int n, const char *name) {
  if (n >= nuser_out_var) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshBlock::SetUserOutputVariableName"
        << std::endl << "User output variable is not allocated." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  user_out_var_names_[n] = name;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn std::size_t MeshBlock::GetBlockSizeInBytes()
//  \brief Calculate the block data size required for restart.

std::size_t MeshBlock::GetBlockSizeInBytes() {
  std::size_t size=0;
  // calculate user MeshBlock data size
  for (int n=0; n<nint_user_meshblock_data_; n++)
    size += iuser_meshblock_data[n].GetSizeInBytes();
  for (int n=0; n<nreal_user_meshblock_data_; n++)
    size += ruser_meshblock_data[n].GetSizeInBytes();

  return size;
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


void MeshBlock::RegisterMeshBlockData(std::shared_ptr<Variable<Real>> pvar_cc) {
  vars_cc_.push_back(pvar_cc);
  return;
}


void MeshBlock::RegisterMeshBlockData(std::shared_ptr<FaceField> pvar_fc) {
  vars_fc_.push_back(pvar_fc);
  return;
}
}
