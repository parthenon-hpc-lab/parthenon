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
//! \file bvals.cpp
//  \brief constructor/destructor and utility functions for BoundaryValues class

#include "bvals/bvals.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "parthenon_mpi.hpp"

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/buffer_utils.hpp"

namespace parthenon {

// BoundaryValues constructor (the first object constructed inside the MeshBlock()
// constructor): sets functions for the appropriate boundary conditions at each of the 6
// dirs of a MeshBlock
BoundaryValues::BoundaryValues(std::weak_ptr<MeshBlock> wpmb, BoundaryFlag *input_bcs,
                               ParameterInput *pin)
    : BoundaryBase(wpmb.lock()->pmy_mesh, wpmb.lock()->loc, wpmb.lock()->block_size,
                   input_bcs),
      pmy_block_(wpmb) {
  // Check BC functions for each of the 6 boundaries in turn ---------------------
  for (int i = 0; i < 6; i++) {
    switch (block_bcs[i]) {
    case BoundaryFlag::reflect:
    case BoundaryFlag::outflow:
      apply_bndry_fn_[i] = true;
      break;
    default: // already initialized to false in class
      break;
    }
  }
  // Inner x1
  nface_ = 2;
  nedge_ = 0;
  CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x1], CoordinateDirection::X1DIR);
  CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x1], CoordinateDirection::X1DIR);

  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  if (pmb->block_size.nx2 > 1) {
    nface_ = 4;
    nedge_ = 4;
    CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x2], CoordinateDirection::X2DIR);
    CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x2], CoordinateDirection::X2DIR);
  }

  if (pmb->block_size.nx3 > 1) {
    nface_ = 6;
    nedge_ = 12;
    CheckBoundaryFlag(block_bcs[BoundaryFace::inner_x3], CoordinateDirection::X3DIR);
    CheckBoundaryFlag(block_bcs[BoundaryFace::outer_x3], CoordinateDirection::X3DIR);
  }

  // prevent reallocation of contiguous memory space for each of 4x possible calls to
  // std::vector<BoundaryVariable *>.push_back() in Field, PassiveScalars
  bvars.reserve(3);

  // Matches initial value of Mesh::next_phys_id_
  // reserve phys=0 for former TAG_AMR=8; now hard-coded in Mesh::CreateAMRMPITag()
  bvars_next_phys_id_ = 1;
}

// destructor

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::SetupPersistentMPI()
//  \brief Setup persistent MPI requests to be reused throughout the entire simulation

void BoundaryValues::SetupPersistentMPI() {
  for (auto bvars_it = bvars.begin(); bvars_it != bvars.end(); ++bvars_it) {
    (*bvars_it)->SetupPersistentMPI();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::StartReceiving(BoundaryCommSubset phase)
//  \brief initiate MPI_Irecv()

void BoundaryValues::StartReceiving(BoundaryCommSubset phase) {
  for (auto bvars_it = bvars.begin(); bvars_it != bvars.end(); ++bvars_it) {
    (*bvars_it)->StartReceiving(phase);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void BoundaryValues::ClearBoundary(BoundaryCommSubset phase)
//  \brief clean up the boundary flags after each loop

void BoundaryValues::ClearBoundary(BoundaryCommSubset phase) {
  // Note BoundaryCommSubset::mesh_init corresponds to initial exchange of conserved fluid
  // variables and magentic fields
  for (auto bvars_it = bvars.begin(); bvars_it != bvars.end(); ++bvars_it) {
    (*bvars_it)->ClearBoundary(phase);
  }
}

// Public function, to be called in MeshBlock ctor for keeping MPI tag bitfields
// consistent across MeshBlocks, even if certain MeshBlocks only construct a subset of
// physical variable classes

int BoundaryValues::AdvanceCounterPhysID(int num_phys) {
#ifdef MPI_PARALLEL
  // TODO(felker): add safety checks? input, output are positive, obey <= 31= MAX_NUM_PHYS
  int start_id = bvars_next_phys_id_;
  bvars_next_phys_id_ += num_phys;
  return start_id;
#else
  return 0;
#endif
}

} // namespace parthenon
