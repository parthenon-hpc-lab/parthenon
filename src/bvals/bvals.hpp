//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//! \file bvals.hpp
//  \brief defines BoundarySwarms 

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "basic_types.hpp"
#include "parthenon_mpi.hpp"

#include "bvals/bvals_interfaces.hpp"
#include "bvals/comms/bnd_info.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "defs.hpp"
#include "mesh/domain.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// forward declarations
// TODO(felker): how many of these foward declarations are actually needed now?
// Can #include "./bvals_interfaces.hpp" suffice?
template <typename T>
class Variable;
class Mesh;
class MeshBlock;
class MeshBlockTree;
class ParameterInput;
struct RegionSize;

// free functions to return boundary flag given input string, and vice versa
BoundaryFlag GetBoundaryFlag(const std::string &input_string);
std::string GetBoundaryString(BoundaryFlag input_flag);
// + confirming that the MeshBlock's boundaries are all valid selections
void CheckBoundaryFlag(BoundaryFlag block_flag, CoordinateDirection dir);

//----------------------------------------------------------------------------------------
//! \class BoundarySwarms
//  \brief centralized class for interacting with each individual swarm boundary data
class BoundarySwarms : public BoundaryCommunication {
 public:
  BoundarySwarms(std::weak_ptr<MeshBlock> pmb, BoundaryFlag *input_bcs,
                 ParameterInput *pin);

  // variable-length arrays of references to all BoundarySwarm instances
  std::vector<std::shared_ptr<BoundarySwarm>> bswarms;

  void SetBoundaryFlags(BoundaryFlag bc_flag[]) {
    for (int i = 0; i < 6; i++) {
      bc_flag[i] = block_bcs[i];
    }
  }

  // inherited functions:
  // ------
  // called before time-stepper:
  void SetupPersistentMPI() final; // setup MPI requests

  // called before and during time-stepper (currently do nothing for swarms):
  void StartReceiving(BoundaryCommSubset phase) final {}
  void ClearBoundary(BoundaryCommSubset phase) final {}

 private:
  // ptr to MeshBlock containing this BoundarySwarms
  std::weak_ptr<MeshBlock> pmy_block_;
  int nface_, nedge_;
  BoundaryFlag block_bcs[6];

  // if a BoundaryPhysics or user fn should be applied at each MeshBlock boundary
  // false --> e.g. block, polar, periodic boundaries
  // bool apply_bndry_fn_[6]{}; // C++11: in-class initializer of non-static member
  // C++11: direct-list-initialization -> value init of array -> zero init of each scalar

  /// Returns shared pointer to a block
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (pmy_block_.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block_.lock();
  }

  friend class Mesh;
  // currently, this class friendship is required for copying send/recv buffers between
  // BoundarySwarm objects within different MeshBlocks on the same MPI rank:
  friend class BoundarySwarm;
};

} // namespace parthenon

#endif // BVALS_BVALS_HPP_
