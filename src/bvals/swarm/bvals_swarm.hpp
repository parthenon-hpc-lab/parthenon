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
#ifndef BVALS_SWARM_BVALS_SWARM_HPP_
#define BVALS_SWARM_BVALS_SWARM_HPP_
//! \file bvals_swarm.hpp
//  \brief handle boundaries for a swarm (representing a set of particles)

#include <memory>

#include "parthenon_mpi.hpp"

#include "bvals/bvals.hpp"
#include "defs.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \class BoundarySwarm
//  \brief

class BoundarySwarm : public BoundaryCommunication {
 public:
  BoundarySwarm(std::weak_ptr<MeshBlock> pmb);
  ~BoundarySwarm() = default;

  std::vector<ParArrayND<int>> vars_int;
  std::vector<ParArrayND<Real>> vars_real;

  // (usuallly the std::size_t unsigned integer type)
  std::vector<BoundaryVariable *>::size_type bvar_index;

  // BoundaryCommunication
  void SetupPersistentMPI() final;
  void StartReceiving(BoundaryCommSubset phase) final;
  void ClearBoundary(BoundaryCommSubset phase) final;
  void Receive(BoundaryCommSubset phase);
  void Send(BoundaryCommSubset phase);

  BoundaryData<> bd_var_;
  std::weak_ptr<MeshBlock> pmy_block;
  Mesh *pmy_mesh_;
  int send_tag[56], recv_tag[56];
  int particle_size, send_size[56], recv_size[56];

 protected:
  int nl_, nu_;
  void InitBoundaryData(BoundaryData<> &bd);

 private:
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (pmy_block.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block.lock();
  }

#ifdef MPI_PARALLEL
  int swarm_id_;
#endif
};

} // namespace parthenon

#endif // BVALS_SWARM_BVALS_SWARM_HPP_
