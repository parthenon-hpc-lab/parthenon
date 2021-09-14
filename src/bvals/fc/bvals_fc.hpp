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
#ifndef BVALS_FC_BVALS_FC_HPP_
#define BVALS_FC_BVALS_FC_HPP_
//! \file bvals_fc.hpp
//  \brief handle boundaries for any FaceField type variable that represents a physical
//         quantity indexed along / located around face-centers of cells

#include <memory>

#include "parthenon_mpi.hpp"

#include "bvals/bvals.hpp"

namespace parthenon {
//----------------------------------------------------------------------------------------
//! \class FaceCenteredBoundaryVariable
//  \brief

class FaceCenteredBoundaryVariable : public BoundaryVariable {
 public:
  FaceCenteredBoundaryVariable(std::weak_ptr<MeshBlock> pmb, FaceField *var,
                               FaceField &coarse_buf, EdgeField &var_flux);
  ~FaceCenteredBoundaryVariable();

  // may want to rebind var_fc to b, b1, b2, etc. Hence ptr member, not reference
  FaceField *var_fc;

  // never need to rebind FaceCentered coarse_buf, so it can be
  // a reference member: ---> must be initialized in initializer list; cannot pass nullptr
  FaceField &coarse_buf;

  // maximum number of reserved unique "physics ID" component of MPI tag bitfield
  // must correspond to the # of "int *phys_id_" private members, below. Convert to array?
  static constexpr int max_phys_id = 5;

  // BoundaryVariable:
  int ComputeVariableBufferSize(const NeighborIndexes &ni, int cng) final;
  int ComputeFluxCorrectionBufferSize(const NeighborIndexes &ni, int cng) final;

  // BoundaryCommunication:
  void SetupPersistentMPI() final;
  void StartReceiving(BoundaryCommSubset phase) final;
  void ClearBoundary(BoundaryCommSubset phase) final;

  // BoundaryBuffer:
  void SendFluxCorrection() final;
  bool ReceiveFluxCorrection(bool is_allocated) final;

 private:
  bool edge_flag_[12];
  int nedge_fine_[12];

  // variable switch used in 2x functions, ReceiveFluxCorrection() and StartReceiving():
  // ready to recv flux from same level and apply correction? false= 2nd pass for fine lvl
  bool recv_flx_same_lvl_;

#ifdef MPI_PARALLEL
  int fc_phys_id_, fc_flx_phys_id_;
#endif

  // BoundaryBuffer:
  int LoadBoundaryBufferSameLevel(BufArray1D<Real> &buf, const NeighborBlock &nb) final;
  void SetBoundarySameLevel(BufArray1D<Real> &buf, const NeighborBlock &nb) final;
  int LoadBoundaryBufferToCoarser(BufArray1D<Real> &buf, const NeighborBlock &nb) final;
  int LoadBoundaryBufferToFiner(BufArray1D<Real> &buf, const NeighborBlock &nb) final;
  void SetBoundaryFromCoarser(BufArray1D<Real> &buf, const NeighborBlock &nb) final;
  void SetBoundaryFromFiner(BufArray1D<Real> &buf, const NeighborBlock &nb) final;

  void CountFineEdges(); // called in SetupPersistentMPI()

  void RemapFlux(const int k, const int jinner, const int jouter, const int i,
                 const Real eps, const ParArrayND<Real> &var, ParArrayND<Real> &flux);
};

} // namespace parthenon

#endif // BVALS_FC_BVALS_FC_HPP_
