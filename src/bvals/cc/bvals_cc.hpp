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
#ifndef BVALS_CC_BVALS_CC_HPP_
#define BVALS_CC_BVALS_CC_HPP_
//! \file bvals_cc.hpp
//  \brief handle boundaries for any ParArrayND type variable that represents a physical
//         quantity indexed along / located around cell-centers

#include <memory>
#include <string>

#include "parthenon_mpi.hpp"

#include "bvals/bvals.hpp"
#include "defs.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//  \brief

class CellCenteredBoundaryVariable : public BoundaryVariable {
 public:
  CellCenteredBoundaryVariable(std::weak_ptr<MeshBlock> pmb, bool is_sparse,
                               const std::string &label, int dim4);
  ~CellCenteredBoundaryVariable();

  // may want to rebind var_cc to u,u1,u2,w,w1, etc. registers for time integrator logic.
  ParArrayND<Real> var_cc;
  ParArrayND<Real> coarse_buf; // may pass nullptr if mesh refinement is unsupported

  // currently, no need to ever switch flux[] ---> keep as reference members (not ptrs)
  // flux[3] w/ 3x empty ParArrayNDs may be passed if mesh refinement is unsupported, but
  // nullptr is not allowed
  ParArrayND<Real> x1flux, x2flux, x3flux;

  // maximum number of reserved unique "physics ID" component of MPI tag bitfield
  // (CellCenteredBoundaryVariable only actually uses 1x if multilevel==false)
  // must correspond to the # of "int *phys_id_" private members, below. Convert to array?
  static constexpr int max_phys_id = 3;

  void Reset(ParArrayND<Real> var, ParArrayND<Real> coarse_var,
             ParArrayND<Real> *var_flux);

  // BoundaryVariable:
  struct VariableBufferSizes {
    int same, c2f, f2c;
  };
  VariableBufferSizes ComputeVariableBufferSizes(const NeighborIndexes &ni, int cng);
  int ComputeVariableBufferSize(const NeighborIndexes &ni, int cng) final;
  int ComputeFluxCorrectionBufferSize(const NeighborIndexes &ni, int cng) final;

  // BoundaryCommunication:
  void SetupPersistentMPI() final;
  void StartReceiving(BoundaryCommSubset phase) final;
  void ClearBoundary(BoundaryCommSubset phase) final;

  // BoundaryBuffer:
  void SendFluxCorrection(bool is_allocated) final;
  bool ReceiveFluxCorrection(bool is_allocated) final;

 protected:
  int nl_, nu_;

#ifdef MPI_PARALLEL
  int cc_phys_id_, cc_flx_phys_id_;
#endif

  void RemapFlux(const int n, const int k, const int jinner, const int jouter,
                 const int i, const Real eps, const ParArrayND<Real> &var,
                 ParArrayND<Real> &flux);
};

} // namespace parthenon

#endif // BVALS_CC_BVALS_CC_HPP_
