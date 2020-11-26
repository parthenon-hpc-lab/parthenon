//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
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

#ifndef BVALS_CC_BVALS_CC_IN_ONE_HPP_
#define BVALS_CC_BVALS_CC_IN_ONE_HPP_

#include <memory>
#include <string>

#include "basic_types.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

template <typename T>
class MeshData;

namespace cell_centered_bvars {
void CalcIndicesSetSame(int ox, int &s, int &e, const IndexRange &bounds);
void CalcIndicesSetFromCoarser(const int &ox, int &s, int &e, const IndexRange &bounds,
                               const std::int64_t &lx, const int &cng, bool include_dim);
void CalcIndicesSetFromFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                             const NeighborBlock &nb, MeshBlock *pmb);
void CalcIndicesLoadSame(int ox, int &s, int &e, const IndexRange &bounds);
void CalcIndicesLoadToFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                            const NeighborBlock &nb, MeshBlock *pmb);
TaskStatus SendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus SetBoundaries(std::shared_ptr<MeshData<Real>> &md);
} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BVALS_CC_IN_ONE_HPP_
