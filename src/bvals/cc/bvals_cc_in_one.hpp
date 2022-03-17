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
#include "bvals/bvals_interfaces.hpp"
#include "coordinates/coordinates.hpp"

namespace parthenon {

template <typename T>
class MeshData;
class IndexRange;
class NeighborBlock;
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

struct BndInfo {
  int si = 0;
  int ei = 0;
  int sj = 0;
  int ej = 0;
  int sk = 0;
  int ek = 0;
  bool allocated = true;
  bool restriction = false;
  Coordinates_t coords, coarse_coords; // coords
  parthenon::BufArray1D<Real> buf;     // comm buffer
};

// We have one CommBndInfo per variable per neighbor per block
struct CommBndInfo : public BndInfo {
  int Nt = 0;
  int Nu = 0;
  int Nv = 0;
  parthenon::ParArray6D<Real> var;    // data variable used for comms
  parthenon::ParArray6D<Real> fine;   // fine data variable for prolongation/restriction
  parthenon::ParArray6D<Real> coarse; // coarse data variable for prolongation/restriction
};

// We have one RefineBndInfo per (l, m, :, :, :, :) element per variable per neighbor per
// block
struct RefineBndInfo : public BndInfo {
  int Nv = 0;
  parthenon::ParArray4D<Real> var;    // data variable used for comms
  parthenon::ParArray4D<Real> fine;   // fine data variable for prolongation/restriction
  parthenon::ParArray4D<Real> coarse; // coarse data variable for prolongation/restriction
};

using CommBufferCache_t = ParArray1D<CommBndInfo>;
using CommBufferCacheHost_t = typename CommBufferCache_t::HostMirror;
using RefineBufferCache_t = ParArray1D<RefineBndInfo>;
using RefineBufferCacheHost_t = typename RefineBufferCache_t::HostMirror;

} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BVALS_CC_IN_ONE_HPP_
