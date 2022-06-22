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
#include <vector>

#include "basic_types.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "coordinates/coordinates.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

template <typename T>
class MeshData;
class IndexRange;
class NeighborBlock;
template <typename T>
class CellVariable;

namespace cell_centered_bvars {
void CalcIndicesSetSame(int ox, int &s, int &e, const IndexRange &bounds);
void CalcIndicesSetFromCoarser(const int &ox, int &s, int &e, const IndexRange &bounds,
                               const std::int64_t &lx, const int &cng, bool include_dim);
void CalcIndicesSetFromFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                             const NeighborBlock &nb, MeshBlock *pmb);
void CalcIndicesLoadSame(int ox, int &s, int &e, const IndexRange &bounds);
void CalcIndicesLoadToFiner(int &si, int &ei, int &sj, int &ej, int &sk, int &ek,
                            const NeighborBlock &nb, MeshBlock *pmb);

int GetBufferSize(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                  std::shared_ptr<CellVariable<Real>> v);

TaskStatus BuildSparseBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);

TaskStatus SendBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus ReceiveBoundaryBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus SetBoundaries(std::shared_ptr<MeshData<Real>> &md);

TaskStatus LoadAndSendSparseFluxCorrectionBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus ReceiveSparseFluxCorrectionBuffers(std::shared_ptr<MeshData<Real>> &md);
TaskStatus SetFluxCorrections(std::shared_ptr<MeshData<Real>> &md);

struct BndInfo {
  int si = 0;
  int ei = 0;
  int sj = 0;
  int ej = 0;
  int sk = 0;
  int ek = 0;

  int Nt = 0;
  int Nu = 0;
  int Nv = 0;

  bool allocated = true;
  bool restriction = false;
  Coordinates_t coords, coarse_coords; // coords

  buf_pool_t<Real>::weak_t buf;       // comm buffer from pool
  parthenon::ParArray6D<Real> var;    // data variable used for comms
  parthenon::ParArray6D<Real> fine;   // fine data variable for prolongation/restriction
  parthenon::ParArray6D<Real> coarse; // coarse data variable for prolongation/restriction

  static BndInfo GetSendBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                                std::shared_ptr<CellVariable<Real>> v);
  static BndInfo GetSetBndInfo(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                               std::shared_ptr<CellVariable<Real>> v);
};

int GetBufferSize(std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
                  std::shared_ptr<CellVariable<Real>> v);

using BufferCache_t = ParArray1D<BndInfo>;
using BufferCacheHost_t = typename BufferCache_t::HostMirror;

// This is just a struct to cleanly hold all of the information it is useful to cache
// for the block boundary communication routines. A copy of it is contained in MeshData.
struct BvarsCache_t {
  void clear() {
    send_buf_vec.clear();
    recv_buf_vec.clear();
    sending_non_zero_flags = ParArray1D<bool>{};
    sending_non_zero_flags_h = ParArray1D<bool>::host_mirror_type{};
    send_bnd_info = BufferCache_t{};
    send_bnd_info_h = BufferCache_t::host_mirror_type{};
    recv_bnd_info = BufferCache_t{};
    recv_bnd_info_h = BufferCache_t::host_mirror_type{};
  }

  std::vector<CommBuffer<buf_pool_t<Real>::owner_t> *> send_buf_vec, recv_buf_vec;
  ParArray1D<bool> sending_non_zero_flags;
  ParArray1D<bool>::host_mirror_type sending_non_zero_flags_h;

  BufferCache_t send_bnd_info{};
  BufferCache_t::host_mirror_type send_bnd_info_h{};

  BufferCache_t recv_bnd_info{};
  BufferCache_t::host_mirror_type recv_bnd_info_h{};
};

} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BVALS_CC_IN_ONE_HPP_
