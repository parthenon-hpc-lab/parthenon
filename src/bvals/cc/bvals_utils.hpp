//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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
#ifndef BVALS_CC_BVALS_UTILS_HPP_
#define BVALS_CC_BVALS_UTILS_HPP_

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace cell_centered_bvars {

namespace impl {

using sp_mb_t = std::shared_ptr<MeshBlock>;
using sp_mbd_t = std::shared_ptr<MeshBlockData<Real>>;
using sp_cv_t = std::shared_ptr<CellVariable<Real>>;
using nb_t = NeighborBlock;

enum class LoopControl { cont, break_out };

template <class F, class... Args>
inline auto func_caller(F func, Args &&...args) -> typename std::enable_if<
    std::is_same<decltype(func(std::declval<Args>()...)), LoopControl>::value,
    LoopControl>::type {
  return func(std::forward<Args>(args)...);
}

template <class F, class... Args>
inline auto func_caller(F func, Args &&...args) -> typename std::enable_if<
    !std::is_same<decltype(func(std::declval<Args>()...)), LoopControl>::value,
    LoopControl>::type {
  func(std::forward<Args>(args)...);
  return LoopControl::cont;
}

template <BoundaryType bound = BoundaryType::any, class F>
inline void ForEachBoundary(std::shared_ptr<MeshData<Real>> &md, F func) {
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; ++n) {
          auto &nb = pmb->pbval->neighbor[n];
          if (bound == BoundaryType::local) {
            if (nb.snb.rank != Globals::my_rank) continue;
          } else if (bound == BoundaryType::nonlocal) {
            if (nb.snb.rank == Globals::my_rank) continue;
          } else if (bound == BoundaryType::reflux_send) {
            // Check if this boundary requires flux correction
            if (nb.snb.level != pmb->loc.level - 1) continue;
            // No flux correction required unless boundaries share a face
            if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) != 1)
              continue;
          } else if (bound == BoundaryType::reflux_recv) {
            // Check if this boundary requires flux correction
            if (nb.snb.level - 1 != pmb->loc.level) continue;
            // No flux correction required unless boundaries share a face
            if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) != 1)
              continue;
          }
          if (func_caller(func, pmb, rc, nb, v) == LoopControl::break_out) return;
        }
      }
    }
  }
}

inline std::tuple<int, int, std::string, int>
SendKey(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb,
        const std::shared_ptr<CellVariable<Real>> &pcv) {
  const int sender_id = pmb->gid;
  const int receiver_id = nb.snb.gid;
  const int location_idx = (1 + nb.ni.ox1) + 3 * (1 + nb.ni.ox2 + 3 * (1 + nb.ni.ox3));
  return {sender_id, receiver_id, pcv->label(), location_idx};
}

inline std::tuple<int, int, std::string, int>
ReceiveKey(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb,
           const std::shared_ptr<CellVariable<Real>> &pcv) {
  const int receiver_id = pmb->gid;
  const int sender_id = nb.snb.gid;
  const int location_idx = (1 - nb.ni.ox1) + 3 * (1 - nb.ni.ox2 + 3 * (1 - nb.ni.ox3));
  return {sender_id, receiver_id, pcv->label(), location_idx};
}

} // namespace impl
} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BVALS_UTILS_HPP_
