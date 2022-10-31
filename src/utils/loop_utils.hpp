//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
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
#ifndef UTILS_LOOP_UTILS_HPP_
#define UTILS_LOOP_UTILS_HPP_

#include <memory>      // smart pointers
#include <type_traits> // std::enable_if
#include <utility>     // std::forward

#include "bvals/cc/bnd_info.hpp" // TODO(JMM): Remove me when possible
#include "mesh/domain.hpp"       // TODO(JMM): Remove me when possible

namespace parthenon {

// forward declarations
class MeshBlock;
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;
template <typename T>
class CellVariable;
class NeighborBlock;

namespace loops {
namespace shorthands {
using sp_mb_t = std::shared_ptr<MeshBlock>;
using sp_mbd_t = std::shared_ptr<MeshBlockData<Real>>;
using sp_cv_t = std::shared_ptr<CellVariable<Real>>;
using nb_t = NeighborBlock;
} // namespace shorthands

enum class LoopControl { cont, break_out };

// Methods for wrapping a function that may or may not return a LoopControl
// object. The first is enabled if the function returns a LoopControl and
// just passes the returned object on. The second just calls the function,
// ignores its return, and returns a LoopControl continue. These wrap the
// function calls in the ForEachBoundary loop template to allow for breaking
// out of the loop if desired
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

// Loop over boundaries (or shared geometric elements) for blocks contained
// in MeshData, calling the passed function func for every boundary. Unifies
// boundary looping that occurs in many places in the boundary communication
// routines and allows for easy selection of a subset of the boundaries based
// on the template parameter BoundaryType. [Really, this probably does not
// need to be a template parameter, it could just be a function argument]
template <BoundaryType bound = BoundaryType::any, class F>
inline void ForEachBoundary(std::shared_ptr<MeshData<Real>> &md, F func) {
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; ++n) {
          auto &nb = pmb->pbval->neighbor[n];
          if constexpr (bound == BoundaryType::local) {
            if (nb.snb.rank != Globals::my_rank) continue;
          } else if constexpr (bound == BoundaryType::nonlocal) {
            if (nb.snb.rank == Globals::my_rank) continue;
          } else if constexpr (bound == BoundaryType::flxcor_send) {
            // Check if this boundary requires flux correction
            if (nb.snb.level != pmb->loc.level - 1) continue;
            // No flux correction required unless boundaries share a face
            if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) != 1)
              continue;
          } else if constexpr (bound == BoundaryType::flxcor_recv) {
            // Check if this boundary requires flux correction
            if (nb.snb.level - 1 != pmb->loc.level) continue;
            // No flux correction required unless boundaries share a face
            if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) != 1)
              continue;
          } else if constexpr (bound == BoundaryType::restricted) {
            // Check if restriction is required
            if (nb.snb.level >= pmb->loc.level) continue;
          }
          if constexpr (bound == BoundaryType::restricted) {
            IndexRange bni, bnj, bnk;
            cell_centered_bvars::ComputeRestrictionBounds(bni, bnj, bnk, nb, pmb);
            // This loop is only over {-1, 0, 1}^3 at most
            for (int nk = bnk.s; nk <= bnk.e; ++nk) {
              for (int nj = bnj.s; nj <= bnj.e; ++nj) {
                for (int ni = bni.s; ni <= bni.e; ++ni) {
                  int ntype = std::abs(ni) + std::abs(nj) + std::abs(nk);
                  // skip myself or coarse levels; only the same level must be restricted
                  if (ntype == 0 ||
                      pmb->pbval->nblevel[nk + 1][nj + 1][ni + 1] != pmb->loc.level)
                    continue;
                  cell_centered_bvars::OffsetIndices offsets(nk, nj, ni);
                  if (func_caller(func, pmb, rc, nb, v, offsets) ==
                      LoopControl::break_out)
                    return;
                }
              }
            }
          } else {
            cell_centered_bvars::OffsetIndices junk;
            if (func_caller(func, pmb, rc, nb, v, junk) == LoopControl::break_out) return;
          }
        }
      }
    }
  }
}

} // namespace loops
} // namespace parthenon

#endif // UTILS_LOOP_UTILS_HPP_
