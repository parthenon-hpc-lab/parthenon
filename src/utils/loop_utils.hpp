//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2024. Triad National Security, LLC. All rights reserved.
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
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//========================================================================================
#ifndef UTILS_LOOP_UTILS_HPP_
#define UTILS_LOOP_UTILS_HPP_

#include <memory>      // smart pointers
#include <type_traits> // std::enable_if
#include <utility>     // std::forward
#include <vector>      // std::vector

#include "bvals/comms/bnd_info.hpp" // TODO(JMM): Remove me when possible
#include "interface/metadata.hpp"
#include "mesh/domain.hpp" // TODO(JMM): Remove me when possible
#include "mesh/mesh.hpp"

namespace parthenon {

typedef struct boundIdx {
  int ib;
  int iv;
  int in;
} boundIdx_t;

// forward declarations
class MeshBlock;
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;
template <typename T>
class Variable;
class NeighborBlock;

namespace loops {
namespace shorthands {
using sp_mb_t = std::shared_ptr<MeshBlock>;
using sp_mbd_t = std::shared_ptr<MeshBlockData<Real>>;
using sp_cv_t = std::shared_ptr<Variable<Real>>;
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
    std::is_same<decltype(func(std::forward<Args>(args)...)), LoopControl>::value,
    LoopControl>::type {
  return func(std::forward<Args>(args)...);
}

template <class F, class... Args>
inline auto func_caller(F func, Args &&...args) -> typename std::enable_if<
    !std::is_same<decltype(func(std::forward<Args>(args)...)), LoopControl>::value,
    LoopControl>::type {
  func(std::forward<Args>(args)...);
  return LoopControl::cont;
}

inline auto &GetNeighborsOnCoarserGMGGrid(MeshBlock *pmb, const GridIdentifier &grid) {
  if (grid.type() == GridType::two_level_composite &&
      pmb->loc.level() != grid.logical_level()) {
    // This is a boundary block on a two-level composite grid, its
    // data is up to date but it needs to send a dummy message to itself
    // on the next coarser grid for synchronization
    return pmb->GetGMGSelfNeighbors();
  }
  return pmb->GetGMGCoarserNeighbors();
}

inline auto &GetNeighborsOnFinerGMGGrid(MeshBlock *pmb, const GridIdentifier &grid) {
  const auto finer_grid = pmb->pmy_mesh->GetGMGGrid(grid.multigrid_level() + 1);
  if (finer_grid.type() == GridType::two_level_composite &&
      finer_grid.block_coarsenings() == grid.block_coarsenings() &&
      pmb->loc.level() == grid.logical_level() && pmb->IsLeafLL()) {
    // This is a boundary block on a two-level composite grid below this
    // one, its data is up to date but it needs to send a dummy message to itself
    // on the next coarser grid for synchronization
    return pmb->GetGMGSelfNeighbors();
  }
  return pmb->GetGMGFinerNeighbors();
}

namespace detail {

template <BoundaryType bound>
inline bool PassesFluxCorrectionFilter(const NeighborBlock &nb, MeshBlock *pmb,
                                       const std::shared_ptr<Variable<Real>> &v) {
  if (nb.loc.level() - (bound == BoundaryType::flxcor_recv) !=
      pmb->loc.level() - (bound == BoundaryType::flxcor_send))
    return false;
  if (nb.offsets.IsFace() && v->IsSet(Metadata::Face)) return true;
  if ((nb.offsets.IsFace() || nb.offsets.IsEdge()) && v->IsSet(Metadata::Edge)) return true;
  if ((nb.offsets.IsFace() || nb.offsets.IsEdge() || nb.offsets.IsNode()) &&
      v->IsSet(Metadata::Node))
    return true;
  return false;
}

// Resolve (ib, iv, in) from a prior ForEachBoundary2 / BuildBoundIndex traversal.
template <BoundaryType bound>
inline bool GetBoundaryAtIndex(const std::shared_ptr<MeshData<Real>> &md,
                               const boundIdx_t &idx, MeshBlock *&pmb, shorthands::sp_mbd_t &rc,
                               const NeighborBlock *&nb, shorthands::sp_cv_t &v) {
  rc = md->GetBlockData(idx.ib);
  pmb = rc->GetBlockPointer();
  auto &varVector = rc->GetVariableVector();
  if (idx.iv < 0 || idx.iv >= static_cast<int>(varVector.size())) return false;
  v = varVector[idx.iv];

  const int fine_level = md->grid.logical_level();

  if constexpr (bound == BoundaryType::gmg_restrict_send) {
    if (!v->IsSet(Metadata::GMGRestrict)) return false;
    auto &neighbors = GetNeighborsOnCoarserGMGGrid(pmb, md->grid);
    if (idx.in < 0 || idx.in >= static_cast<int>(neighbors.size())) return false;
    nb = &neighbors[idx.in];
    return true;
  } else if constexpr (bound == BoundaryType::gmg_restrict_recv) {
    if (!v->IsSet(Metadata::GMGRestrict)) return false;
    auto &neighbors = GetNeighborsOnFinerGMGGrid(pmb, md->grid);
    if (idx.in < 0 || idx.in >= static_cast<int>(neighbors.size())) return false;
    nb = &neighbors[idx.in];
    return true;
  } else if constexpr (bound == BoundaryType::gmg_prolongate_send) {
    if (!v->IsSet(Metadata::GMGProlongate)) return false;
    auto &neighbors = GetNeighborsOnFinerGMGGrid(pmb, md->grid);
    if (idx.in < 0 || idx.in >= static_cast<int>(neighbors.size())) return false;
    nb = &neighbors[idx.in];
    return true;
  } else if constexpr (bound == BoundaryType::gmg_prolongate_recv) {
    if (!v->IsSet(Metadata::GMGProlongate)) return false;
    auto &neighbors = GetNeighborsOnCoarserGMGGrid(pmb, md->grid);
    if (idx.in < 0 || idx.in >= static_cast<int>(neighbors.size())) return false;
    nb = &neighbors[idx.in];
    return true;
  } else if constexpr (bound == BoundaryType::gmg_same) {
    if (!v->IsSet(Metadata::FillGhost)) return false;
    if (md->grid.type() == GridType::two_level_composite) {
      const auto &gmg_same = pmb->loc.level() == md->grid.logical_level()
                                 ? pmb->GetGMGSameNeighbors()
                                 : pmb->GetGMGCompositeFinerNeighbors();
      if (idx.in < 0 || idx.in >= static_cast<int>(gmg_same.size())) return false;
      nb = &gmg_same[idx.in];
      return (pmb->loc.level() == fine_level || nb->loc.level() == fine_level);
    }
    auto &neighbors = pmb->GetNeighbors();
    if (idx.in < 0 || idx.in >= static_cast<int>(neighbors.size())) return false;
    nb = &neighbors[idx.in];
    return true;
  } else {
    [[maybe_unused]] constexpr bool flx_bound =
        bound == BoundaryType::flxcor_send || bound == BoundaryType::flxcor_recv;
    if (!v->IsSet(Metadata::FillGhost) && !v->IsSet(Metadata::Flux)) return false;
    auto &neighbors = pmb->GetNeighbors();
    if (idx.in < 0 || idx.in >= static_cast<int>(neighbors.size())) return false;
    nb = &neighbors[idx.in];
    if constexpr (bound == BoundaryType::local) {
      if (!v->IsSet(Metadata::FillGhost)) return false;
      if (nb->rank != Globals::my_rank) return false;
    } else if constexpr (bound == BoundaryType::nonlocal) {
      if (!v->IsSet(Metadata::FillGhost)) return false;
      if (nb->rank == Globals::my_rank) return false;
    } else if constexpr (bound == BoundaryType::any) {
      if (!v->IsSet(Metadata::FillGhost)) return false;
    } else if constexpr (flx_bound) {
      if (!v->IsSet(Metadata::Flux)) return false;
      if (!PassesFluxCorrectionFilter<bound>(*nb, pmb, v)) return false;
    }
    return true;
  }
}

// Canonical boundary enumeration. Visitor is invoked for each selected boundary as
// (block, iv, neighbor_index, pmb, rc, nb, v) and may return LoopControl::break_out.
template <BoundaryType bound, class Visitor>
inline LoopControl ForEachBoundaryIndexed(std::shared_ptr<MeshData<Real>> &md,
                                          Visitor &&visit) {
  const int fine_level = md->grid.logical_level();
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    const auto &gmg_same = pmb->loc.level() == md->grid.logical_level()
                                 ? pmb->GetGMGSameNeighbors()
                                 : pmb->GetGMGCompositeFinerNeighbors();
    const auto &varVector = rc->GetVariableVector();
    for (int iv = 0; iv < static_cast<int>(varVector.size()); ++iv) {
      const auto &v = varVector[iv];
      if constexpr (bound == BoundaryType::gmg_restrict_send) {
        if (v->IsSet(Metadata::GMGRestrict)) {
          auto &neighbors = GetNeighborsOnCoarserGMGGrid(pmb, md->grid);
          for (int n = 0; n < static_cast<int>(neighbors.size()); ++n) {
            auto &nb = neighbors[n];
            if (visit(block, iv, n, pmb, rc, nb, v) == LoopControl::break_out) {
              return LoopControl::break_out;
            }
          }
        }
      } else if constexpr (bound == BoundaryType::gmg_restrict_recv) {
        if (v->IsSet(Metadata::GMGRestrict)) {
          auto &neighbors = GetNeighborsOnFinerGMGGrid(pmb, md->grid);
          for (int n = 0; n < static_cast<int>(neighbors.size()); ++n) {
            auto &nb = neighbors[n];
            if (visit(block, iv, n, pmb, rc, nb, v) == LoopControl::break_out) {
              return LoopControl::break_out;
            }
          }
        }
      } else if constexpr (bound == BoundaryType::gmg_prolongate_send) {
        if (v->IsSet(Metadata::GMGProlongate)) {
          auto &neighbors = GetNeighborsOnFinerGMGGrid(pmb, md->grid);
          for (int n = 0; n < static_cast<int>(neighbors.size()); ++n) {
            auto &nb = neighbors[n];
            if (visit(block, iv, n, pmb, rc, nb, v) == LoopControl::break_out) {
              return LoopControl::break_out;
            }
          }
        }
      } else if constexpr (bound == BoundaryType::gmg_prolongate_recv) {
        if (v->IsSet(Metadata::GMGProlongate)) {
          auto &neighbors = GetNeighborsOnCoarserGMGGrid(pmb, md->grid);
          for (int n = 0; n < static_cast<int>(neighbors.size()); ++n) {
            auto &nb = neighbors[n];
            if (visit(block, iv, n, pmb, rc, nb, v) == LoopControl::break_out) {
              return LoopControl::break_out;
            }
          }
        }
      } else if constexpr (bound == BoundaryType::gmg_same) {
        if (v->IsSet(Metadata::FillGhost)) {
          if (md->grid.type() == GridType::two_level_composite) {
            for (int n = 0; n < static_cast<int>(gmg_same.size()); ++n) {
              auto &nb = gmg_same[n];
              if (pmb->loc.level() == fine_level || nb.loc.level() == fine_level) {
                if (visit(block, iv, n, pmb, rc, nb, v) == LoopControl::break_out) {
                  return LoopControl::break_out;
                }
              }
            }
          } else {
            auto &neighbors = pmb->GetNeighbors();
            for (int n = 0; n < static_cast<int>(neighbors.size()); ++n) {
              auto &nb = neighbors[n];
              if (visit(block, iv, n, pmb, rc, nb, v) == LoopControl::break_out) {
                return LoopControl::break_out;
              }
            }
          }
        }
      } else {
        if (v->IsSet(Metadata::FillGhost) || v->IsSet(Metadata::Flux)) {
          [[maybe_unused]] constexpr bool flx_bound =
              bound == BoundaryType::flxcor_send || bound == BoundaryType::flxcor_recv;
          auto &neighbors = pmb->GetNeighbors();
          for (int n = 0; n < static_cast<int>(neighbors.size()); ++n) {
            auto &nb = neighbors[n];
            if constexpr (bound == BoundaryType::local) {
              if (!v->IsSet(Metadata::FillGhost)) continue;
              if (nb.rank != Globals::my_rank) continue;
            } else if constexpr (bound == BoundaryType::nonlocal) {
              if (!v->IsSet(Metadata::FillGhost)) continue;
              if (nb.rank == Globals::my_rank) continue;
            } else if constexpr (bound == BoundaryType::any) {
              if (!v->IsSet(Metadata::FillGhost)) continue;
            } else if constexpr (flx_bound) {
              if (!v->IsSet(Metadata::Flux)) continue;
              if (!PassesFluxCorrectionFilter<bound>(nb, pmb, v)) continue;
            }
            if (visit(block, iv, n, pmb, rc, nb, v) == LoopControl::break_out) {
              return LoopControl::break_out;
            }
          }
        }
      }
    }
  }
  return LoopControl::cont;
}

} // namespace detail

// Loop over boundaries (or shared geometric elements) for blocks contained
// in MeshData, calling the passed function func for every boundary. Unifies
// boundary looping that occurs in many places in the boundary communication
// routines and allows for easy selection of a subset of the boundaries based
// on the template parameter BoundaryType.
template <BoundaryType bound = BoundaryType::any, class F>
inline void ForEachBoundary(std::shared_ptr<MeshData<Real>> &md, F func) {
  PARTHENON_INSTRUMENT
  detail::ForEachBoundaryIndexed<bound>(
      md, [&](int /*block*/, int /*iv*/, int /*n*/, MeshBlock *pmb, shorthands::sp_mbd_t &rc,
              const NeighborBlock &nb, const shorthands::sp_cv_t &v) {
        if (func_caller(func, pmb, rc, nb, v) == LoopControl::break_out) {
          return LoopControl::break_out;
        }
        return LoopControl::cont;
      });
}

// Same traversal order as ForEachBoundary, but passes mesh indices (ib, iv, in).
template <BoundaryType bound = BoundaryType::any, class F>
inline void ForEachBoundary2(std::shared_ptr<MeshData<Real>> &md, F func) {
  PARTHENON_INSTRUMENT
  detail::ForEachBoundaryIndexed<bound>(
      md, [&](int block, int iv, int n, MeshBlock * /*pmb*/, shorthands::sp_mbd_t & /*rc*/,
              const NeighborBlock & /*nb*/, const shorthands::sp_cv_t & /*v*/) {
        if (func_caller(func, block, iv, n) == LoopControl::break_out) {
          return LoopControl::break_out;
        }
        return LoopControl::cont;
      });
}

template <BoundaryType bound = BoundaryType::any>
inline std::vector<boundIdx_t> BuildBoundIndex(std::shared_ptr<MeshData<Real>> &md) {
  PARTHENON_INSTRUMENT
  //WIP: This was previously done in two passes -- first to count malloc indices all at once, second to fill
  std::vector<boundIdx_t> indices;
  ForEachBoundary2<bound>(md, [&](int ib, int iv, int in) {
    indices.push_back({ib, iv, in});
  });
  return indices;
}

// OpenMP over a flat boundary list built by BuildBoundIndex / ForEachBoundary2.
template <BoundaryType bound = BoundaryType::any, class F>
inline void ForEachBoundaryOMP1(std::shared_ptr<MeshData<Real>> &md,
                                const std::vector<boundIdx_t> &bound_indices, F func) {
  PARTHENON_INSTRUMENT
  const int ibound = static_cast<int>(bound_indices.size());
#pragma omp parallel for
  for (int i = 0; i < ibound; i++) {
    MeshBlock *pmb = nullptr;
    shorthands::sp_mbd_t rc;
    const NeighborBlock *nb = nullptr;
    shorthands::sp_cv_t v;
    if (!detail::GetBoundaryAtIndex<bound>(md, bound_indices[i], pmb, rc, nb, v)) {
      continue;
    }
    func_caller(func, pmb, rc, *nb, v, i);
  }
}

} // namespace loops
} // namespace parthenon

#endif // UTILS_LOOP_UTILS_HPP_
