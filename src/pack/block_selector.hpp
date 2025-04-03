//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
#ifndef PACK_BLOCK_SELECTOR_HPP_
#define PACK_BLOCK_SELECTOR_HPP_

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/variable.hpp"
#include "pack/pack_utils.hpp"
#include "pack/sparse_pack_base.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/type_list.hpp"
#include "utils/utils.hpp"

namespace parthenon {

// A block selector function can be passed SparsePack::Descriptor::GetPack to
// select a subset of the blocks contained in a MeshData object based on a
// chosen criteria
using block_selector_func_t = std::function<bool(MeshBlockData<Real> *)>;

// Predefined block selector functions
namespace GetBlockSelector {

// Find blocks that have at least one neighbor that is coarser than they are
inline block_selector_func_t WithCoarserNeighbors(const MeshData<Real> *pmd) {
  const bool tl_comp = (pmd->grid.type == GridType::two_level_composite);
  const int level_comp = pmd->grid.logical_level;
  return [tl_comp, level_comp](MeshBlockData<Real> *pmbd) {
    auto pmb = pmbd->GetParentPointer();
    // Coarser blocks on two-level composite grids can only have same and finer neighbors
    if (tl_comp && pmb->loc.level() != level_comp) return false;
    auto *pneighbors = tl_comp ? &(pmb->gmg_same_neighbors) : &(pmb->neighbors);
    for (const auto &neighbor : *pneighbors) {
      if (neighbor.loc.level() < level_comp) return true;
    }
    return false;
  };
}

inline block_selector_func_t OnPhysicalBoundary(BoundaryFace bf) {
  return [bf](MeshBlockData<Real> *pmbd) {
    return pmbd->GetParentPointer()->IsPhysicalBoundary(bf);
  };
}

inline block_selector_func_t OnPhysicalBoundary() {
  return [](MeshBlockData<Real> *pmbd) {
    return pmbd->GetParentPointer()->IsPhysicalBoundary();
  };
}

inline block_selector_func_t FineOnCompositeGrid(const MeshData<Real> *pmd) {
  if (pmd->grid.type == GridType::two_level_composite) {
    const int fine_level = pmd->grid.logical_level;
    return [fine_level](MeshBlockData<Real> *pmbd) {
      return fine_level == pmbd->GetBlockPointer()->loc.level();
    };
  }
  return block_selector_func_t{};
}

} // namespace GetBlockSelector

} // namespace parthenon

#endif // PACK_BLOCK_SELECTOR_HPP_
