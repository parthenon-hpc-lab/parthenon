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

#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "pack/block_selector.hpp"
#include "pack/sparse_pack.hpp"

namespace parthenon {
namespace impl {

template <class T>
inline void GetBlockSelection(T *pmd, const block_selector_func_t &block_selector,
                              std::vector<bool> &include_block,
                              bool only_fine_two_level_composite_blocks) {
  PARTHENON_REQUIRE(include_block.size() == pmd->NumBlocks() ||
                        (include_block.size() == 0),
                    "Must specify inclusion status for all blocks.");

  // LFR: For multi-grid, we want to select only fine blocks on two-level composite
  // grids by default since only the fine grid cells are "active" (the coarse level
  // blocks just provide necessary boundary information during comms)
  block_selector_func_t fbc_selector{};
  if constexpr (std::is_same<T, MeshData<Real>>::value) {
    if (only_fine_two_level_composite_blocks)
      fbc_selector = GetBlockSelector::FineOnCompositeGrid(pmd);
  }

  // Select blocks for inclusion based on the specified user functor and possibly
  // the two-level composite selector
  if (block_selector || fbc_selector) {
    if (include_block.size() == 0) include_block.resize(pmd->NumBlocks(), true);
    ForEachBlock(pmd, std::vector<bool>{}, [&](int b, MeshBlockData<Real> *pmbd) {
      const bool bs = block_selector ? block_selector(pmbd) : true;
      const bool fbcs = fbc_selector ? fbc_selector(pmbd) : true;
      include_block[b] = include_block[b] && bs && fbcs;
    });
  }
}

template void GetBlockSelection<MeshData<Real>>(
    MeshData<Real> *pmd, const block_selector_func_t &block_selector,
    std::vector<bool> &include_block, bool only_fine_two_level_composite_blocks);

template void GetBlockSelection<MeshBlockData<Real>>(
    MeshBlockData<Real> *pmd, const block_selector_func_t &block_selector,
    std::vector<bool> &include_block, bool only_fine_two_level_composite_blocks);

} // namespace impl
} // namespace parthenon
