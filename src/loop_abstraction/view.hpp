//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LOOP_ABSTRACTION_VIEW_HPP_
#define LOOP_ABSTRACTION_VIEW_HPP_

// This file was made in part with generative AI.

// View over a single variable of a raw Kokkos view (as opposed to a SparsePack; see
// pack_view.hpp for the pack-backed views). GetView binds one
// (block, var) slice of a 5D view to the current InnerIndexRange, and the returned
// view_view_t accepts the same index forms as the loop body (flat int, MemoryOffset,
// Index3, or explicit k, j, i) so a kernel reads the same way regardless of loop tag.
//
// NOTE: this is not a primary way to use the loop abstraction. It exists mainly to
// support the raw-Kokkos-view kernels in the loop benchmarks (which are not part of
// this PR). Production kernels should prefer the SparsePack-backed views in
// pack_view.hpp. This path is kept minimal and may be revisited.

#include "pack/sparse_pack/sparse_pack.hpp"
#include "utils/type_list.hpp"

#include "base.hpp"

namespace parthenon::loop_abstraction {

// Primary template: flat-index access is relative to the memory origin implied by the
// loop slice (flattened_offset). Specialized below for bovi (origin is the current
// chunk) and boiv (a single logical cell).
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
struct view_view_t {
 public:
  parthenon::Real *data = nullptr;
  int flattened_offset = 0;
  const parthenon::Indexer3D *memory_indexer = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx + flattened_offset]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(MemoryOffset idx) const { return (*this)(idx.flat); }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[memory_indexer->GetFlatIdx(in.k, in.j, in.i) + flattened_offset];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }
};

template <inner_tag INNER_TAG>
struct view_view_t<loop_tag::bovi, INNER_TAG> {
 public:
  parthenon::Real *data = nullptr;
  int shift = 0;
  const parthenon::Indexer3D *memory_indexer = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(MemoryOffset idx) const { return (*this)(idx.flat); }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[memory_indexer->GetFlatIdx(in.k, in.j, in.i) - shift];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }
};

template <inner_tag INNER_TAG>
struct view_view_t<loop_tag::boiv, INNER_TAG> {
 public:
  parthenon::Real *data = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    (void)in;
    return *data;
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const {
    (void)idx;
    return *data;
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(MemoryOffset idx) const {
    (void)idx;
    return *data;
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }
};

template <class IndexSpaceType, class ViewType>
KOKKOS_INLINE_FUNCTION auto GetView(const InnerIndexRange<IndexSpaceType> &idx_range,
                                    ViewType &in, int var,
                                    std::array<int, 3> offset = {0, 0, 0}) {
  constexpr loop_tag LOOP_TAG = IndexSpaceType::loop_tag_v;
  constexpr inner_tag INNER_TAG = IndexSpaceType::inner_tag_v;
  if constexpr (LOOP_TAG == loop_tag::boiv) {
    static_assert(INNER_TAG == inner_tag::logical_flat ||
                      INNER_TAG == inner_tag::logical_coords,
                  "boiv currently expects logical inner coordinates");
    return view_view_t<LOOP_TAG, INNER_TAG>{
        &in(idx_range.block, var, idx_range.ks + offset[0], idx_range.js + offset[1],
            idx_range.is + offset[2])};
  } else {
    const auto &memory_indexer = idx_range.pidx_space->GetMemoryIndexer();
    const int shift = memory_indexer.GetFlatIdx(
        idx_range.ks + offset[0], idx_range.js + offset[1], idx_range.is + offset[2]);
    return view_view_t<LOOP_TAG, INNER_TAG>{
        &in(idx_range.block, var, idx_range.ks + offset[0], idx_range.js + offset[1],
            idx_range.is + offset[2]),
        shift, &memory_indexer};
  }
}

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_VIEW_HPP_
