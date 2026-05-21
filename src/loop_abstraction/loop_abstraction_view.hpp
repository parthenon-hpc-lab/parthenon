#pragma once

#include "pack/sparse_pack/sparse_pack.hpp"
#include "utils/type_list.hpp"

#include "loop_abstraction_base.hpp"

namespace loop_abstraction {

template <class IndexSpaceType>
struct var_view_t {
 public:
  parthenon::Real *data = nullptr;
  int flattened_offset = 0;
  const IndexSpaceType *pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx + flattened_offset]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) +
                flattened_offset];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }
};

template <inner_tag INNER_TAG>
struct var_view_t<IndexSpace<loop_tag::bovi, INNER_TAG>> {
 public:
  parthenon::Real *data = nullptr;
  int shift = 0;
  const IndexSpace<loop_tag::bovi, INNER_TAG> *pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) - shift];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }
};

template <inner_tag INNER_TAG>
struct var_view_t<IndexSpace<loop_tag::boiv, INNER_TAG>> {
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
  parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }
};

template <class IndexSpaceType, class ViewType>
KOKKOS_INLINE_FUNCTION auto GetView(const InnerIndexRange<IndexSpaceType> &idx_range,
                                    ViewType &in, int var,
                                    std::array<int, 3> offset = {0, 0, 0}) {
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    static_assert(IndexSpaceType::inner_tag_v == inner_tag::logical_flat ||
                      IndexSpaceType::inner_tag_v == inner_tag::logical_coords,
                  "boiv currently expects logical inner coordinates");
    return var_view_t<IndexSpaceType>{&in(idx_range.block, var, idx_range.ks + offset[0],
                                          idx_range.js + offset[1],
                                          idx_range.is + offset[2])};
  } else {
    const int shift = idx_range.pidx_space->GetMemoryIndexer().GetFlatIdx(
        idx_range.ks + offset[0], idx_range.js + offset[1], idx_range.is + offset[2]);
    return var_view_t<IndexSpaceType>{&in(idx_range.block, var, idx_range.ks + offset[0],
                                          idx_range.js + offset[1],
                                          idx_range.is + offset[2]),
                                      shift, idx_range.pidx_space};
  }
}

} // namespace loop_abstraction
