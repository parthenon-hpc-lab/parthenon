#pragma once

#include "loop_abstraction_base.hpp"

namespace plb2 {

namespace loop_abstraction {

template <class IndexSpaceType>
struct var_view_t {
 public:
  parthenon::Real *data = nullptr;
  int shift;
  const IndexSpaceType *pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx + shift]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) + shift];
  }
};

template <>
struct var_view_t<IndexSpace<loop_tag::boiv, inner_tag::logical>> {
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
};

template <loop_tag LOOP_TAG, inner_tag INNER_TAG, class ViewType>
KOKKOS_INLINE_FUNCTION auto GetInnerView(const IndexSpace<LOOP_TAG, INNER_TAG> &idx_space,
                                         ViewType &in, int block, int var,
                                         std::array<int, 3> offset = {0, 0, 0}) {
  return var_view_t<IndexSpace<LOOP_TAG, INNER_TAG>>{&in(block, var, 0, 0, 0),
                                                     static_cast<int>(idx_space.GetMemoryIndexer()
                                                                          .GetFlatIdx(
                                                                              offset[0],
                                                                              offset[1],
                                                                              offset[2])),
                                                     &idx_space};
}

template <class IndexSpaceType, class ViewType>
KOKKOS_INLINE_FUNCTION auto GetView(const InnerIndexRange<IndexSpaceType> &idx_range,
                                    ViewType &in, int var,
                                    std::array<int, 3> offset = {0, 0, 0}) {
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    static_assert(IndexSpaceType::inner_tag_v == inner_tag::logical,
                  "boiv currently expects logical inner coordinates");
    return var_view_t<IndexSpaceType>{
        &in(idx_range.block, var, idx_range.payload_.k + offset[0], idx_range.payload_.j + offset[1],
            idx_range.payload_.i + offset[2])};
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi &&
                       IndexSpaceType::inner_tag_v == inner_tag::memory) {
    return var_view_t<IndexSpaceType>{
        &in(idx_range.block, var, idx_range.payload_.ks + offset[0], idx_range.payload_.js + offset[1],
            idx_range.payload_.is + offset[2]),
        0, idx_range.pidx_space};
  } else {
    return GetInnerView(*idx_range.pidx_space, in, idx_range.block, var, offset);
  }
}

} // namespace loop_abstraction

} // namespace plb2
