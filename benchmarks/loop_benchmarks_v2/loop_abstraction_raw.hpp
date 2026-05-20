#pragma once

#include "loop_abstraction_base.hpp"

namespace plb2 {

namespace loop_abstraction::impl {

template <class IndexSpaceType, class F>
KOKKOS_INLINE_FUNCTION void outer_raw_for(IndexSpaceType idx_space, F &&f) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    for (int b = 0; b < idx_space.nblocks; ++b) {
      InnerIndexRangeType idx_range;
      idx_range.pidx_space = &idx_space;
      idx_range.block = b;
      f(idx_range, b);
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int nouter = GetNOuter(idx_space);
    for (int b = 0; b < idx_space.nblocks; ++b) {
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * idx_space.ninner;
        const int logical_end = std::min((o + 1) * idx_space.ninner - 1,
                                         static_cast<int>(idx_space.logical_kji.size()) - 1);
        const auto idx_range =
            InnerIndexRangeType::FlatRange(idx_space, b, logical_start, logical_end);
        f(idx_range, b);
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    static_assert(IndexSpaceType::inner_tag_v == inner_tag::logical,
                  "Probably don't want to do boiv over memory");
    const int ks = idx_space.logical_kji.template StartIdx<0>();
    const int ke = idx_space.logical_kji.template EndIdx<0>();
    const int js = idx_space.logical_kji.template StartIdx<1>();
    const int je = idx_space.logical_kji.template EndIdx<1>();
    const int is = idx_space.logical_kji.template StartIdx<2>();
    const int ie = idx_space.logical_kji.template EndIdx<2>();
    InnerIndexRangeType idx_range;
    idx_range.pidx_space = &idx_space;
    for (idx_range.block = 0; idx_range.block < idx_space.nblocks; ++idx_range.block) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            idx_range.payload_.k = k;
            idx_range.payload_.j = j;
            idx_range.payload_.i = i;
            f(idx_range, idx_range.block);
          }
        }
      }
    }
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner_raw_for(const InnerIndexRangeType &idx_range, F &&f) {
  using IndexSpaceType =
      std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  const auto &idx_space = *(idx_range.pidx_space);
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical) {
      const int ks = idx_space.logical_kji.template StartIdx<0>();
      const int ke = idx_space.logical_kji.template EndIdx<0>();
      const int js = idx_space.logical_kji.template StartIdx<1>();
      const int je = idx_space.logical_kji.template EndIdx<1>();
      const int is = idx_space.logical_kji.template StartIdx<2>();
      const int ie = idx_space.logical_kji.template EndIdx<2>();
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            f(idx_space.memory_kji.GetFlatIdx(k, j, i));
          }
        }
      }
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      const int nouter = GetNOuter(idx_space);
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * idx_space.ninner;
        const int logical_end = std::min((o + 1) * idx_space.ninner - 1,
                                         static_cast<int>(idx_space.logical_kji.size()) - 1);
        const auto inner_range =
            InnerIndexRangeType::FlatRange(idx_space, idx_range.block, logical_start, logical_end);
#pragma omp simd
        for (int idx = inner_range.payload_.flat_start; idx <= inner_range.payload_.flat_end;
             ++idx) {
          f(idx);
        }
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int start = idx_range.payload_.flat_start;
    const int end_exclusive = idx_range.payload_.flat_end + 1 - start;
#pragma omp simd
    for (int idx = 0; idx < end_exclusive; ++idx) {
      if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
        f(idx);
      } else {
        const auto [k, j, i] = idx_space.logical_kji(idx + start);
        f(Index3{k, j, i});
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    f(Index3{idx_range.payload_.k, idx_range.payload_.j, idx_range.payload_.i});
  }
}

} // namespace loop_abstraction::impl

} // namespace plb2
