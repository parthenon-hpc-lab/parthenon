#pragma once

#include "loop_abstraction_base.hpp"
#include "loop_abstraction_range.hpp"

namespace plb2 {

namespace loop_abstraction::impl {

template <class IndexSpaceType, class F>
KOKKOS_INLINE_FUNCTION void outer_raw_for(IndexSpaceType idx_space, F &&f) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
      InnerIndexRangeType idx_range;
      idx_range.pidx_space = &idx_space;
      idx_range.block = b;
      f(idx_range, b);
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int nouter = GetNOuter(idx_space);
    for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * idx_space.GetNInner();
        const int logical_end = std::min((o + 1) * idx_space.GetNInner() - 1,
                                         static_cast<int>(idx_space.GetLogicalIndexer().size()) -
                                             1);
        const auto idx_range = FlatRange(idx_space, b, logical_start, logical_end);
        f(idx_range, b);
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    const auto &logical_kji = idx_space.GetLogicalIndexer();
    const int ks = logical_kji.template StartIdx<0>();
    const int ke = logical_kji.template EndIdx<0>();
    const int js = logical_kji.template StartIdx<1>();
    const int je = logical_kji.template EndIdx<1>();
    const int is = logical_kji.template StartIdx<2>();
    const int ie = logical_kji.template EndIdx<2>();
    InnerIndexRangeType idx_range;
    idx_range.pidx_space = &idx_space;
    for (idx_range.block = 0; idx_range.block < idx_space.GetNBlocks(); ++idx_range.block) {
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            idx_range.k = k;
            idx_range.j = j;
            idx_range.i = i;
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
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat
               || IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
      const auto &logical_kji = idx_space.GetLogicalIndexer();
      const int ks = logical_kji.template StartIdx<0>();
      const int ke = logical_kji.template EndIdx<0>();
      const int js = logical_kji.template StartIdx<1>();
      const int je = logical_kji.template EndIdx<1>();
      const int is = logical_kji.template StartIdx<2>();
      const int ie = logical_kji.template EndIdx<2>();
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
              f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i));
            } else {
              f(Index3{k, j, i});
            }
          }
        }
      }
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      const int nouter = GetNOuter(idx_space);
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * idx_space.GetNInner();
        const int logical_end = std::min((o + 1) * idx_space.GetNInner() - 1,
                                         static_cast<int>(idx_space.GetLogicalIndexer().size()) -
                                             1);
        const auto inner_range = FlatRange(idx_space, idx_range.block, logical_start, logical_end);
#pragma omp simd
        for (int idx = inner_range.flat_start; idx <= inner_range.flat_end; ++idx) {
          f(idx);
        }
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int start = idx_range.flat_start;
    const int end_exclusive = idx_range.flat_end + 1 - start;
#pragma omp simd
    for (int idx = 0; idx < end_exclusive; ++idx) {
      if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
        f(idx);
      } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
        const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
        f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i));
      } else {
        const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
        f(Index3{k, j, i});
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
      f(idx_space.GetLogicalIndexer().GetFlatIdx(idx_range.k, idx_range.j, idx_range.i));
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
      f(Index3{idx_range.k, idx_range.j, idx_range.i});
    }
  }
}

} // namespace loop_abstraction::impl

} // namespace plb2
