#pragma once

#include "loop_abstraction_base.hpp"
#include "loop_abstraction_scratch_primitives.hpp"
#include "loop_abstraction_kokkos.hpp"
#include "loop_abstraction_pack_view.hpp"
#include "loop_abstraction_raw.hpp"
#include "loop_abstraction_view.hpp"

namespace loop_abstraction {

template <class IndexSpaceType, class F>
void outer(IndexSpaceType idx_space, F &&f) {
  if constexpr (IndexSpaceType::backend_v == loop_backend::raw) {
    impl::outer_raw_for(idx_space, std::forward<F>(f));
  } else if constexpr (IndexSpaceType::backend_v == loop_backend::kokkos) {
    impl::outer_kokkos(idx_space, std::forward<F>(f));
  } else {
    static_assert(always_false<IndexSpaceType>,
                  "Unsupported loop backend for outer().");
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner(const InnerIndexRangeType &idx_range, F &&f) {
  if constexpr (InnerIndexRangeType::index_space_t::backend_v == loop_backend::raw) {
    impl::inner_raw_for(idx_range, std::forward<F>(f));
  } else if constexpr (InnerIndexRangeType::index_space_t::backend_v ==
                       loop_backend::kokkos) {
    impl::inner_kokkos(idx_range, std::forward<F>(f));
  } else {
    static_assert(always_false<InnerIndexRangeType>,
                  "Unsupported loop backend for inner().");
  }
}

template <class Halo, class InnerIndexRangeType>
KOKKOS_INLINE_FUNCTION auto AddHalo(const InnerIndexRangeType &idx_range) {
  return idx_range.template AddHalo<Halo>();
}

} // namespace loop_abstraction
