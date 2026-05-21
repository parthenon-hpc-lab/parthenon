#pragma once

#include "loop_abstraction_base.hpp"
#include "loop_abstraction_kokkos.hpp"
#include "loop_abstraction_range.hpp"
#include "loop_abstraction_raw.hpp"
#include "loop_abstraction_view.hpp"

namespace plb2 {

namespace loop_abstraction {

template <class IndexSpaceType, class F>
void outer(IndexSpaceType idx_space, F &&f) {
  if constexpr (impl::use_raw_for_v) {
    impl::outer_raw_for(idx_space, std::forward<F>(f));
  } else {
    impl::outer_kokkos(idx_space, std::forward<F>(f));
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner(const InnerIndexRangeType &idx_range, F &&f) {
  if constexpr (impl::use_raw_for_v) {
    impl::inner_raw_for(idx_range, std::forward<F>(f));
  } else {
    impl::inner_kokkos(idx_range, std::forward<F>(f));
  }
}

} // namespace loop_abstraction

} // namespace plb2
