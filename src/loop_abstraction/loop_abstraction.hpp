//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LOOP_ABSTRACTION_LOOP_ABSTRACTION_HPP_
#define LOOP_ABSTRACTION_LOOP_ABSTRACTION_HPP_

// This file was made in part with generative AI.

#include "loop_abstraction_base.hpp"
#include "loop_abstraction_scratch.hpp"
#include "loop_abstraction_kokkos.hpp"
#include "loop_abstraction_pack_view.hpp"
#include "loop_abstraction_raw.hpp"
#include "loop_abstraction_view.hpp"

namespace parthenon::loop_abstraction {

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

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_HPP_
