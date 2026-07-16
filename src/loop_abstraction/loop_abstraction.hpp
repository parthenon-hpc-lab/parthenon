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

// Umbrella header for the loop abstraction and the intended include point for users.
// It pulls in the core types and all backends/helpers, then defines the public
// outer()/inner()/AddHalo() entry points that dispatch to the backend selected at
// compile time by IndexSpace::backend_v. See LOOP_ABSTRACTION_CONTRACTS.md for the
// semantics of the loop tags, inner tags, halos, and scratch.

#include "loop_abstraction_base.hpp"
#include "loop_abstraction_scratch.hpp"
#include "loop_abstraction_scratch_indexed.hpp"
#include "loop_abstraction_kokkos.hpp"
#include "loop_abstraction_pack_view.hpp"
#include "loop_abstraction_flux_view.hpp"
#include "loop_abstraction_raw.hpp"
#include "loop_abstraction_view.hpp"

namespace parthenon::loop_abstraction {

// Launch the outer loop over an IndexSpace (blocks and, for bovi/boiv, chunks of the
// kji space). Dispatches to the raw or Kokkos backend at compile time.
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

// Run the inner loop over one InnerIndexRange slice. The traversal (flat/coords/
// memory span, plus any halo) follows the range's tags; dispatches to the backend at
// compile time.
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

// Extend an inner range's visited logical points by the shifted copies named by
// Halo (see loop_abstraction_halo.hpp and LOOP_ABSTRACTION_CONTRACTS.md).
template <class Halo, class InnerIndexRangeType>
KOKKOS_INLINE_FUNCTION auto AddHalo(const InnerIndexRangeType &idx_range) {
  return idx_range.template AddHalo<Halo>();
}

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_HPP_
