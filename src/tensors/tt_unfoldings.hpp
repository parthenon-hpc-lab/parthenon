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

#ifndef TENSORS_TT_UNFOLDINGS_HPP
#define TENSORS_TT_UNFOLDINGS_HPP

#include "kokkos_abstraction.hpp"

namespace parthenon {
namespace tensor2 {

// ==============================================================================
// TENSOR UNFOLDINGS
// ==============================================================================
// These structs provide 2D matrix views of 3D tensor cores for use in
// matrix operations. They have no dependencies on TensorCore or TensorTraits,
// so they can be defined independently.

// Vertical unfolding: reshape [lr][dd][rr] as [lr*dd, rr]
// Layout-agnostic since d is always the fastest moving row index
template <class CoreLike, bool transpose = false>
struct vertical_unfolding {
  const CoreLike &core;
  int nl, nd, nr;

  vertical_unfolding(const CoreLike &core_in)
      : core(core_in), nl(core_in.LR()), nd(core_in.DD()), nr(core_in.RR()) {}

  vertical_unfolding(const CoreLike &core_in, int nl, int nd, int nr)
      : core(core_in), nl(nl), nd(nd), nr(nr) {}

  KOKKOS_FORCEINLINE_FUNCTION
  decltype(auto) operator()(int j, int i) const {
    if constexpr (transpose) {
      const int rl = i / nd;
      const int d = i % nd;
      return core(rl, d, j);
    } else {
      const int rl = j / nd;
      const int d = j % nd;
      return core(rl, d, i);
    }
  }
};

// Horizontal unfolding: reshape [lr][dd][rr] as [lr, dd*rr]
// Layout-aware: index arithmetic depends on which dimension is stride-1
template <class CoreLike, bool transpose = false, bool d_fastest_moving = true>
struct horizontal_unfolding {
  const CoreLike &core;
  int nl, nd, nr;

  horizontal_unfolding(const CoreLike &core_in)
      : core(core_in), nl(core_in.LR()), nd(core_in.DD()), nr(core_in.RR()) {}

  horizontal_unfolding(const CoreLike &core_in, int nl, int nd, int nr)
      : core(core_in), nl(nl), nd(nd), nr(nr) {}

  KOKKOS_FORCEINLINE_FUNCTION
  decltype(auto) operator()(int row, int col) const {
    if constexpr (transpose) {
      if constexpr (d_fastest_moving) {
        // dd is stride-1: [lr][rr][dd]
        const int rr = row / nd;
        const int d = row % nd;
        return core(col, d, rr);
      } else {
        // rr is stride-1: [lr][dd][rr]
        const int d = row / nr;
        const int rr = row % nr;
        return core(col, d, rr);
      }
    } else {
      if constexpr (d_fastest_moving) {
        // dd is stride-1: [lr][rr][dd]
        const int rr = col / nd;
        const int d = col % nd;
        return core(row, d, rr);
      } else {
        // rr is stride-1: [lr][dd][rr]
        const int d = col / nr;
        const int rr = col % nr;
        return core(row, d, rr);
      }
    }
  }
};

// Helper functions to get matrix dimensions
template <class T, bool transpose>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const vertical_unfolding<T, transpose> &m) {
  return transpose ? m.nr : m.nd * m.nl;
}

template <class T, bool transpose>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const vertical_unfolding<T, transpose> &m) {
  return transpose ? m.nd * m.nl : m.nr;
}

template <class T, bool transpose, bool d_fastest_moving>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const horizontal_unfolding<T, transpose, d_fastest_moving> &m) {
  return transpose ? m.nd * m.nr : m.nl;
}

template <class T, bool transpose, bool d_fastest_moving>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const horizontal_unfolding<T, transpose, d_fastest_moving> &m) {
  return transpose ? m.nl : m.nd * m.nr;
}

} // namespace tensor2
} // namespace parthenon

#endif // TENSORS_TT_UNFOLDINGS_HPP
