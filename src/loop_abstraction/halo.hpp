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
#ifndef LOOP_ABSTRACTION_HALO_HPP_
#define LOOP_ABSTRACTION_HALO_HPP_

// This file was made in part with generative AI.

// Halo offset types and the compile-time contract checks they must satisfy. A halo
// is a compile-time set of logical offsets naming the shifted copies of an inner
// range that a producer loop must also visit (see LOOP_ABSTRACTION_CONTRACTS.md).
// Also provides AddHaloToIndexer, which extends a base indexer's bounds to cover
// every shifted copy.

#include <algorithm>
#include <array>

#include "utils/indexer.hpp"

#include "loop_abstraction/types.hpp"

namespace parthenon::loop_abstraction {

namespace halo {
// Halo types enumerate the shifted copies of an inner range that should be
// visited, including the identity offset {0,0,0}. Offsets must be ordered by
// increasing flat index in the halo-extended logical indexer.
//
// This ordering lets the bovi implementation build the halo range with a
// single linear merge pass: each candidate span only needs to be compared with
// the last emitted span. In other words, halo construction stays device-friendly
// and avoids sorting or dynamic storage before every inner loop.
struct none_t {
  static constexpr int npoints = 1;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct plus_i_t {
  static constexpr int npoints = 2;
  // Sorted by flat offset: identity, then +j.
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n == 0 ? 0 : 1; }
};

struct plus_j_t {
  static constexpr int npoints = 2;
  // Sorted by flat offset: identity, then +j.
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return n == 0 ? 0 : 1; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct plus_k_t {
  static constexpr int npoints = 2;
  // Sorted by flat offset: identity, then +j.
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return n == 0 ? 0 : 1; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct minus_i_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return -1 * (n == 0); }
};

struct minus_j_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return -1 * (n == 0); }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct minus_k_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return -1 * (n == 0); }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

// ---------------------------------------------------------------------------
//  Asymmetric halo definitions – needed when plasma‑viscosity is enabled
// ---------------------------------------------------------------------------
// Parthenon requires the halo points to be listed in strict lexicographic
// order of (dk, dj, di) and to contain exactly one identity offset (0,0,0).
// The points below satisfy that rule for each sweep direction.

// Sweep direction = X‑axis (i)
struct asym_i_t {
  static constexpr int npoints = 7;
  // (dk,dj,di) for n = 0…6:
  // 0: (-1,  0,  0)   lower‑k neighbour
  // 1: ( 0, -1,  0)   lower‑j neighbour
  // 2: ( 0,  0, -2)   two cells back in i
  // 3: ( 0,  0,  0)   identity
  // 4: ( 0,  0, +1)   forward i neighbour
  // 5: ( 0, +1,  0)   upper‑j neighbour
  // 6: (+1,  0,  0)   upper‑k neighbour
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return (n == 0) ? -1 : (n == 6) ? 1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return (n == 1) ? -1 : (n == 5) ? 1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return (n == 2) ? -2 : (n == 4) ? 1 : 0; }
};

// Sweep direction = Y‑axis (j)
struct asym_j_t {
  static constexpr int npoints = 7;
  // (dk,dj,di) for n = 0…6:
  // 0: (-1,  0,  0)   lower‑k neighbour
  // 1: ( 0, -2,  0)   two cells back in j
  // 2: ( 0,  0, -1)   lower‑i neighbour
  // 3: ( 0,  0,  0)   identity
  // 4: ( 0,  0, +1)   upper‑i neighbour
  // 5: ( 0, +1,  0)   upper‑j neighbour
  // 6: (+1,  0,  0)   upper‑k neighbour
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return (n == 0) ? -1 : (n == 6) ? 1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return (n == 1) ? -2 : (n == 5) ? 1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return (n == 2) ? -1 : (n == 4) ? 1 : 0; }
};

// Sweep direction = Z‑axis (k)
struct asym_k_t {
  static constexpr int npoints = 7;

  // Lexicographic ordering of points (dk,dj,di):
  // 0: (-2,  0,  0)   two cells back in k
  // 1: ( 0, -1,  0)   lower‑j neighbour
  // 2: ( 0,  0, -1)   lower‑i neighbour
  // 3: ( 0,  0,  0)   identity
  // 4: ( 0,  0, +1)   upper‑i neighbour
  // 5: ( 0, +1,  0)   upper‑j neighbour
  // 6: (+1,  0,  0)   forward k neighbour

  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) {
    // dk = -2 for point 0, +1 for point 6, otherwise 0
    return (n == 0) ? -2 : (n == 6) ? 1 : 0;
  }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) {
    // dj = -1 for point 1, +1 for point 5, otherwise 0
    return (n == 1) ? -1 : (n == 5) ? 1 : 0;
  }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) {
    // di = -1 for point 2, +1 for point 4, otherwise 0
    return (n == 2) ? -1 : (n == 4) ? 1 : 0;
  }
};

} // namespace halo

namespace impl {
constexpr bool HaloOffsetLess(const int dk0, const int dj0, const int di0, const int dk1,
                              const int dj1, const int di1) {
  if (dk0 != dk1) return dk0 < dk1;
  if (dj0 != dj1) return dj0 < dj1;
  return di0 < di1;
}

template <class Halo>
constexpr bool HaloHasUniqueIdentity() {
  if constexpr (Halo::npoints <= 0) {
    return false;
  } else {
    int count = 0;
    for (int n = 0; n < Halo::npoints; ++n) {
      if (Halo::dk(n) == 0 && Halo::dj(n) == 0 && Halo::di(n) == 0) {
        ++count;
      }
    }
    return count == 1;
  }
}

template <class Halo>
constexpr bool HaloOffsetsAreStrictlySorted() {
  if constexpr (Halo::npoints <= 0) {
    return false;
  } else {
    for (int n = 1; n < Halo::npoints; ++n) {
      if (!HaloOffsetLess(Halo::dk(n - 1), Halo::dj(n - 1), Halo::di(n - 1), Halo::dk(n),
                          Halo::dj(n), Halo::di(n))) {
        return false;
      }
    }
    return true;
  }
}

template <class Halo>
constexpr bool HaloSatisfiesContract() {
  return HaloHasUniqueIdentity<Halo>() && HaloOffsetsAreStrictlySorted<Halo>();
}

} // namespace impl

// Compile-time bounding box of a halo's offsets: the per-dimension min/max shift and
// the resulting extents. Used by the per-point scratch storage to size a dense local
// buffer that covers every shifted copy the halo names.
template <class Halo>
struct HaloBox {
  static constexpr int min_k = [] {
    int out = Halo::dk(0);
    for (int n = 1; n < Halo::npoints; ++n)
      out = std::min(out, Halo::dk(n));
    return out;
  }();
  static constexpr int max_k = [] {
    int out = Halo::dk(0);
    for (int n = 1; n < Halo::npoints; ++n)
      out = std::max(out, Halo::dk(n));
    return out;
  }();
  static constexpr int min_j = [] {
    int out = Halo::dj(0);
    for (int n = 1; n < Halo::npoints; ++n)
      out = std::min(out, Halo::dj(n));
    return out;
  }();
  static constexpr int max_j = [] {
    int out = Halo::dj(0);
    for (int n = 1; n < Halo::npoints; ++n)
      out = std::max(out, Halo::dj(n));
    return out;
  }();
  static constexpr int min_i = [] {
    int out = Halo::di(0);
    for (int n = 1; n < Halo::npoints; ++n)
      out = std::min(out, Halo::di(n));
    return out;
  }();
  static constexpr int max_i = [] {
    int out = Halo::di(0);
    for (int n = 1; n < Halo::npoints; ++n)
      out = std::max(out, Halo::di(n));
    return out;
  }();
  static constexpr int nk = max_k - min_k + 1;
  static constexpr int nj = max_j - min_j + 1;
  static constexpr int ni = max_i - min_i + 1;
  static constexpr int size = nk * nj * ni;
};

template <class Halo>
KOKKOS_INLINE_FUNCTION auto AddHaloToIndexer(const parthenon::Indexer3D &idxer) {
  std::array<int, 3> extend_low{0, 0, 0}, extend_up{0, 0, 0};
  for (int p = 0; p < Halo::npoints; ++p) {
    extend_low[0] = std::max(extend_low[0], -Halo::dk(p));
    extend_low[1] = std::max(extend_low[1], -Halo::dj(p));
    extend_low[2] = std::max(extend_low[2], -Halo::di(p));

    extend_up[0] = std::max(extend_up[0], Halo::dk(p));
    extend_up[1] = std::max(extend_up[1], Halo::dj(p));
    extend_up[2] = std::max(extend_up[2], Halo::di(p));
  }

  return parthenon::Indexer3D({idxer.template StartIdx<0>() - extend_low[0],
                               idxer.template EndIdx<0>() + extend_up[0]},
                              {idxer.template StartIdx<1>() - extend_low[1],
                               idxer.template EndIdx<1>() + extend_up[1]},
                              {idxer.template StartIdx<2>() - extend_low[2],
                               idxer.template EndIdx<2>() + extend_up[2]});
}

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_HALO_HPP_
