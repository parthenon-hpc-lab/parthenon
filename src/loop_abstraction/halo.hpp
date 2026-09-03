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

// Projection-closure check for reduced-dimension (2D/1D) runs.
//
// In a run where some direction is degenerate (extent 1), a halo offset with a
// nonzero component in that direction names a shifted copy of the inner range that
// does not exist. The reduced-dimension policy is DROP: keep only offsets whose
// components in every inactive direction are zero. DROP is a cheap special case of
// the semantically correct operation PROJECT (zero the inactive components of every
// offset). DROP == PROJECT exactly when the halo is closed under projection onto the
// active directions -- i.e. every offset's projection is itself a declared offset.
//
// This holds for every physical stencil we expect (dense corner boxes, single-
// direction extensions, standard 3/5/7-point stencils). It fails only for sparse
// L-shaped / pure-diagonal halos, whose fix is to add the missing projection point
// (filling an unused scratch cell has no side effects). We enforce closure at compile
// time; PROJECT is a deliberately deferred extension.
//
// ndim follows the Parthenon convention: i is active for ndim >= 1, j for ndim >= 2,
// k for ndim >= 3.
template <class Halo>
constexpr bool HaloProjectionClosedForNdim(int ndim) {
  for (int n = 0; n < Halo::npoints; ++n) {
    const int pk = (ndim > 2) ? Halo::dk(n) : 0;
    const int pj = (ndim > 1) ? Halo::dj(n) : 0;
    const int pi = (ndim > 0) ? Halo::di(n) : 0;
    bool found = false;
    for (int m = 0; m < Halo::npoints; ++m) {
      if (Halo::dk(m) == pk && Halo::dj(m) == pj && Halo::di(m) == pi) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

// Check the only degenerations Parthenon can produce: 2D (k inactive) and 1D
// (k and j inactive). A halo closed under both is safe in every run, independent of
// the runtime dimensionality, so the check is fully compile-time.
template <class Halo>
constexpr bool HaloIsProjectionClosed() {
  return HaloProjectionClosedForNdim<Halo>(2) && HaloProjectionClosedForNdim<Halo>(1);
}

} // namespace impl

// The contiguous run [begin, end) of halo offsets kept in a reduced-dimension run.
// Offsets are sorted lexicographically by (dk, dj, di); the inactive directions are
// always the most-significant sort keys (k, then j), so the DROP-kept set -- offsets
// with zero component in every inactive direction -- is a single contiguous run. The
// identity {0,0,0} is always kept. Because a sub-range of a sorted array is still
// sorted, the merge in BuildRegions stays valid when seeded from begin.
//
// ndim follows the Parthenon convention: i active for ndim >= 1, j for ndim >= 2,
// k for ndim >= 3.
struct HaloRange {
  int begin;
  int end;
};

template <class Halo>
KOKKOS_INLINE_FUNCTION constexpr HaloRange HaloReducedRange(int ndim) {
  int begin = Halo::npoints;
  int end = 0;
  for (int n = 0; n < Halo::npoints; ++n) {
    const bool keep = (ndim > 2 || Halo::dk(n) == 0) && (ndim > 1 || Halo::dj(n) == 0) &&
                      (ndim > 0 || Halo::di(n) == 0);
    if (keep) {
      if (n < begin) begin = n;
      end = n + 1;
    }
  }
  if (begin > end) begin = end; // the identity always survives, so end > begin
  return {begin, end};
}

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

// Extend a base indexer's bounds to cover every shifted copy the halo names, using
// only the offsets kept in a reduced-dimension run (the [begin, end) run). Offsets
// dropped in a degenerate direction never extend that direction, so the extended
// indexer stays inside the real logical space.
template <class Halo>
KOKKOS_INLINE_FUNCTION auto AddHaloToIndexer(const parthenon::Indexer3D &idxer,
                                             HaloRange range) {
  std::array<int, 3> extend_low{0, 0, 0}, extend_up{0, 0, 0};
  for (int p = range.begin; p < range.end; ++p) {
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
