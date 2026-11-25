//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights
// reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================
#ifndef SRC_UTILITIES_THREE_VEC_HPP_
#define SRC_UTILITIES_THREE_VEC_HPP_

#include <parthenon/parthenon.hpp>

namespace parthenon {
namespace utils {

struct ThreeVec {
  std::array<Real, 3> vals;

  KOKKOS_FORCEINLINE_FUNCTION
  ThreeVec() = default;

  KOKKOS_FORCEINLINE_FUNCTION
  ThreeVec(Real a1, Real a2, Real a3) : vals{a1, a2, a3} {}

  template <class var_t, class pack_t>
  KOKKOS_FORCEINLINE_FUNCTION ThreeVec(const pack_t &pack, int b, var_t, int k, int j,
                                       int i)
      : vals{pack(b, var_t(0), k, j, i), pack(b, var_t(1), k, j, i),
             pack(b, var_t(2), k, j, i)} {}

  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator[](CoordinateDirection dir) { return vals[dir - 1]; }

  KOKKOS_FORCEINLINE_FUNCTION
  const Real &operator[](CoordinateDirection dir) const { return vals[dir - 1]; }
};

KOKKOS_FORCEINLINE_FUNCTION
ThreeVec operator+(const ThreeVec &a, const ThreeVec &b) {
  ThreeVec out;
  for (const auto dir : {X1DIR, X2DIR, X3DIR})
    out[dir] = a[dir] + b[dir];
  return out;
}

KOKKOS_FORCEINLINE_FUNCTION
ThreeVec operator-(const ThreeVec &a, const ThreeVec &b) {
  ThreeVec out;
  for (const auto dir : {X1DIR, X2DIR, X3DIR})
    out[dir] = a[dir] - b[dir];
  return out;
}

KOKKOS_FORCEINLINE_FUNCTION
ThreeVec operator*(Real a, const ThreeVec &b) {
  ThreeVec out;
  for (const auto dir : {X1DIR, X2DIR, X3DIR})
    out[dir] = a * b[dir];
  return out;
}

KOKKOS_FORCEINLINE_FUNCTION
ThreeVec operator*(const ThreeVec &b, Real a) {
  ThreeVec out;
  for (const auto dir : {X1DIR, X2DIR, X3DIR})
    out[dir] = b[dir] * a;
  return out;
}

KOKKOS_FORCEINLINE_FUNCTION
ThreeVec operator/(const ThreeVec &b, Real a) {
  ThreeVec out;
  for (const auto dir : {X1DIR, X2DIR, X3DIR})
    out[dir] = b[dir] / a;
  return out;
}

KOKKOS_FORCEINLINE_FUNCTION
Real DotProduct(const ThreeVec &a, const ThreeVec &b) {
  return a[X1DIR] * b[X1DIR] + a[X2DIR] * b[X2DIR] + a[X3DIR] * b[X3DIR];
}

KOKKOS_FORCEINLINE_FUNCTION
ThreeVec CrossProduct(const ThreeVec &a, const ThreeVec &b) {
  ThreeVec out;
  out[X1DIR] = a[X2DIR] * b[X3DIR] - a[X3DIR] * b[X2DIR];
  out[X2DIR] = a[X3DIR] * b[X1DIR] - a[X1DIR] * b[X3DIR];
  out[X3DIR] = a[X1DIR] * b[X2DIR] - a[X2DIR] * b[X1DIR];
  return out;
}

} // namespace utils
} // namespace parthenon

#endif // SRC_UTILITIES_THREE_VEC_HPP_
