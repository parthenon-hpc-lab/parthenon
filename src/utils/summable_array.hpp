//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_SUMMABLE_ARRAY_HPP_
#define UTILS_SUMMABLE_ARRAY_HPP_
namespace parthenon {

template <class T, std::size_t N>
struct summable_array_t {
  using value_type = T;
  value_type data_[N];

  // Kokkos reduction requirements
  KOKKOS_INLINE_FUNCTION
  summable_array_t() { init(); }

  KOKKOS_INLINE_FUNCTION
  summable_array_t(const summable_array_t &rhs) {
    for (int i = 0; i < N; i++)
      data_[i] = rhs.data_[i];
  }

  KOKKOS_INLINE_FUNCTION
  void init() {
    for (int i = 0; i < N; i++)
      data_[i] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  summable_array_t &operator+=(const summable_array_t &src) {
    for (int i = 0; i < N; i++)
      data_[i] += src.data_[i];
    return *this;
  }

  value_type &operator[](std::size_t i) { return data_[i]; }
  const value_type &operator[](std::size_t i) const { return data_[i]; }

  // Contiguous container requirements
  KOKKOS_INLINE_FUNCTION
  std::size_t size() const { return N; }

  value_type *data() { return data_; }
};

} // namespace parthenon

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<parthenon::summable_array_t<parthenon::Real, 2>> {
  KOKKOS_FORCEINLINE_FUNCTION static parthenon::summable_array_t<parthenon::Real, 2>
  sum() {
    return parthenon::summable_array_t<parthenon::Real, 2>();
  }
};

} // namespace Kokkos

#endif // UTILS_SUMMABLE_ARRAY_HPP_
