//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_NAN_PAYLOAD_TAG_HPP_
#define UTILS_NAN_PAYLOAD_TAG_HPP_

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>

#include "kokkos_abstraction.hpp"
#include "utils/concepts_lite.hpp"

namespace parthenon {

namespace impl {

template <int NBYTES>
struct contiguous_bitset {
  KOKKOS_DEFAULTED_FUNCTION
  contiguous_bitset() = default;
  KOKKOS_INLINE_FUNCTION
  explicit contiguous_bitset(char val) : bytes{val} {}
  char bytes[NBYTES];

  static constexpr int char_bit_size = sizeof('a') * 8;

  KOKKOS_INLINE_FUNCTION
  void SetOne(const std::size_t idx) {
    char &byte = bytes[idx / char_bit_size];
    byte = byte | (1 << (idx % char_bit_size));
  }

  KOKKOS_INLINE_FUNCTION
  void SetZero(const std::size_t idx) {
    char &byte = bytes[idx / char_bit_size];
    byte = byte & ~(1 << (idx % char_bit_size));
  }

  KOKKOS_INLINE_FUNCTION
  void Flip(const std::size_t idx) {
    char &byte = bytes[idx / char_bit_size];
    byte = byte ^ (1 << (idx % char_bit_size));
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void SetEndBytes(T val) {
    static_assert(sizeof(val) <= NBYTES * sizeof(std::declval<char>()),
                  "Input type is too large for given contiguous_bitset.");
    std::memcpy(bytes, &val, sizeof(val));
  }
};
} // namespace impl

template <class T, class U,
          typename std::enable_if<sizeof(T) == sizeof(U), int>::type = 0>
KOKKOS_INLINE_FUNCTION bool BitwiseCompare(const T &a, const U &b) {
  auto &a_bits = reinterpret_cast<const impl::contiguous_bitset<sizeof(a)> &>(a);
  auto &b_bits = reinterpret_cast<const impl::contiguous_bitset<sizeof(b)> &>(b);
  bool val = true;
  for (int i = 0; i < sizeof(U); ++i) {
    val = val & (b_bits.bytes[i] == a_bits.bytes[i]);
  }
  return val;
  // memcmp returns zero if the memory is the same
  // which is a little confusing, it is also not available on device
  // so we don't use std::memcmp(&a, &b, sizeof(a))
}

template <class T, REQUIRES(std::numeric_limits<T>::is_iec559)>
KOKKOS_INLINE_FUNCTION int GetNaNTag(T val) {
  uint8_t tag;
  std::memcpy(&tag, &val, sizeof(tag));
  return tag;
}

template <class T, REQUIRES(!std::numeric_limits<T>::is_iec559)>
KOKKOS_INLINE_FUNCTION int GetNaNTag(T val) {
  return -1;
}

template <class T, REQUIRES(std::numeric_limits<T>::is_iec559)>
T GetNaNWithPayloadTag(uint8_t tag = 1) {
  double flag_nan = std::numeric_limits<T>::quiet_NaN();
  auto &flag_bits =
      reinterpret_cast<impl::contiguous_bitset<sizeof(flag_nan)> &>(flag_nan);

  // tag must be > 0 otherwise since tag = 0 just keeps the value of
  // the default quiet NaN
  assert(tag > 0);
  flag_bits.SetEndBytes(tag);

  // Do a few quick checks to make sure there isn't anything
  // weird going on
  assert(std::isnan(flag_nan));
  assert(!BitwiseCompare(std::numeric_limits<T>::signaling_NaN(), flag_nan));
  assert(!BitwiseCompare(std::numeric_limits<T>::quiet_NaN(), flag_nan));

  return flag_nan;
}

template <class T, REQUIRES(!std::numeric_limits<T>::is_iec559)>
T GetNaNWithPayloadTag(uint8_t tag = 1) {
  PARTHENON_DEBUG_WARN("Trying to use NaN payload tags without IEEE support.");
  return std::numeric_limits<T>::quiet_NaN();
}

} // namespace parthenon

#endif // UTILS_NAN_PAYLOAD_TAG_HPP_
