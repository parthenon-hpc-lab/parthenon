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

#ifndef UTILS_BYTE_UTILS_HPP_
#define UTILS_BYTE_UTILS_HPP_

// This file was made in part with generative AI.

#include <array>
#include <type_traits>

#include <Kokkos_BitManipulation.hpp>

#include "parthenon_arrays.hpp"

namespace parthenon {

namespace byte_utils {

template <typename T>
KOKKOS_INLINE_FUNCTION void PackValue(const ParArray1D<char> &buffer, int &offset,
                                      const T &value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Packing requires trivially copyable values.");
  const auto bytes = Kokkos::bit_cast<std::array<unsigned char, sizeof(T)>>(value);
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    buffer(offset + i) = static_cast<char>(bytes[i]);
  }
  offset += sizeof(T);
}

template <typename T>
KOKKOS_INLINE_FUNCTION T UnpackValue(const ParArray1D<char> &buffer, int &offset) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Unpacking requires trivially copyable values.");
  std::array<unsigned char, sizeof(T)> bytes{};
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    bytes[i] = static_cast<unsigned char>(buffer(offset + i));
  }
  offset += sizeof(T);
  return Kokkos::bit_cast<T>(bytes);
}

} // namespace byte_utils

} // namespace parthenon

#endif // UTILS_BYTE_UTILS_HPP_
