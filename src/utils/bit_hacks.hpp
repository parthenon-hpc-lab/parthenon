//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_BIT_HACKS_HPP_
#define UTILS_BIT_HACKS_HPP_

namespace parthenon {
namespace impl {
template <int NDIM = 3>
inline constexpr uint64_t GetInterleaveConstant(int power) {
  // For power = 2, NDIM = 3, this should return
  // ...000011000011
  // For power = 1, NDIM = 3, this should return
  // ...001001001001
  // For power = 2, NDIM = 2, this should return
  // ...001100110011
  // etc.
  constexpr int type_bit_size = sizeof(uint64_t) * 8;
  if (power >= type_bit_size) return ~0ULL; // Return with all bits set
  uint64_t i_const = ~((~0ULL) << power);   // std::pow(2, power) - 1;
  int cur_shift = type_bit_size * NDIM; // Works for anything that will fit in uint64_t
  while (cur_shift >= NDIM * power) {
    if (cur_shift < type_bit_size) i_const = (i_const << cur_shift) | i_const;
    cur_shift /= 2;
  }
  return i_const;
}
} // namespace impl

template <int NDIM = 3, int N_VALID_BITS = 21>
inline uint64_t InterleaveZeros(uint64_t x) {
  // This is a standard bithack for interleaving zeros in binary numbers to make a Morton
  // number
  if constexpr (N_VALID_BITS >= 64)
    x = (x | x << 64 * (NDIM - 1)) & impl::GetInterleaveConstant<NDIM>(64);
  if constexpr (N_VALID_BITS >= 32)
    x = (x | x << 32 * (NDIM - 1)) & impl::GetInterleaveConstant<NDIM>(32);
  if constexpr (N_VALID_BITS >= 16)
    x = (x | x << 16 * (NDIM - 1)) & impl::GetInterleaveConstant<NDIM>(16);
  if constexpr (N_VALID_BITS >= 8)
    x = (x | x << 8 * (NDIM - 1)) & impl::GetInterleaveConstant<NDIM>(8);
  if constexpr (N_VALID_BITS >= 4)
    x = (x | x << 4 * (NDIM - 1)) & impl::GetInterleaveConstant<NDIM>(4);
  if constexpr (N_VALID_BITS >= 2)
    x = (x | x << 2 * (NDIM - 1)) & impl::GetInterleaveConstant<NDIM>(2);
  if constexpr (N_VALID_BITS >= 1)
    x = (x | x << 1 * (NDIM - 1)) & impl::GetInterleaveConstant<NDIM>(1);
  return x;
}

inline int NumberOfBinaryTrailingZeros(std::uint64_t val) {
  int n = 0;
  if (val == 0) return sizeof(val) * 8;
  if ((val & 0xFFFFFFFF) == 0) {
    n += 32;
    val = val >> 32;
  }
  if ((val & 0x0000FFFF) == 0) {
    n += 16;
    val = val >> 16;
  }
  if ((val & 0x000000FF) == 0) {
    n += 8;
    val = val >> 8;
  }
  if ((val & 0x0000000F) == 0) {
    n += 4;
    val = val >> 4;
  }
  if ((val & 0x00000003) == 0) {
    n += 2;
    val = val >> 2;
  }
  if ((val & 0x00000001) == 0) {
    n += 1;
    val = val >> 1;
  }
  return n;
}

inline int MaximumPowerOf2Divisor(int in) { return in & (~(in - 1)); }

inline uint IntegerLog2Ceil(uint in) {
  uint log2 = 0;
  uint in_temp = in;
  while (in_temp >>= 1) {
    log2++;
  }
  uint pow = 1U << log2;
  return log2 + (pow != in);
}

inline uint IntegerLog2Floor(uint in) {
  uint log2 = 0;
  while (in >>= 1) {
    log2++;
  }
  return log2;
}

} // namespace parthenon

#endif // UTILS_BIT_HACKS_HPP_
