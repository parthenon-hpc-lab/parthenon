//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_MORTON_NUMBER_HPP_
#define UTILS_MORTON_NUMBER_HPP_

#include <memory>
#include <vector>

namespace parthenon {
namespace impl {
template <int NDIM = 3>
constexpr uint64_t GetInterleaveConstant(int power) {
  // For power = 2, NDIM = 3, this should return
  // ...000011000011
  // For power = 1, NDIM = 3, this should return
  // ...001001001001
  // For power = 2, NDIM = 2, this should return
  // ...001100110011
  // etc.
  uint64_t i_const = ~((~static_cast<uint64_t>(0)) << power); // std::pow(2, power) - 1;
  int cur_shift =
      sizeof(uint64_t) * 8 * NDIM; // Works for anything that will fit in uint64_t
  while (cur_shift >= NDIM * power) {
    if (cur_shift < sizeof(uint64_t) * 8) i_const = (i_const << cur_shift) | i_const;
    cur_shift /= 2;
  }
  return i_const;
}

template <int NDIM = 3, int N_VALID_BITS = 21>
uint64_t InterleaveZeros(uint64_t x) {
  // This is a standard bithack for interleaving zeros in binary numbers to make a Morton
  // number
  if constexpr (N_VALID_BITS >= 64)
    x = (x | x << 64 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(64);
  if constexpr (N_VALID_BITS >= 32)
    x = (x | x << 32 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(32);
  if constexpr (N_VALID_BITS >= 16)
    x = (x | x << 16 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(16);
  if constexpr (N_VALID_BITS >= 8)
    x = (x | x << 8 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(8);
  if constexpr (N_VALID_BITS >= 4)
    x = (x | x << 4 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(4);
  if constexpr (N_VALID_BITS >= 2)
    x = (x | x << 2 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(2);
  if constexpr (N_VALID_BITS >= 1)
    x = (x | x << 1 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(1);
  return x;
}

inline uint64_t GetMortonBits(int level, uint64_t x, uint64_t y, uint64_t z, int chunk) {
  constexpr int NBITS = 21;
  constexpr uint64_t lowest_nbits_mask = ~((~static_cast<uint64_t>(0)) << NBITS);

  // Shift the by level location to the global location
  x = x << (3 * NBITS - level);
  y = y << (3 * NBITS - level);
  z = z << (3 * NBITS - level);

  // Get the chunk signifigance NBITS bits of each direction
  x = (x >> (chunk * NBITS)) & lowest_nbits_mask;
  y = (y >> (chunk * NBITS)) & lowest_nbits_mask;
  z = (z >> (chunk * NBITS)) & lowest_nbits_mask;

  // Return the interleaved section of the morton number
  return InterleaveZeros<3, NBITS>(z) << 2 | InterleaveZeros<3, NBITS>(y) << 1 |
         InterleaveZeros<3, NBITS>(x);
}
} // namespace impl

struct MortonNumber {
  uint64_t most, mid, least;

  MortonNumber(int level, uint64_t x, uint64_t y, uint64_t z)
      : most(GetMortonBits(level, x, y, z, 2)), mid(GetMortonBits(level, x, y, z, 1)),
        least(GetMortonBits(level, x, y, z, 0)) {}
};

inline bool operator<(const MortonNumber &lhs, const MortonNumber &rhs) {
  if (lhs.most == rhs.most && lhs.mid == rhs.mid) return lhs.least < rhs.least;
  if (lhs.most == rhs.most) return lhs.mid < rhs.mid;
  return lhs.most < rhs.most;
}

inline bool operator>(const MortonNumber &lhs, const MortonNumber &rhs) {
  if (lhs.most == rhs.most && lhs.mid == rhs.mid) return lhs.least > rhs.least;
  if (lhs.most == rhs.most) return lhs.mid > rhs.mid;
  return lhs.most > rhs.most;
}

inline bool operator==(const MortonNumber &lhs, const MortonNumber &rhs) {
  return (lhs.most == rhs.most) && (lhs.mid == rhs.mid) && (lhs.least == rhs.least);
}

} // namespace parthenon

#endif // UTILS_MORTON_NUMBER_HPP_
