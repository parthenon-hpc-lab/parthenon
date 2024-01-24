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

#include "utils/bit_hacks.hpp"

namespace parthenon {
namespace impl {
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
  // Bits of the Morton number going from most to least significant
  uint64_t bits[3];

  MortonNumber(int level, uint64_t x, uint64_t y, uint64_t z)
      : bits{impl::GetMortonBits(level, x, y, z, 2),
             impl::GetMortonBits(level, x, y, z, 1),
             impl::GetMortonBits(level, x, y, z, 0)} {}
};

inline bool operator<(const MortonNumber &lhs, const MortonNumber &rhs) {
  if (lhs.bits[0] == rhs.bits[0] && lhs.bits[1] == rhs.bits[1])
    return lhs.bits[2] < rhs.bits[2];
  if (lhs.bits[0] == rhs.bits[0]) return lhs.bits[1] < rhs.bits[1];
  return lhs.bits[0] < rhs.bits[0];
}

inline bool operator>(const MortonNumber &lhs, const MortonNumber &rhs) {
  if (lhs.bits[0] == rhs.bits[0] && lhs.bits[1] == rhs.bits[1])
    return lhs.bits[2] > rhs.bits[2];
  if (lhs.bits[0] == rhs.bits[0]) return lhs.bits[1] > rhs.bits[1];
  return lhs.bits[0] > rhs.bits[0];
}

inline bool operator==(const MortonNumber &lhs, const MortonNumber &rhs) {
  return (lhs.bits[2] == rhs.bits[2]) && (lhs.bits[1] == rhs.bits[1]) &&
         (lhs.bits[0] == rhs.bits[0]);
}

} // namespace parthenon

#endif // UTILS_MORTON_NUMBER_HPP_
