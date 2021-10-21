//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_ALIAS_METHOD_HPP_
#define UTILS_ALIAS_METHOD_HPP_
//! \file alias_method.hpp
//  \brief Construct tables used to sample from a discrete probability distribution using
//  the alias method

#include <vector>

#include <kokkos_abstraction.hpp>

#include "defs.hpp"

namespace parthenon {
namespace AliasMethod {

//! AliasMethod
//  \brief Implement the alias method to sample from a discrete probability distribution
//
//  This class implements the alias method that allows sampling from a discrete
//  probability distribution in O(1). For details, see:
//  https://en.wikipedia.org/wiki/Alias_method
//  https://www.keithschwarz.com/darts-dice-coins/
class AliasMethod {
 public:
  Kokkos::View<Real *> prob_table;
  Kokkos::View<int *> alias_table;

  // Construct the AliasMethod with the given discrete probabilities
  explicit AliasMethod(const std::vector<Real> &probabilities);

  // Sample from the discrete probability distribution defined by the probabilities given
  // to the constructor. The returned value is the zero-based index of the sampled
  // element. rand1 and rand2 are two independent random variables drawn from the uniform
  // distribution [0,1) that need to be provided.
  // This function can be called inside Kokkos loops on the device.
  KOKKOS_INLINE_FUNCTION int Sample(Real rand1, Real rand2) const {
    int idx = static_cast<int>(rand1 * prob_table.size());
    if ((rand2 >= prob_table(idx)) && (alias_table(idx) != -1))
      return alias_table(idx);
    else
      return idx;
  }
};

} // namespace AliasMethod
} // namespace parthenon

#endif // UTILS_ALIAS_METHOD_HPP_
