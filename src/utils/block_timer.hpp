//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#ifndef UTILS_BLOCK_TIMER_HPP_
#define UTILS_BLOCK_TIMER_HPP_

#include <limits>

#include <impl/Kokkos_ClockTic.hpp>

namespace parthenon {

using cost_t = AtomicParArray1D<double>;
class BlockTimer {
 public:
  BlockTimer() = delete;
  KOKKOS_INLINE_FUNCTION
  BlockTimer(const int lid, cost_t &cost) : lid_(lid), cost_(cost) {
    start_ = Kokkos::Impl::clock_tic();
  }
  KOKKOS_INLINE_FUNCTION
  ~BlockTimer() {
    stop_ = Kokkos::Impl::clock_tic();
    // deal with overflow of clock
    cost_(lid) += (stop_ < start_ ?
                  static_cast<double>(std::numeric_limits<uint64_t>::max() - start_)
                    + static_cast<double>(stop_) :
                  static_cast<double>(stop_ - start_));
  }
 private:
  const int lid_;
  const cost_t &cost_;
  uint64_t start_, stop_;
  ;
};

} // namespace parthenon

#endif
