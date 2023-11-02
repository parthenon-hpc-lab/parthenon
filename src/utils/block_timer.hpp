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

#include "config.hpp"
#include "kokkos_abstraction.hpp"

namespace parthenon {

class BlockTimer {
#ifdef ENABLE_LB_TIMERS
 public:
  BlockTimer() = delete;
  // constructor for team policies when there is no pack
  KOKKOS_INLINE_FUNCTION
  BlockTimer(team_mbr_t &member, double *cost)
      : member_(&member), cost_(cost), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for non-team policies when there is no pack
  KOKKOS_INLINE_FUNCTION
  explicit BlockTimer(double *cost)
      : member_(nullptr), cost_(cost), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for team policies and packs
  template <typename T>
  KOKKOS_INLINE_FUNCTION BlockTimer(team_mbr_t &member, const T &pack, const int b)
      : member_(&member), cost_(&pack.GetCost(b)), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for non-team policies and packs
  template <typename T>
  KOKKOS_INLINE_FUNCTION BlockTimer(const T &pack, const int b)
      : member_(nullptr), cost_(&pack.GetCost(b)), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for team policies without a block index
  KOKKOS_INLINE_FUNCTION
  ~BlockTimer() {
    auto stop = Kokkos::Impl::clock_tic();
    // deal with overflow of clock
    auto diff =
        (stop < start_
             ? static_cast<double>(std::numeric_limits<uint64_t>::max() - start_) +
                   static_cast<double>(stop)
             : static_cast<double>(stop - start_));
    if (member_ == nullptr) {
      Kokkos::atomic_add(cost_, diff);
    } else {
      Kokkos::single(Kokkos::PerTeam(*member_),
                     [&]() { Kokkos::atomic_add(cost_, diff); });
    }
  }

 private:
  const team_mbr_t *member_;
  double *cost_;
  const uint64_t start_;
#else // no timers, so just stub this out
 public:
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION BlockTimer(Args &&...args) {}
#endif
};

} // namespace parthenon

#endif
