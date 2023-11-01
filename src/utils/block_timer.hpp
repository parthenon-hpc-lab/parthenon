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

template <typename T = void>
class BlockTimer {
#ifdef ENABLE_LB_TIMERS
 public:
  BlockTimer() = delete;
  // constructor for team policies when there is no pack
  KOKKOS_INLINE_FUNCTION
  BlockTimer(team_mbr_t &member, T cost)
    : member_(&member), pack_(cost), b_(0), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for non-team policies when there is no pack
  KOKKOS_INLINE_FUNCTION
  explicit BlockTimer(T cost)
    : member_(nullptr), pack_(cost), b_(0), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for team policies and packs
  KOKKOS_INLINE_FUNCTION
  BlockTimer(team_mbr_t &member, const T &pack, const int b)
    : member_(&member), pack_(pack), b_(b), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for non-team policies and packs
  KOKKOS_INLINE_FUNCTION
  BlockTimer(const T &pack, const int b)
    : member_(nullptr), pack_(pack), b_(b), start_(Kokkos::Impl::clock_tic()) {}
  // constructor for team policies without a block index
  KOKKOS_INLINE_FUNCTION
  ~BlockTimer() {
    auto stop = Kokkos::Impl::clock_tic();
    // deal with overflow of clock
    auto diff = (stop < start_ ?
                  static_cast<double>(std::numeric_limits<uint64_t>::max() - start_)
                    + static_cast<double>(stop) :
                  static_cast<double>(stop - start_));
    if (member_ == nullptr) {
      if constexpr (std::is_same<T, double *>::value) {
        Kokkos::atomic_add(pack_, diff);
      } else {
        Kokkos::atomic_add(&(pack_.GetCost(b_)), diff);
      }
    } else {
      if constexpr (std::is_same<T, double *>::value) {
        Kokkos::single(Kokkos::PerTeam(*member_), [&] () {
          Kokkos::atomic_add(pack_, diff);
        });
      } else {
        Kokkos::single(Kokkos::PerTeam(*member_), [&] () {
          Kokkos::atomic_add(&(pack_.GetCost(b_)), diff);
        });
      }
    }
  }
 private:
  const team_mbr_t *member_;
  T &pack_;
  const int b_;
  const uint64_t start_;
#else // no timers, so just stub this out
 public:
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION
  BlockTimer(Args &&... args) {}
#endif
};

} // namespace parthenon

#endif
