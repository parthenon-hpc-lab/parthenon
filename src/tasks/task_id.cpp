//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
//! \file tasks.cpp
//  \brief implementation of the TaskID class

#include "utils/error_checking.hpp"
#include "tasks/task_id.hpp"

#include <algorithm>
#include <bitset>
#include <stdexcept>
#include <string>
#include <utility>

namespace parthenon {

// TaskID constructor. Default id = 0.
TaskID::TaskID(int id) { Set(id); }

void TaskID::Set(int id) {
  if (id < 0) throw std::invalid_argument("TaskID requires integer arguments >= 0");
  if (id == 0) {
    bitblocks.resize(1);
    return;
  }
  id--;
  const int n_myblocks = id / BITBLOCK + 1;
  // grow if necessary.  never shrink
  if (n_myblocks > bitblocks.size()) bitblocks.resize(n_myblocks);
  bitblocks[n_myblocks - 1] |= (static_cast<uint64_t>(1) << (id % BITBLOCK));
  bit = id; 
  nbits_set++;
}

void TaskID::clear() {
  for (auto &bset : bitblocks) {
    bset = 0;
  }
}

bool TaskID::CheckDependencies(const TaskID &rhs) const {
  const int n_myblocks = bitblocks.size();
  const int n_srcblocks = rhs.bitblocks.size();
  if (n_myblocks == n_srcblocks) {
    for (int i = 0; i < n_myblocks; i++) {
      if ((bitblocks[i] & rhs.bitblocks[i]) != rhs.bitblocks[i]) return false;
    }
  } else if (n_myblocks > n_srcblocks) {
    for (int i = 0; i < n_srcblocks; i++) {
      if ((bitblocks[i] & rhs.bitblocks[i]) != rhs.bitblocks[i]) return false;
    }
  } else {
    for (int i = 0; i < n_myblocks; i++) {
      if ((bitblocks[i] & rhs.bitblocks[i]) != rhs.bitblocks[i]) return false;
    }
    for (int i = n_myblocks; i < n_srcblocks; i++) {
      if (rhs.bitblocks[i] > 0) return false;
    }
  }
  return true;
}

void TaskID::SetFinished(const TaskID &rhs) {
  const int n_myblocks = bitblocks.size();
  const int n_srcblocks = rhs.bitblocks.size();
  if (n_myblocks == n_srcblocks) {
    for (int i = 0; i < n_myblocks; i++) {
      bitblocks[i] ^= rhs.bitblocks[i];
    }
  } else if (n_myblocks > n_srcblocks) {
    for (int i = 0; i < n_srcblocks; i++) {
      bitblocks[i] ^= rhs.bitblocks[i];
    }
  } else {
    for (int i = 0; i < n_myblocks; i++) {
      bitblocks[i] ^= rhs.bitblocks[i];
    }
    for (int i = n_myblocks; i < n_srcblocks; i++) {
      bitblocks.push_back(rhs.bitblocks[i]);
    }
  }
}

bool TaskID::operator==(const TaskID &rhs) const {
  if (nbits_set != rhs.nbits_set) return false;
  
  const int n_myblocks = bitblocks.size();
  const int n_srcblocks = rhs.bitblocks.size();
  if (n_myblocks == n_srcblocks) {
    for (int i = 0; i < n_myblocks; i++) {
      if (bitblocks[i] != rhs.bitblocks[i]) return false;
    }
  } else if (n_myblocks > n_srcblocks) {
    for (int i = 0; i < n_srcblocks; i++) {
      if (bitblocks[i] != rhs.bitblocks[i]) return false;
    }
    for (int i = n_srcblocks; i < n_myblocks; i++) {
      if (bitblocks[i] > 0) return false;
    }
  } else {
    for (int i = 0; i < n_myblocks; i++) {
      if (bitblocks[i] != rhs.bitblocks[i]) return false;
    }
    for (int i = n_myblocks; i < n_srcblocks; i++) {
      if (rhs.bitblocks[i] > 0) return false;
    }
  }
  return true;
}

bool TaskID::operator!=(const TaskID &rhs) const { return !operator==(rhs); }

TaskID TaskID::operator|(const TaskID &rhs) const {
  TaskID res;
  const int n_myblocks = bitblocks.size();
  const int n_srcblocks = rhs.bitblocks.size();
  res.bitblocks.resize(std::max(n_myblocks, n_srcblocks));
  if (n_myblocks == n_srcblocks) {
    for (int i = 0; i < n_myblocks; i++) {
      res.bitblocks[i] = bitblocks[i] | rhs.bitblocks[i];
    }
  } else if (n_myblocks > n_srcblocks) {
    for (int i = 0; i < n_srcblocks; i++) {
      res.bitblocks[i] = bitblocks[i] | rhs.bitblocks[i];
    }
    for (int i = n_srcblocks; i < n_myblocks; i++) {
      res.bitblocks[i] = bitblocks[i];
    }
  } else {
    for (int i = 0; i < n_myblocks; i++) {
      res.bitblocks[i] = bitblocks[i] | rhs.bitblocks[i];
    }
    for (int i = n_myblocks; i < n_srcblocks; i++) {
      res.bitblocks[i] = rhs.bitblocks[i];
    }
  }
  return res;
}

std::string TaskID::to_string() const {
  std::string bs;
  for (int i = bitblocks.size() - 1; i >= 0; i--) {
    //bs += bitblocks[i].to_string();
  }
  return bs;
}

} // namespace parthenon
