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

#ifndef TASKS_TASK_ID_HPP_
#define TASKS_TASK_ID_HPP_

#include <bitset>
#include <string>
#include <vector>

#include "basic_types.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief generalization of bit fields for Task IDs, status, and dependencies.

#define BITBLOCK 64

class TaskID {
 public:
  TaskID() : nbits_set(0), bit(-1) { Set(0); }
  explicit TaskID(int id);

  void Set(int id);
  void clear();
  bool CheckDependencies(const TaskID &rhs) const;
  void SetFinished(const TaskID &rhs);
  bool operator==(const TaskID &rhs) const;
  bool operator!=(const TaskID &rhs) const;
  TaskID operator|(const TaskID &rhs) const;
  std::string to_string() const;

 private:
  int nbits_set; 
  int bit;
  std::vector<uint64_t> bitblocks;
};

} // namespace parthenon

#endif // TASKS_TASK_ID_HPP_
