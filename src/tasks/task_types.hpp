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

#ifndef TASKS_TASK_TYPES_HPP_
#define TASKS_TASK_TYPES_HPP_

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"

namespace parthenon {

class Task {
 public:
  Task(const TaskID &id, TaskID dep, std::function<TaskStatus()> func)
      : myid_(id), dep_(dep), func_(std::move(func)) {}
  TaskStatus operator()() { return func_(); }
  TaskID GetID() { return myid_; }
  TaskID GetDependency() { return dep_; }
  void SetComplete() { complete_ = true; }
  bool IsComplete() { return complete_; }

 private:
  TaskID myid_, dep_;
  bool lb_time, complete_ = false;
  std::function<TaskStatus()> func_;
};

} // namespace parthenon

#endif // TASKS_TASK_TYPES_HPP_
