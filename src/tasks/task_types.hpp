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

#ifndef TASKS_TASK_TYPES_HPP_
#define TASKS_TASK_TYPES_HPP_

#include <chrono> // NOLINT [build/c++11]
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "globals.hpp"

namespace parthenon {

enum class TaskType { single, iterative, completion_criteria };

class Task {
 public:
  Task(const TaskID &id, const TaskID &dep, std::function<TaskStatus()> func)
      : myid_(id), dep_(dep), type_(TaskType::single), key_(-1), func_(std::move(func)),
        interval_(1) {}
  Task(const TaskID &id, const TaskID &dep, std::function<TaskStatus()> func,
       const TaskType &type, const int key)
      : myid_(id), dep_(dep), type_(type), key_(key), func_(std::move(func)),
        interval_(1) {
    assert(key_ >= 0);
    assert(type_ != TaskType::single);
  }
  Task(const TaskID &id, const TaskID &dep, std::function<TaskStatus()> func,
       const TaskType &type, const int key, const int interval)
      : myid_(id), dep_(dep), type_(type), key_(key), func_(std::move(func)),
        interval_(interval) {
    assert(key_ >= 0);
    assert(type_ != TaskType::single);
    assert(interval_ > 0);
  }
  void operator()() {
    if (calls_ == 0) {
      // on first call, set start time
      start_time_ = std::chrono::high_resolution_clock::now();
    }

    calls_++;
    if (calls_ % interval_ == 0) {
      // set total runtime of current task, must go into Global namespace because
      // functions called by the task functor don't have access to the task itself and
      // they may want to check if the task has been running for too long indicating that
      // it got stuck in an infinite loop
      Globals::current_task_runtime_sec =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now() - start_time_)
              .count() *
          1e-9;
      status_ = func_();
      Globals::current_task_runtime_sec = 0.0;
    } else {
      status_ = TaskStatus::skip;
    }
  }
  void SetID(const TaskID &id) { myid_ = id; }
  TaskID GetID() const { return myid_; }
  TaskID GetDependency() const { return dep_; }
  TaskStatus GetStatus() const { return status_; }
  void SetStatus(const TaskStatus &status) { status_ = status; }
  TaskType GetType() const { return type_; }
  int GetKey() const { return key_; }
  void SetRegional() { regional_ = true; }
  bool IsRegional() const { return regional_; }

 private:
  TaskID myid_;
  const TaskID dep_;
  const TaskType type_;
  const int key_;
  TaskStatus status_ = TaskStatus::incomplete;
  bool regional_ = false;
  std::function<TaskStatus()> func_;
  int calls_ = 0;
  const int interval_;

  // this is used to record the start time of the task so that we can check for how long
  // the task been running and detect potential hangs, infinite loops, etc.
  std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace parthenon

#endif // TASKS_TASK_TYPES_HPP_
