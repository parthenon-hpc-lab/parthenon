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

#ifndef TASKS_TASK_LIST_HPP_
#define TASKS_TASK_LIST_HPP_

#include <bitset>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "task_id.hpp"
#include "task_types.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

enum class TaskListStatus { running, stuck, complete, nothing_to_do };

class TaskList {
 public:
  TaskList() = default;
  bool IsComplete() { return task_list_.empty(); }
  int Size() { return task_list_.size(); }
  void Reset() {
    tasks_added_ = 0;
    task_list_.clear();
    tasks_completed_.clear();
  }
  void MarkTaskComplete(TaskID id) { tasks_completed_.SetFinished(id); }
  void ClearComplete() {
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if (task->GetStatus() == TaskStatus::complete
         && task->GetType() != TaskType::iterative) {
        task = task_list_.erase(task);
      } else {
        ++task;
      }
    }
  }
  void IterationComplete(const std::string &label) {
    completed_iters_.insert(label);
    for (auto &task : task_list_) {
      if (task.GetLabel() == label) {
        task.SetStatus(TaskStatus::complete);
      }
    }
  }
  void ClearIteration(const std::string &label) {
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if (task->GetLabel() == label) {
        task = task_list_.erase(task);
      } else {
        ++task;
      }
    }
  }
  void ResetIteration(const std::string &label) {
    count_[label]++;
    if (count_[label] == max_iterations_[label]) {
      if (throw_with_max_iters_[label]) {
        PARTHENON_THROW("Iteration " + label
                      + " reached maximum allowed cycles without convergence.");
      }
      if (warn_with_max_iters_[label]) {
        PARTHENON_WARN("Iteration " + label
                     + " reached maximum allowed cycles without convergence.");
      }
      IterationComplete(label);
      return;
    }
    for (auto &task : task_list_) {
      if (task.GetLabel() == label) {
        if (tasks_completed_.CheckDependencies(task.GetID())) {
          tasks_completed_.SetFinished(task.GetID());
        }
        task.SetStatus(TaskStatus::incomplete);
      }
    }
  }
  TaskListStatus DoAvailable() {
    for (auto &task : task_list_) {
      // first skip task if it's complete.  Only possible for TaskType::iterative
      if (task.GetStatus() == TaskStatus::complete) continue;
      auto dep = task.GetDependency();
      if (tasks_completed_.CheckDependencies(dep)) {
        task();
        if (task.GetStatus() == TaskStatus::complete) {
          MarkTaskComplete(task.GetID());
          if (task.GetType() == TaskType::completion_criteria) {
            IterationComplete(task.GetLabel());
          }
        } else if (task.GetStatus() == TaskStatus::iterate
                  && task.GetType() == TaskType::completion_criteria) {
          ResetIteration(task.GetLabel());
        }
      }
    }
    for (auto &label : completed_iters_) {
      ClearIteration(label);
    }
    completed_iters_.clear();
    ClearComplete();
    if (IsComplete()) return TaskListStatus::complete;
    return TaskListStatus::running;
  }
  bool Validate() {
    std::set<std::string> iters;
    for (auto &task : task_list_) {
      if (task.GetType() == TaskType::iterative) iters.insert(task.GetLabel());
    }
    int num_iters = iters.size();
    int found = 0;
    for (auto &iter : iters) {
      for (auto &task : task_list_) {
        if (task.GetType() == TaskType::completion_criteria && task.GetLabel() == iter) {
          found++;
          break;
        }
      }
    }
    return (found == num_iters);
  }

  template <class F, class... Args>
  TaskID AddTask(TaskID const &dep, F &&func, Args &&... args) {
    TaskID id(tasks_added_ + 1);
    task_list_.push_back(
        Task(id, dep, [=, func = std::forward<F>(func)]() mutable -> TaskStatus {
          return func(args...);
        }));
    tasks_added_++;
    return id;
  }

  // overload to add member functions of class T to task list
  // NOTE: we must capture the object pointer
  template <class T, class... Args>
  TaskID AddTask(TaskID const &dep, TaskStatus (T::*func)(Args...), T *obj,
                 Args &&... args) {
    return this->AddTask(dep,
                         [=]() mutable -> TaskStatus { return (obj->*func)(args...); });
  }

  template <class F, class... Args>
  TaskID AddIterativeTask(const TaskType &type, const std::string &label,
                          TaskID const &dep, F &&func, Args &&... args) {
    if (max_iterations_.count(label) == 0) {
      max_iterations_[label] = std::numeric_limits<unsigned int>::max();
      count_[label] = 0;
    }
    if (throw_with_max_iters_.count(label) == 0) {
      throw_with_max_iters_[label] = false;
    }
    if (warn_with_max_iters_.count(label) == 0) {
      warn_with_max_iters_[label] = true;
    }
    TaskID id(tasks_added_ + 1);
    task_list_.push_back(
        Task(id, dep, [=, func = std::forward<F>(func)]() mutable -> TaskStatus {
          return func(args...);
        }, type, label));
    tasks_added_++;
    return id;
  }

  // overload to add member functions of class T to task list
  // NOTE: we must capture the object pointer
  template <class T, class... Args>
  TaskID AddIterativeTask(const TaskType &type, const std::string &label,
                          TaskID const &dep, TaskStatus (T::*func)(Args...), T *obj,
                          Args &&... args) {
    return this->AddIterativeTask(type, label, dep,
                         [=]() mutable -> TaskStatus { return (obj->*func)(args...); });
  }

  void SetMaxIterations(const std::string &label, const int max) {
    max_iterations_[label] = max;
  }

  void SetFailWithMaxIterations(const std::string &label, bool flag) { 
    throw_with_max_iters_[label] = flag;
  }

  void SetWarnWithMaxIterations(const std::string &label, bool flag) { 
    warn_with_max_iters_[label] = flag;
  }

  void Print() {
    int i = 0;
    std::cout << "TaskList::Print():" << std::endl;
    for (auto &t : task_list_) {
      std::cout << "  " << i << "  " << t.GetID().to_string() << "  "
                << t.GetDependency().to_string() << std::endl;
      i++;
    }
  }

 protected:
  std::list<Task> task_list_;
  int tasks_added_ = 0;
  TaskID tasks_completed_;
  std::map<std::string, unsigned int> max_iterations_;
  std::map<std::string, unsigned int> count_;
  std::set<std::string> completed_iters_;
  std::map<std::string, bool> throw_with_max_iters_;
  std::map<std::string, bool> warn_with_max_iters_;
};

using TaskRegion = std::vector<TaskList>;

struct TaskCollection {
  TaskCollection() = default;
  TaskRegion &AddRegion(const int num_lists) {
    regions.push_back(TaskRegion(num_lists));
    return regions.back();
  }
  TaskListStatus Execute() {
    assert(Validate());
    for (auto &region : regions) {
      int complete_cnt = 0;
      auto num_lists = region.size();
      while (complete_cnt != num_lists) {
        // TODO(pgrete): need to let Kokkos::PartitionManager handle this
        for (auto i = 0; i < num_lists; ++i) {
          if (!region[i].IsComplete()) {
            auto status = region[i].DoAvailable();
            if (status == TaskListStatus::complete) {
              complete_cnt++;
            }
          }
        }
      }
    }
    return TaskListStatus::complete;
  }
  bool Validate() {
    for (auto &region : regions) {
      for (auto &list : region) {
        if (!list.Validate()) return false;
      }
    }
    return true;
  }

  std::vector<TaskRegion> regions;
};

} // namespace parthenon

#endif // TASKS_TASK_LIST_HPP_
