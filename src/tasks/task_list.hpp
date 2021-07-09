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
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
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

class TaskList;
namespace task_list_impl {
TaskID AddTaskHelper(TaskList *, Task);
} // namespace task_list_impl

class IterativeTasks {
 public:
  IterativeTasks(TaskList *tl, int key) : tl_(tl), key_(key) {
    max_iterations_ = std::numeric_limits<unsigned int>::max();
  }

  template <class T, class... Args>
  TaskID AddTask(TaskID const &dep, T &&func, Args &&... args) {
    return AddTask_(TaskType::iterative, dep, std::forward<T>(func),
                    std::forward<Args>(args)...);
  }
  // overload to add member functions of class T to task list
  // NOTE: we must capture the object pointer
  template <class T, class... Args>
  TaskID AddTask(TaskID const &dep, TaskStatus (T::*func)(Args...), T *obj,
                 Args &&... args) {
    return this->AddTask_(TaskType::iterative, dep, [=]() mutable -> TaskStatus {
      return (obj->*func)(std::forward<Args>(args)...);
    });
  }

  template <class T, class... Args>
  TaskID AddCompletionTask(TaskID const &dep, T &&func, Args &&... args) {
    return AddTask_(TaskType::completion_criteria, dep, std::forward<T>(func),
                    std::forward<Args>(args)...);
  }
  template <class T, class... Args>
  TaskID AddCompletionTask(TaskID const &dep, TaskStatus (T::*func)(Args...), T *obj,
                           Args &&... args) {
    return this->AddTask_(TaskType::completion_criteria, dep,
                          [=]() mutable -> TaskStatus {
                            return (obj->*func)(std::forward<Args>(args)...);
                          });
  }

  void SetMaxIterations(const unsigned int max) { max_iterations_ = max; }
  void SetFailWithMaxIterations(const bool flag) { throw_with_max_iters_ = flag; }
  void SetWarnWithMaxIterations(const bool flag) { warn_with_max_iters_ = flag; }
  bool ShouldThrowWithMax() const { return throw_with_max_iters_; }
  bool ShouldWarnWithMax() const { return warn_with_max_iters_; }
  unsigned int GetMaxIterations() const { return max_iterations_; }
  unsigned int GetIterationCount() const { return count_; }
  void IncrementCount() { count_++; }

 private:
  template <class F, class... Args>
  TaskID AddTask_(const TaskType &type, TaskID const &dep, F &&func, Args &&... args) {
    TaskID id(0);
    id = task_list_impl::AddTaskHelper(tl_, Task(
        id, dep,
        [=, func = std::forward<F>(func)]() mutable -> TaskStatus {
          return func(std::forward<Args>(args)...);
        },
        type, key_));
    return id;
  }
  TaskList *tl_;
  int key_;
  unsigned int max_iterations_;
  unsigned int count_ = 0;
  bool throw_with_max_iters_ = false;
  bool warn_with_max_iters_ = true;
};

class TaskList {
 public:
  TaskList() = default;
  bool IsComplete() { return task_list_.empty(); }
  int Size() { return task_list_.size(); }
  void MarkTaskComplete(TaskID id) { tasks_completed_.SetFinished(id); }
  void ClearComplete() {
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if (task->GetStatus() == TaskStatus::complete &&
          task->GetType() != TaskType::iterative) {
        task = task_list_.erase(task);
      } else {
        ++task;
      }
    }
  }
  void IterationComplete(const int key) {
    completed_iters_.insert(key);
    for (auto &task : task_list_) {
      if (task.GetKey() == key) {
        task.SetStatus(TaskStatus::complete);
      }
    }
  }
  void ClearIteration(const int key) {
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if (task->GetKey() == key) {
        task = task_list_.erase(task);
      } else {
        ++task;
      }
    }
  }
  void ResetIteration(const int key) {
    iter_tasks[key].IncrementCount();
    if (iter_tasks[key].GetIterationCount() == iter_tasks[key].GetMaxIterations()) {
      if (iter_tasks[key].ShouldThrowWithMax()) {
        PARTHENON_THROW("Iteration " + std::to_string(key) +
                        " reached maximum allowed cycles without convergence.");
      }
      if (iter_tasks[key].ShouldWarnWithMax()) {
        PARTHENON_WARN("Iteration " + std::to_string(key) +
                       " reached maximum allowed cycles without convergence.");
      }
      IterationComplete(key);
      return;
    }
    for (auto &task : task_list_) {
      if (task.GetKey() == key) {
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
            IterationComplete(task.GetKey());
          }
        } else if (task.GetStatus() == TaskStatus::iterate &&
                   task.GetType() == TaskType::completion_criteria) {
          ResetIteration(task.GetKey());
        }
      }
    }
    for (auto &key : completed_iters_) {
      ClearIteration(key);
    }
    completed_iters_.clear();
    ClearComplete();
    if (IsComplete()) return TaskListStatus::complete;
    return TaskListStatus::running;
  }
  bool Validate() const {
    std::set<int> iters;
    for (auto &task : task_list_) {
      if (task.GetType() == TaskType::iterative) iters.insert(task.GetKey());
    }
    int num_iters = iters.size();
    int found = 0;
    for (auto &iter : iters) {
      for (auto &task : task_list_) {
        if (task.GetType() == TaskType::completion_criteria && task.GetKey() == iter) {
          found++;
          break;
        }
      }
    }
    bool valid = (found == num_iters);
    PARTHENON_REQUIRE_THROWS(
        valid,
        "Task list validation found iterative tasks without a completion criteria");
    return valid;
  }

  TaskID AddTask(Task tsk) {
    TaskID id(tasks_added_ + 1);
    tsk.SetID(id);
    task_list_.push_back(tsk);
    tasks_added_++;
    return id;
  }

  template <class F, class... Args>
  TaskID AddTask(TaskID const &dep, F &&func, Args &&... args) {
    TaskID id(tasks_added_ + 1);
    task_list_.push_back(
        Task(id, dep, [=, func = std::forward<F>(func)]() mutable -> TaskStatus {
          return func(std::forward<Args>(args)...);
        }));
    tasks_added_++;
    return id;
  }

  // overload to add member functions of class T to task list
  // NOTE: we must capture the object pointer
  template <class T, class... Args>
  TaskID AddTask(TaskID const &dep, TaskStatus (T::*func)(Args...), T *obj,
                 Args &&... args) {
    return this->AddTask(dep, [=]() mutable -> TaskStatus {
      return (obj->*func)(std::forward<Args>(args)...);
    });
  }


  IterativeTasks &AddIteration() {
    int key = iter_tasks.size();
    iter_tasks.push_back(IterativeTasks(this, key));
    return iter_tasks.back();
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
  std::vector<IterativeTasks> iter_tasks;
  std::list<Task> task_list_;
  int tasks_added_ = 0;
  TaskID tasks_completed_;
  std::set<int> completed_iters_;
};

namespace task_list_impl {
// helper function to avoid having to call a member function of TaskList from
// IterativeTasks before TaskList has been defined
inline TaskID AddTaskHelper(TaskList *tl, Task tsk) {
  return tl->AddTask(tsk);
}
} // namespace task_list_impl

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
  bool Validate() const {
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
