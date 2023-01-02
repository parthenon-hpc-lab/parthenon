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
#include "utils/reductions.hpp"

namespace parthenon {

enum class TaskListStatus { running, stuck, complete, nothing_to_do };

class TaskList;
namespace task_list_impl {
TaskID AddTaskHelper(TaskList *, Task);
} // namespace task_list_impl

class IterativeTasks {
 public:
  IterativeTasks(TaskList *tl, int key) : tl_(tl), key_(key) {
    max_iterations_ = std::numeric_limits<int>::max();
  }

  // overload to add member functions of class T to task list
  // NOTE: we must capture the object pointer
  template <class T, class U, class... Args>
  TaskID AddTask(TaskID const &dep, TaskStatus (T::*func)(Args...), U *obj,
                 Args &&...args) {
    return this->AddTask_(TaskType::iterative, dep, [=]() mutable -> TaskStatus {
      return (obj->*func)(std::forward<Args>(args)...);
    });
  }

  template <class T, class... Args>
  TaskID AddTask(TaskID const &dep, T &&func, Args &&...args) {
    return AddTask_(TaskType::iterative, dep, std::forward<T>(func),
                    std::forward<Args>(args)...);
  }

  template <class T, class U, class... Args>
  TaskID SetCompletionTask(TaskID const &dep, TaskStatus (T::*func)(Args...), U *obj,
                           Args &&...args) {
    return AddTask_(TaskType::completion_criteria, dep, [=]() mutable -> TaskStatus {
      return (obj->*func)(std::forward<Args>(args)...);
    });
  }

  template <class T, class... Args>
  TaskID SetCompletionTask(TaskID const &dep, T &&func, Args &&...args) {
    return AddTask_(TaskType::completion_criteria, dep, std::forward<T>(func),
                    std::forward<Args>(args)...);
  }

  void SetMaxIterations(const int max) {
    assert(max > 0);
    max_iterations_ = max;
  }
  void SetFailWithMaxIterations(const bool flag) { throw_with_max_iters_ = flag; }
  void SetWarnWithMaxIterations(const bool flag) { warn_with_max_iters_ = flag; }
  bool ShouldThrowWithMax() const { return throw_with_max_iters_; }
  bool ShouldWarnWithMax() const { return warn_with_max_iters_; }
  int GetMaxIterations() const { return max_iterations_; }
  int GetIterationCount() const { return count_; }
  void IncrementCount() { count_++; }

 private:
  template <class F, class... Args>
  TaskID AddTask_(const TaskType &type, TaskID const &dep, F &&func, Args &&...args) {
    TaskID id(0);
    id = task_list_impl::AddTaskHelper(
        tl_, Task(
                 id, dep,
                 [=, func = std::forward<F>(func)]() mutable -> TaskStatus {
                   return func(std::forward<Args>(args)...);
                 },
                 type, key_));
    return id;
  }
  TaskList *tl_;
  int key_;
  int max_iterations_;
  unsigned int count_ = 0;
  bool throw_with_max_iters_ = false;
  bool warn_with_max_iters_ = true;
};

class TaskList {
 public:
  TaskList() = default;
  bool IsComplete() { return task_list_.empty(); }
  int Size() { return task_list_.size(); }
  void MarkRegional(const TaskID &id) {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        task.SetRegional();
        break;
      }
    }
  }
  void MarkTaskComplete(const TaskID &id) { tasks_completed_.SetFinished(id); }
  bool CheckDependencies(const TaskID &id) const {
    return tasks_completed_.CheckDependencies(id);
  }
  bool CheckTaskRan(TaskID id) const {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        return (task.GetStatus() != TaskStatus::incomplete &&
                task.GetStatus() != TaskStatus::skip);
      }
    }
    return false;
  }
  bool CheckStatus(const TaskID &id, TaskStatus status) const {
    for (auto &task : task_list_) {
      if (task.GetID() == id) return (task.GetStatus() == status);
    }
    return true;
  }
  bool CheckTaskCompletion(const TaskID &id) const {
    return CheckStatus(id, TaskStatus::complete);
  }
  void ClearComplete() {
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if (task->GetStatus() == TaskStatus::complete &&
          task->GetType() != TaskType::iterative &&
          task->GetType() != TaskType::completion_criteria && !task->IsRegional()) {
        task = task_list_.erase(task);
      } else {
        ++task;
      }
    }
    std::set<int> completed_iters;
    for (auto &tsk : task_list_) {
      if (tsk.GetType() == TaskType::completion_criteria &&
          tsk.GetStatus() == TaskStatus::complete && !tsk.IsRegional()) {
        completed_iters.insert(tsk.GetKey());
      }
    }
    for (const auto &key : completed_iters) {
      ClearIteration(key);
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
    // TODO(the reviewers) how important is it that we can set key to -1 rather than
    // making it a size_t/unit32_t in the first place?
    assert(key >= 0);
    PARTHENON_REQUIRE_THROWS((decltype(iter_tasks.size()))key < iter_tasks.size(),
                             "Invalid iteration key");
    iter_tasks[key].IncrementCount();
    if (iter_tasks[key].GetIterationCount() == iter_tasks[key].GetMaxIterations()) {
      if (iter_tasks[key].ShouldThrowWithMax()) {
        PARTHENON_THROW("Iteration " + iter_labels[key] +
                        " reached maximum allowed cycles without convergence.");
      }
      if (iter_tasks[key].ShouldWarnWithMax()) {
        PARTHENON_WARN("Iteration " + iter_labels[key] +
                       " reached maximum allowed cycles without convergence.");
      }
      ClearIteration(key);
      return;
    }
    for (auto &task : task_list_) {
      if (task.GetKey() == key) {
        if (CheckDependencies(task.GetID())) {
          MarkTaskComplete(task.GetID());
        }
        task.SetStatus(TaskStatus::incomplete);
      }
    }
  }
  void ResetIfNeeded(const TaskID &id) {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        if (task.GetType() == TaskType::completion_criteria) {
          ResetIteration(task.GetKey());
        }
        break;
      }
    }
  }
  void CompleteIfNeeded(const TaskID &id) {
    MarkTaskComplete(id);
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if (task->GetID() == id) {
        if (task->GetType() == TaskType::completion_criteria) {
          ClearIteration(task->GetKey());
        } else if (task->GetType() == TaskType::single) {
          task_list_.erase(task);
        }
        break;
      } else {
        ++task;
      }
    }
  }
  void DoAvailable() {
    for (auto &task : task_list_) {
      // first skip task if it's complete.  Possible for iterative tasks
      if (task.GetStatus() != TaskStatus::incomplete) continue;
      auto dep = task.GetDependency();
      if (CheckDependencies(dep)) {
        task();
        if (task.GetStatus() == TaskStatus::complete && !task.IsRegional()) {
          MarkTaskComplete(task.GetID());
        } else if (task.GetStatus() == TaskStatus::skip &&
                   task.GetType() == TaskType::completion_criteria) {
          ResetIteration(task.GetKey());
        }
      }
    }
    ClearComplete();
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

  // overload to add member functions of class T to task list
  // NOTE: we must capture the object pointer
  template <class T, class U, class... Args>
  TaskID AddTask(TaskID const &dep, TaskStatus (T::*func)(Args...), U *obj,
                 Args &&...args) {
    return this->AddTask(dep, [=]() mutable -> TaskStatus {
      return (obj->*func)(std::forward<Args>(args)...);
    });
  }

  template <class F, class... Args>
  TaskID AddTask(TaskID const &dep, F &&func, Args &&...args) {
    TaskID id(tasks_added_ + 1);
    task_list_.push_back(
        Task(id, dep, [=, func = std::forward<F>(func)]() mutable -> TaskStatus {
          return func(std::forward<Args>(args)...);
        }));
    tasks_added_++;
    return id;
  }

  IterativeTasks &AddIteration(const std::string &label) {
    int key = iter_tasks.size();
    iter_tasks.push_back(IterativeTasks(this, key));
    iter_labels.push_back(label);
    return iter_tasks.back();
  }

  void Print() {
    int i = 0;
    std::cout << "TaskList::Print():" << std::endl;
    for (auto &t : task_list_) {
      std::cout << "  " << i << "  " << t.GetID().to_string() << "  "
                << t.GetDependency().to_string() << " "
                << (t.GetStatus() == TaskStatus::incomplete)
                << (t.GetStatus() == TaskStatus::complete)
                << (t.GetStatus() == TaskStatus::skip)
                << (t.GetStatus() == TaskStatus::iterate)
                << (t.GetStatus() == TaskStatus::fail) << std::endl;

      i++;
    }
  }

 protected:
  std::vector<IterativeTasks> iter_tasks;
  std::vector<std::string> iter_labels;
  std::list<Task> task_list_;
  int tasks_added_ = 0;
  TaskID tasks_completed_;
};

namespace task_list_impl {
// helper function to avoid having to call a member function of TaskList from
// IterativeTasks before TaskList has been defined
inline TaskID AddTaskHelper(TaskList *tl, Task tsk) { return tl->AddTask(tsk); }
} // namespace task_list_impl

class TaskRegion {
 public:
  explicit TaskRegion(const int size) : lists(size) {}
  void AddRegionalDependencies(const int reg_dep_id, const int list_index, TaskID id) {
    auto task_pair = std::make_pair(list_index, id);
    id_for_reg[reg_dep_id].push_back(task_pair);
    lists[list_index].MarkRegional(id);
    all_done[reg_dep_id].val = 0;
  }

  TaskList &operator[](int i) { return lists[i]; }

  int size() const { return lists.size(); }

  bool CheckAndUpdate() {
    for (auto &reg_dep : id_for_reg) {
      auto reg_id = reg_dep.first;
      if (HasRun(reg_id) && !all_done[reg_id].active) {
        all_done[reg_id].val = IsComplete(reg_id);
        all_done[reg_id].StartReduce(MPI_MIN);
      }
      if (all_done[reg_id].active) {
        auto status = all_done[reg_id].CheckReduce();
        if (status == TaskStatus::complete) {
          if (all_done[reg_id].val) {
            for (auto &lst : reg_dep.second) {
              lists[lst.first].CompleteIfNeeded(lst.second);
            }
          } else {
            for (auto &lst : reg_dep.second) {
              lists[lst.first].ResetIfNeeded(lst.second);
            }
            all_done[reg_id].val = 0;
          }
        }
      }
    }
    int complete_cnt = 0;
    const int num_lists = size();
    for (auto i = 0; i < num_lists; ++i) {
      if (lists[i].IsComplete()) complete_cnt++;
    }
    return (complete_cnt == num_lists);
  }

  bool Validate() const {
    for (auto &list : lists) {
      if (!list.Validate()) return false;
    }
    return true;
  }

 private:
  bool HasRun(const int reg_id) {
    auto &lvec = id_for_reg[reg_id];
    int n_to_run = lvec.size();
    int n_ran = 0;
    for (auto &pair : lvec) {
      int list_index = pair.first;
      TaskID id = pair.second;
      if (lists[list_index].CheckTaskRan(id)) {
        n_ran++;
      }
    }
    return n_ran == n_to_run;
  }
  bool IsComplete(const int reg_id) {
    auto &lvec = id_for_reg[reg_id];
    int n_to_finish = lvec.size();
    int n_finished = 0;
    for (auto &pair : lvec) {
      int list_index = pair.first;
      TaskID id = pair.second;
      if (lists[list_index].CheckTaskCompletion(id)) {
        n_finished++;
      }
    }
    return n_finished == n_to_finish;
  }

  // id_for_reg[region_id] = std::pair<>(task_list_index, task_id_of_regional_task)
  std::map<int, std::vector<std::pair<int, TaskID>>> id_for_reg;
  std::vector<TaskList> lists;
  std::map<int, AllReduce<int>> all_done;
};

class TaskCollection {
 public:
  TaskCollection() = default;
  TaskRegion &AddRegion(const int num_lists) {
    regions.push_back(TaskRegion(num_lists));
    return regions.back();
  }
  TaskListStatus Execute() {
    assert(Validate());
    for (auto &region : regions) {
      bool complete = false;
      auto num_lists = region.size();
      while (!complete) {
        // TODO(pgrete): need to let Kokkos::PartitionManager handle this
        for (auto i = 0; i < num_lists; ++i) {
          if (!region[i].IsComplete()) {
            region[i].DoAvailable();
          }
        }
        complete = region.CheckAndUpdate();
      }
    }
    return TaskListStatus::complete;
  }

 private:
  bool Validate() const {
    for (auto &region : regions) {
      if (!region.Validate()) return false;
    }
    return true;
  }

  std::vector<TaskRegion> regions;
};

} // namespace parthenon

#endif // TASKS_TASK_LIST_HPP_
