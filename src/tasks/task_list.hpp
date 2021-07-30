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
#include <chrono>
using namespace std::chrono;

#include "basic_types.hpp"
#include "globals.hpp"
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
    return AddTask_(TaskType::iterative, 1, dep, std::forward<T>(func),
                    std::forward<Args>(args)...);
  }
  // overload to add member functions of class T to task list
  // NOTE: we must capture the object pointer
  template <class T, class... Args>
  TaskID AddTask(TaskID const &dep, TaskStatus (T::*func)(Args...), T *obj,
                 Args &&... args) {
    return this->AddTask_(TaskType::iterative, 1, dep, [=]() mutable -> TaskStatus {
      return (obj->*func)(std::forward<Args>(args)...);
    });
  }

  template <class T, class... Args>
  TaskID SetCompletionTask(TaskID const &dep, T &&func, Args &&... args) {
    return AddTask_(TaskType::completion_criteria, check_interval_, dep,
                    std::forward<T>(func), std::forward<Args>(args)...);
  }
  template <class T, class... Args>
  TaskID SetCompletionTask(TaskID const &dep, TaskStatus (T::*func)(Args...), T *obj,
                           Args &&... args) {
    return AddTask_(TaskType::completion_criteria, check_interval_, dep,
                    [=]() mutable -> TaskStatus {
                      return (obj->*func)(std::forward<Args>(args)...);
                    });
  }

  void SetMaxIterations(const unsigned int max) { max_iterations_ = max; }
  void SetCheckInterval(const unsigned int chk) { check_interval_ = chk; }
  void SetFailWithMaxIterations(const bool flag) { throw_with_max_iters_ = flag; }
  void SetWarnWithMaxIterations(const bool flag) { warn_with_max_iters_ = flag; }
  bool ShouldThrowWithMax() const { return throw_with_max_iters_; }
  bool ShouldWarnWithMax() const { return warn_with_max_iters_; }
  unsigned int GetMaxIterations() const { return max_iterations_; }
  unsigned int GetIterationCount() const { return count_; }
  unsigned int GetCheckInterval() const { return check_interval_; }
  void IncrementCount() { count_++; }
  bool Locked() const { return locked_; }
  void Lock() { locked_ = true; }
  void Unlock() { locked_ = false; }

 private:
  template <class F, class... Args>
  TaskID AddTask_(const TaskType &type, const int interval, TaskID const &dep, F &&func,
                  Args &&... args) {
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
  unsigned int max_iterations_;
  unsigned int count_ = 0;
  unsigned int check_interval_ = 1;
  bool throw_with_max_iters_ = false;
  bool warn_with_max_iters_ = true;
  bool locked_ = false;
};

class TaskList {
 public:
  TaskList() = default;
  bool IsComplete() { return task_list_.empty(); }
  int Size() { return task_list_.size(); }
  void MarkRegional(TaskID id) {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        task.SetRegional();
        break;
      }
    }
  }
  void MarkTaskComplete(TaskID id) { tasks_completed_.SetFinished(id); }
  bool CheckDependencies(TaskID id) const {
    return tasks_completed_.CheckDependencies(id);
  }
  bool CheckTaskRan(TaskID id) const {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        return (task.GetStatus() != TaskStatus::incomplete &&
                task.GetStatus() != TaskStatus::skip);
      }
    }
    return true;
  }
  bool CheckStatus(TaskID id, TaskStatus status) const {
    for (auto &task : task_list_) {
      if (task.GetID() == id) return (task.GetStatus() == status);
    }
    std::cout << "Aghhhh!!!" << std::endl;
    return true;
  }
  bool CheckTaskCompletion(TaskID id) const {
    return CheckStatus(id, TaskStatus::complete);
  }
  void ClearComplete() {
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if (task->GetStatus() == TaskStatus::complete &&
          task->GetType() != TaskType::iterative &&
          task->GetType() != TaskType::completion_criteria) {
        task = task_list_.erase(task);
      } else {
        ++task;
      }
    }
    std::set<int> completed_iters;
    for (auto &tsk : task_list_) {
      if (tsk.GetType() == TaskType::completion_criteria &&
          tsk.GetStatus() == TaskStatus::complete &&
          !tsk.IsRegional()) {
        completed_iters.insert(tsk.GetKey());
      }
    }
    for (const auto &key : completed_iters) {
      ClearIteration(key);
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
    iter_tasks[key].Unlock();
    iter_tasks[key].IncrementCount();
    if (iter_tasks[key].GetIterationCount() % 100 == 0) std::cout << Globals::my_rank << "  iter count = " << iter_tasks[key].GetIterationCount() << std::endl;
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
        if (CheckDependencies(task.GetID())) {
          MarkTaskComplete(task.GetID());
        } else if (task.GetType() != TaskType::completion_criteria) {
          std::cout << "What the actual f*ck" << std::endl;
        }
        task.SetStatus(TaskStatus::incomplete);
      }
    }
  }
  void ResetIfNeeded(TaskID id) {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        auto status = task.GetStatus();
        if (task.GetType() == TaskType::completion_criteria &&
            status == TaskStatus::skip) {
          std::cout << Globals::my_rank << " resetting " << iter_tasks[task.GetKey()].GetIterationCount() << std::endl;
          ResetIteration(task.GetKey());
        } else {
          std::cout << "Should not be here..." << std::endl;
        }
        break;
      }
    }
  }
  void CompleteIfNeeded(TaskID id) {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        ClearIteration(task.GetKey());
        break;
      }
    }
  }
  void ResetFromID(TaskID id) {
    for (auto &task : task_list_) {
      if (task.GetID() == id) {
        //std::cout << Globals::my_rank << " force resetting " << iter_tasks[task.GetKey()].GetIterationCount() << std::endl;
        ResetIteration(task.GetKey());
        break;
      }
    }
  }
  TaskListStatus DoAvailable() {
    for (auto &task : task_list_) {
      // first skip task if it's complete.  Only possible for TaskType::iterative
      if (task.GetStatus() != TaskStatus::incomplete) continue;
      auto dep = task.GetDependency();
      if (CheckDependencies(dep)) {
        task();
        if (task.GetStatus() == TaskStatus::complete) {
          MarkTaskComplete(task.GetID());
        }
      }
    }
    ClearComplete();
    //if (IsComplete()) return TaskListStatus::complete;
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
  std::list<Task> task_list_;
  int tasks_added_ = 0;
  TaskID tasks_completed_;
  std::set<int> completed_iters_;
};

namespace task_list_impl {
// helper function to avoid having to call a member function of TaskList from
// IterativeTasks before TaskList has been defined
inline TaskID AddTaskHelper(TaskList *tl, Task tsk) { return tl->AddTask(tsk); }
} // namespace task_list_impl

//using TaskRegion = std::vector<TaskList>;

struct TaskRegion {
  TaskRegion(const int size) : lists(size) {}
  void AddRegionalDependencies(const int reg_dep_id, const int list_index, TaskID id) {
    auto task_pair = std::make_pair(list_index, id);
    id_for_reg[reg_dep_id].push_back(task_pair);
    lists[list_index].MarkRegional(id);
  }
  TaskList & operator[](int i) {
    return lists[i];
  }
  int size() const { return lists.size(); }

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
    //std::cout << n_ran << "/" << n_to_run << std::endl;
    return n_ran == n_to_run;
  }
  bool Skip(const int reg_id) {
    auto &lvec = id_for_reg[reg_id];
    int n_to_run = lvec.size();
    int n_ran = 0;
    for (auto &pair : lvec) {
      int list_index = pair.first;
      TaskID id = pair.second;
      if (lists[list_index].CheckStatus(id, TaskStatus::skip)) {
        n_ran++;
      }
    }
    //std::cout << n_ran << "/" << n_to_run << std::endl;
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

  std::map<int, std::vector<std::pair<int, TaskID>>> id_for_reg;
  std::vector<TaskList> lists;
};

struct TaskCollection {
  TaskCollection() = default;
  TaskRegion &AddRegion(const int num_lists) {
    regions.push_back(TaskRegion(num_lists));
    return regions.back();
  }
  TaskListStatus Execute() {
    assert(Validate());
    auto time = high_resolution_clock::now();
    auto total_time = duration_cast<microseconds>(time-time);
    for (auto &region : regions) {
      int complete_cnt = 0;
      auto num_lists = region.size();
      int cycle = 0;
      while (complete_cnt != num_lists) {
        // TODO(pgrete): need to let Kokkos::PartitionManager handle this
        for (auto i = 0; i < num_lists; ++i) {
          if (!region[i].IsComplete()) {
            //std::cout << Globals::my_rank << " " << i << std::endl;
            auto status = region[i].DoAvailable();
          }
        }
        /*std::string line;
        std::getline(std::cin, line);
        if (line == "p") {
          for (int i = 0; i < num_lists; ++i) {
            region[i].Print();
          }
        }*/
        for (auto &reg : region.id_for_reg) {
          auto reg_id = reg.first;
          if (region.HasRun(reg_id)) {
            //std::cout << "if  HasRun cycle " << cycle << std::endl;
            bool done = region.IsComplete(reg_id);
#ifdef MPI_PARALLEL
            int all_done = done;
            int global_done;
            auto start = high_resolution_clock::now();
            MPI_Allreduce(&all_done, &global_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            auto stop = high_resolution_clock::now();
            total_time += duration_cast<microseconds>(stop-start);
            //std::cout << n_finished << "/" << n_to_finish << "   " << global_done << std::endl;
#else
            int global_done = done;
#endif

            //std::cout << Globals::my_rank << " global_done = " << global_done << std::endl;
            if (global_done) {
              for (auto &lst : reg.second) {
                region.lists[lst.first].CompleteIfNeeded(lst.second);
              }
              std::cout << "time in MPI_Allreduce was " << total_time.count() << std::endl;
            } else {
              //std::cout << "Resetting..." << std::endl;
              for (auto &lst : reg.second) {
                region.lists[lst.first].ResetFromID(lst.second);
              }
            }
          } else if (region.Skip(reg_id)) {
            //std::cout << "else HasRun cycle " << cycle << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
            for (auto &lst : reg.second) {
              region.lists[lst.first].ResetIfNeeded(lst.second);
            }
          }
        }
        cycle++;
        complete_cnt = 0;
        for (auto i = 0; i < num_lists; ++i) {
          if (region[i].IsComplete()) complete_cnt++;
        }
      }
    }
    return TaskListStatus::complete;
  }
  bool Validate() const {
    for (auto &region : regions) {
      for (auto &list : region.lists) {
        if (!list.Validate()) return false;
      }
    }
    return true;
  }

  std::vector<TaskRegion> regions;
};

} // namespace parthenon

#endif // TASKS_TASK_LIST_HPP_
