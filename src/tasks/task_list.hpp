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
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "task_id.hpp"
#include "task_types.hpp"

namespace parthenon {

enum class TaskListStatus { running, stuck, complete, nothing_to_do };

class TaskList {
 public:
  bool IsComplete() { return task_list_.empty(); }
  int Size() { return task_list_.size(); }
  void Reset() {
    tasks_added_ = 0;
    task_list_.clear();
    dependencies_.clear();
    tasks_completed_.clear();
  }
  bool IsReady() {
    for (auto &l : dependencies_) {
      if (!l->IsComplete()) {
        return false;
      }
    }
    return true;
  }
  void MarkTaskComplete(TaskID id) { tasks_completed_.SetFinished(id); }
  void ClearComplete() {
    auto task = task_list_.begin();
    while (task != task_list_.end()) {
      if ((*task)->IsComplete()) {
        task = task_list_.erase(task);
      } else {
        ++task;
      }
    }
  }
  TaskListStatus DoAvailable() {
    for (auto &task : task_list_) {
      auto dep = task->GetDependency();
      if (tasks_completed_.CheckDependencies(dep)) {
        /*std::cerr << "Task dependency met:" << std::endl
                  << dep.to_string() << std::endl
                  << tasks_completed_.to_string() << std::endl
                  << task->GetID().to_string() << std::endl << std::endl;*/
        TaskStatus status = (*task)();
        if (status == TaskStatus::complete) {
          task->SetComplete();
          MarkTaskComplete(task->GetID());
          /*std::cerr << "Task complete:" << std::endl
                    << task->GetID().to_string() << std::endl
                    << tasks_completed_.to_string() << std::endl << std::endl;*/
        }
      }
    }
    ClearComplete();
    if (IsComplete()) return TaskListStatus::complete;
    return TaskListStatus::running;
  }

//R( T::*pmf )( ARGS... )

  template <class... Args>
  TaskID NewAddTask(TaskStatus(&func)(Args...), TaskID &dep, Args &&... args) {
    TaskID id(tasks_added_ + 1);
    task_list_.push_back(std::make_unique<Task<Args...>>(id, dep, func, std::forward<Args>(args)...));
    tasks_added_++;
    return id;
  }
  template <typename T, class... Args>
  TaskID AddTask(Args &&... args) {
    TaskID id(tasks_added_ + 1);
    task_list_.push_back(std::make_unique<T>(id, std::forward<Args>(args)...));
    tasks_added_++;
    return id;
  }
  void Print() {
    int i = 0;
    std::cout << "TaskList::Print():" << std::endl;
    for (auto &t : task_list_) {
      std::cout << "  " << i << "  " << t->GetID().to_string() << "  "
                << t->GetDependency().to_string() << std::endl;
      i++;
    }
  }

 protected:
  std::list<std::unique_ptr<BaseTask>> task_list_;
  int tasks_added_ = 0;
  std::vector<TaskList *> dependencies_;
  TaskID tasks_completed_;
};

} // namespace parthenon

#endif // TASK_TASK_LIST_HPP_
