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

#ifndef TASK_LIST_TASKS_HPP_
#define TASK_LIST_TASKS_HPP_

#include <bitset>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"

namespace parthenon {

class MeshBlock;
struct Integrator;

enum class TaskListStatus { running, stuck, complete, nothing_to_do };

using SimpleTaskFunc = std::function<TaskStatus()>;
using BlockTaskFunc = std::function<TaskStatus(MeshBlock *)>;
using BlockStageTaskFunc = std::function<TaskStatus(MeshBlock *, int)>;
using BlockStageNamesTaskFunc =
    std::function<TaskStatus(MeshBlock *, int, std::vector<std::string> &)>;
using BlockStageNamesIntegratorTaskFunc =
    std::function<TaskStatus(MeshBlock *, int, std::vector<std::string> &, Integrator *)>;

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief generalization of bit fields for Task IDs, status, and dependencies.

#define BITBLOCK 16

class TaskID {
 public:
  TaskID() { Set(0); }
  explicit TaskID(int id);

  void Set(int id);
  void clear();
  bool CheckDependencies(const TaskID &rhs) const;
  void SetFinished(const TaskID &rhs);
  bool operator==(const TaskID &rhs) const;
  TaskID operator|(const TaskID &rhs) const;
  std::string to_string();

 private:
  std::vector<std::bitset<BITBLOCK>> bitblocks;
};

class BaseTask {
 public:
  BaseTask(TaskID id, TaskID dep) : myid_(id), dep_(dep) {}
  virtual ~BaseTask() = default;
  virtual TaskStatus operator()() = 0;
  TaskID GetID() { return myid_; }
  TaskID GetDependency() { return dep_; }
  void SetComplete() { complete_ = true; }
  bool IsComplete() { return complete_; }

 protected:
  TaskID myid_, dep_;
  bool lb_time, complete_ = false;
};

class SimpleTask : public BaseTask {
 public:
  SimpleTask(TaskID id, SimpleTaskFunc func, TaskID dep)
      : BaseTask(id, dep), func_(func) {}
  TaskStatus operator()() { return func_(); }

 private:
  SimpleTaskFunc func_;
};

class BlockTask : public BaseTask {
 public:
  BlockTask(TaskID id, BlockTaskFunc func, TaskID dep, MeshBlock *pmb)
      : BaseTask(id, dep), func_(func), pblock_(pmb) {}
  TaskStatus operator()() { return func_(pblock_); }

 private:
  BlockTaskFunc func_;
  MeshBlock *pblock_;
};

class BlockStageTask : public BaseTask {
 public:
  BlockStageTask(TaskID id, BlockStageTaskFunc func, TaskID dep, MeshBlock *pmb,
                 int stage)
      : BaseTask(id, dep), func_(func), pblock_(pmb), stage_(stage) {}
  TaskStatus operator()() { return func_(pblock_, stage_); }

 private:
  BlockStageTaskFunc func_;
  MeshBlock *pblock_;
  int stage_;
};

class BlockStageNamesTask : public BaseTask {
 public:
  BlockStageNamesTask(TaskID id, BlockStageNamesTaskFunc func, TaskID dep, MeshBlock *pmb,
                      int stage, const std::vector<std::string> &sname)
      : BaseTask(id, dep), func_(func), pblock_(pmb), stage_(stage), sname_(sname) {}
  TaskStatus operator()() { return func_(pblock_, stage_, sname_); }

 private:
  BlockStageNamesTaskFunc func_;
  MeshBlock *pblock_;
  int stage_;
  std::vector<std::string> sname_;
};

class BlockStageNamesIntegratorTask : public BaseTask {
 public:
  BlockStageNamesIntegratorTask(TaskID id, BlockStageNamesIntegratorTaskFunc func,
                                TaskID dep, MeshBlock *pmb, int stage,
                                const std::vector<std::string> &sname, Integrator *integ)
      : BaseTask(id, dep), func_(func), pblock_(pmb), stage_(stage), sname_(sname),
        int_(integ) {}
  TaskStatus operator()() { return func_(pblock_, stage_, sname_, int_); }

 private:
  BlockStageNamesIntegratorTaskFunc func_;
  MeshBlock *pblock_;
  int stage_;
  std::vector<std::string> sname_;
  Integrator *int_;
};

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

#endif // TASK_LIST_TASKS_HPP_
