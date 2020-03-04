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
#ifndef TASK_LIST_TASK_LIST_HPP_
#define TASK_LIST_TASK_LIST_HPP_
//!   \file task_list.hpp
//    \brief provides functionality to control dynamic execution using tasks

// C headers

// C++ headers
#include <bitset>
#include <list>
#include <cstdint>      // std::uint64_t
#include <string>       // std::string

// Athena++ headers
#include "athena.hpp"

#define MAX_TASKS 64

namespace parthenon {
// forward declarations
class Mesh;
class MeshBlock;
class TaskList;
class TaskID;

// TODO(felker): these 4x declarations can be nested in TaskList if MGTaskList is derived

// constants = return codes for functions working on individual Tasks and TaskList
enum class TaskStatus {fail, success, next};
enum class TaskListStatus {running, stuck, complete, nothing_to_do};

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief generalization of bit fields for Task IDs, status, and dependencies.

class TaskID {  // POD but not aggregate (there is a user-provided ctor)
 public:
  TaskID() = default;
  explicit TaskID(unsigned int id);
  void Clear();
  bool IsUnfinished(const TaskID& id) const;
  bool CheckDependencies(const TaskID& dep) const;
  void SetFinished(const TaskID& id);

  bool operator== (const TaskID& rhs) const;
  TaskID operator| (const TaskID& rhs) const;

  void Print(const std::string label = "");

 private:
  std::bitset<MAX_TASKS> bitfld_;

  friend class TaskList;
};


//----------------------------------------------------------------------------------------
//! \struct Task
//  \brief data and function pointer for an individual Task

struct Task { // aggregate and POD
  TaskID task_id;    // encodes task with bit positions in HydroIntegratorTaskNames
  TaskID dependency; // encodes dependencies to other tasks using " " " "
  TaskStatus (*TaskFunc)(MeshBlock*, int);  // ptr to member function
  bool lb_time; // flag for automatic load balancing based on timing
};

struct BetterTask {
  BetterTask() :
      task(nullptr),
      task_block(nullptr)
  {};

  BetterTask(void (*task_func)(MeshBlock*,int),
             MeshBlock *pmb,
             std::bitset<MAX_TASKS> dep) :
      task(task_func),
      task_block(pmb),
      dependency(dep) { };
  void (*task)(MeshBlock *pmb, int stage);
  void  DoTaskList(const int stage);
  MeshBlock *task_block;
  uint32_t id;
  std::bitset<MAX_TASKS> dependency;
};

//---------------------------------------------------------------------------------------
//! \struct TaskStates
//  \brief container for task states on a single MeshBlock

struct TaskStates { // aggregate and POD
  TaskID finished_tasks;
  int indx_first_task, num_tasks_left;
  void Reset(int ntasks) {
    indx_first_task = 0;
    num_tasks_left = ntasks;
    finished_tasks.Clear();
  }
};

//----------------------------------------------------------------------------------------
//! \class TaskList
//  \brief data and function definitions for task list base class

class TaskList {
 public:
  TaskList() : ntasks(0), task_list_{} {} // 2x direct + zero initialization
  // rule of five:
  virtual ~TaskList() = default;

  // data
  int ntasks;     // number of tasks in this list

  // functions
  TaskListStatus DoAllAvailableTasks(MeshBlock *pmb, int stage, TaskStates &ts);
  void DoTaskListOneStage(Mesh *pmesh, int stage);
  void DoBetterTaskList(const int stage);


  void IncrementTaskCount(int delta=1) {
    _task_count ++;
  }

  int TaskCount() { return _task_count;}
 protected:
  // TODO(felker): rename to avoid confusion with class name
  Task task_list_[MAX_TASKS];
  int _task_count;
  std::list<BetterTask> _task;
  std::bitset<MAX_TASKS> _null_task;
  std::bitset<MAX_TASKS> _running;
  std::bitset<MAX_TASKS> _complete;

 private:
  virtual TaskID AddTask(TaskStatus (*)(MeshBlock *pmb, int stage), const TaskID& dep, const bool load_balance_timer = false) = 0;
  virtual void StartupTaskList(MeshBlock *pmb, int stage) = 0;
};
}
#endif  // TASK_LIST_TASK_LIST_HPP_
