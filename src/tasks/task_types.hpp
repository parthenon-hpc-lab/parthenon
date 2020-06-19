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
#include "interface/container.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

class Integrator;

using SimpleTaskFunc = std::function<TaskStatus()>;
using BlockTaskFunc = std::function<TaskStatus(MeshBlock *)>;
using BlockStageTaskFunc = std::function<TaskStatus(MeshBlock *, int)>;
using BlockStageNamesTaskFunc =
    std::function<TaskStatus(MeshBlock *, int, std::vector<std::string> &)>;
using BlockStageNamesIntegratorTaskFunc =
    std::function<TaskStatus(MeshBlock *, int, std::vector<std::string> &, Integrator *)>;
using ContainerTaskFunc = std::function<TaskStatus(Container<Real> &)>;
using TwoContainerTaskFunc =
    std::function<TaskStatus(Container<Real> &, Container<Real> &)>;

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

class ContainerTask : public BaseTask {
 public:
  ContainerTask(TaskID id, ContainerTaskFunc func, TaskID dep, Container<Real> &rc)
      : BaseTask(id, dep), _func(func), _cont(&rc) {}
  TaskStatus operator()() { return _func(*_cont); }

 private:
  ContainerTaskFunc _func;
  Container<Real> *_cont;
};

class TwoContainerTask : public BaseTask {
 public:
  TwoContainerTask(TaskID id, TwoContainerTaskFunc func, TaskID dep, Container<Real> &rc1,
                   Container<Real> &rc2)
      : BaseTask(id, dep), _func(func), _cont1(&rc1), _cont2(&rc2) {}
  TaskStatus operator()() { return _func(*_cont1, *_cont2); }

 private:
  TwoContainerTaskFunc _func;
  Container<Real> *_cont1;
  Container<Real> *_cont2;
};

} // namespace parthenon

#endif // TASKS_TASK_TYPES_HPP_