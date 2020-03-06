
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

#include "tasks.hpp"

namespace parthenon {

namespace TaskFactory {

std::unique_ptr<BaseTask> NewTask(TaskID id, BaseTaskFunc* func, TaskID dep) {
  return std::unique_ptr<BaseTask>(new BaseTask(func, id, dep));
}

std::unique_ptr<BaseTask> NewTask(TaskID id, BlockTaskFunc* func, TaskID dep, MeshBlock *pmb) {
  return std::unique_ptr<BlockTask>(new BlockTask(func, id, dep, pmb));
}

std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageTaskFunc* func, TaskID dep, MeshBlock *pmb,
    int stage) {
  return std::unique_ptr<BlockStageTask>(new BlockStageTask(func, id, dep, pmb, stage));
}

std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageNamesTaskFunc* func, TaskID dep,
    MeshBlock *pmb, int stage, const std::vector<std::string>& sname) {
  return std::unique_ptr<BlockStageNamesTask>(
    new BlockStageNamesTask(func, id, dep, pmb, stage, sname));
}

std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageNamesIntegratorTaskFunc* func, TaskID dep,
    MeshBlock *pmb, int stage, const std::vector<std::string>& sname, Integrator *integ) {
  return std::unique_ptr<BlockStageNamesIntegratorTask>(
    new BlockStageNamesIntegratorTask(func, id, dep, pmb, stage, sname, integ));
}

} // namespace TaskFactory
} // namespace parthenon