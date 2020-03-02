
#include "tasks.hpp"

namespace TaskFactory {
  std::unique_ptr<BaseTask> NewTask(TaskID id, BaseTaskFunc* func, TaskID dep) { 
    return std::unique_ptr<BaseTask>(new BaseTask(func, id, dep)); 
  }
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockTaskFunc* func, TaskID dep, MeshBlock *pmb) {
    return std::unique_ptr<BlockTask>(new BlockTask(func, id, dep, pmb));
  }
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageTaskFunc* func, TaskID dep, MeshBlock *pmb, int stage) {
    return std::unique_ptr<BlockStageTask>(new BlockStageTask(func, id, dep, pmb, stage));
  }
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageNamesTaskFunc* func, TaskID dep, MeshBlock *pmb, int stage, 
                                    const std::vector<std::string>& sname) {
    return std::unique_ptr<BlockStageNamesTask>(new BlockStageNamesTask(func, id, dep, pmb, stage, sname));
  }
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageNamesIntegratorTaskFunc* func, TaskID dep, MeshBlock *pmb, int stage, 
                                    const std::vector<std::string>& sname, Integrator *integ) {
    return std::unique_ptr<BlockStageNamesIntegratorTask>(new BlockStageNamesIntegratorTask(func, id, dep, pmb, stage, sname, integ));
  }
};