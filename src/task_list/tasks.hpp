#ifndef BETTER_TASK_HPP
#define BETTER_TASK_HPP

#include <list>
#include <vector>
#include <bitset>
#include <string>
#include <memory>
#include <iostream>
#include "mesh/mesh.hpp"
//#include "driver/multistage.hpp"

class Integrator;

#define MAX_TASKS 64

enum class TaskStatus {fail, success, next};
enum class TaskListStatus {running, stuck, complete, nothing_to_do};

using BaseTaskFunc = TaskStatus ();
using BlockTaskFunc = TaskStatus (MeshBlock*);
using BlockStageTaskFunc = TaskStatus (MeshBlock*, int);
using BlockStageNamesTaskFunc = TaskStatus (MeshBlock*, int, std::vector<std::string>&);
using BlockStageNamesIntegratorTaskFunc = TaskStatus (MeshBlock*, int, std::vector<std::string>&, Integrator*);

//----------------------------------------------------------------------------------------
//! \class TaskID
//  \brief generalization of bit fields for Task IDs, status, and dependencies.

class TaskID {  // POD but not aggregate (there is a user-provided ctor)
 public:
  TaskID() = default;
  explicit TaskID(unsigned int id);

  void clear();
  bool IsUnfinished(const TaskID& id) const;
  bool CheckDependencies(const TaskID& dep) const;
  void SetFinished(const TaskID& id);
  std::bitset<MAX_TASKS> GetBitFld() const {return bitfld_;}

  bool operator== (const TaskID& rhs) const;
  TaskID operator| (const TaskID& rhs) const;

  void Print(const std::string label = "");

 private:
  std::bitset<MAX_TASKS> bitfld_;
};

class BaseTask {
  public:
    BaseTask() {}
    BaseTask(BaseTaskFunc* func) : _func(func) {}
    BaseTask(BaseTaskFunc* func, TaskID id, TaskID dep) : _func(func), _myid(id), _dep(dep) {}
    virtual TaskStatus operator () () {return _func();}
    TaskID GetID() { return _myid; }
    TaskID GetDependency() { return _dep; }
    void SetComplete() { _complete = true; }
    bool IsComplete() { return _complete; }
  protected:
    TaskID _myid, _dep;
    bool lb_time, _complete=false;
  private:
    BaseTaskFunc* _func;
};

class BlockTask : public BaseTask {
  public:
    BlockTask(BlockTaskFunc* func, TaskID id, TaskID dep, MeshBlock *pmb) 
        : _func(func), _pblock(pmb) { _myid=id; _dep=dep; }
    TaskStatus operator () () { return _func(_pblock); }
  private:
    BlockTaskFunc* _func;
    MeshBlock *_pblock;
};

class BlockStageTask : public BaseTask {
  public:
    BlockStageTask(BlockStageTaskFunc* func, TaskID id, TaskID dep, MeshBlock *pmb, int stage) 
        : _func(func), _pblock(pmb), _stage(stage) { _myid=id; _dep=dep; }
    TaskStatus operator () () { return _func(_pblock, _stage); }
  private:
    BlockStageTaskFunc* _func;
    MeshBlock *_pblock;
    int _stage;
};

class BlockStageNamesTask : public BaseTask {
  public:
    BlockStageNamesTask(BlockStageNamesTaskFunc* func, TaskID id, TaskID dep, MeshBlock *pmb, int stage, const std::vector<std::string>& sname) 
        : _func(func), _pblock(pmb), _stage(stage), _sname(sname) { _myid=id; _dep=dep; }
    TaskStatus operator () () { return _func(_pblock, _stage, _sname); }
  private:
    BlockStageNamesTaskFunc* _func;
    MeshBlock *_pblock;
    int _stage;
    std::vector<std::string> _sname;
};

class BlockStageNamesIntegratorTask : public BaseTask {
  public:
    BlockStageNamesIntegratorTask(BlockStageNamesIntegratorTaskFunc* func, TaskID id, TaskID dep, MeshBlock *pmb, int stage, const std::vector<std::string>& sname, Integrator* integ) 
        : _func(func), _pblock(pmb), _stage(stage), _sname(sname), _int(integ) { _myid=id; _dep=dep; }
    TaskStatus operator () () { return _func(_pblock, _stage, _sname, _int); }
  private:
    BlockStageNamesIntegratorTaskFunc* _func;
    MeshBlock *_pblock;
    int _stage;
    std::vector<std::string> _sname;
    Integrator *_int;
};

namespace TaskFactory {
  std::unique_ptr<BaseTask> NewTask(TaskID id, BaseTaskFunc* func, TaskID dep);
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockTaskFunc* func, TaskID dep, MeshBlock *pmb);
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageTaskFunc* func, TaskID dep, MeshBlock *pmb, int stage);
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageNamesTaskFunc* func, TaskID dep, MeshBlock *pmb, int stage, const std::vector<std::string>& sname);
  std::unique_ptr<BaseTask> NewTask(TaskID id, BlockStageNamesIntegratorTaskFunc* func, TaskID dep, MeshBlock *pmb, int stage, const std::vector<std::string>& sname, Integrator *integ);
};

class TaskList {
  public:
    bool IsComplete() { return _task_list.empty(); }
    int Size() { return _task_list.size(); }
    void Reset() { 
      _tasks_added = 0;
      _task_list.clear();
      _dependencies.clear();
      _tasks_completed.clear();
    }
    bool IsReady() {
      for (auto & l : _dependencies) {
        if (!l->IsComplete()) {
          return false;
        }
      }
      return true;
    }
    void MarkTaskComplete(TaskID id) { _tasks_completed.SetFinished(id); }
    int ClearComplete() {
      auto task = _task_list.begin();
      int completed = 0;
      while (task != _task_list.end()) {
        if ((*task)->IsComplete()) {
          task = _task_list.erase(task);
          completed++;
        } else {
          ++task;
        }
      }
      return completed;
    }
    TaskListStatus DoAvailable() {
      for (auto & task : _task_list) {
        auto dep = task->GetDependency();
        if(_tasks_completed.CheckDependencies(dep)) {
          TaskStatus status = (*task)();
          if (status == TaskStatus::success) {
            task->SetComplete();
            MarkTaskComplete(task->GetID());
          } 
        }
      }
      int completed = ClearComplete();
      if (IsComplete()) return TaskListStatus::complete;
      return TaskListStatus::running;
    }
    template<class...Args>
    TaskID AddTask(Args... args) {
      if (_tasks_added == MAX_TASKS) {
        // Do error checking
      }
      TaskID id(_tasks_added+1);
      _task_list.push_back(
        TaskFactory::NewTask(id, std::forward<Args>(args)...)
      );
      _tasks_added++;
      return id;
    }
    void Print() {
      int i = 0;
      std::cout << "TaskList::Print():" << std::endl;
      for (auto& t : _task_list) {
        std::cout << "  " << i << "  " << t->GetID().GetBitFld().to_string() << "  " << t->GetDependency().GetBitFld().to_string() << std::endl;
        i++;
      }
    }

  protected:
    std::list<std::unique_ptr<BaseTask>> _task_list;
    int _tasks_added=0;
    std::vector<TaskList*> _dependencies;
    TaskID _tasks_completed;
};

#endif