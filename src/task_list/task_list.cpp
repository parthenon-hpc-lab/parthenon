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
//! \file task_list.cpp
//  \brief functions for TaskList base class


// C headers

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "task_list.hpp"

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn TaskListStatus TaskList::DoAllAvailableTasks
//  \brief do all tasks that can be done (are not waiting for a dependency to be
//  cleared) in this TaskList, return status.

TaskListStatus TaskList::DoAllAvailableTasks(MeshBlock *pmb, int stage, TaskStates &ts) {
  int skip = 0;
  TaskStatus ret;
  if (ts.num_tasks_left == 0) return TaskListStatus::nothing_to_do;

  for (int i=ts.indx_first_task; i<ntasks; i++) {
    Task &taski = task_list_[i];
    if (ts.finished_tasks.IsUnfinished(taski.task_id)) { // task not done
      // check if dependency clear
      if (ts.finished_tasks.CheckDependencies(taski.dependency)) {
        if (taski.lb_time) pmb->StartTimeMeasurement();
        ret = task_list_[i].TaskFunc(pmb, stage);
        if (taski.lb_time) pmb->StopTimeMeasurement();
        if (ret != TaskStatus::fail) { // success
          ts.num_tasks_left--;
          ts.finished_tasks.SetFinished(taski.task_id);
          if (skip == 0) ts.indx_first_task++;
          if (ts.num_tasks_left == 0) return TaskListStatus::complete;
          if (ret == TaskStatus::next) continue;
          return TaskListStatus::running;
        }
      }
      skip++; // increment number of tasks processed

    } else if (skip == 0) { // this task is already done AND it is at the top of the list
      ts.indx_first_task++;
    }
  }
  // there are still tasks to do but nothing can be done now
  return TaskListStatus::stuck;
}
#if 0
void TaskList::DoBetterTaskList(const int stage)
{

  _running.reset();
  _complete.reset();

  #pragma omp parallel shared(_running, _complete)
  #pragma omp single
  {
    for(auto it = _task.begin(); it != _task.end(); ++it) {
      if ((!(_running.test(it->id) || _complete.test(it->id))) &&
	  (it->dependency.to_ulong()&_complete.to_ulong()==it->dependency.to_ulong())
	  ) {
        _running.set(it->id);
        #pragma omp task firstprivate(it)
        {
          it->task(it->task_block, stage);
          #pragma omp atomic update
          _complete.set(it->id);
        }
      }
    }
  }
}
#endif
//----------------------------------------------------------------------------------------
//! \fn void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage)
//  \brief completes all tasks in this list, will not return until all are tasks done

void TaskList::DoTaskListOneStage(Mesh *pmesh, int stage) {
  int nthreads = pmesh->GetNumMeshThreads();
  int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);

  // Not sure what you're trying to do here
  // are you tring to insert all the tasks into the tasks for htis block?
  // is this to accommodate variable tasks per block?
#if 0 // SS commented out
  MeshBlock *pmb = pmesh->pblock;
  while (pmb != nullptr) {
    _task.splice(_task.begin(), pmb->task_list.GetTasks());
    pmb = pmb->next;
  }

  DoTaskList(stage);
#endif // SS commented out

  // construct the MeshBlock array on this process
  MeshBlock **pmb_array = new MeshBlock*[nmb];
  MeshBlock *pmb = pmesh->pblock;
  for (int n=0; n < nmb; ++n) {
    pmb_array[n] = pmb;
    pmb = pmb->next;
  }

  // clear the task states, startup the integrator and initialize mpi calls
#pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
  for (int i=0; i<nmb; ++i) {
    pmb_array[i]->tasks.Reset(ntasks);
    StartupTaskList(pmb_array[i], stage);
  }

  int nmb_left = nmb;
  // cycle through all MeshBlocks and perform all tasks possible
  while (nmb_left > 0) {
    // KNOWN ISSUE: Workaround for unknown OpenMP race condition. See #183 on GitHub.
#pragma omp parallel for reduction(- : nmb_left) num_threads(nthreads) schedule(dynamic,1)
    for (int i=0; i<nmb; ++i) {
      if (DoAllAvailableTasks(pmb_array[i],stage,pmb_array[i]->tasks)
          == TaskListStatus::complete) {
        nmb_left--;
      }
    }
  }
  delete [] pmb_array;
  return;
}
