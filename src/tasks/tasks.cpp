//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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

#include <cstdio>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#if __has_include(<cxxabi.h>)
#include <cxxabi.h> //NOLINT
#define HAS_CXX_ABI
#endif

#include "tasks.hpp"
#include "thread_pool.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
TaskID TaskID::operator|(const TaskID &other) const {
  // calling this operator means you're building a TaskID to hold a dependency
  TaskID result;
  if (task != nullptr)
    result.dep.push_back(task);
  else
    result.dep.insert(result.dep.end(), dep.begin(), dep.end());
  if (other.task != nullptr)
    result.dep.push_back(other.task);
  else
    result.dep.insert(result.dep.end(), other.dep.begin(), other.dep.end());
  return result;
}

TaskStatus Task::operator()() {
  auto status = f();
  if (verbose_level_ > 0)
    printf("%s [status = %i, rank = %i]\n", label_.c_str(), static_cast<int>(status),
           Globals::my_rank);
  if (task_type == TaskType::completion) {
    // keep track of how many times it's been called
    num_calls += (status == TaskStatus::iterate || status == TaskStatus::complete);
    // enforce minimum number of iterations
    if (num_calls < exec_limits.first && status == TaskStatus::complete)
      status = TaskStatus::iterate;
    // enforce maximum number of iterations
    if (num_calls == exec_limits.second) status = TaskStatus::complete;
  }
  // save the status in the Task object
  SetStatus(status);
  return status;
}

bool Task::ready() {
  // check that no dependency is incomplete
  bool go = true;
  for (auto &dep : dependencies) {
    go = go && (dep->GetStatus() != TaskStatus::incomplete);
  }
  return go;
}

std::ostream &WriteTaskGraph(std::ostream &stream,
                             const std::vector<std::shared_ptr<Task>> &tasks) {
#ifndef HAS_CXX_ABI
  std::cout << "Warning: task graph output will not include function"
               "signatures since libcxxabi is unavailable.\n";
#endif
  std::vector<std::pair<std::regex, std::string>> replacements;
  replacements.emplace_back("parthenon::", "");
  replacements.emplace_back("std::", "");
  replacements.emplace_back("MeshData<[^>]*>", "MD");
  replacements.emplace_back("MeshBlockData<[^>]*>", "MBD");
  replacements.emplace_back("shared_ptr", "sptr");
  replacements.emplace_back("TaskStatus ", "");
  replacements.emplace_back("BoundaryType::", "");

  stream << "digraph {\n";
  stream << "node [fontname=\"Helvetica,Arial,sans-serif\"]\n";
  stream << "edge [fontname=\"Helvetica,Arial,sans-serif\"]\n";
  constexpr int kBufSize = 1024;
  char buf[kBufSize];
  for (auto &ptask : tasks) {
    std::string cleaned_label = ptask->GetLabel();
    for (auto &[re, str] : replacements)
      cleaned_label = std::regex_replace(cleaned_label, re, str);
    snprintf(buf, kBufSize, " n%p [label=\"%s\"];\n", ptask->GetID().GetTask(),
             cleaned_label.c_str());
    stream << std::string(buf);
  }
  for (auto &ptask : tasks) {
    for (auto &pdtask : ptask->GetDependent(TaskStatus::complete)) {
      snprintf(buf, kBufSize, " n%p -> n%p [style=\"solid\"];\n",
               ptask->GetID().GetTask(), pdtask->GetID().GetTask());
      stream << std::string(buf);
    }
  }
  for (auto &ptask : tasks) {
    for (auto &pdtask : ptask->GetDependent(TaskStatus::iterate)) {
      snprintf(buf, kBufSize, " n%p -> n%p [style=\"dashed\"];\n",
               ptask->GetID().GetTask(), pdtask->GetID().GetTask());
      stream << std::string(buf);
    }
  }
  stream << "}\n";
  return stream;
}

TaskListStatus TaskRegion::Execute(Pool_t &pool) {
  std::atomic<bool> task_failed = false;

  // for now, require a pool with one thread
  PARTHENON_REQUIRE_THROWS(pool.size() == 1,
                           "Pool_t size != 1 is not currently supported.")

  // first, if needed, finish building the graph
  if (!graph_built) BuildGraph();

  // declare this so it can call itself
  std::function<TaskStatus(Task *)> ProcessTask;
  ProcessTask = [&pool, &ProcessTask, &task_failed](Task *task) -> TaskStatus {
    if (task_failed.load(std::memory_order_acquire)) {
      return TaskStatus::fail;
    }

    auto status = task->operator()();
    if (status == TaskStatus::fail) {
      task_failed.store(true, std::memory_order_release);
      return status;
    }

    auto next_up = task->GetDependent(status);
    for (auto t : next_up) {
      if (t->ready()) {
        pool.enqueue([t, &ProcessTask]() { return ProcessTask(t); });
      }
    }

    return status;
  };

  // now enqueue the "first_task" for all task lists
  for (auto &tl : task_lists) {
    auto t = tl.GetStartupTask();
    pool.enqueue([t, &ProcessTask]() { return ProcessTask(t); });
  }

  // then wait until everything is done
  const TaskStatus status = pool.wait();

  // Check the results, so as to fire any exceptions from threads
  // Return failure if a task failed
  if (status != TaskStatus::complete || task_failed.load(std::memory_order_acquire)) {
    return TaskListStatus::fail;
  } else {
    return TaskListStatus::complete;
  }
}

void TaskRegion::AppendTasks(std::vector<std::shared_ptr<Task>> &tasks_inout) {
  BuildGraph();
  for (const auto &tl : task_lists) {
    tl.AppendTasks(tasks_inout);
  }
}

void TaskRegion::AddRegionalDependencies(const std::vector<TaskList *> &tls) {
  const auto num_lists = tls.size();
  const auto num_regional = tls.front()->NumRegional();
  std::vector<Task *> tasks(num_lists);
  for (int i = 0; i < num_regional; i++) {
    for (int j = 0; j < num_lists; j++) {
      tasks[j] = tls[j]->Regional(i);
    }
    std::vector<std::vector<Task *>> reg_dep;
    for (int j = 0; j < num_lists; j++) {
      reg_dep.push_back(std::vector<Task *>());
      for (auto t : tasks[j]->GetDependent(TaskStatus::complete)) {
        reg_dep[j].push_back(t);
      }
    }
    for (int j = 0; j < num_lists; j++) {
      for (auto t : reg_dep[j]) {
        for (int k = 0; k < num_lists; k++) {
          if (j == k) continue;
          t->AddDependency(tasks[k]);
          tasks[k]->AddDependent(t, TaskStatus::complete);
        }
      }
    }
  }
}

void TaskRegion::BuildGraph() {
  // first handle regional dependencies by getting a vector of pointers
  // to every sub-TaskList of each of the main TaskLists in the region
  // (and also including a pointer to the main TaskLists). Match these
  // TaskLists up across the region and insert their regional dependencies
  std::vector<std::vector<TaskList *>> tls;
  for (auto &tl : task_lists)
    tls.emplace_back(tl.GetAllTaskLists());

  int num_sublists = tls.front().size();
  std::vector<TaskList *> matching_lists(task_lists.size());
  for (int sl = 0; sl < num_sublists; ++sl) {
    for (int i = 0; i < task_lists.size(); ++i)
      matching_lists[i] = tls[i][sl];
    AddRegionalDependencies(matching_lists);
  }

  // now hook up iterations
  for (auto &tl : task_lists) {
    tl.ConnectIteration();
  }

  graph_built = true;
  for (auto &tl : task_lists) {
    tl.SetGraphBuilt();
  }
}

} // namespace parthenon
