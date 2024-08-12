//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef TASKS_TASKS_HPP_
#define TASKS_TASKS_HPP_

#if __has_include(<cxxabi.h>)
#include <cxxabi.h> //NOLINT
#define HAS_CXX_ABI
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <basic_types.hpp>
#include <parthenon_mpi.hpp>

#include "globals.hpp"
#include "thread_pool.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"

// Macro for decorating functions passed to AddTask so that their names
// are stored for outputing task graphs
#define TF(...) std::string(#__VA_ARGS__), __VA_ARGS__

template <class T>
struct is_tuple_t : std::false_type {};

template <class... Ts>
struct is_tuple_t<std::tuple<Ts...>> : std::true_type {};

namespace parthenon {

enum class TaskListStatus { complete, fail };
enum class TaskType { normal, completion };

class TaskQualifier {
 public:
  using qualifier_t = uint64_t;
  TaskQualifier() = delete;
  TaskQualifier(const qualifier_t n) : flags(n) {} // NOLINT(runtime/explicit)

  static inline constexpr qualifier_t normal{0};
  static inline constexpr qualifier_t local_sync{1 << 0};
  static inline constexpr qualifier_t global_sync{1 << 1};
  static inline constexpr qualifier_t completion{1 << 2};
  static inline constexpr qualifier_t once_per_region{1 << 3};

  bool LocalSync() const { return flags & local_sync; }
  bool GlobalSync() const { return flags & global_sync; }
  bool Completion() const { return flags & completion; }
  bool Once() const { return flags & once_per_region; }

 private:
  qualifier_t flags;
};

// forward declare Task for TaskID
class Task;
class TaskID {
 public:
  TaskID() : task(nullptr) {}
  // pointers to Task are implicitly convertible to TaskID
  TaskID(Task *t) : task(t) {} // NOLINT(runtime/explicit)

  TaskID operator|(const TaskID &other) const {
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

  const std::vector<Task *> &GetIDs() const { return std::cref(dep); }

  bool empty() const { return (!task && dep.size() == 0); }
  Task *GetTask() { return task; }

 private:
  Task *task = nullptr;
  std::vector<Task *> dep;
};

class Task {
 public:
  Task() = default;
  template <typename TID>
  Task(TID &&dep, const std::string &label, const std::function<TaskStatus()> &func,
       std::pair<int, int> limits = {1, 1})
      : Task(std::forward<TID>(dep), label, 0, func, limits) {}
  template <typename TID>
  Task(TID &&dep, const std::string &label, int verbose_level,
       const std::function<TaskStatus()> &func, std::pair<int, int> limits = {1, 1})
      : label_(label), verbose_level_(verbose_level), f(func), exec_limits(limits) {
    if (dep.GetIDs().size() == 0 && dep.GetTask()) {
      dependencies.insert(dep.GetTask());
    } else {
      for (auto &d : dep.GetIDs()) {
        dependencies.insert(d);
      }
    }
    // always add "this" to repeat task if it's incomplete
    dependent[static_cast<int>(TaskStatus::incomplete)].push_back(this);
  }

  TaskStatus operator()() {
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
  TaskID GetID() { return this; }
  std::string GetLabel() const { return label_; }
  bool ready() {
    // check that no dependency is incomplete
    bool go = true;
    for (auto &dep : dependencies) {
      go = go && (dep->GetStatus() != TaskStatus::incomplete);
    }
    return go;
  }
  void AddDependency(Task *t) { dependencies.insert(t); }
  std::unordered_set<Task *> &GetDependencies() { return dependencies; }
  void AddDependent(Task *t, TaskStatus status) {
    dependent[static_cast<int>(status)].push_back(t);
  }
  std::vector<Task *> &GetDependent(TaskStatus status = TaskStatus::complete) {
    return dependent[static_cast<int>(status)];
  }
  void SetType(TaskType type) { task_type = type; }
  TaskType GetType() { return task_type; }
  void SetStatus(TaskStatus status) {
    std::lock_guard<std::mutex> lock(mutex);
    task_status = status;
  }
  TaskStatus GetStatus() {
    std::lock_guard<std::mutex> lock(mutex);
    return task_status;
  }
  void reset_iteration() { num_calls = 0; }

 private:
  std::function<TaskStatus()> f;
  // store a list of tasks that might be available to
  // run for each possible status this task returns
  std::array<std::vector<Task *>, 3> dependent;
  std::unordered_set<Task *> dependencies;
  std::pair<int, int> exec_limits;
  TaskType task_type = TaskType::normal;
  int num_calls = 0;
  TaskStatus task_status = TaskStatus::incomplete;
  std::mutex mutex;
  int verbose_level_;
  std::string label_;
};

inline std::ostream &WriteTaskGraph(std::ostream &stream,
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

class TaskRegion;
class TaskList {
  friend class TaskRegion;

 public:
  TaskList() : TaskList(TaskID(), {1, 1}) {}
  explicit TaskList(const TaskID &dep, std::pair<int, int> limits)
      : dependency(dep), exec_limits(limits), graph_built{false} {
    // make a trivial first_task after which others will get launched
    // simplifies logic for iteration and startup
    tasks.push_back(std::make_shared<Task>(
        dependency, "first_task",
        [&tasks = tasks]() {
          for (auto &t : tasks) {
            t->SetStatus(TaskStatus::incomplete);
          }
          return TaskStatus::complete;
        },
        exec_limits));
    first_task = tasks.back().get();
    // connect list dependencies to this list's first_task
    for (auto t : first_task->GetDependencies()) {
      t->AddDependent(first_task, TaskStatus::complete);
    }

    // make a trivial last_task that tasks dependent on this list's execution
    // can depend on.  Also simplifies exiting completed iterations
    tasks.push_back(std::make_shared<Task>(
        TaskID(), "last_task",
        [&completion_tasks = completion_tasks]() {
          for (auto t : completion_tasks) {
            t->reset_iteration();
          }
          return TaskStatus::complete;
        },
        exec_limits));
    last_task = tasks.back().get();
  }

  template <class... Args>
  TaskID AddTask(TaskID dep, Args &&...args) {
    return AddTask(TaskQualifier::normal, dep, std::forward<Args>(args)...);
  }

  template <class Arg1, class... Args>
  TaskID AddTask(const TaskQualifier tq, TaskID dep, Arg1 &&arg1, Args &&...args) {
    if constexpr (std::is_invocable<Arg1, Args...>::value) {
      return AddTaskImpl(tq, dep, {}, 0, std::forward<Arg1>(arg1),
                         std::forward<Args>(args)...);
    } else if constexpr (std::is_convertible<Arg1, std::string>::value) {
      return AddTaskImpl(tq, dep, std::forward<Arg1>(arg1), 0,
                         std::forward<Args>(args)...);
    } else if constexpr (is_tuple_t<Arg1>::value) {
      return AddTaskImpl(tq, dep, std::get<0>(arg1), std::get<1>(arg1), std::get<2>(arg1),
                         std::forward<Args>(args)...);
    } else {
      static_assert(always_false<Arg1>, "Bad signature for AddTask.");
      return TaskID();
    }
    // Stops some compilers from complaining
    return TaskID();
  }

  template <class... Args>
  TaskID AddTaskImpl(const TaskQualifier tq, TaskID dep,
                     const std::optional<std::string> &label, int verbose,
                     Args &&...args) {
    if (graph_built) {
      PARTHENON_FAIL("Trying to add a task to a TaskList that has already"
                     " been built into a completed TaskRegion graph.");
    }
    // user-space tasks always depend on something. if no dependencies are given,
    // make the task dependent on the list's first_task
    if (dep.empty()) dep = TaskID(first_task);

    if (!tq.Once() || (tq.Once() && unique_id == 0)) {
      AddUserTask(dep, label, verbose, std::forward<Args>(args)...);
    } else {
      tasks.push_back(std::make_shared<Task>(
          dep, "once task", [=]() { return TaskStatus::complete; }, exec_limits));
    }

    Task *my_task = tasks.back().get();
    TaskID id(my_task);

    if (tq.LocalSync() || tq.GlobalSync() || tq.Once()) {
      regional_tasks.push_back(my_task);
    }

    if (tq.GlobalSync()) {
      bool do_mpi = false;
#ifdef MPI_PARALLEL
      // make status, request, and comm for this global task
      global_status.push_back(std::make_shared<int>(0));
      global_request.push_back(std::make_shared<MPI_Request>(MPI_REQUEST_NULL));
      // be careful about the custom deleter so it doesn't call
      // an MPI function after Finalize
      global_comm.emplace_back(new MPI_Comm, [&](MPI_Comm *d) {
        int finalized;
        PARTHENON_MPI_CHECK(MPI_Finalized(&finalized));
        if (!finalized) PARTHENON_MPI_CHECK(MPI_Comm_free(d));
      });
      // we need another communicator to support multiple in flight non-blocking
      // collectives where we can't guarantee calling order across ranks
      PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, global_comm.back().get()));
      do_mpi = true;
#endif // MPI_PARALLEL
      TaskID start;
      // only call MPI once per region, on the list with unique_id = 0
      if (unique_id == 0 && do_mpi) {
#ifdef MPI_PARALLEL
        // add a task that starts the Iallreduce on the task statuses
        tasks.push_back(std::make_shared<Task>(
            id, "GlobalSync start",
            [my_task, &stat = *global_status.back(), &req = *global_request.back(),
             &comm = *global_comm.back()]() {
              // jump through a couple hoops to figure out statuses of all instances of
              // my_task accross all lists in the enclosing TaskRegion
              auto dependent = my_task->GetDependent(TaskStatus::complete);
              assert(dependent.size() == 1);
              auto mytask = *dependent.begin();
              stat = 0;
              for (auto dep : mytask->GetDependencies()) {
                stat = std::max(stat, static_cast<int>(dep->GetStatus()));
              }
              PARTHENON_MPI_CHECK(
                  MPI_Iallreduce(MPI_IN_PLACE, &stat, 1, MPI_INT, MPI_MAX, comm, &req));
              return TaskStatus::complete;
            },
            exec_limits));
        start = TaskID(tasks.back().get());
        // add a task that tests for completion of the Iallreduces of statuses
        tasks.push_back(std::make_shared<Task>(
            start, "GlobalSync check completion",
            [&stat = *global_status.back(), &req = *global_request.back()]() {
              int check;
              PARTHENON_MPI_CHECK(MPI_Test(&req, &check, MPI_STATUS_IGNORE));
              if (check) {
                return static_cast<TaskStatus>(stat);
              }
              return TaskStatus::incomplete;
            },
            exec_limits));
#endif         // MPI_PARALLEL
      } else { // unique_id != 0
        // just add empty tasks
        tasks.push_back(std::make_shared<Task>(
            id, "empty GlobalSync task", [&]() { return TaskStatus::complete; },
            exec_limits));
        start = TaskID(tasks.back().get());
        tasks.push_back(std::make_shared<Task>(
            start, "empty GlobalSync task", [my_task]() { return my_task->GetStatus(); },
            exec_limits));
      }
      // reset id so it now points at the task that finishes the Iallreduce
      id = TaskID(tasks.back().get());
      // make the task that starts the Iallreduce point at the one that finishes it
      start.GetTask()->AddDependent(id.GetTask(), TaskStatus::complete);
      // for any status != incomplete, my_task should point at the mpi reduction
      my_task->AddDependent(start.GetTask(), TaskStatus::complete);
      my_task->AddDependent(start.GetTask(), TaskStatus::iterate);
      // make the finish Iallreduce task finish on all lists before moving on
      regional_tasks.push_back(id.GetTask());
    }

    // connect completion tasks to last_task
    if (tq.Completion()) {
      auto t = id.GetTask();
      t->SetType(TaskType::completion);
      t->AddDependent(last_task, TaskStatus::complete);
      completion_tasks.push_back(t);
    }

    // make connections so tasks point to this task to run next
    for (auto d : my_task->GetDependencies()) {
      if (d->GetType() == TaskType::completion) {
        d->AddDependent(my_task, TaskStatus::iterate);
      } else {
        d->AddDependent(my_task, TaskStatus::complete);
      }
    }
    return id;
  }

  template <typename TID>
  std::pair<TaskList &, TaskID> AddSublist(TID &&dep, std::pair<int, int> minmax_iters) {
    sublists.push_back(std::make_shared<TaskList>(dep, minmax_iters));
    auto &tl = *sublists.back();
    tl.SetID(unique_id);
    return std::make_pair(std::ref(tl), TaskID(tl.last_task));
  }

  inline friend std::ostream &operator<<(std::ostream &stream, const TaskList &tl) {
    std::vector<std::shared_ptr<Task>> tasks;
    tl.AppendTasks(tasks);
    return WriteTaskGraph(stream, tasks);
  }

  std::vector<TaskList *> GetAllTaskLists() {
    std::vector<TaskList *> list;
    GetAllTaskListsInternal(list);
    return list;
  }

 private:
  TaskID dependency;
  std::pair<int, int> exec_limits;
  // put these in shared_ptrs so copying TaskList works as expected
  std::vector<std::shared_ptr<Task>> tasks;
  std::vector<std::shared_ptr<TaskList>> sublists;
#ifdef MPI_PARALLEL
  std::vector<std::shared_ptr<int>> global_status;
  std::vector<std::shared_ptr<MPI_Request>> global_request;
  std::vector<std::shared_ptr<MPI_Comm>> global_comm;
#endif // MPI_PARALLEL
  // vectors are fine for these
  std::vector<Task *> regional_tasks;
  std::vector<Task *> global_tasks;
  std::vector<Task *> completion_tasks;
  // special startup and takedown tasks auto added to lists
  Task *first_task;
  Task *last_task;
  // a unique id to support tasks that should only get executed once per region
  int unique_id;
  bool graph_built;

  void GetAllTaskListsInternal(std::vector<TaskList *> &list) {
    list.emplace_back(this);
    for (auto &ptl : sublists)
      ptl->GetAllTaskListsInternal(list);
  }

  void AppendTasks(std::vector<std::shared_ptr<Task>> &tasks_inout) const {
    tasks_inout.insert(tasks_inout.end(), tasks.begin(), tasks.end());
    for (const auto &stl : sublists) {
      tasks_inout.insert(tasks_inout.end(), stl->tasks.begin(), stl->tasks.end());
    }
  }

  void SetGraphBuilt() {
    graph_built = true;
    for (auto &tl : sublists)
      tl->SetGraphBuilt();
  }

  Task *GetStartupTask() { return first_task; }
  size_t NumRegional() const { return regional_tasks.size(); }
  Task *Regional(const int i) { return regional_tasks[i]; }
  void SetID(const int id) { unique_id = id; }

  void ConnectIteration() {
    if (completion_tasks.size() != 0) {
      auto last = completion_tasks.back();
      last->AddDependent(first_task, TaskStatus::iterate);
    }
    for (auto &tl : sublists)
      tl->ConnectIteration();
  }

  template <class F>
  std::string MakeUserTaskLabel(std::optional<std::string> label) {
    if (!label.has_value()) label = "anon";
#ifdef HAS_CXX_ABI
    int status;
    // _cxa_demangle calls malloc so we must call free
    char *signature_name = abi::__cxa_demangle(typeid(F).name(), NULL, NULL, &status);
    auto signature = std::string(signature_name);
    std::free(signature_name);
#else
    std::string signature = " (...)";
#endif
    auto n = signature.find('(');
    signature.insert(n--, *label);
    return signature;
  }

  template <class T, class U, class... Args1, class... Args2>
  void AddUserTask(TaskID &dep, const std::optional<std::string> &label, int verbose,
                   TaskStatus (T::*func)(Args1...), U *obj, Args2 &&...args) {
    tasks.push_back(std::make_shared<Task>(
        dep, MakeUserTaskLabel<decltype(func)>(label), verbose,
        [=]() mutable -> TaskStatus {
          return (obj->*func)(std::forward<Args2>(args)...);
        },
        exec_limits));
  }

  template <class F, class... Args>
  void AddUserTask(TaskID &dep, const std::optional<std::string> &label, int verbose,
                   F &&func, Args &&...args) {
    tasks.push_back(std::make_shared<Task>(
        dep, MakeUserTaskLabel<F>(label), verbose,
        [=, func = std::forward<F>(func)]() mutable -> TaskStatus {
          return func(std::forward<Args>(args)...);
        },
        exec_limits));
  }
};

class TaskCollection;
class TaskRegion {
  friend TaskCollection;

 public:
  TaskRegion() = delete;
  explicit TaskRegion(const int num_lists) : task_lists(num_lists) {
    for (int i = 0; i < num_lists; i++)
      task_lists[i].SetID(i);
  }

  TaskListStatus Execute(ThreadPool &pool) {
    // for now, require a pool with one thread
    PARTHENON_REQUIRE_THROWS(pool.size() == 1,
                             "ThreadPool size != 1 is not currently supported.")

    // first, if needed, finish building the graph
    if (!graph_built) BuildGraph();

    // declare this so it can call itself
    std::function<TaskStatus(Task *)> ProcessTask;
    ProcessTask = [&pool, &ProcessTask](Task *task) -> TaskStatus {
      auto status = task->operator()();
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
    pool.wait();

    // Check the results, so as to fire any exceptions from threads
    // Return failure if a task failed
    return (pool.check_task_returns() == TaskStatus::complete) ? TaskListStatus::complete
                                                               : TaskListStatus::fail;
  }

  TaskList &operator[](const int i) { return task_lists[i]; }

  size_t size() const { return task_lists.size(); }

  inline friend std::ostream &operator<<(std::ostream &stream, TaskRegion &region) {
    std::vector<std::shared_ptr<Task>> tasks;
    region.AppendTasks(tasks);
    return WriteTaskGraph(stream, tasks);
  }

 private:
  std::vector<TaskList> task_lists;
  bool graph_built = false;

  void AppendTasks(std::vector<std::shared_ptr<Task>> &tasks_inout) {
    BuildGraph();
    for (const auto &tl : task_lists) {
      tl.AppendTasks(tasks_inout);
    }
  }

  void AddRegionalDependencies(const std::vector<TaskList *> &tls) {
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

  void BuildGraph() {
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
};

class TaskCollection {
 public:
  TaskCollection() = default;

  TaskRegion &AddRegion(const int num_lists) {
    regions.emplace_back(num_lists);
    return regions.back();
  }
  TaskListStatus Execute() {
    static ThreadPool pool(1);
    return Execute(pool);
  }
  TaskListStatus Execute(ThreadPool &pool) {
    TaskListStatus status;
    for (auto &region : regions) {
      status = region.Execute(pool);
      if (status != TaskListStatus::complete) return status;
    }
    return TaskListStatus::complete;
  }

  inline friend std::ostream &operator<<(std::ostream &stream, TaskCollection &tc) {
    std::vector<std::shared_ptr<Task>> tasks;
    tc.AppendTasks(tasks);
    return WriteTaskGraph(stream, tasks);
  }

 private:
  std::list<TaskRegion> regions;

  void AppendTasks(std::vector<std::shared_ptr<Task>> &tasks_inout) {
    for (auto &region : regions) {
      region.AppendTasks(tasks_inout);
    }
  }
};

} // namespace parthenon

#endif // TASKS_TASKS_HPP_
