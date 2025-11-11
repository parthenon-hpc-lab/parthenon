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

// STL Includes
#include <memory>
#include <set>

// Third Party Includes
#include <catch2/catch.hpp>

// Internal Includes
#include "basic_types.hpp"
#include "tasks/tasks.hpp"

using parthenon::TaskID;
using parthenon::Task;
using parthenon::TaskList;
using parthenon::TaskStatus;

TEST_CASE("Task Object Lifecycle", "[TaskList][AddTask]") {
  GIVEN("A TaskList") {
    // This weak_ptr is just used to make sure TaskList destroys its objects when it
    // goes out of scope.
    std::weak_ptr<int> track_destruction;

    {
      auto obj = std::make_shared<int>(0);

      // A weak ptr is taken to the shared ptr to check that it is destroyed later.
      track_destruction = obj;

      TaskList task_list;
      task_list.AddTask(TaskID{}, [obj] { return TaskStatus::complete; });

      // Task objects should still be alive here.
      REQUIRE(!track_destruction.expired());
    }

    // Task objects are now destroyed
    REQUIRE(track_destruction.expired());
  }
}

// TaskChecker: Provides functionality to build and verify task dependency graphs
struct TaskChecker {
  parthenon::TaskCollection tc;  // Underlying task collection being tested
  std::vector<parthenon::TaskRegion*> regions;  // List of task regions in tcs

  // Task tracking and dependency management
  std::size_t current_global_task_id{0};  // Counter for assigning unique task IDs
  std::vector<std::set<std::size_t>> dag_dependencies;  // Stores dependencies for each task
  std::vector<std::vector<std::size_t>> region_tasks;  // Maps region index to list of tasks in that region
  std::map<Task*, std::size_t> task_to_id;  // Maps Task pointer to its unique ID
  std::map<std::size_t, Task*> id_to_task;  // Maps unique ID to Task pointer
  std::map<std::size_t, bool> task_complete;  // Tracks completion status of tasks

  // Qualifier: Handles task qualification properties, particularly synchronization requirements
  struct Qualifier { 
    std::set<std::size_t> task_ids;  // Task IDs this qualifier applies to
    parthenon::TaskQualifier pqualifier;  // Underlying Parthenon qualifier
    TaskChecker *tc;  // Parent TaskChecker reference
    
    // Default constructor creates a normal qualifier
    Qualifier() : pqualifier(parthenon::TaskQualifier::normal) {}

    // Factory method for creating a local synchronization qualifier
    static Qualifier LocalSync(TaskChecker &tc) {
      Qualifier qual;
      qual.tc = &tc;
      qual.pqualifier = parthenon::TaskQualifier::local_sync; 
      return qual;
    }

    // Resolve: Implements synchronization logic based on qualifier type
    void Resolve() { 
      if (pqualifier.LocalSync()) {
        // A task list sync implies that all downstream tasks require
        // all of the sync marked tasks be completed. 
        
        // First, collect all dependencies from all tasks in this sync group
        std::set<std::size_t> combined_dependencies; 
        combined_dependencies.insert(task_ids.begin(), task_ids.end());
        for (auto task : task_ids)
          combined_dependencies.insert(tc->dag_dependencies[task].begin(), tc->dag_dependencies[task].end());
        
        printf("combined_dependencies = %i\n", combined_dependencies.size()); 
        
        // Do a brute force search through all tasks and check if they depend on 
        // any of the tasks in this local sync region
        for (std::size_t task = 0; task < tc->current_global_task_id; ++task) { 
          auto &deps = tc->dag_dependencies[task];
          bool depends_on{false};
          
          // Check if this task depends on any task in the sync group
          for (auto dep : deps) {
            for (auto task : task_ids) {
              if (task == dep) depends_on = true;
            }
          }
          
          // If it depends on the sync group, make it depend on all tasks in the combined dependencies
          if (depends_on)
            deps.insert(combined_dependencies.begin(), combined_dependencies.end());
        }
      }
    }
  };
  
  // Default qualifier for tasks that don't specify one
  inline static Qualifier default_qualifier{};

  // Execute all tasks in the collection
  auto Execute() {
    return tc.Execute();
  }
  
  // Add a new region with the specified size and return its index
  std::size_t AddRegion(int region_size) {
    regions.emplace_back(&tc.AddRegion(region_size));
    region_tasks.emplace_back();  // Initialize empty task list for this region
    return regions.size() - 1;  // Return index of the newly added region
  }
  
  // Add a task to the specified region and task list with given dependencies
  std::size_t AddTask(std::size_t region, std::size_t task_list, std::vector<std::size_t> deps, Qualifier &qualifier = default_qualifier) { 
    // Build the dependency task ID by combining all dependent tasks
    TaskID tid(0); 
    for (auto dep : deps) { 
      tid = tid | id_to_task[dep];
    }

    // Get the requested region and task list
    auto &tl = (*regions[region])[task_list]; 
    
    // Add the task with a lambda that checks if all dependencies are complete
    auto id_out = tl.AddTask(qualifier.pqualifier, tid, [&](std::size_t task_id, TaskChecker *task_checker){
      bool all_dependencies_complete{true};
      for (auto &task : task_checker->dag_dependencies[task_id]) { 
        all_dependencies_complete = all_dependencies_complete && task_checker->task_complete[task];
      }
      printf("running task %i, all_dependencies_complete = %i (%i)\n", task_id, all_dependencies_complete, dag_dependencies[task_id].size());
      if (all_dependencies_complete) {
        task_complete[task_id] = true;
        return parthenon::TaskStatus::complete;
      }
        return parthenon::TaskStatus::fail;
    }, current_global_task_id, this);
    
    // Register the new task in our tracking structures
    task_to_id[id_out.GetTask()] = current_global_task_id;
    id_to_task[current_global_task_id] = id_out.GetTask(); 
    task_complete[current_global_task_id] = false;
    region_tasks[region].push_back(current_global_task_id); 

    // Build comprehensive dependencies for this task:
    // 1. First from explicitly stated dependencies
    dag_dependencies.emplace_back();
    auto &cur_task_deps = dag_dependencies.back();
    cur_task_deps.insert(deps.begin(), deps.end());
    
    // 2. Add transitive dependencies (dependencies of dependencies)
    for (auto dep : deps)
      cur_task_deps.insert(dag_dependencies[dep].begin(), dag_dependencies[dep].end());

    // 3. Add dependencies to all tasks from previous regions
    for (int r = 0; r < region; ++r) {
      for (auto &t : region_tasks[r]) {
        cur_task_deps.insert(t);
      } 
    }

    // 4. Register this task with the qualifier for later resolution
    qualifier.task_ids.insert(current_global_task_id);

    // Increment to prepare for next task
    current_global_task_id++;

    // Return the ID of the newly created task
    return current_global_task_id - 1;
  }
};

TEST_CASE("TaskCollection dependence", "[TaskList][AddTask]") {
  GIVEN("A TaskCollection") {

    TaskChecker tc; 
    int region1_size = 3;
    auto r1 = tc.AddRegion(region1_size);
    auto local_sync1 = TaskChecker::Qualifier::LocalSync(tc);
    for (int l = 0; l < region1_size; ++l) {
      auto t0 = tc.AddTask(r1, l, {}); 
      if (l == 0) t0 = tc.AddTask(r1, l, {t0});
      auto t1 = tc.AddTask(r1, l, {t0}, local_sync1);
      tc.AddTask(r1, l, {t1});
    }
    local_sync1.Resolve();

    int region2_size = 2;
    auto r2 = tc.AddRegion(region2_size);
    for (int l = 0; l < region2_size; ++l) { 
      auto t1 = tc.AddTask(r2, l, {});
      tc.AddTask(r2, l, {t1});
    }

    auto status = tc.Execute();
    REQUIRE(status == parthenon::TaskListStatus::complete);
  }
}


TEST_CASE("TaskCollection timeout", "[TaskList][AddTask]") {
  GIVEN("A TaskCollection") {
    parthenon::TaskCollection tc;
    parthenon::TaskRegion &region = tc.AddRegion(1);
    region[0].AddTask(TaskID(0), []() { return TaskStatus::incomplete; });
    const std::size_t timeout_in_seconds = 4;
    parthenon::TaskListStatus status = tc.Execute(timeout_in_seconds);
    REQUIRE(status == parthenon::TaskListStatus::fail);
  }
}
