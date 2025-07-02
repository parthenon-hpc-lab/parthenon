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

// Third Party Includes
#include <catch2/catch.hpp>

// Internal Includes
#include "basic_types.hpp"
#include "tasks/tasks.hpp"

using parthenon::TaskCollection;
using parthenon::TaskID;
using parthenon::TaskList;
using parthenon::TaskListStatus;
using parthenon::TaskStatus;

TEST_CASE("Task failure handling", "[TaskList][AddTask]") {
  GIVEN("A TaskList") {
    TaskCollection tc;

    // create task region and add a task that returns TaskStatus::fail
    auto &region = tc.AddRegion(1);
    auto &tl = region[0];
    TaskID task0 = tl.AddTask(TaskID{}, [] { return TaskStatus::fail; });
    TaskID task1 = tl.AddTask(task0, [] { return TaskStatus::complete; });

    // Execute the task collection
    TaskListStatus status = tc.Execute();

    // Confirm failed Task results in a failed TaskList
    REQUIRE(status == TaskListStatus::fail);
  }
}
