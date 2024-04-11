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

using parthenon::TaskID;
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
