//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "utils/callbacks.hpp"

using Catch::Matchers::Contains;
using parthenon::CallbackDAG;
using Registration = parthenon::CallbackRegistration<std::function<void()>>;

namespace {
std::ptrdiff_t Position(const std::vector<std::string> &order, const std::string &name) {
  return std::find(order.begin(), order.end(), name) - order.begin();
}

} // namespace

TEST_CASE("CallbackDAG topological sort", "[CallbackDAG]") {
  CallbackDAG dag;

  SECTION("Empty graph") { REQUIRE(dag.TopologicalSort().empty()); }

  SECTION("Single node") {
    dag.AddNode("a");
    auto order = dag.TopologicalSort();
    REQUIRE(order.size() == 1);
    REQUIRE(order[0] == "a");
  }

  SECTION("Linear order") {
    // a -> b -> c
    dag.AddEdge("a", "b");
    dag.AddEdge("b", "c");

    auto order = dag.TopologicalSort();
    REQUIRE(order.size() == 3);
    REQUIRE(order[0] == "a");
    REQUIRE(order[1] == "b");
    REQUIRE(order[2] == "c");
  }

  SECTION("Diamond dependency") {
    // Diamond:
    //     a
    //    / \.
    //   b   c
    //    \ /
    //     d
    dag.AddEdge("a", "b");
    dag.AddEdge("a", "c");
    dag.AddEdge("b", "d");
    dag.AddEdge("c", "d");

    auto order = dag.TopologicalSort();
    REQUIRE(order.size() == 4);
    REQUIRE(order[0] == "a");
    REQUIRE(order[3] == "d");
    // 'b' and 'c' can be in either order, but both must come after 'a' and before 'd'
    REQUIRE(
        ((order[1] == "b" && order[2] == "c") || (order[1] == "c" && order[2] == "b")));
  }

  SECTION("Disconnected components") {
    // Two separate chains: a -> b and c -> d
    dag.AddEdge("a", "b");
    dag.AddEdge("c", "d");

    auto order = dag.TopologicalSort();
    REQUIRE(order.size() == 4);
    REQUIRE(Position(order, "a") < Position(order, "b"));
    REQUIRE(Position(order, "c") < Position(order, "d"));
  }

  SECTION("Complex DAG") {
    //   grid -> hydro -> eos
    //              |      |
    //              v      v
    //         multispecies
    //              |
    //              v
    //           driver
    dag.AddEdge("grid", "hydro");
    dag.AddEdge("hydro", "eos");
    dag.AddEdge("hydro", "multispecies");
    dag.AddEdge("eos", "multispecies");
    dag.AddEdge("multispecies", "driver");

    auto order = dag.TopologicalSort();
    REQUIRE(order.size() == 5);
    REQUIRE(Position(order, "grid") < Position(order, "hydro"));
    REQUIRE(Position(order, "hydro") < Position(order, "eos"));
    REQUIRE(Position(order, "hydro") < Position(order, "multispecies"));
    REQUIRE(Position(order, "eos") < Position(order, "multispecies"));
    REQUIRE(Position(order, "multispecies") < Position(order, "driver"));
  }
}

TEST_CASE("CallbackDAG cycle detection", "[CallbackDAG]") {
  CallbackDAG dag;

  SECTION("Simple cycle") {
    // a -> b -> a
    dag.AddEdge("a", "b");
    dag.AddEdge("b", "a");
    REQUIRE_THROWS_AS(dag.TopologicalSort(), std::runtime_error);
  }

  SECTION("Three node cycle") {
    // a -> b -> c -> a
    dag.AddEdge("a", "b");
    dag.AddEdge("b", "c");
    dag.AddEdge("c", "a");
    REQUIRE_THROWS_AS(dag.TopologicalSort(), std::runtime_error);
  }

  SECTION("Cycle with separate node") {
    // a -> b -> c -> a, plus unconnected d
    dag.AddEdge("a", "b");
    dag.AddEdge("b", "c");
    dag.AddEdge("c", "a");
    dag.AddNode("d");
    REQUIRE_THROWS_AS(dag.TopologicalSort(), std::runtime_error);
  }

  SECTION("Cycle error message names the nodes in the cycle") {
    dag.AddEdge("hydro", "eos");
    dag.AddEdge("eos", "multispecies");
    dag.AddEdge("multispecies", "hydro");
    REQUIRE_THROWS_WITH(dag.TopologicalSort(), Contains("Cyclic dependency") &&
                                                   Contains("hydro") && Contains("eos") &&
                                                   Contains("multispecies"));
  }
}

TEST_CASE("CallbackDAG GraphViz output", "[CallbackDAG]") {
  CallbackDAG dag;
  std::ostringstream oss;

  SECTION("WriteGraphViz") {
    dag.AddEdge("a", "b");
    dag.AddEdge("b", "c");
    dag.WriteGraphViz(oss);

    const std::string output = oss.str();
    REQUIRE_THAT(output, Contains("digraph"));
    REQUIRE_THAT(output, Contains("\"a\""));
    REQUIRE_THAT(output, Contains("\"b\""));
    REQUIRE_THAT(output, Contains("\"c\""));
    REQUIRE_THAT(output, Contains("\"a\" -> \"b\""));
    REQUIRE_THAT(output, Contains("\"b\" -> \"c\""));
  }

  SECTION("Stream operator") {
    dag.AddEdge("x", "y");
    oss << dag;

    const std::string output = oss.str();
    REQUIRE_THAT(output, Contains("digraph"));
    REQUIRE_THAT(output, Contains("\"x\" -> \"y\""));
  }
}

TEST_CASE("CallbackRegistration dependencies", "[CallbackRegistration]") {
  auto noop = []() {};
  Registration reg;

  SECTION("Default registration is empty") {
    REQUIRE_FALSE(reg.IsRegistered());
    REQUIRE_FALSE(static_cast<bool>(reg));
    REQUIRE(reg == nullptr);
    REQUIRE(reg.depends_on.empty());
    REQUIRE(reg.required_by.empty());
  }

  SECTION("Register stores after/before dependencies") {
    reg.Register(noop, {"eos", "grid"}, {"driver"});
    REQUIRE(reg.IsRegistered());
    REQUIRE(reg != nullptr);
    REQUIRE(reg.depends_on == std::vector<std::string>{"eos", "grid"});
    REQUIRE(reg.required_by == std::vector<std::string>{"driver"});
  }

  SECTION("Register without dependencies") {
    reg.Register(noop);
    REQUIRE(reg.IsRegistered());
    REQUIRE(reg.depends_on.empty());
    REQUIRE(reg.required_by.empty());
  }

  SECTION("Assignment clears previous dependencies") {
    reg.Register(noop, {"eos"}, {"driver"});
    reg = noop;
    REQUIRE(reg.IsRegistered());
    REQUIRE(reg.depends_on.empty());
    REQUIRE(reg.required_by.empty());
  }

  SECTION("Re-registering replaces previous dependencies") {
    reg.Register(noop, {"eos"}, {"driver"});
    reg.Register(noop, {"grid"});
    REQUIRE(reg.depends_on == std::vector<std::string>{"grid"});
    REQUIRE(reg.required_by.empty());
  }

  SECTION("Before/after registrations order a CallbackDAG") {
    // hydro runs after grid; eos declares it runs before hydro
    std::map<std::string, Registration> regs;
    regs["grid"].Register(noop);
    regs["hydro"].Register(noop, {"grid"});
    regs["eos"].Register(noop, {}, {"hydro"});

    CallbackDAG dag;
    for (const auto &[name, r] : regs) {
      dag.AddNode(name);
      for (const auto &dep : r.depends_on) {
        dag.AddEdge(dep, name);
      }
      for (const auto &req : r.required_by) {
        dag.AddEdge(name, req);
      }
    }

    auto order = dag.TopologicalSort();
    REQUIRE(order.size() == 3);
    REQUIRE(Position(order, "grid") < Position(order, "hydro"));
    REQUIRE(Position(order, "eos") < Position(order, "hydro"));
  }
}

TEST_CASE("Packages_t ordered callbacks", "[Packages_t][CallbackDAG]") {
  using parthenon::Packages_t;
  using parthenon::StateDescriptor;
  using SD = StateDescriptor;

  // only CallbackRegistration members can be ordered
  static_assert(parthenon::CallbackMemberOf<&SD::FillDerivedMesh, SD>);
  static_assert(!parthenon::CallbackMemberOf<&SD::amr_criteria, SD>);
  static_assert(!parthenon::CallbackMemberOf<0, SD>);

  // FillDerivedMesh callbacks record the name of their package when called
  std::vector<std::string> called;
  auto record = [&](const std::string &name) {
    return [&called, name](parthenon::MeshData<parthenon::Real> *) {
      called.push_back(name);
    };
  };

  Packages_t packages;
  auto add = [&](const std::string &name) {
    auto pkg = std::make_shared<StateDescriptor>(name);
    packages.Add(pkg);
    return pkg;
  };

  // eos runs after grid and before hydro, hydro runs after grid, io has no callback
  auto grid = add("grid");
  auto hydro = add("hydro");
  auto eos = add("eos");
  add("io");
  grid->FillDerivedMesh.Register(record("grid"));
  hydro->FillDerivedMesh.Register(record("hydro"), /*after=*/{"grid"});
  eos->FillDerivedMesh.Register(record("eos"), /*after=*/{"grid"}, /*before=*/{"hydro"});

  // Packages whose FillDerivedMesh ran, in order
  auto fill_derived = [&]() {
    called.clear();
    for (const auto &fill_derived_mesh : packages.Callbacks<&SD::FillDerivedMesh>()) {
      fill_derived_mesh(nullptr);
    }
    return called;
  };
  const std::vector<std::string> expected{"grid", "eos", "hydro"};

  SECTION("Callbacks yields the registrations in order") {
    REQUIRE(packages.Callbacks<&SD::FillDerivedMesh>().size() == 3);
    REQUIRE(fill_derived() == expected);
    // other callbacks have their own order
    REQUIRE(packages.Callbacks<&SD::FillDerivedBlock>().size() == 0);
  }

  SECTION("Dependencies on packages without the callback are ignored") {
    add("driver")->FillDerivedMesh.Register(record("driver"), /*after=*/{"io"},
                                            /*before=*/{"missing"});
    REQUIRE(fill_derived().size() == 4);
  }

  SECTION("Cycles throw") {
    grid->FillDerivedMesh.Register(record("grid"), /*after=*/{"hydro"});
    REQUIRE_THROWS_WITH(fill_derived(), Contains("StateDescriptor callbacks") &&
                                            Contains("Cyclic dependency"));
  }

  SECTION("Adding a package updates the order") {
    REQUIRE(fill_derived() == expected);
    add("driver")->FillDerivedMesh.Register(record("driver"), /*after=*/{"hydro"});
    REQUIRE(fill_derived() == std::vector<std::string>{"grid", "eos", "hydro", "driver"});
  }

  SECTION("Changing registrations changes the order") {
    REQUIRE(fill_derived() == expected);
    // unregistering a package
    eos->FillDerivedMesh = nullptr;
    REQUIRE(fill_derived() == std::vector<std::string>{"grid", "hydro"});
    // changing dependencies
    hydro->FillDerivedMesh.Register(record("hydro"), /*after=*/{}, /*before=*/{"grid"});
    REQUIRE(fill_derived() == std::vector<std::string>{"hydro", "grid"});
    // swapping the function keeps the order
    grid->FillDerivedMesh = record("grid");
    REQUIRE(fill_derived() == std::vector<std::string>{"hydro", "grid"});
  }

  SECTION("WriteCallbackGraph") {
    std::ostringstream oss;
    packages.WriteCallbackGraph<&SD::FillDerivedMesh>(oss, "FillDerivedMesh");
    const std::string output = oss.str();
    REQUIRE_THAT(output, Contains("FillDerivedMesh"));
    REQUIRE_THAT(output, Contains("\"grid\" -> \"eos\""));
    REQUIRE_THAT(output, Contains("\"eos\" -> \"hydro\""));
    REQUIRE_THAT(output, !Contains("\"io\""));
  }
}

// Downstream codes can order the callbacks of their own types the same way
struct Unit {
  std::string name;
  Registration Setup;
  Registration Finalize;
};

TEST_CASE("CallbackOrder of a downstream type", "[CallbackOrder][CallbackDAG]") {
  using parthenon::CallbackOrder;
  std::vector<std::string> called;
  auto record = [&](const std::string &name) {
    return [&called, name]() { called.push_back(name); };
  };

  std::map<std::string, std::shared_ptr<Unit>> units;
  for (const auto &name : {"eos", "grid", "hydro"}) {
    units[name] = std::make_shared<Unit>(Unit{name});
  }
  units["grid"]->Setup.Register(record("grid"));
  units["hydro"]->Setup.Register(record("hydro"), /*after=*/{"grid"});
  units["eos"]->Setup.Register(record("eos"), /*after=*/{"grid"}, /*before=*/{"hydro"});
  units["hydro"]->Finalize.Register(record("hydro"));

  SECTION("Callbacks run in dependency order") {
    const CallbackOrder<Unit> order(units, "Unit");
    for (const auto &setup : order.Callbacks<&Unit::Setup>()) {
      setup();
    }
    REQUIRE(called == std::vector<std::string>{"grid", "eos", "hydro"});

    called.clear();
    for (const auto &finalize : order.Callbacks<&Unit::Finalize>()) {
      finalize();
    }
    REQUIRE(called == std::vector<std::string>{"hydro"});
  }

  SECTION("Cycles name the type") {
    units["grid"]->Setup.Register(record("grid"), /*after=*/{"hydro"});
    const CallbackOrder<Unit> order(units, "Unit");
    REQUIRE_THROWS_WITH(order.Callbacks<&Unit::Setup>(),
                        Contains("Unit callbacks") && Contains("Cyclic dependency"));
  }
}
