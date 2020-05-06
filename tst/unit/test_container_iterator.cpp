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
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "interface/container.hpp"
#include "interface/container_iterator.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using parthenon::CellVariable;
using parthenon::CellVariableVector;
using parthenon::Container;
using parthenon::ContainerIterator;
using parthenon::DevExecSpace;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::Metadata;
using parthenon::MetadataFlag;
using parthenon::PackIndexMap;
using parthenon::par_for;
using parthenon::ParArray4D;
using parthenon::ParArrayND;
using parthenon::Real;

TEST_CASE("Can pull variables from containers based on Metadata", "[ContainerIterator]") {
  GIVEN("A Container with a set of variables initialized to zero") {
    Container<Real> rc;
    Metadata m_in({Metadata::Independent, Metadata::FillGhost});
    Metadata m_out;
    std::vector<int> scalar_block_size{16, 16, 16};
    std::vector<int> vector_block_size{16, 16, 16, 3};
    // Make some variables
    rc.Add("v1", m_in, scalar_block_size);
    rc.Add("v2", m_out, scalar_block_size);
    rc.Add("v3", m_in, vector_block_size);
    rc.Add("v4", m_out, vector_block_size);
    rc.Add("v5", m_in, scalar_block_size);
    rc.Add("v6", m_out, scalar_block_size);

    WHEN("We extract a subcontainer") {
      auto subcontainer = Container<Real>(rc, {"v1", "v3", "v5"});
      THEN("The container has the names in the right order") {
        auto vars = subcontainer.GetCellVariableVector();
        REQUIRE(vars[0]->label() == "v1");
        REQUIRE(vars[1]->label() == "v3");
        REQUIRE(vars[2]->label() == "v5");
      }
    }

    auto v = rc.PackVariables();
    par_for(
        "Initialize variables", DevExecSpace(), 0, v.GetDim(4) - 1, 0, v.GetDim(3) - 1, 0,
        v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          v(l, k, j, i) = 0.0;
        });

    WHEN("we check them") {
      // set them all to zero
      const CellVariableVector<Real> &cv = rc.GetCellVariableVector();
      for (int n = 0; n < cv.size(); n++) {
        ParArrayND<Real> v = cv[n]->data;
        par_for(
            "Initialize variables", DevExecSpace(), 0, v.GetDim(4) - 1, 0,
            v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
              v(l, k, j, i) = 0.0;
            });
      }
      THEN("they should sum to zero") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        Real total = 0.0;
        Real sum = 1.0;
        Kokkos::parallel_reduce(
            policy4D({0, 0, 0, 0}, {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                          Real &vsum) { vsum += v(l, k, j, i); },
            sum);
        total += sum;
        REQUIRE(total == 0.0);
      }

      AND_THEN("we touch the right number of elements") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        int total = 0;
        int sum = 1;
        Kokkos::parallel_reduce(
            policy4D({0, 0, 0, 0}, {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i, int &cnt) {
              cnt++;
            },
            sum);
        total += sum;
        REQUIRE(total == 40960);
      }
    }

    WHEN("we set Independent variables to one") {
      // set "Independent" variables to one
      auto v = rc.PackVariables({Metadata::Independent});
      par_for(
          "Set independent variables", DevExecSpace(), 0, v.GetDim(4) - 1, 0,
          v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            v(l, k, j, i) = 1.0;
          });

      THEN("they should sum appropriately") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        Real total = 0.0;
        Real sum = 1.0;
        Kokkos::parallel_reduce(
            policy4D({0, 0, 0, 0}, {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                          Real &vsum) { vsum += v(l, k, j, i); },
            sum);
        total += sum;
        REQUIRE(std::abs(total - 20480.0) < 1.e-14);
      }
      AND_THEN("pulling out a subset by name should work") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        auto v = rc.PackVariables({"v2", "v3", "v5"});
        Real total = 0.0;
        Real sum = 1.0;
        Kokkos::parallel_reduce(
            policy4D({0, 0, 0, 0}, {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                          Real &vsum) { vsum += v(l, k, j, i); },
            sum);
        total += sum;
        REQUIRE(std::abs(total - 16384.0) < 1.e-14);
      }
    }

    WHEN("we set individual fields by index") {
      PackIndexMap vmap;
      auto v = rc.PackVariables(std::vector<std::string>({"v3", "v6"}), vmap);
      const int iv3lo = vmap["v3"].first;
      const int iv3hi = vmap["v3"].second;
      const int iv6 = vmap["v6"].first;
      par_for(
          "Initialize variables", DevExecSpace(), 0, v.GetDim(3) - 1, 0, v.GetDim(2) - 1,
          0, v.GetDim(1) - 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
            v(1, k, j, i) = 1.0; // should correspond with iv3lo+1
            v(3, k, j, i) = 3.0; // should correspond with iv6
          });
      THEN("the values should as we expect") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        Real total = 0.0;
        Real sum = 1.0;
        Kokkos::parallel_reduce(
            policy4D({0, 0, 0, 0}, {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                          Real &vsum) {
              bool check3 = (l == iv3lo + 1);
              bool check6 = (l == iv6);
              vsum += (check3 && v(l, k, j, i) != 1.0);
              vsum += (check6 && v(l, k, j, i) != 3.0);
            },
            sum);
        total += sum;
        REQUIRE(total == 0.0);
      }
      AND_THEN("summing up everything should still work") {
        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        auto v = rc.PackVariables();
        Real total = 0.0;
        Real sum = 1.0;
        Kokkos::parallel_reduce(
            policy4D({0, 0, 0, 0}, {v.GetDim(4), v.GetDim(3), v.GetDim(2), v.GetDim(1)}),
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                          Real &vsum) { vsum += v(l, k, j, i); },
            sum);
        total += sum;
        REQUIRE(std::abs(total - 16384.0) < 1.e-14);
      }
    }

    WHEN("we set fluxes of independent variables") {
      auto vf = rc.PackVariablesAndFluxes({Metadata::Independent, Metadata::FillGhost});
      par_for(
          "Set fluxes", DevExecSpace(), 0, vf.GetDim(4) - 1, 0, vf.GetDim(3) - 1, 0,
          vf.GetDim(2) - 1, 0, vf.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            vf(l, k, j, i) = 0.0;
            vf.flux(0, l, k, j, i) = 16.0 - i;
            vf.flux(1, l, k, j, i) = 16.0 - j;
            vf.flux(2, l, k, j, i) = 16.0 - k;
          });
      THEN("adding in the fluxes should change the values appropriately") {
        par_for(
            "Update vars", DevExecSpace(), 0, vf.GetDim(4) - 1, 0, vf.GetDim(3) - 2, 0,
            vf.GetDim(2) - 2, 0, vf.GetDim(1) - 2,
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
              vf(l, k, j, i) -= ((vf.flux(0, l, k, j, i + 1) - vf.flux(0, l, k, j, i)) +
                                 (vf.flux(1, l, k, j + 1, i) - vf.flux(1, l, k, j, i)) +
                                 (vf.flux(2, l, k + 1, j, i) - vf.flux(2, l, k, j, i)));
            });

        using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;
        Real total = 0.0;
        Real sum = 1.0;
        Kokkos::parallel_reduce(
            policy4D({0, 0, 0, 0},
                     {v.GetDim(4), v.GetDim(3) - 1, v.GetDim(2) - 1, v.GetDim(1) - 1}),
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i,
                          Real &vsum) { vsum += v(l, k, j, i); },
            sum);
        total += sum;
        REQUIRE(std::abs(total - 50625.0) < 1.e-14);
      }
    }

    WHEN("we add sparse fields") {
      Metadata m_sparse;
      m_sparse = Metadata({Metadata::Sparse}, 1);
      rc.Add("vsparse", m_sparse, scalar_block_size);
      m_sparse = Metadata({Metadata::Sparse}, 13);
      rc.Add("vsparse", m_sparse, scalar_block_size);
      m_sparse = Metadata({Metadata::Sparse}, 42);
      rc.Add("vsparse", m_sparse, scalar_block_size);
      THEN("the low and high index bounds are correct as returned by PackVariables") {
        PackIndexMap imap;
        auto v = rc.PackVariables({"v3", "v6", "vsparse"}, imap);
        REQUIRE(imap["vsparse"].first == 4);
        REQUIRE(imap["vsparse"].second == 6);
      }
      AND_THEN("bounds are still correct if I get just a subset of the sparse fields") {
        PackIndexMap imap;
        auto v = rc.PackVariables({"v3", "vsparse"}, {1, 42}, imap);
        REQUIRE(imap["vsparse"].first == 3);
        REQUIRE(imap["vsparse"].second == 4);
        REQUIRE(imap["vsparse_42"].first == 4);
      }
    }
  }
}
// Test wrapper to run a function multiple times
template <typename InitFunc, typename PerfFunc>
double performance_test_wrapper(const int n_burn, const int n_perf, InitFunc init_func,
                                PerfFunc perf_func) {
  // Initialize the timer and test
  Kokkos::Timer timer;
  init_func();

  for (int i_run = 0; i_run < n_burn; i_run++) {
    // burn
    perf_func();
  }
  Kokkos::fence();
  timer.reset();
  for (int i_run = 0; i_run < n_perf; i_run++) {
    // time
    perf_func();
  }

  // Time it
  Kokkos::fence();
  double perf_time = timer.seconds();

  // FIXME?
  // Validate results?

  return perf_time;
}

TEST_CASE("Container Iterator Performance", "[ContainerIterator][performance]") {
  const int N = 32; // Dimensions of blocks
  const int Nvar = 10;
  const int n_burn = 500; // Num times to burn in before timing
  const int n_perf = 500; // Num times to run while timing

  // Make a raw ParArray4D for closest to bare metal looping
  ParArrayND<Real> raw_array("raw_array", Nvar, N, N, N);

  // Make a function for initializing the raw ParArray4D
  auto init_raw_array = [&]() {
    par_for(
        "Initialize ", DevExecSpace(), 0, Nvar - 1, 0, N - 1, 0, N - 1, 0, N - 1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          raw_array(l, k, j, i) =
              static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
        });
  };

  // Test performance iterating over variables (we should aim for this performance)
  double time_raw_array = performance_test_wrapper(n_burn, n_perf, init_raw_array, [&]() {
    par_for(
        "Raw Array Perf", DevExecSpace(), 0, Nvar - 1, 0, N - 1, 0, N - 1, 0, N - 1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          raw_array(l, k, j, i) *=
              raw_array(l, k, j, i); // Do something trivial, square each term
        });
  });

  // Make a container for testing performance
  Container<Real> container;
  Metadata m_in({Metadata::Independent});
  Metadata m_out;
  std::vector<int> scalar_block_size{N, N, N};
  std::vector<int> vector_block_size{N, N, N, 3};

  // make some variables - 5 in all, 2 3-vectors, total 10 fields
  container.Add("v0", m_in, scalar_block_size);
  container.Add("v1", m_in, scalar_block_size);
  container.Add("v2", m_in, vector_block_size);
  container.Add("v3", m_in, scalar_block_size);
  container.Add("v4", m_in, vector_block_size);
  container.Add("v5", m_in, scalar_block_size);
  // Make a function for initializing the container variables
  auto init_container = [&]() {
    const CellVariableVector<Real> &cv = container.GetCellVariableVector();
    for (int n = 0; n < cv.size(); n++) {
      ParArrayND<Real> v = cv[n]->data;
      par_for(
          "Initialize variables", DevExecSpace(), 0, v.GetDim(4) - 1, 0, v.GetDim(3) - 1,
          0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            v(l, k, j, i) = static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
          });
    }
  };

  // Test performance iterating over variables in container
  double time_iterate_variables =
      performance_test_wrapper(n_burn, n_perf, init_container, [&]() {
        const CellVariableVector<Real> &cv = container.GetCellVariableVector();
        for (int n = 0; n < cv.size(); n++) {
          ParArrayND<Real> v = cv[n]->data;
          par_for(
              "Iterate Variables Perf", DevExecSpace(), 0, v.GetDim(4) - 1, 0,
              v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                v(l, k, j, i) *= v(l, k, j, i); // Do something trivial, square each term
              });
        }
      });

  { // Grab variables by mask and do timing tests
    auto var_view = container.PackVariables({Metadata::Independent});

    auto init_view_of_views = [&]() {
      par_for(
          "Initialize ", DevExecSpace(), 0, var_view.GetDim(4) - 1, 0,
          var_view.GetDim(3) - 1, 0, var_view.GetDim(2) - 1, 0, var_view.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            var_view(l, k, j, i) =
                static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
          });
    };

    // Test performance of view of views VariablePack implementation
    double time_view_of_views =
        performance_test_wrapper(n_burn, n_perf, init_view_of_views, [&]() {
          par_for(
              "Flat Container Array Perf", DevExecSpace(), 0, var_view.GetDim(4) - 1, 0,
              var_view.GetDim(3) - 1, 0, var_view.GetDim(2) - 1, 0,
              var_view.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                auto &var = var_view(l);
                var(k, j, i) *= var(k, j, i); // Do something trivial, square each term
              });
        });

    std::cout << "Mask: raw_array performance: " << time_raw_array << std::endl;
    std::cout << "Mask: iterate_variables performance: " << time_iterate_variables
              << std::endl;
    std::cout << "Mask: iterate_variables/raw_array "
              << time_iterate_variables / time_raw_array << std::endl;
    std::cout << "Mask: view_of_views performance: " << time_view_of_views << std::endl;
    std::cout << "Mask: view_of_views/raw_array " << time_view_of_views / time_raw_array
              << std::endl;
  }
  { // Grab variables by name and do timing tests
    std::vector<std::string> names({"v0", "v1", "v2", "v3", "v4", "v5"});
    auto var_view_named = container.PackVariables(names);

    auto init_view_of_views = [&]() {
      par_for(
          "Initialize ", DevExecSpace(), 0, var_view_named.GetDim(4) - 1, 0,
          var_view_named.GetDim(3) - 1, 0, var_view_named.GetDim(2) - 1, 0,
          var_view_named.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            var_view_named(l, k, j, i) =
                static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
          });
    };

    // Test performance of view of views VariablePack implementation
    double time_view_of_views =
        performance_test_wrapper(n_burn, n_perf, init_view_of_views, [&]() {
          par_for(
              "Flat Container Array Perf", DevExecSpace(), 0,
              var_view_named.GetDim(4) - 1, 0, var_view_named.GetDim(3) - 1, 0,
              var_view_named.GetDim(2) - 1, 0, var_view_named.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                var_view_named(l, k, j, i) *=
                    var_view_named(l, k, j, i); // Do something trivial, square each term
              });
        });

    // Test performance of view of views when the pack is built every time
    // tests caching
    double time_always_pack =
        performance_test_wrapper(n_burn, n_perf, init_view_of_views, [&]() {
          auto var_view_named = container.PackVariables(names);
          par_for(
              "Always pack Perf", DevExecSpace(), 0, var_view_named.GetDim(4) - 1, 0,
              var_view_named.GetDim(3) - 1, 0, var_view_named.GetDim(2) - 1, 0,
              var_view_named.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                var_view_named(l, k, j, i) *=
                    var_view_named(l, k, j, i); // Do something trivial, square each term
              });
        });

    std::cout << "Named: raw_array performance: " << time_raw_array << std::endl;
    std::cout << "Named: iterate_variables performance: " << time_iterate_variables
              << std::endl;
    std::cout << "Named: iterate_variables/raw_array "
              << time_iterate_variables / time_raw_array << std::endl;
    std::cout << "Named: view_of_views performance: " << time_view_of_views << std::endl;
    std::cout << "Named: view_of_views/raw_array " << time_view_of_views / time_raw_array
              << std::endl;
    std::cout << "Named: always pack performance: " << time_always_pack << std::endl;
    std::cout << "Named: always_pack/raw_array " << time_always_pack / time_raw_array
              << std::endl;
  }
  { // Grab some variables by name with indexing and do timing tests
    PackIndexMap imap;
    auto vsub = container.PackVariables({"v1", "v2", "v5"}, imap);

    auto init_view_of_views = [&]() {
      par_for(
          "Initialize ", DevExecSpace(), 0, vsub.GetDim(4) - 1, 0, vsub.GetDim(3) - 1, 0,
          vsub.GetDim(2) - 1, 0, vsub.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            vsub(l, k, j, i) = static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
          });
    };

    // Test performance of view of views VariablePack implementation
    double time_view_of_views =
        performance_test_wrapper(n_burn, n_perf, init_view_of_views, [&]() {
          par_for(
              "Flat Container Array Perf", DevExecSpace(), 0, vsub.GetDim(4) - 1, 0,
              vsub.GetDim(3) - 1, 0, vsub.GetDim(2) - 1, 0, vsub.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                vsub(l, k, j, i) *=
                    vsub(l, k, j, i); // Do something trivial, square each term
              });
        });
    // we only did half as many variables, so multiply by two for something quick and
    // dirty
    time_view_of_views *= 2.0;

    std::cout << "Indexed: raw_array performance: " << time_raw_array << std::endl;
    std::cout << "Indexed: iterate_variables performance: " << time_iterate_variables
              << std::endl;
    std::cout << "Indexed: iterate_variables/raw_array "
              << time_iterate_variables / time_raw_array << std::endl;
    std::cout << "Indexed: view_of_views performance: " << time_view_of_views
              << std::endl;
    std::cout << "Indexed: view_of_views/raw_array "
              << time_view_of_views / time_raw_array << std::endl;
  }
}
