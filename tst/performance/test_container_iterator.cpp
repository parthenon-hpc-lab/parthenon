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
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "athena.hpp"
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
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

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
