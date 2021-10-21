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
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/meshblock_data_iterator.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

using parthenon::CellVariable;
using parthenon::CellVariableVector;
using parthenon::DevExecSpace;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::MeshBlockData;
using parthenon::MeshBlockDataIterator;
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

// File scope variables
constexpr int N = 32; // Dimensions of blocks
constexpr int Nvar = 10;
constexpr int N_kernels_to_launch_per_test = 100;

template <typename InitFunc, typename PerfFunc>
void performance_test_wrapper(const std::string &test_name, InitFunc init_func,
                              PerfFunc perf_func) {
  BENCHMARK_ADVANCED(test_name.c_str())(Catch::Benchmark::Chronometer meter) {
    init_func();
    Kokkos::fence();
    meter.measure([&]() {
      for (int i = 0; i < N_kernels_to_launch_per_test; ++i) {
        perf_func();
      }
      Kokkos::fence();
    });
  };
}

static MeshBlockData<Real> createTestContainer() {
  // Make a container for testing performance
  MeshBlockData<Real> container;
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
  return container;
}

// std::function<void()> createLambdaRaw(ParArrayND<Real> &raw_array) {
template <class T>
std::function<void()> createLambdaRaw(T &raw_array) {
  return [&]() {
    par_for(
        DEFAULT_LOOP_PATTERN, "Initialize ", DevExecSpace(), 0, Nvar - 1, 0, N - 1, 0,
        N - 1, 0, N - 1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          raw_array(l, k, j, i) =
              static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
        });
  };
}

std::function<void()> createLambdaContainer(MeshBlockData<Real> &container) {
  return [&]() {
    const CellVariableVector<Real> &cv = container.GetCellVariableVector();
    for (int n = 0; n < cv.size(); n++) {
      ParArrayND<Real> v = cv[n]->data;
      par_for(
          DEFAULT_LOOP_PATTERN, "Initialize variables", DevExecSpace(), 0,
          v.GetDim(4) - 1, 0, v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            v(l, k, j, i) = static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
          });
    }
    return container;
  };
}

std::function<void()> createLambdaContainerCellVar(MeshBlockData<Real> &container,
                                                   std::vector<std::string> &names) {
  return [&]() {
    for (int n = 0; n < names.size(); n++) {
      CellVariable<Real> &v = container.Get(names[n]);
      auto data = v.data;
      par_for(
          DEFAULT_LOOP_PATTERN, "Initialize variables", DevExecSpace(), 0,
          v.GetDim(4) - 1, 0, v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
          KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
            data(l, k, j, i) = static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
          });
    }
    return container;
  };
}

std::function<void()>
createLambdaInitViewOfViews(parthenon::VariablePack<Real> &var_view) {
  return [&]() {
    par_for(
        DEFAULT_LOOP_PATTERN, "Initialize ", DevExecSpace(), 0, var_view.GetDim(4) - 1, 0,
        var_view.GetDim(3) - 1, 0, var_view.GetDim(2) - 1, 0, var_view.GetDim(1) - 1,
        KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
          var_view(l, k, j, i) = static_cast<Real>((l + 1) * (k + 1) * (j + 1) * (i + 1));
        });
  };
}

TEST_CASE("Catch2 Container Iterator Performance",
          "[MeshBlockDataIterator][performance]") {
  SECTION("Raw Array") {
    GIVEN("A raw ParArray4d") {
      // Make a raw ParArray4D for closest to bare metal looping
      ParArray4D<Real> raw_array("raw_array", Nvar, N, N, N);
      auto init_raw_array = createLambdaRaw(raw_array);
      // Make a function for initializing the raw ParArray4D
      performance_test_wrapper("Mask: Raw Array Perf", init_raw_array, [&]() {
        par_for(
            DEFAULT_LOOP_PATTERN, "Raw Array Perf", DevExecSpace(), 0, Nvar - 1, 0, N - 1,
            0, N - 1, 0, N - 1,
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
              raw_array(l, k, j, i) *=
                  raw_array(l, k, j, i); // Do something trivial, square each term
            });
      });
    } // GIVEN
    GIVEN("A ParArrayNd") {
      // Make a raw ParArray4D for closest to bare metal looping
      ParArrayND<Real> nd_array("nd_array", Nvar, N, N, N);
      auto init_nd_array = createLambdaRaw(nd_array);
      // Make a function for initializing the raw ParArray4D
      performance_test_wrapper("Mask: Nd Array Perf", init_nd_array, [&]() {
        par_for(
            DEFAULT_LOOP_PATTERN, "Nd Array Perf", DevExecSpace(), 0, Nvar - 1, 0, N - 1,
            0, N - 1, 0, N - 1,
            KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
              nd_array(l, k, j, i) *=
                  nd_array(l, k, j, i); // Do something trivial, square each term
            });
      });
    } // GIVEN
  }   // SECTION

  SECTION("Iterate Variables") {
    GIVEN("A container.") {
      MeshBlockData<Real> container = createTestContainer();
      auto init_container = createLambdaContainer(container);

      // Make a function for initializing the container variables
      performance_test_wrapper("Mask: Iterate Variables Perf", init_container, [&]() {
        const CellVariableVector<Real> &cv = container.GetCellVariableVector();
        for (int n = 0; n < cv.size(); n++) {
          ParArrayND<Real> v = cv[n]->data;
          par_for(
              DEFAULT_LOOP_PATTERN, "Iterate Variables Perf", DevExecSpace(), 0,
              v.GetDim(4) - 1, 0, v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                v(l, k, j, i) *= v(l, k, j, i); // Do something trivial, square each term
              });
        }
      });
    } // GIVEN
    GIVEN("A container cellvar.") {
      MeshBlockData<Real> container = createTestContainer();
      std::vector<std::string> names({"v0", "v1", "v2", "v3", "v4", "v5"});
      auto init_container = createLambdaContainerCellVar(container, names);

      // Make a function for initializing the container variables
      performance_test_wrapper("Mask: Iterate Variables Perf", init_container, [&]() {
        for (int n = 0; n < names.size(); n++) {
          CellVariable<Real> &v = container.Get(names[n]);
          // Do something trivial, square each term
          par_for(
              DEFAULT_LOOP_PATTERN, "Iterate CellVariables Perf", DevExecSpace(), 0,
              v.GetDim(4) - 1, 0, v.GetDim(3) - 1, 0, v.GetDim(2) - 1, 0, v.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                v.data(l, k, j, i) *= v.data(l, k, j, i);
              });
        }
      });
    } // GIVEN
  }   // SECTION

  SECTION("View of Views") {
    GIVEN("A container.") {
      MeshBlockData<Real> container = createTestContainer();
      WHEN("The view of views does not have any names.") {
        parthenon::VariablePack<Real> var_view =
            container.PackVariables({Metadata::Independent});
        auto init_view_of_views = createLambdaInitViewOfViews(var_view);
        // Test performance of view of views VariablePack implementation
        performance_test_wrapper("Mask: View of Views Perf", init_view_of_views, [&]() {
          par_for(
              DEFAULT_LOOP_PATTERN, "Flat Container Array Perf", DevExecSpace(), 0,
              var_view.GetDim(4) - 1, 0, var_view.GetDim(3) - 1, 0,
              var_view.GetDim(2) - 1, 0, var_view.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                auto &var = var_view(l);
                var(k, j, i) *= var(k, j, i); // Do something trivial, square each term
              });
        });
      } // WHEN

      WHEN("The view of views is implemented with names.") {
        std::vector<std::string> names({"v0", "v1", "v2", "v3", "v4", "v5"});
        parthenon::VariablePack<Real> var_view_named = container.PackVariables(names);
        auto init_view_of_views = createLambdaInitViewOfViews(var_view_named);
        // Test performance of view of views VariablePack implementation
        performance_test_wrapper("Named: View of views", init_view_of_views, [&]() {
          par_for(
              DEFAULT_LOOP_PATTERN, "Flat Container Array Perf", DevExecSpace(), 0,
              var_view_named.GetDim(4) - 1, 0, var_view_named.GetDim(3) - 1, 0,
              var_view_named.GetDim(2) - 1, 0, var_view_named.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                var_view_named(l, k, j, i) *=
                    var_view_named(l, k, j, i); // Do something trivial, square each term
              });
        });
      } // WHEN

      // The pack is built every time, tests caching
      WHEN("The view of views is implemented with names and construction of Pack "
           "Variables is included in the timing.") {
        std::vector<std::string> names({"v0", "v1", "v2", "v3", "v4", "v5"});
        auto var_view_named = container.PackVariables(names);
        auto init_view_of_views = createLambdaInitViewOfViews(var_view_named);
        performance_test_wrapper("View of views", init_view_of_views, [&]() {
          auto var_view_named = container.PackVariables(names);
          par_for(
              DEFAULT_LOOP_PATTERN, "Always pack Perf", DevExecSpace(), 0,
              var_view_named.GetDim(4) - 1, 0, var_view_named.GetDim(3) - 1, 0,
              var_view_named.GetDim(2) - 1, 0, var_view_named.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                var_view_named(l, k, j, i) *=
                    var_view_named(l, k, j, i); // Do something trivial, square each term
              });
        });
      } // WHEN

      WHEN("The view of views is implemented with names and indices.") {
        PackIndexMap imap;
        auto vsub = container.PackVariables({"v0", "v1", "v2", "v3", "v4", "v5"}, imap);
        auto init_view_of_views = createLambdaInitViewOfViews(vsub);
        performance_test_wrapper("View of views", init_view_of_views, [&]() {
          par_for(
              DEFAULT_LOOP_PATTERN, "Flat Container Array Perf", DevExecSpace(), 0,
              vsub.GetDim(4) - 1, 0, vsub.GetDim(3) - 1, 0, vsub.GetDim(2) - 1, 0,
              vsub.GetDim(1) - 1,
              KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
                vsub(l, k, j, i) *=
                    vsub(l, k, j, i); // Do something trivial, square each term
              });
        });
      }
    } // GIVEN
  }   // SECTION
} // TEST_CASE
