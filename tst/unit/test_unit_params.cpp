//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "interface/params.hpp"
#include "kokkos_abstraction.hpp"
#include "openPMD/Series.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "outputs/parthenon_opmd.hpp"
#include "outputs/restart_hdf5.hpp"
#include "outputs/restart_opmd.hpp"
#include "parthenon_array_generic.hpp"

using parthenon::Params;
using parthenon::Real;

TEST_CASE("Add, Get, and Update are called", "[Add,Get,Update]") {
  GIVEN("A key with some value") {
    Params params;
    std::string key = "test_key";
    double value = -2.0;
    THEN("we can add it to Params") {
      params.Add(key, value);
      AND_THEN("and retreive it with Get") {
        double output = params.Get<double>(key);
        REQUIRE(output == Approx(value));
      }
      WHEN("Trying to update a non-mutable param") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.Update<double>(key, 2.0), std::runtime_error);
        }
      }
      WHEN("the same key is provided a second time") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.Add(key, value), std::runtime_error);
        }
      }
      WHEN("attempting to get the key but casting to a different type") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.Get<int>(key), std::runtime_error);
        }
      }
      WHEN("attempting to get the pointer with GetMutable") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.GetMutable<double>(key), std::runtime_error);
        }
      }
    }
    WHEN("We add it to params as mutable") {
      params.Add(key, value, true);
      THEN("We can retrieve the pointer to the object with GetMutable") {
        double *pval = params.GetMutable<double>(key);
        REQUIRE(*pval == Approx(value));
        AND_THEN("We can modify the value by dereferencing the pointer") {
          double new_val = 5;
          *pval = new_val;
          AND_THEN("params.get reflects the new value") {
            double output = params.Get<double>(key);
            REQUIRE(output == Approx(new_val));
          }
        }
      }
      THEN("We can update the mutable param") {
        params.Update<double>(key, 2.0);
        AND_THEN("The new value is reflected in Get") {
          double output = params.Get<double>(key);
          REQUIRE(output == Approx(2.0));
        }
      }
      WHEN("trying to Update with a wrong type") {
        THEN("an error is thrown") {
          REQUIRE_THROWS_AS(params.Update<int>(key, 2), std::runtime_error);
        }
      }
    }
  }

  GIVEN("An empty params structure") {
    Params params;
    std::string non_existent_key = "key";
    WHEN(" attempting to get a key that does not exist ") {
      THEN("an error is thrown") {
        REQUIRE_THROWS_AS(params.Get<double>(non_existent_key), std::runtime_error);
      }
    }
    WHEN(" attempting to update a key that does not exist ") {
      THEN("an error is thrown") {
        REQUIRE_THROWS_AS(params.Update<double>(non_existent_key, 2.0),
                          std::runtime_error);
      }
    }
  }
}

TEST_CASE("reset is called", "[reset]") {
  GIVEN("A key is added") {
    Params params;
    std::string key = "test_key";
    double value = -2.0;
    params.Add(key, value);
    WHEN("the params are reset") {
      params.reset();
      REQUIRE_THROWS_AS(params.Get<double>(key), std::runtime_error);
    }
  }
}

TEST_CASE("when hasKey is called", "[hasKey]") {
  GIVEN("A key is added") {
    Params params;
    std::string key = "test_key";
    double value = -2.0;
    params.Add(key, value);

    REQUIRE(params.hasKey(key) == true);

    WHEN("the params are reset") {
      params.reset();
      REQUIRE(params.hasKey(key) == false);
    }
  }
}

#if defined(ENABLE_HDF5) && defined(PARTHENON_ENABLE_OPENPMD)
using parthenon::RestartReaderHDF5;
using parthenon::RestartReaderOPMD;
using OutputTypes = std::tuple<RestartReaderHDF5, RestartReaderOPMD>;
#elif defined(ENABLE_HDF5)
using parthenon::RestartReaderHDF5;
using OutputTypes = std::tuple<RestartReaderHDF5>;
#elif defined(PARTHENON_ENABLE_OPENPMD)
using parthenon::RestartReaderOPMD;
using OutputTypes = std::tuple<RestartReaderOPMD>;
#else
using OutputTypes = std::tuple<>;
#endif

TEMPLATE_LIST_TEST_CASE("A set of params can be dumped to file", "[params][output]",
                        OutputTypes) {
  GIVEN("A params object with a few kinds of objects") {
    Params params;
    const auto restart = Params::Mutability::Restart;
    const auto only_mutable = Params::Mutability::Mutable;

    Real scalar = 3.0;
    params.Add("scalar", scalar, restart);

    bool boolscalar = false;
    params.Add("boolscalar", boolscalar, restart);

    std::vector<int> vector = {0, 1, 2};
    params.Add("vector", vector, only_mutable);

    parthenon::ParArray2D<Real> arr2d("myarr", 3, 2);
    auto arr2d_h = Kokkos::create_mirror_view(arr2d);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        arr2d_h(i, j) = 2 * i + j;
      }
    }
    Kokkos::deep_copy(arr2d, arr2d_h);
    params.Add("arr2d", arr2d);

    parthenon::HostArray2D<Real> hostarr2d("hostarr2d", 2, 3);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        hostarr2d(i, j) = 2 * i + j + 1;
      }
    }
    params.Add("hostarr2d", hostarr2d, restart);

    THEN("We can output") {
      std::string filename;
      const std::string groupname = "Params";
      const std::string prefix = "test_pkg";
      if constexpr (std::is_same_v<RestartReaderHDF5, TestType>) {
        using namespace parthenon::HDF5;
        filename = "params_test.h5";

        H5F file = H5F::FromHIDCheck(
            H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
        auto group = MakeGroup(file, groupname);
        params.WriteAllToHDF5(prefix, group);
      } else if constexpr (std::is_same_v<RestartReaderOPMD, TestType>) {
        filename = ("params_test.%05T.bp");
        auto series = openPMD::Series(filename, openPMD::Access::CREATE);
        series.setIterationEncoding(openPMD::IterationEncoding::fileBased);
        auto it = series.iterations[0];
        parthenon::OpenPMDUtils::WriteAllParams(params, prefix, &it);
      } else {
        FAIL("This logic is flawed. I should not be here.");
      }
      AND_THEN("We can directly read the relevant data from the file") {
        Real in_scalar;
        std::vector<int> in_vector;
        // deliberately the wrong size
        parthenon::ParArray2D<Real> in_arr2d("myarr", 1, 1);
        parthenon::HostArray2D<Real> in_hostarr2d("hostarr2d", 2, 3);

        if constexpr (std::is_same_v<RestartReaderHDF5, TestType>) {
          H5F file =
              H5F::FromHIDCheck(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
          const H5O obj =
              H5O::FromHIDCheck(H5Oopen(file, groupname.c_str(), H5P_DEFAULT));

          HDF5ReadAttribute(obj, prefix + "/scalar", in_scalar);
          HDF5ReadAttribute(obj, prefix + "/vector", in_vector);
          HDF5ReadAttribute(obj, prefix + "/arr2d", in_arr2d);
          HDF5ReadAttribute(obj, prefix + "/hostarr2d", in_hostarr2d);
        } else if constexpr (std::is_same_v<RestartReaderOPMD, TestType>) {
          auto series = openPMD::Series(filename, openPMD::Access::READ_ONLY);
          auto it = series.iterations[0];
          // Note that we're explicitly using `delim` here which tests the character
          // replacement of '/' in the WriteAllParams function.
          using parthenon::OpenPMDUtils::delim;

          in_scalar =
              it.getAttribute(groupname + delim + prefix + delim + "scalar").get<Real>();

          in_vector = it.getAttribute(groupname + delim + prefix + delim + "vector")
                          .get<std::vector<int>>();

          // Technically, we're not reading "directly" here but the restart reader ctor
          // literally just opens the file.
          auto resfile = RestartReaderOPMD(filename.c_str());
          auto &in_arr2d_v = in_arr2d.KokkosView();
          resfile.RestoreViewAttribute(groupname + delim + prefix + delim + "arr2d",
                                       in_arr2d_v);

          auto &in_hostarr2d_v = in_hostarr2d.KokkosView();
          resfile.RestoreViewAttribute(groupname + delim + prefix + delim + "hostarr2d",
                                       in_hostarr2d_v);
        }
        REQUIRE(scalar == in_scalar);

        for (int i = 0; i < vector.size(); ++i) {
          REQUIRE(in_vector[i] == vector[i]);
        }

        REQUIRE(in_arr2d.extent_int(0) == arr2d.extent_int(0));
        REQUIRE(in_arr2d.extent_int(1) == arr2d.extent_int(1));
        int nwrong = 1;
        parthenon::par_reduce(
            parthenon::loop_pattern_mdrange_tag, "test arr2d", parthenon::DevExecSpace(),
            0, arr2d.extent_int(0) - 1, 0, arr2d.extent_int(1) - 1,
            KOKKOS_LAMBDA(const int i, const int j, int &nw) {
              nw += (in_arr2d(i, j) != arr2d(i, j));
            },
            nwrong);
        REQUIRE(nwrong == 0);

        REQUIRE(in_hostarr2d.extent_int(0) == hostarr2d.extent_int(0));
        REQUIRE(in_hostarr2d.extent_int(1) == hostarr2d.extent_int(1));
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 3; ++j) {
            REQUIRE(hostarr2d(i, j) == in_hostarr2d(i, j));
          }
        }
      }

      AND_THEN("We can restart a params object from the file") {
        Params rparams;

        // init the params object to restart into
        Real test_scalar = 0.0;
        rparams.Add("scalar", test_scalar, restart);

        bool test_bool = true;
        rparams.Add("boolscalar", test_bool, restart);

        std::vector<int> test_vector;
        rparams.Add("vector", test_vector, only_mutable);

        parthenon::ParArray2D<Real> test_arr2d("myarr", 1, 1);
        rparams.Add("arr2d", test_arr2d);

        parthenon::HostArray2D<Real> test_hostarr("hostarr2d", 1, 1);
        rparams.Add("hostarr2d", test_hostarr, restart);

        if constexpr (std::is_same_v<RestartReaderHDF5, TestType>) {
          H5F file =
              H5F::FromHIDCheck(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
          const H5G obj =
              H5G::FromHIDCheck(H5Oopen(file, groupname.c_str(), H5P_DEFAULT));
          rparams.ReadFromRestart(prefix, obj);
        } else if constexpr (std::is_same_v<RestartReaderOPMD, TestType>) {
          auto resfile = RestartReaderOPMD(filename.c_str());
          resfile.ReadParams(prefix, rparams);
        }

        AND_THEN("The values for the restartable params are updated to match the file") {
          auto test_scalar = rparams.Get<Real>("scalar");
          REQUIRE(test_scalar == scalar);

          auto test_bool = rparams.Get<bool>("boolscalar");
          REQUIRE(test_bool == boolscalar);

          auto test_hostarr = rparams.Get<parthenon::HostArray2D<Real>>("hostarr2d");
          REQUIRE(test_hostarr.extent_int(0) == hostarr2d.extent_int(0));
          REQUIRE(test_hostarr.extent_int(1) == hostarr2d.extent_int(1));
          for (int i = 0; i < hostarr2d.extent_int(0); ++i) {
            for (int j = 0; j < hostarr2d.extent_int(1); ++j) {
              REQUIRE(test_hostarr(i, j) == hostarr2d(i, j));
            }
          }
        }
        AND_THEN("The values for non-restartable params have not been updated") {
          auto test_vector = rparams.Get<std::vector<int>>("vector");
          REQUIRE(test_vector.size() == 0);
          auto test_arr2d = rparams.Get<parthenon::ParArray2D<Real>>("arr2d");
          REQUIRE(test_arr2d.extent_int(0) == 1);
          REQUIRE(test_arr2d.extent_int(1) == 1);
        }
      }
    }
  }
}
