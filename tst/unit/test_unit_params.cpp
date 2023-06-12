 //========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
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
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "kokkos_abstraction.hpp"
#include "interface/params.hpp"
#include "outputs/parthenon_hdf5.hpp"

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

#ifdef ENABLE_HDF5

TEST_CASE("A set of params can be dumped to file", "[params][output]") {
  GIVEN("A params object with a few kinds of objects") {
    Params params;

    Real scalar = 3.0;
    params.Add("scalar", scalar);

    std::vector<int> vector = {0, 1, 2};
    params.Add("vector", vector);

    parthenon::ParArray2D<Real> arr2d("myarr", 3, 2);
    auto arr2d_h = Kokkos::create_mirror_view(arr2d);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        arr2d_h(i, j) = 2 * i + j;
      }
    }
    Kokkos::deep_copy(arr2d, arr2d_h);
    params.Add("arr2d", arr2d);

    parthenon::HostArray2D<Real> hostarr("hostarr2d", 2, 3);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        hostarr(i, j) = 2 * i + j + 1;
      }
    }
    params.Add("hostarr2d", hostarr);

    THEN("We can output to hdf5") {
      const std::string filename = "params_test.h5";
      const std::string groupname = "params";
      const std::string prefix = "test_pkg";
      using namespace parthenon::HDF5;
      {
        H5F file = H5F::FromHIDCheck(
            H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
        auto group = MakeGroup(file, groupname);
        params.WriteAllToHDF5(prefix, group);
      }
      AND_THEN("We can directly read the relevant data from the hdf5 file") {
	H5F file = H5F::FromHIDCheck(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
	const H5O obj = H5O::FromHIDCheck(H5Oopen(file, groupname.c_str(), H5P_DEFAULT));

	Real in_scalar;
	HDF5ReadAttribute(obj, prefix + "/scalar", in_scalar);
	REQUIRE(std::abs(scalar - in_scalar) <= 1e-10);

	std::vector<int> in_vector;
	HDF5ReadAttribute(obj, prefix + "/vector", in_vector);
	for (int i = 0; i < vector.size(); ++i) {
	  REQUIRE(in_vector[i] == vector[i]);
	}
	
	// deliberately the wrong size
	parthenon::ParArray2D<Real> in_arr2d("myarr", 1, 1);
	
      }
    }
  }
}

#endif // ENABLE_HDF5
