//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "utils/concepts_lite.hpp"

template <class T, class VAL_TYPE>
bool SatisfiesContainerRequirements(T &&in, VAL_TYPE val, int size_in) {
  // Check that the value_type of the container is what we would expect
  using cont_val_type = decltype(contiguous_container::value_type(in));
  bool test = std::is_same<cont_val_type, VAL_TYPE>::value;

  int size = contiguous_container::size(in);
  test = test && (size == size_in);

  // Check that we can access the data and they all have value val
  int *pin = contiguous_container::data(in);
  for (int i = 0; i < size; ++i)
    test = test && (val == pin[i]);

  return test;
}

TEST_CASE("Check that the contiguous container concept works", "") {
  GIVEN("Some containers and some data") {
    int SIZE = 10;
    int val = 2;

    int my_int = val;
    std::vector<int> my_vec(SIZE, val);
    // This should also work fine for objects defined on device, but
    // it is more work to write a generic function for checking values
    // via pointer on device and on host
    Kokkos::View<int *, parthenon::LayoutWrapper, parthenon::HostMemSpace> my_view("Test",
                                                                                   SIZE);
    // We expect that ParArrays should also conform to the contiguous_container
    // interface
    parthenon::ParArray1D<int>::host_mirror_type my_pararr("Test", SIZE);

    for (int i = 0; i < SIZE; ++i) {
      my_pararr(i) = val;
      my_view(i) = val;
    }

    REQUIRE(SatisfiesContainerRequirements(my_int, val, 1));
    REQUIRE(SatisfiesContainerRequirements(my_vec, val, SIZE));
    REQUIRE(SatisfiesContainerRequirements(my_view, val, SIZE));
    REQUIRE(SatisfiesContainerRequirements(my_pararr, val, SIZE));
  }
}
