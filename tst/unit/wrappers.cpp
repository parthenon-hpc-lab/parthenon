#include <catch2/catch.hpp>
#include "../../src/athena.hpp"

template<class T = int>
bool test_wrapper(T loop_pattern) {
  const int N = 32;
  AthenaArray3D<> arr_dev("device",N,N,N);
  auto arr_host = Kokkos::create_mirror_view(arr_dev);

  for (int k = 0; k < N; k++)
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        arr_host(k,j,i) = 1.0;
  

  return std::is_same<T, int>::value;
  //return true;
}

TEST_CASE("default", "[wrapper]") {
  REQUIRE(test_wrapper(0) == true);
}

TEST_CASE("other", "[wrapper]") {
  REQUIRE(test_wrapper(loop_pattern_mdrange_tag) == false);
}
