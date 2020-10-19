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
//=======================================================================================
#ifndef UTILS_PARTITION_STL_CONTAINERS_HPP_
#define UTILS_PARTITION_STL_CONTAINERS_HPP_

#include <string>
#include <vector>

#include "error_checking.hpp"

namespace parthenon {
namespace partition {
// Note the interface here has objects in partition are now
// COPIED, not pointed at.
// Change is due to using std::shared_ptr<MeshBlock>
template <typename T>
using Partition_t = std::vector<std::vector<T>>;

namespace partition_impl {
// x/y rounded up
// See discussion here for limitations and alternatives
// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
KOKKOS_INLINE_FUNCTION
int IntCeil(int x, int y) {
  PARTHENON_DEBUG_REQUIRE(x >= 0, "ceil only works for x >= 0");
  PARTHENON_DEBUG_REQUIRE(y > 0, "ceil only works for y > 0");
  // avoids overflow in x+y
  return x > 0 ? 1 + ((x - 1) / y) : 0;
}
} // namespace partition_impl

// Takes container of elements and fills partitions
// of size N with pointers to elements.
// Assumes Container_t has STL-style iterators defined
template <typename T, template <class...> typename Container_t, class... extra>
auto ToSizeN(Container_t<T, extra...> &container, const int N) {
  using std::to_string;
  using namespace partition_impl;

  PARTHENON_REQUIRE_THROWS(N > 0, "Your partition must be at least size 1");

  int nelements = container.size();
  int npartitions = IntCeil(nelements, N);

  Partition_t<T> partitions(npartitions);
  for (auto &p : partitions) {
    p.reserve(N);
    p.clear();
  }

  int p = 0;
  int b = 0;
  for (auto &element : container) {
    partitions[p].push_back(element);
    if (++b >= N) {
      ++p;
      b = 0;
    }
  }
  return partitions;
}

// Takes container of elements and fills N partitions
// with pointers to elements.
// Assumes Container_t has STL-style iterators defined
template <typename T, template <class...> typename Container_t, class... extra>
auto ToNPartitions(Container_t<T, extra...> &container, const int N) {
  using namespace partition_impl;
  int nelements = container.size();
  int partition_size = IntCeil(nelements, N);
  return ToSizeN(container, partition_size);
}
} // namespace partition
} // namespace parthenon

#endif // UTILS_PARTITION_STL_CONTAINERS_HPP_
