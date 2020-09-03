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
#ifndef UTILS_PARTITION_CONTAINERS_HPP_
#define UTILS_PARTITION_CONTAINERS_HPP_

#include <string>
#include <vector>

#include "error_checking.hpp"

namespace parthenon {
namespace Partition {
// TODO(JMM): The templated type safety with T* and T may need to be
// changed if we move to sufficiently general container objects.
template <typename T>
using Partition_t = std::vector<std::vector<T *>>;

// x/y rounded up
// See discussion here for limitations and alternatives
// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
KOKKOS_INLINE_FUNCTION
int IntCeil(int x, int y) {
  PARTHENON_DEBUG_REQUIRE(x > 0 && y > 0, "ceil only works for x,y > 0");
  // avoids overflow in x+y
  return 1 + ((x - 1) / y);
}

// Takes container of elements and fills partitions
// of size N with pointers to elements.
// Assumes Container_t has STL-style iterators defined
template <typename Container_t, typename T>
void ToSizeN(Container_t &container, const int N, Partition_t<T> &partitions) {
  using std::to_string;

  PARTHENON_REQUIRE_THROWS(N > 0, "You must have at least 1 partition");
  int nelements = container.size();
  PARTHENON_REQUIRE_THROWS(nelements > 0,
                           "You must have at least 1 element to partition");
  std::string msg = ("Cannot partition " + to_string(nelements) +
                     " elements into partitions of size " + to_string(N) + ".");
  PARTHENON_REQUIRE_THROWS(nelements >= N, msg);

  int npartitions = IntCeil(nelements, N);
  partitions.resize(npartitions);
  for (auto &p : partitions) {
    p.reserve(N);
    p.clear();
  }

  int p = 0;
  int b = 0;
  for (auto &element : container) {
    partitions[p].push_back(&element);
    if (++b >= N) {
      ++p;
      b = 0;
    }
  }
}

// Takes container of elements and fills N partitions
// with pointers to elements.
// Assumes Container_t has STL-style iterators defined
template <typename Container_t, typename T>
void ToNPartitions(Container_t &container,
                   const int N, Partition_t<T> &partitions) {
  int nelements = container.size();
  int partition_size = IntCeil(nelements, N);
  ToSizeN(container, partition_size, partitions);
}
} // namespace Partition
} // namespace parthenon

#endif // UTILS_PARTITION_CONTAINERS_HPP_
