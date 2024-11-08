//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_SORT_HPP_
#define UTILS_SORT_HPP_

//! \file sort.hpp
//  \brief Contains functions for sorting data according to a provided comparator
//  See tst/unit/test_unit_sort.cpp for example usage.

#include "defs.hpp"
#include "parthenon_arrays.hpp"

#include <Kokkos_Sort.hpp>

#ifdef KOKKOS_ENABLE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

#include <algorithm>

namespace parthenon {

// Returns the upper bound (or the array size if value has not been found)
// Could/Should be replaced with a Kokkos std version once available (currently schedule
// for 4.2 release).
// Note, the API follows the std::upper_bound with the difference of taking an
// array/view as input rather than first and last Iterators, and returning an index
// rather than an Iterator.
template <class T>
KOKKOS_INLINE_FUNCTION int upper_bound(const T &arr, Real val) {
  int l = 0;
  int r = arr.extent_int(0);
  int m;
  while (l < r) {
    m = l + (r - l) / 2;
    if (val >= arr(m)) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  if (l < arr.extent_int(0) && val >= arr(l)) {
    l++;
  }
  return l;
}

template <class Key, class KeyComparator>
void sort(ParArray1D<Key> data, KeyComparator comparator, size_t min_idx,
          size_t max_idx) {
  PARTHENON_DEBUG_REQUIRE(min_idx < data.extent(0), "Invalid minimum sort index!");
  PARTHENON_DEBUG_REQUIRE(max_idx < data.extent(0), "Invalid maximum sort index!");
#if defined(KOKKOS_ENABLE_CUDA) && !defined(__clang__)
  thrust::device_ptr<Key> first_d = thrust::device_pointer_cast(data.data()) + min_idx;
  thrust::device_ptr<Key> last_d = thrust::device_pointer_cast(data.data()) + max_idx + 1;
  thrust::sort(first_d, last_d, comparator);
#else
  auto sub_data = Kokkos::subview(data, std::make_pair(min_idx, max_idx + 1));
  Kokkos::sort(sub_data, comparator);
#endif // KOKKOS_ENABLE_CUDA
}

template <class Key>
void sort(ParArray1D<Key> data, size_t min_idx, size_t max_idx) {
  PARTHENON_DEBUG_REQUIRE(min_idx < data.extent(0), "Invalid minimum sort index!");
  PARTHENON_DEBUG_REQUIRE(max_idx < data.extent(0), "Invalid maximum sort index!");
#if defined(KOKKOS_ENABLE_CUDA) && !defined(__clang__)
  thrust::device_ptr<Key> first_d = thrust::device_pointer_cast(data.data()) + min_idx;
  thrust::device_ptr<Key> last_d = thrust::device_pointer_cast(data.data()) + max_idx + 1;
  thrust::sort(first_d, last_d);
#else
  auto sub_data = Kokkos::subview(data, std::make_pair(min_idx, max_idx + 1));
  Kokkos::sort(sub_data);
#endif // KOKKOS_ENABLE_CUDA
}

template <class Key, class KeyComparator>
void sort(ParArray1D<Key> data, KeyComparator comparator) {
  sort(data, comparator, 0, data.extent(0) - 1);
}

template <class Key>
void sort(ParArray1D<Key> data) {
  sort(data, 0, data.extent(0) - 1);
}

} // namespace parthenon

#endif // UTILS_SORT_HPP_
