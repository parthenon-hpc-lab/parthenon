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
#ifdef KOKKOS_ENABLE_CUDA
#ifdef __clang__
  PARTHENON_FAIL("sort is using thrust and there exists an incompatibility with clang, "
                 "see https://github.com/lanl/parthenon/issues/647 for more details. We "
                 "won't fix it because eventually the Parthenon sort should make use of "
                 "Kokkos::sort once a performant implementation is availabe. If you see "
                 "this message and need sort on CUDA devices with clang compiler please "
                 "get in touch by opening an issue on the Parthenon GitHub repo.");
#else
  thrust::device_ptr<Key> first_d = thrust::device_pointer_cast(data.data()) + min_idx;
  thrust::device_ptr<Key> last_d = thrust::device_pointer_cast(data.data()) + max_idx + 1;
  thrust::sort(first_d, last_d, comparator);
#endif
#else
  if (std::is_same<DevExecSpace, HostExecSpace>::value) {
    std::sort(data.data() + min_idx, data.data() + max_idx + 1, comparator);
  } else {
    PARTHENON_FAIL("sort is not supported outside of CPU or NVIDIA GPU. If you need sort "
                   "support on other devices, e.g., AMD or Intel GPUs, please get in "
                   "touch by opening an issue on the Parthenon GitHub.");
  }
#endif // KOKKOS_ENABLE_CUDA
}

template <class Key>
void sort(ParArray1D<Key> data, size_t min_idx, size_t max_idx) {
  PARTHENON_DEBUG_REQUIRE(min_idx < data.extent(0), "Invalid minimum sort index!");
  PARTHENON_DEBUG_REQUIRE(max_idx < data.extent(0), "Invalid maximum sort index!");
#ifdef KOKKOS_ENABLE_CUDA
#ifdef __clang__
  PARTHENON_FAIL("sort is using thrust and there exists an incompatibility with clang, "
                 "see https://github.com/lanl/parthenon/issues/647 for more details. We "
                 "won't fix it because eventually the Parthenon sort should make use of "
                 "Kokkos::sort once a performant implementation is availabe. If you see "
                 "this message and need sort on CUDA devices with clang compiler please "
                 "get in touch by opening an issue on the Parthenon GitHub repo.");
#else
  thrust::device_ptr<Key> first_d = thrust::device_pointer_cast(data.data()) + min_idx;
  thrust::device_ptr<Key> last_d = thrust::device_pointer_cast(data.data()) + max_idx + 1;
  thrust::sort(first_d, last_d);
#endif
#else
  if (std::is_same<DevExecSpace, HostExecSpace>::value) {
    std::sort(data.data() + min_idx, data.data() + max_idx + 1);
  } else {
    PARTHENON_FAIL("sort is not supported outside of CPU or NVIDIA GPU. If you need sort "
                   "support on other devices, e.g., AMD or Intel GPUs, please get in "
                   "touch by opening an issue on the Parthenon GitHub.");
  }
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
