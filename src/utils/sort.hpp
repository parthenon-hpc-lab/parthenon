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

#ifdef KOKKOS_ENABLE_CUDA
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

namespace parthenon {

template <class Key, class KeyComparator>
void sort(ParArray1D<Key> data, KeyComparator comparator, size_t min_idx,
          size_t max_idx) {
  PARTHENON_DEBUG_REQUIRE(min_idx >= 0 && min_idx < data.extent(0),
                          "Invalid minimum sort index!");
  PARTHENON_DEBUG_REQUIRE(max_idx >= 0 && max_idx < data.extent(0),
                          "Invalid maximum sort index!");
#ifdef KOKKOS_ENABLE_CUDA
  thrust::device_ptr<Key> first_d = thrust::device_pointer_cast(data.data()) + min_idx;
  thrust::device_ptr<Key> last_d = thrust::device_pointer_cast(data.data()) + max_idx + 1;
  thrust::sort(first_d, last_d, comparator);
#else
  std::sort(data.data() + min_idx, data.data() + max_idx + 1, comparator);
#endif // KOKKOS_ENABLE_CUDA
}

template <class Key>
void sort(ParArray1D<Key> data, size_t min_idx, size_t max_idx) {
  PARTHENON_DEBUG_REQUIRE(min_idx >= 0 && min_idx < data.extent(0),
                          "Invalid minimum sort index!");
  PARTHENON_DEBUG_REQUIRE(max_idx >= 0 && max_idx < data.extent(0),
                          "Invalid maximum sort index!");
#ifdef KOKKOS_ENABLE_CUDA
  thrust::device_ptr<Key> first_d = thrust::device_pointer_cast(data.data()) + min_idx;
  thrust::device_ptr<Key> last_d = thrust::device_pointer_cast(data.data()) + max_idx + 1;
  thrust::sort(first_d, last_d);
#else
  std::sort(data.data() + min_idx, data.data() + max_idx + 1);
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
