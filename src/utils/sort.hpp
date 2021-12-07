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
//  See src/interface/swarm.hpp for an example KeyIterator and KeyComparator, SwarmKey and
//  SwarmKeyCompare

#include "defs.hpp"
#include "parthenon_arrays.hpp"

#ifdef KOKKOS_ENABLE_CUDA
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif

namespace parthenon {

template<class KeyIterator, class KeyComparator>
void sort(KeyIterator first, KeyIterator last, KeyComparator comparator) {
#ifdef KOKKOS_ENABLE_CUDA
  thrust::device_ptr<KeyIterator> first_d = thrust::device_pointer_cast(first);
  thrust::device_ptr<KeyIterator> last_d = thrust::device_pointer_cast(last);
  thrust::sort(first_d, last_d, comparator);
#else
  std::sort(first, last, comparator);
#endif // KOKKOS_ENABLE_CUDA
}

template<class KeyIterator>
void sort(KeyIterator first, KeyIterator last) {
#ifdef KOKKOS_ENABLE_CUDA
  thrust::device_ptr<KeyIterator> first_d = thrust::device_pointer_cast(first);
  thrust::device_ptr<KeyIterator> last_d = thrust::device_pointer_cast(last);
  thrust::sort(first_d, last_d);
#else
  std::sort(first, last);
#endif // KOKKOS_ENABLE_CUDA
}

} // namespace parthenon

#endif // UTILS_SORT_HPP_
