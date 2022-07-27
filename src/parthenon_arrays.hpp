//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#ifndef PARTHENON_ARRAYS_HPP_
#define PARTHENON_ARRAYS_HPP_

#include <cassert>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"

// Macro for automatically creating a useful name
#define PARARRAY_TEMP                                                                    \
  "ParArrayND:" + std::string(__FILE__) + ":" + std::to_string(__LINE__)

namespace parthenon {

template <typename T, typename Layout = LayoutWrapper>
using device_view_t = Kokkos::View<T ******, Layout, DevMemSpace>;

template <typename T, typename Layout = LayoutWrapper>
using host_view_t = typename device_view_t<T, Layout>::HostMirror;

template <typename T, typename State = empty_state_t, typename Layout = LayoutWrapper>
using ParArrayND = ParArrayGeneric<device_view_t<T, Layout>, State>;

template <typename T, typename State = empty_state_t, typename Layout = LayoutWrapper>
using ParArrayHost = ParArrayGeneric<host_view_t<T, Layout>, State>;

template <typename T>
struct FaceArray {
  ParArrayND<T> x1f, x2f, x3f;
  FaceArray() = default;
  FaceArray(const std::string &label, int ncells3, int ncells2, int ncells1)
      : x1f(label + "x1f", ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells3 + 1, ncells2, ncells1) {}
  FaceArray(const std::string &label, int ncells4, int ncells3, int ncells2, int ncells1)
      : x1f(label + "x1f", ncells4, ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells4, ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells4, ncells3 + 1, ncells2, ncells1) {}
  FaceArray(const std::string &label, int ncells5, int ncells4, int ncells3, int ncells2,
            int ncells1)
      : x1f(label + "x1f", ncells5, ncells4, ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells5, ncells4, ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells5, ncells4, ncells3 + 1, ncells2, ncells1) {}
  FaceArray(const std::string &label, int ncells6, int ncells5, int ncells4, int ncells3,
            int ncells2, int ncells1)
      : x1f(label + "x1f", ncells6, ncells5, ncells4, ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells6, ncells5, ncells4, ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells6, ncells5, ncells4, ncells3 + 1, ncells2, ncells1) {}
  __attribute__((nothrow)) ~FaceArray() = default;

  // TODO(JMM): should this be 0,1,2?
  // Should we return the reference? Or something else?
  KOKKOS_FORCEINLINE_FUNCTION
  ParArrayND<T> &Get(int i) {
    assert(1 <= i && i <= 3);
    if (i == 1) return (x1f);
    if (i == 2)
      return (x2f);
    else
      return (x3f); // i == 3
  }
  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(int dir, Args... args) const {
    assert(1 <= dir && dir <= 3);
    if (dir == 1) return x1f(std::forward<Args>(args)...);
    if (dir == 2)
      return x2f(std::forward<Args>(args)...);
    else
      return x3f(std::forward<Args>(args)...); // i == 3
  }
};

// this is for backward compatibility with Athena++ functionality
using FaceField = FaceArray<Real>;

template <typename T>
struct EdgeArray {
  ParArrayND<T> x1e, x2e, x3e;
  EdgeArray() = default;
  EdgeArray(const std::string &label, int ncells3, int ncells2, int ncells1)
      : x1e(label + "x1e", ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells3, ncells2 + 1, ncells1 + 1) {}
  EdgeArray(const std::string &label, int ncells4, int ncells3, int ncells2, int ncells1)
      : x1e(label + "x1e", ncells4, ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells4, ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells4, ncells3, ncells2 + 1, ncells1 + 1) {}
  EdgeArray(const std::string &label, int ncells5, int ncells4, int ncells3, int ncells2,
            int ncells1)
      : x1e(label + "x1e", ncells5, ncells4, ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells5, ncells4, ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells5, ncells4, ncells3, ncells2 + 1, ncells1 + 1) {}
  EdgeArray(const std::string &label, int ncells6, int ncells5, int ncells4, int ncells3,
            int ncells2, int ncells1)
      : x1e(label + "x1e", ncells6, ncells5, ncells4, ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells6, ncells5, ncells4, ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells6, ncells5, ncells4, ncells3, ncells2 + 1, ncells1 + 1) {
  }
  __attribute__((nothrow)) ~EdgeArray() = default;
};

// backwards compatibility with Athena++ functionality
using EdgeField = EdgeArray<Real>;

template <typename T>
struct NodeArray : public ParArrayND<T> {
  NodeArray() = default;
  NodeArray(const std::string &label, int ncells3, int ncells2, int ncells1)
      : ParArrayND<T>(label, ncells3 + 1, ncells2 + 1, ncells1 + 1){}
  NodeArray(const std::string &label, int ncells4, int ncells3, int ncells2, int ncells1)
      : ParArrayND<T>(label, ncells4, ncells3 + 1, ncells2 + 1, ncells1 + 1) {}
  NodeArray(const std::string &label, int ncells5, int ncells4, int ncells3, int ncells2,
            int ncells1)
      : ParArrayND<T>(label, ncells5, ncells4, ncells3 + 1, ncells2 + 1, ncells1 + 1) {}
  NodeArray(const std::string &label, int ncells6, int ncells5, int ncells4, int ncells3,
            int ncells2, int ncells1)
      : ParArrayND<T>(label, ncells6, ncells5, ncells4, ncells3 + 1, ncells2 + 1, ncells1 + 1) {}

  __attribute__((nothrow)) ~NodeArray() = default;
};

// backwards compatibility with Athena++ functionality
using NodeField = NodeArray<Real>;


} // namespace parthenon

#endif // PARTHENON_ARRAYS_HPP_
