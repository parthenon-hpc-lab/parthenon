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
#include "variable_dimensions.hpp"

// Macro for automatically creating a useful name
#define PARARRAY_TEMP                                                                    \
  "ParArrayND:" + std::string(__FILE__) + ":" + std::to_string(__LINE__)

namespace parthenon {

template <typename T, typename Layout = LayoutWrapper>
using device_view_t = Kokkos::View<multi_pointer_t<T>, Layout, DevMemSpace>;

template <typename T, typename Layout = LayoutWrapper>
using host_view_t = typename device_view_t<T, Layout>::HostMirror;

template <typename T, typename State = empty_state_t, typename Layout = LayoutWrapper>
using ParArrayND = ParArrayGeneric<device_view_t<T, Layout>, State>;

template <typename T, typename State = empty_state_t, typename Layout = LayoutWrapper>
using ParArrayHost = ParArrayGeneric<host_view_t<T, Layout>, State>;

} // namespace parthenon

#endif // PARTHENON_ARRAYS_HPP_
