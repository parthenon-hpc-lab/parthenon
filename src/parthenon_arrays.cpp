//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include "parthenon_arrays.hpp"

namespace parthenon {

#define PARTHENON_ARRAY_SPEC(T)                                                          \
  template class ParArrayGeneric<device_view_t<T, LayoutWrapper>, empty_state_t>

PARTHENON_ARRAY_SPEC(float);
PARTHENON_ARRAY_SPEC(double);

#undef PARTHENON_ARRAY_SPEC

} // namespace parthenon

#ifdef PARTHENON_PRE_INSTANTIATE_KOKKOS_VIEWS
namespace Kokkos {
// the most common ones
#define PARTHENON_VIEW_TYPE_INSTANTIATION(T)                                             \
  template class View<T *, parthenon::LayoutWrapper, parthenon::DevMemSpace>;            \
  template class View<T **, parthenon::LayoutWrapper, parthenon::DevMemSpace>;           \
  template class View<T ***, parthenon::LayoutWrapper, parthenon::DevMemSpace>;          \
  template class View<T ****, parthenon::LayoutWrapper, parthenon::DevMemSpace>;         \
  template class View<parthenon::multi_pointer_t<T, parthenon::MAX_VARIABLE_DIMENSION>,  \
                      parthenon::LayoutWrapper, parthenon::DevMemSpace>

PARTHENON_VIEW_TYPE_INSTANTIATION(float);
PARTHENON_VIEW_TYPE_INSTANTIATION(double);

#undef PARTHENON_VIEW_TYPE_INSTANTIATION
} // namespace Kokkos
#endif // PARTHENON_PRE_INSTANTIATE_KOKKOS_VIEWS
