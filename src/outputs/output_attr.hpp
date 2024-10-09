//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023-2024 The Parthenon collaboration
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

#ifndef OUTPUTS_OUTPUT_ATTR_HPP_
#define OUTPUTS_OUTPUT_ATTR_HPP_

// JMM: This could probably be done with template magic but I think
// using a macro is honestly the simplest and cleanest solution here.
// Template solution would be to define a variatic class to conain the
// list of types and then a hierarchy of structs/functions to turn
// that into function calls. Preprocessor seems easier, given we're
// not manipulating this list in any way.
// The following types are the ones we allow to be stored as attributes in outputs
// (specifically within Params).
#define PARTHENON_ATTR_VALID_VEC_TYPES(T)                                                \
  T, std::vector<T>, ParArray1D<T>, ParArray2D<T>, ParArray3D<T>, HostArray1D<T>,        \
      HostArray2D<T>, HostArray3D<T>, Kokkos::View<T *>, Kokkos::View<T **>,             \
      ParArrayND<T>, ParArrayHost<T>
// JMM: This is the list of template specializations we
// "pre-instantiate" We only pre-instantiate device memory, not host
// memory. The reason is that when building with the Kokkos serial
// backend, DevMemSpace and HostMemSpace are the same and so this
// resolves to the same type in the macro, which causes problems.
#define PARTHENON_ATTR_FOREACH_VECTOR_TYPE(T)                                            \
  PARTHENON_ATTR_APPLY(T);                                                               \
  PARTHENON_ATTR_APPLY(Kokkos::View<T *, LayoutWrapper, DevMemSpace>);                   \
  PARTHENON_ATTR_APPLY(Kokkos::View<T **, LayoutWrapper, DevMemSpace>);                  \
  PARTHENON_ATTR_APPLY(Kokkos::View<T ***, LayoutWrapper, DevMemSpace>);                 \
  PARTHENON_ATTR_APPLY(device_view_t<T>)

#endif // OUTPUTS_OUTPUT_ATTR_HPP_