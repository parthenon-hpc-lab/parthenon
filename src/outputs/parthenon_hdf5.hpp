//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
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
#ifndef OUTPUTS_PARTHENON_HDF5_HPP_
#define OUTPUTS_PARTHENON_HDF5_HPP_

#include "config.hpp"

#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

// JMM: This could probably be done with template magic but I think
// using a macro is honestly the simplest and cleanest solution here.
// Template solution would be to define a variatic class to conain the
// list of types and then a hierarchy of structs/functions to turn
// that into function calls. Preprocessor seems easier, given we're
// not manipulating this list in any way.
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
// Only proceed if HDF5 output enabled
#ifdef ENABLE_HDF5

// Definitions common to parthenon restart and parthenon output for HDF5

#include <hdf5.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "outputs/parthenon_hdf5_types.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace HDF5 {

//  Implemented in CPP file as it's complex
hid_t GenerateFileAccessProps();

inline H5G MakeGroup(hid_t file, const std::string &name) {
  return H5G::FromHIDCheck(
      H5Gcreate(file, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
}

template <typename T>
void HDF5WriteND(hid_t location, const std::string &name, const T *data, int rank,
                 const hsize_t *local_offset, const hsize_t *local_count,
                 const hsize_t *global_count, hid_t plist_xfer, hid_t plist_dcreate) {
  const H5S local_space = H5S::FromHIDCheck(H5Screate_simple(rank, local_count, NULL));
  const H5S global_space = H5S::FromHIDCheck(H5Screate_simple(rank, global_count, NULL));

  auto type = getHDF5Type(data);
  const H5D gDSet =
      H5D::FromHIDCheck(H5Dcreate(location, name.c_str(), type, global_space, H5P_DEFAULT,
                                  plist_dcreate, H5P_DEFAULT));
  PARTHENON_HDF5_CHECK(H5Sselect_hyperslab(global_space, H5S_SELECT_SET, local_offset,
                                           NULL, local_count, NULL));
  PARTHENON_HDF5_CHECK(
      H5Dwrite(gDSet, type, local_space, global_space, plist_xfer, data));
}

template <typename T>
void HDF5Write1D(hid_t location, const std::string &name, const T *data,
                 const hsize_t *local_offset, const hsize_t *local_count,
                 const hsize_t *global_count, const H5P &plist_xfer) {
  HDF5WriteND(location, name, data, 1, local_offset, local_count, global_count,
              plist_xfer, H5P_DEFAULT);
}

template <typename T>
void HDF5Write2D(hid_t location, const std::string &name, const T *data,
                 const hsize_t *local_offset, const hsize_t *local_count,
                 const hsize_t *global_count, const H5P &plist_xfer) {
  HDF5WriteND(location, name, data, 2, local_offset, local_count, global_count,
              plist_xfer, H5P_DEFAULT);
}

template <typename T>
void HDF5WriteAttribute(const std::string &name, size_t num_values, const T *data,
                        hid_t location) {
  // can't write 0-size attributes
  if (num_values == 0) return;

  const hsize_t dim[1] = {num_values};
  const H5S data_space = H5S::FromHIDCheck(dim[0] == 1 ? H5Screate(H5S_SCALAR)
                                                       : H5Screate_simple(1, dim, dim));

  auto type = getHDF5Type(data);

  H5A const attribute = H5A::FromHIDCheck(
      H5Acreate(location, name.c_str(), type, data_space, H5P_DEFAULT, H5P_DEFAULT));
  PARTHENON_HDF5_CHECK(H5Awrite(attribute, type, data));
}

// In CPP file
void HDF5WriteAttribute(const std::string &name, const std::string &value,
                        hid_t location);

template <typename T>
void HDF5WriteAttribute(const std::string &name, const std::vector<T> &values,
                        hid_t location) {
  HDF5WriteAttribute(name, values.size(), values.data(), location);
}

// template specialization for std::string (must go into cpp file)
template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<std::string> &values,
                        hid_t location);

// template specialization for bool (must go into cpp file)
template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<bool> &values,
                        hid_t location);

template <typename T, REQUIRES(implements<kokkos_view(T)>::value)>
void HDF5WriteAttribute(const std::string &name, const T &view, hid_t location) {
  PARTHENON_REQUIRE(view.span_is_contiguous(), "Only works for contiguous views");

  // cpplint demands compile constants be all caps
  constexpr size_t RANK = static_cast<size_t>(T::rank);
  hsize_t dim[RANK];
  for (size_t d = 0; d < RANK; ++d) {
    dim[d] = view.extent_int(d);
  }
  const H5S data_space = H5S::FromHIDCheck(H5Screate_simple(RANK, dim, dim));
  // works regardless of memory space of the view
  auto *pdata = view.data();
  auto view_h = Kokkos::create_mirror_view(view);
  if constexpr (!std::is_same<typename T::memory_space, Kokkos::HostSpace>::value) {
    Kokkos::deep_copy(view_h, view);
    pdata = view_h.data();
  }
  auto type = getHDF5Type(pdata);
  H5A const attribute = H5A::FromHIDCheck(
      H5Acreate(location, name.c_str(), type, data_space, H5P_DEFAULT, H5P_DEFAULT));
  PARTHENON_HDF5_CHECK(H5Awrite(attribute, type, pdata));
}

template <typename T, REQUIRES(implements<scalar(T)>::value)>
void HDF5WriteAttribute(const std::string &name, const T &value, hid_t location) {
  std::vector<T> vec(1);
  vec[0] = value;
  HDF5WriteAttribute(name, vec, location);
}

template <typename D, typename S>
void HDF5WriteAttribute(const std::string &name, const ParArrayGeneric<D, S> &view,
                        hid_t location) {
  return HDF5WriteAttribute(name, view.KokkosView(), location);
}

template <typename T, REQUIRES(implements<scalar(T)>::value)>
void HDF5ReadAttribute(hid_t location, const std::string &name, T &val) {
  auto vec = HDF5ReadAttributeVec<T>(location, name);
  val = vec[0];
}

template <typename T, REQUIRES(implements<kokkos_view(T)>::value)>
void HDF5ReadAttribute(hid_t location, const std::string &name, T &view) {
  static_assert(std::is_same<typename T::array_layout, Kokkos::LayoutLeft>::value ||
                    std::is_same<typename T::array_layout, Kokkos::LayoutRight>::value,
                "Currently can only read from contiguous views");
  // attribute info
  H5A attr;
  auto [rank, dim, size] = HDF5GetAttributeInfo(location, name, attr);

  // check rank
  int view_rank = static_cast<size_t>(T::rank);
  PARTHENON_REQUIRE(rank == view_rank, "input and output view are same rank");

  // Resize view.
  typename T::array_layout layout;
  for (int d = 0; d < rank; ++d) {
    layout.dimension[d] = dim[d];
  }
  view = T(view.label(), layout);

  // pull out data pointer
  auto *pdata = view.data();
  auto view_h = Kokkos::create_mirror_view(view);
  if constexpr (!std::is_same<typename T::memory_space, Kokkos::HostSpace>::value) {
    // JMM: I need the pointer to point at host memory. But right now,
    // only the type of the memory matters, not its contents.
    pdata = view_h.data();
  }

  // check type
  auto type = getHDF5Type(pdata);
  const H5T hdf5_type = H5T::FromHIDCheck(H5Aget_type(attr));
  auto status = PARTHENON_HDF5_CHECK(H5Tequal(type, hdf5_type));
  PARTHENON_REQUIRE_THROWS(status > 0, "Type mismatch for attribute " + name);

  // Read attribute from file
  PARTHENON_HDF5_CHECK(H5Aread(attr, type, pdata));

  if constexpr (!std::is_same<typename T::memory_space, Kokkos::HostSpace>::value) {
    Kokkos::deep_copy(view, view_h);
  }
}

template <typename D, typename S>
void HDF5ReadAttribute(hid_t location, const std::string &name,
                       ParArrayGeneric<D, S> &pararray) {
  // forces compiler to pass by non-const reference into the next function
  // Note this loses information stored in the `State` of the pararray.
  D &view = pararray.KokkosView();
  HDF5ReadAttribute(location, name, view);
}

template <typename T>
void HDF5ReadAttribute(hid_t location, const std::string &name, std::vector<T> &vec) {
  vec = HDF5ReadAttributeVec<T>(location, name);
}

// Template extern declarations ensuring these are instantiated elsewhere
#define PARTHENON_ATTR_APPLY(...)                                                        \
  extern template void HDF5ReadAttribute<__VA_ARGS__>(                                   \
      hid_t location, const std::string &name, __VA_ARGS__ &val);                        \
  extern template void HDF5WriteAttribute<__VA_ARGS__>(                                  \
      const std::string &name, const __VA_ARGS__ &value, hid_t location)

PARTHENON_ATTR_FOREACH_VECTOR_TYPE(bool);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(int32_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(int64_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(uint32_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(uint64_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(float);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(double);

#undef PARTHENON_ATTR_APPLY

} // namespace HDF5
} // namespace parthenon

#endif // ifdef ENABLE_HDF5

#endif // OUTPUTS_PARTHENON_HDF5_HPP_
