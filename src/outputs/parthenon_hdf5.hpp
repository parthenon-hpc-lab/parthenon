//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
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

#include "kokkos_abstraction.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace HDF5 {

// Number of dimension of HDF5 field data sets (block x nv x nu x nt x nz x ny x nx)
static constexpr size_t H5_NDIM = 7;

static constexpr int OUTPUT_VERSION_FORMAT = 3;

/**
 * @brief RAII handles for HDF5. Use the typedefs directly (e.g. `H5A`, `H5D`, etc.)
 *
 * @tparam CloseFn - function pointer to destructor for HDF5 object
 */
template <herr_t (*CloseFn)(hid_t)>
class H5Handle {
 public:
  H5Handle() = default;

  H5Handle(H5Handle const &) = delete;
  H5Handle &operator=(H5Handle const &) = delete;

  H5Handle(H5Handle &&other) : hid_(other.Release()) {}
  H5Handle &operator=(H5Handle &&other) {
    Reset();
    hid_ = other.Release();
    return *this;
  }

  static H5Handle FromHIDCheck(hid_t const hid) {
    PARTHENON_REQUIRE_THROWS(hid >= 0, "H5 FromHIDCheck failed");

    H5Handle handle;
    handle.hid_ = hid;
    return handle;
  }

  void Reset() {
    if (*this) {
      PARTHENON_HDF5_CHECK(CloseFn(hid_));
      hid_ = -1;
    }
  }

  hid_t Release() {
    auto hid = hid_;
    hid_ = -1;
    return hid;
  }

  ~H5Handle() { Reset(); }

  // Implicit conversion to hid_t for convenience
  operator hid_t() const { return hid_; }
  explicit operator bool() const { return hid_ >= 0; }

 private:
  hid_t hid_ = -1;
};

using H5A = H5Handle<&H5Aclose>;
using H5D = H5Handle<&H5Dclose>;
using H5F = H5Handle<&H5Fclose>;
using H5G = H5Handle<&H5Gclose>;
using H5O = H5Handle<&H5Oclose>;
using H5P = H5Handle<&H5Pclose>;
using H5T = H5Handle<&H5Tclose>;
using H5S = H5Handle<&H5Sclose>;

// Static functions to return HDF type
static hid_t getHDF5Type(const hbool_t *) { return H5T_NATIVE_HBOOL; }
static hid_t getHDF5Type(const int32_t *) { return H5T_NATIVE_INT32; }
static hid_t getHDF5Type(const int64_t *) { return H5T_NATIVE_INT64; }
static hid_t getHDF5Type(const uint32_t *) { return H5T_NATIVE_UINT32; }
static hid_t getHDF5Type(const uint64_t *) { return H5T_NATIVE_UINT64; }
static hid_t getHDF5Type(const float *) { return H5T_NATIVE_FLOAT; }
static hid_t getHDF5Type(const double *) { return H5T_NATIVE_DOUBLE; }
static hid_t getHDF5Type(const char *) { return H5T_NATIVE_CHAR; }

// On MacOS size_t is "unsigned long" and uint64_t is != "unsigned long".
// Thus, size_t is not captured by the overload above and needs to selectively enabled.
template <typename T,
          typename std::enable_if<std::is_same<T, unsigned long>::value && // NOLINT
                                      !std::is_same<T, uint64_t>::value,
                                  bool>::type = true>
static hid_t getHDF5Type(const T *) {
  return H5T_NATIVE_ULONG;
}

static H5T getHDF5Type(const char *const *) {
  H5T var_string_type = H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
  PARTHENON_HDF5_CHECK(H5Tset_size(var_string_type, H5T_VARIABLE));
  return var_string_type;
}

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
  auto pdata = view.data();
  if constexpr (!std::is_same<typename T::memory_space, Kokkos::HostSpace>::value) {
    auto view_h = Kokkos::create_mirror_view_and_copy(view);
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

std::tuple<int, std::vector<hsize_t>, std::size_t>
HDF5GetAttributeInfo(hid_t location, const std::string &name, H5A &attr);

template <typename T>
std::vector<T> HDF5ReadAttributeVec(hid_t location, const std::string &name) {
  H5A attr;
  auto [rank, dim, size] = HDF5GetAttributeInfo(location, name, attr);
  std::vector<T> res(size);

  // Check type
  auto type = getHDF5Type(res.data());
  const H5T hdf5_type = H5T::FromHIDCheck(H5Aget_type(attr));
  auto status = PARTHENON_HDF5_CHECK(H5Tequal(type, hdf5_type));
  PARTHENON_REQUIRE_THROWS(status > 0, "Type mismatch for attribute " + name);

  // Read data from file
  PARTHENON_HDF5_CHECK(H5Aread(attr, type, res.data()));

  return res;
}

// template specialization for std::string (must go into cpp file)
template <>
std::vector<std::string> HDF5ReadAttributeVec(hid_t location, const std::string &name);

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

  // resize view
  typename T::array_layout layout;
  for (int d = 0; d < rank; ++d) {
    layout.dimension[d] = dim[d];
  }
  Kokkos::resize(view, layout);

  // pull out data pointer
  auto *pdata = view.data();
  if constexpr (!std::is_same<typename T::memory_space, Kokkos::HostSpace>::value) {
    auto view_h = Kokkos::create_mirror_view_and_copy(view);
    pdata = view_h.data();
  }

  // check type
  auto type = getHDF5Type(pdata);
  const H5T hdf5_type = H5T::FromHIDCheck(H5Aget_type(attr));
  auto status = PARTHENON_HDF5_CHECK(H5Tequal(type, hdf5_type));
  PARTHENON_REQUIRE_THROWS(status > 0, "Type mismatch for attribute " + name);

  // Read attribute from file
  PARTHENON_HDF5_CHECK(H5Aread(attr, type, pdata));
}

template <typename T>
void HDF5ReadAttribute(hid_t location, const std::string &name, std::vector<T> &vec) {
  vec = HDF5ReadAttributeVec<T>(location, name);
}

} // namespace HDF5
} // namespace parthenon

#endif // ifdef ENABLE_HDF5

#endif // OUTPUTS_PARTHENON_HDF5_HPP_
