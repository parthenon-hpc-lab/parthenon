//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
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

#include "outputs/parthenon_hdf5.hpp"

namespace parthenon {
namespace HDF5 {

std::tuple<int, std::vector<hsize_t>, std::size_t>
HDF5GetAttributeInfo(hid_t location, const std::string &name, H5A &attr) {
  // check if attribute exists
  auto status = PARTHENON_HDF5_CHECK(H5Aexists(location, name.c_str()));
  PARTHENON_REQUIRE_THROWS(status > 0, "Attribute '" + name + "' does not exist");

  // Open attribute
  attr = H5A::FromHIDCheck(H5Aopen(location, name.c_str(), H5P_DEFAULT));

  // Get attribute shape
  const H5S dataspace = H5S::FromHIDCheck(H5Aget_space(attr));
  int rank = PARTHENON_HDF5_CHECK(H5Sget_simple_extent_ndims(dataspace));
  std::size_t size = 1;
  std::vector<hsize_t> dim;
  if (rank > 0) {
    dim.resize(rank);
    PARTHENON_HDF5_CHECK(H5Sget_simple_extent_dims(dataspace, dim.data(), NULL));
    for (int d = 0; d < rank; ++d) {
      size *= dim[d];
    }
    if (size == 0) {
      PARTHENON_THROW("Attribute " + name + " has no value");
    }
  } else { // scalar quantity
    dim.resize(1);
    dim[0] = 1;
  }
  // JMM: H5Handle doesn't play nice with returning a tuple/structured
  // binding, which is why it's not in the tuple. I think the issue is
  // that H5Handle doesn't have a copy assignment operator, only a
  // move operator. That probably implies not great things about the
  // performance of returning the dim array by value here, but
  // whatever. This isn't performance critical code.
  return std::make_tuple(rank, dim, size);
}

// template specializations for std::string and bool
void HDF5WriteAttribute(const std::string &name, const std::string &value,
                        hid_t location) {
  HDF5WriteAttribute(name, value.c_str(), location);
}

template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<std::string> &values,
                        hid_t location) {
  std::vector<const char *> char_ptrs(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    char_ptrs[i] = values[i].c_str();
  }
  HDF5WriteAttribute(name, char_ptrs, location);
}

template <>
std::vector<std::string> HDF5ReadAttributeVec(hid_t location, const std::string &name) {
  // get strings as char pointers, HDF5 will allocate the memory and we need to free it
  auto char_ptrs = HDF5ReadAttributeVec<char *>(location, name);

  // make strings out of char pointers, which copies the memory and then free the memeory
  std::vector<std::string> res(char_ptrs.size());
  for (size_t i = 0; i < res.size(); ++i) {
    res[i] = std::string(char_ptrs[i]);
    free(char_ptrs[i]);
  }

  return res;
}

// JMM: A little circular but it works.
template <>
std::vector<bool> HDF5ReadAttributeVec(hid_t location, const std::string &name) {
  H5A attr;
  auto [rank, dim, size] = HDF5GetAttributeInfo(location, name, attr);

  // Check type
  const hid_t type = H5T_NATIVE_HBOOL;
  const H5T hdf5_type = H5T::FromHIDCheck(H5Aget_type(attr));
  auto status = PARTHENON_HDF5_CHECK(H5Tequal(type, hdf5_type));
  PARTHENON_REQUIRE_THROWS(status > 0, "Type mismatch for attribute " + name);

  // Read data from file
  // can't use std::vector here because std::vector<bool>  doesn't have .data() member
  std::unique_ptr<hbool_t[]> data(new hbool_t[size]);
  PARTHENON_HDF5_CHECK(H5Aread(attr, type, data.get()));

  std::vector<bool> res(size);
  for (size_t i = 0; i < res.size(); ++i) {
    res[i] = data[i];
  }

  return res;
}

template <>
void HDF5WriteAttribute(const std::string &name, const std::vector<bool> &values,
                        hid_t location) {
  // can't use std::vector here because std::vector<bool>  doesn't have .data() member
  std::unique_ptr<hbool_t[]> data(new hbool_t[values.size()]);
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  HDF5WriteAttribute(name, values.size(), data.get(), location);
}

void HDF5ReadAttribute(hid_t location, const std::string &name, std::string &val) {
  std::vector<std::string> vec = HDF5ReadAttributeVec<std::string>(location, name);
  val = vec[0];
}

} // namespace HDF5
} // namespace parthenon
#endif // ENABLE_HDF5
