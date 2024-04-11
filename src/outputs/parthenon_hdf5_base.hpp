//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
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
#ifndef OUTPUTS_PARTHENON_HDF5_BASE_HPP_
#define OUTPUTS_PARTHENON_HDF5_BASE_HPP_

#include "config.hpp"

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

#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace HDF5 {
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

} // namespace HDF5
} // namespace parthenon

#endif // ENABLE_HDF5
#endif // OUTPUTS_PARTHENON_HDF5_BASE_HPP_
