// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
// Definitions common to parthenon restart and parthenon output for HDF5

// options for building
#include "config.hpp"

#ifdef HDF5OUTPUT
#include <hdf5.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/meshblock_data_iterator.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

#include "parthenon_mpi.hpp"

#define PREDINT32 H5T_NATIVE_INT32
#define PREDFLOAT64 H5T_NATIVE_DOUBLE
#define PREDCHAR H5T_NATIVE_CHAR

using parthenon::Real;
#ifndef HDF5OUTPUT
#define LOADVARIABLEALL(dst, pmb, var, is, ie, js, je, ks, ke)
#define LOADVARIABLEONE(index, dst, var, is, ie, js, je, ks, ke, vlen)
#define UNLOADVARIABLEONE(index, src, var, is, ie, js, je, ks, ke, vlen)
#else

namespace parthenon {
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
using H5P = H5Handle<&H5Pclose>;
using H5T = H5Handle<&H5Tclose>;
using H5S = H5Handle<&H5Sclose>;
} // namespace parthenon

#define LOADVARIABLERAW(index, dst, var, i1, i2, i3, i4, i5, i6)                         \
  do {                                                                                   \
    for (int i = 0; i < i1; i++) {                                                       \
      for (int j = 0; j < i2; j++) {                                                     \
        for (int k = 0; k < i3; k++) {                                                   \
          for (int l = 0; l < i4; l++) {                                                 \
            for (int m = 0; m < i5; m++) {                                               \
              for (int n = 0; n < i6; n++) {                                             \
                dst[index] = var(i, j, k, l, m, n);                                      \
                index++;                                                                 \
              }                                                                          \
            }                                                                            \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
    }                                                                                    \
  } while (false)

#define LOADVARIABLEONE(index, dst, var, is, ie, js, je, ks, ke, vlen)                   \
  do {                                                                                   \
    for (int k = ks; k <= ke; k++) {                                                     \
      for (int j = js; j <= je; j++) {                                                   \
        for (int i = is; i <= ie; i++) {                                                 \
          for (int l = 0; l < vlen; l++) {                                               \
            dst[index] = var(l, k, j, i);                                                \
            index++;                                                                     \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
    }                                                                                    \
  } while (false)

// loads a variable
#define LOADVARIABLEALL(dst, pm, var, is, ie, js, je, ks, ke)                            \
  do {                                                                                   \
    int index = 0;                                                                       \
    for (auto &pmb : pm->block_list) {                                                   \
      for (int k = ks; k <= ke; k++) {                                                   \
        for (int j = js; j <= je; j++) {                                                 \
          for (int i = is; i <= ie; i++) {                                               \
            dst[index] = var(k, j, i);                                                   \
            index++;                                                                     \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
    }                                                                                    \
  } while (false)

#define UNLOADVARIABLEONE(index, src, var, is, ie, js, je, ks, ke, vlen)                 \
  do {                                                                                   \
    for (int k = ks; k <= ke; k++) {                                                     \
      for (int j = js; j <= je; j++) {                                                   \
        for (int i = is; i <= ie; i++) {                                                 \
          for (int l = 0; l < vlen; l++) {                                               \
            var(l, k, j, i) = src[index];                                                \
            index++;                                                                     \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
    }                                                                                    \
  } while (false)

#define WRITEH5SLAB2(name, pData, theLocation, Starts, Counts, lDSpace, gDSpace, plist)  \
  do {                                                                                   \
    ::parthenon::H5D const gDSet = ::parthenon::H5D::FromHIDCheck(                       \
        H5Dcreate(theLocation, name, H5T_NATIVE_DOUBLE, gDSpace, H5P_DEFAULT,            \
                  H5P_DEFAULT, H5P_DEFAULT));                                            \
    PARTHENON_HDF5_CHECK(                                                                \
        H5Sselect_hyperslab(gDSpace, H5S_SELECT_SET, Starts, NULL, Counts, NULL));       \
    PARTHENON_HDF5_CHECK(                                                                \
        H5Dwrite(gDSet, H5T_NATIVE_DOUBLE, lDSpace, gDSpace, plist, pData));             \
  } while (false)
#define WRITEH5SLAB(name, pData, theLocation, localStart, localCount, globalCount,       \
                    plist)                                                               \
  do {                                                                                   \
    ::parthenon::H5S const lDSpace =                                                     \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, localCount, NULL));           \
    ::parthenon::H5S const gDSpace =                                                     \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, globalCount, NULL));          \
    WRITEH5SLAB2(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,     \
                 plist);                                                                 \
  } while (false)

#define WRITEH5SLAB_X2(name, pData, theLocation, Starts, Counts, lDSpace, gDSpace,       \
                       plist, theType)                                                   \
  do {                                                                                   \
    ::parthenon::H5D const gDSet = ::parthenon::H5D::FromHIDCheck(H5Dcreate(             \
        theLocation, name, theType, gDSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));    \
    PARTHENON_HDF5_CHECK(                                                                \
        H5Sselect_hyperslab(gDSpace, H5S_SELECT_SET, Starts, NULL, Counts, NULL));       \
    PARTHENON_HDF5_CHECK(H5Dwrite(gDSet, theType, lDSpace, gDSpace, plist, pData));      \
  } while (false)

#define WRITEH5SLAB_X(name, pData, theLocation, Starts, Counts, lDSpace, gDSpace, plist, \
                      theType)                                                           \
  do {                                                                                   \
    ::parthenon::H5D const gDSet = ::parthenon::H5D::FromHIDCheck(H5Dcreate(             \
        theLocation, name, theType, gDSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));    \
    PARTHENON_HDF5_CHECK(                                                                \
        H5Sselect_hyperslab(gDSpace, H5S_SELECT_SET, Starts, NULL, Counts, NULL));       \
    PARTHENON_HDF5_CHECK(H5Dwrite(gDSet, theType, lDSpace, gDSpace, plist, pData));      \
  } while (false)

#define WRITEH5SLABI32(name, pData, theLocation, localStart, localCount, globalCount,    \
                       plist)                                                            \
  do {                                                                                   \
    ::parthenon::H5S const lDSpace =                                                     \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, localCount, NULL));           \
    ::parthenon::H5S const gDSpace =                                                     \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, globalCount, NULL));          \
    WRITEH5SLAB_X(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,    \
                  plist, H5T_NATIVE_INT);                                                \
  } while (false)

#define WRITEH5SLABI64(name, pData, theLocation, localStart, localCount, globalCount,    \
                       plist)                                                            \
  do {                                                                                   \
    ::parthenon::H5S lDSpace =                                                           \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, localCount, NULL));           \
    ::parthenon::H5S gDSpace =                                                           \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, globalCount, NULL));          \
    WRITEH5SLAB_X(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,    \
                  plist, H5T_NATIVE_LLONG);                                              \
  } while (false)

#define WRITEH5SLABDOUBLE(name, pData, theLocation, localStart, localCount, globalCount, \
                          plist)                                                         \
  do {                                                                                   \
    ::parthenon::H5S const lDSpace =                                                     \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, localCount, NULL));           \
    ::parthenon::H5S const gDSpace =                                                     \
        ::parthenon::H5S::FromHIDCheck(H5Screate_simple(2, globalCount, NULL));          \
    WRITEH5SLAB_X(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,    \
                  plist, H5T_NATIVE_DOUBLE);                                             \
  } while (false)

static void writeH5AI32(const char *name, const int *pData, const hid_t &dSpace,
                        const hid_t &dSet) {
  // write an attribute to file
  ::parthenon::H5A const attribute = ::parthenon::H5A::FromHIDCheck(
      H5Acreate(dSet, name, PREDINT32, dSpace, H5P_DEFAULT, H5P_DEFAULT));
  PARTHENON_HDF5_CHECK(H5Awrite(attribute, PREDINT32, pData));
}

static void writeH5AF64(const char *name, const Real *pData, const hid_t &dSpace,
                        const hid_t &dSet) {
  // write an attribute to file
  ::parthenon::H5A const attribute = ::parthenon::H5A::FromHIDCheck(
      H5Acreate(dSet, name, PREDFLOAT64, dSpace, H5P_DEFAULT, H5P_DEFAULT));
  PARTHENON_HDF5_CHECK(H5Awrite(attribute, PREDFLOAT64, pData));
}

static void writeH5ASTRING(const char *name, const std::string &pData,
                           const hid_t &dSpace, const hid_t &dSet) {
  ::parthenon::H5T const atype = ::parthenon::H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
  PARTHENON_HDF5_CHECK(H5Tset_size(atype, pData.length()));
  PARTHENON_HDF5_CHECK(H5Tset_strpad(atype, H5T_STR_NULLTERM));
  ::parthenon::H5A const attribute = ::parthenon::H5A::FromHIDCheck(
      H5Acreate(dSet, name, atype, dSpace, H5P_DEFAULT, H5P_DEFAULT));
  PARTHENON_HDF5_CHECK(H5Awrite(attribute, atype, pData.c_str()));
}

static void writeH5ASTRINGS(const char *name, const std::vector<std::string> &pData,
                            const hid_t &dSpace, const hid_t &dSet) {
  int max_name_length = 0;
  for (const auto &s : pData) {
    max_name_length = std::max(max_name_length, static_cast<int>(s.length()));
  }

  std::vector<const char *> c_strs;
  c_strs.reserve(pData.size());
  for (int i = 0; i < pData.size(); i++) {
    // Copy pData[i] into c_strs[i], including a null terminator
    c_strs.push_back(pData[i].c_str());
  }

  ::parthenon::H5T const atype = ::parthenon::H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
  PARTHENON_HDF5_CHECK(H5Tset_size(atype, H5T_VARIABLE));

  ::parthenon::H5A const attribute = ::parthenon::H5A::FromHIDCheck(
      H5Acreate(dSet, name, atype, dSpace, H5P_DEFAULT, H5P_DEFAULT));
  PARTHENON_HDF5_CHECK(H5Awrite(attribute, atype, c_strs.data()));
}

// Static functions to return HDF type
static hid_t getHdfType(float *x) { return H5T_NATIVE_FLOAT; }
static hid_t getHdfType(double *x) { return H5T_NATIVE_DOUBLE; }
static hid_t getHdfType(int32_t *x) { return H5T_NATIVE_INT32; }
static hid_t getHdfType(int64_t *x) { return H5T_NATIVE_INT64; }
static hid_t getHdfType(std::string *x) { return H5T_C_S1; }

#endif
#endif // OUTPUTS_PARTHENON_HDF5_HPP_
