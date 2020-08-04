// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef OUTPUTS_PAETHENON_HDF5_HPP_
#define OUTPUTS_PAETHENON_HDF5_HPP_
// Definitions common to parthenon restart and parthenon output for HDF5

#include <cstdlib>
#include <fstream>
#include <hdf5.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "outputs/outputs.hpp"

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/container_iterator.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

#include "parthenon_mpi.hpp"

#define PREDINT32 H5T_NATIVE_INT32
#define PREDFLOAT64 H5T_NATIVE_DOUBLE
#define PREDCHAR H5T_NATIVE_CHAR

using parthenon::Real;
#ifdef HDF5OUTPUT

// loads a variable
#define LOADVARIABLEALL(dst, pmb, var, is, ie, js, je, ks, ke)                           \
  {                                                                                      \
    int index = 0;                                                                       \
    while (pmb != nullptr) {                                                             \
      for (int k = ks; k <= ke; k++) {                                                   \
        for (int j = js; j <= je; j++) {                                                 \
          for (int i = is; i <= ie; i++) {                                               \
            dst[index] = var(k, j, i);                                                   \
            index++;                                                                     \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
      pmb = pmb->next;                                                                   \
    }                                                                                    \
  }

#define LOADVARIABLEONE(index, dst, var, is, ie, js, je, ks, ke, vlen)                   \
  {                                                                                      \
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
  }

#define UNLOADVARIABLEONE(index, src, var, is, ie, js, je, ks, ke, vlen)                 \
  {                                                                                      \
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
  }

#define WRITEH5SLAB2(name, pData, theLocation, Starts, Counts, lDSpace, gDSpace, plist)	\
  {                                                                                      \
    hid_t gDSet = H5Dcreate(theLocation, name, H5T_NATIVE_DOUBLE, gDSpace, H5P_DEFAULT,  \
                            H5P_DEFAULT, H5P_DEFAULT);                                   \
    H5Sselect_hyperslab(gDSpace, H5S_SELECT_SET, Starts, NULL, Counts, NULL);            \
    H5Dwrite(gDSet, H5T_NATIVE_DOUBLE, lDSpace, gDSpace, plist, pData);                  \
    H5Dclose(gDSet);                                                                     \
  }
#define WRITEH5SLAB(name, pData, theLocation, localStart, localCount, globalCount,       \
                    plist)                                                               \
  {                                                                                      \
    hid_t lDSpace = H5Screate_simple(2, localCount, NULL);                               \
    hid_t gDSpace = H5Screate_simple(2, globalCount, NULL);                              \
    WRITEH5SLAB2(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,     \
                 plist);                                                                 \
    H5Sclose(gDSpace);                                                                   \
    H5Sclose(lDSpace);                                                                   \
  }

#define WRITEH5SLAB_X2(name, pData, theLocation, Starts, Counts, lDSpace, gDSpace,       \
                       plist, theType)                                                   \
  {                                                                                      \
    hid_t gDSet = H5Dcreate(theLocation, name, theType, gDSpace, H5P_DEFAULT,            \
                            H5P_DEFAULT, H5P_DEFAULT);                                   \
    H5Sselect_hyperslab(gDSpace, H5S_SELECT_SET, Starts, NULL, Counts, NULL);            \
    H5Dwrite(gDSet, theType, lDSpace, gDSpace, plist, pData);                            \
    H5Dclose(gDSet);                                                                     \
  }

#define WRITEH5SLAB_X(name, pData, theLocation, Starts, Counts, lDSpace, gDSpace, plist, \
                      theType)                                                           \
  {                                                                                      \
    hid_t gDSet = H5Dcreate(theLocation, name, theType, gDSpace, H5P_DEFAULT,            \
                            H5P_DEFAULT, H5P_DEFAULT);                                   \
    H5Sselect_hyperslab(gDSpace, H5S_SELECT_SET, Starts, NULL, Counts, NULL);            \
    H5Dwrite(gDSet, theType, lDSpace, gDSpace, plist, pData);                            \
    H5Dclose(gDSet);                                                                     \
  }

#define WRITEH5SLABI32(name, pData, theLocation, localStart, localCount, globalCount,    \
                       plist)                                                            \
  {                                                                                      \
    hid_t lDSpace = H5Screate_simple(2, localCount, NULL);                               \
    hid_t gDSpace = H5Screate_simple(2, globalCount, NULL);                              \
    WRITEH5SLAB_X(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,    \
                  plist, H5T_NATIVE_INT);                                                \
    H5Sclose(gDSpace);                                                                   \
    H5Sclose(lDSpace);                                                                   \
  }

#define WRITEH5SLABI64(name, pData, theLocation, localStart, localCount, globalCount,    \
                       plist)                                                            \
  {                                                                                      \
    hid_t lDSpace = H5Screate_simple(2, localCount, NULL);                               \
    hid_t gDSpace = H5Screate_simple(2, globalCount, NULL);                              \
    WRITEH5SLAB_X(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,    \
                  plist, H5T_NATIVE_LLONG);                                              \
    H5Sclose(gDSpace);                                                                   \
    H5Sclose(lDSpace);                                                                   \
  }

#define WRITEH5SLABDOUBLE(name, pData, theLocation, localStart, localCount, globalCount, \
                          plist)                                                         \
  {                                                                                      \
    hid_t lDSpace = H5Screate_simple(2, localCount, NULL);                               \
    hid_t gDSpace = H5Screate_simple(2, globalCount, NULL);                              \
    WRITEH5SLAB_X(name, pData, theLocation, localStart, localCount, lDSpace, gDSpace,    \
                  plist, H5T_NATIVE_DOUBLE);                                             \
    H5Sclose(gDSpace);                                                                   \
    H5Sclose(lDSpace);                                                                   \
  }

static herr_t writeH5AI32(const char *name, const int *pData, hid_t &file,
                          const hid_t &dSpace, const hid_t &dSet) {
  // write an attribute to file
  herr_t status; // assumption that multiple errors are stacked in calls.
  hid_t attribute;
  attribute = H5Acreate(dSet, name, PREDINT32, dSpace, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute, PREDINT32, pData);
  status = H5Aclose(attribute);
  return status;
}

static herr_t writeH5AF64(const char *name, const Real *pData, hid_t &file,
                          const hid_t &dSpace, const hid_t &dSet) {
  // write an attribute to file
  herr_t status; // assumption that multiple errors are stacked in calls.
  hid_t attribute;
  attribute = H5Acreate(dSet, name, PREDFLOAT64, dSpace, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute, PREDFLOAT64, pData);
  status = H5Aclose(attribute);
  return status;
}

static herr_t writeH5ASTRING(const char *name, const std::string pData, hid_t &file,
                             const hid_t &dSpace, const hid_t &dSet) {
  auto atype = H5Tcopy(H5T_C_S1);
  auto status = H5Tset_size(atype, pData.length());
  status = H5Tset_strpad(atype, H5T_STR_NULLTERM);
  auto attribute = H5Acreate(dSet, name, atype, dSpace, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute, atype, pData.c_str());
  status = H5Aclose(attribute);
  return status;
}

// Static functions to return HDF type
static hid_t getHdfType(float *x) { return H5T_NATIVE_FLOAT; }
static hid_t getHdfType(double *x) { return H5T_NATIVE_DOUBLE; }
static hid_t getHdfType(int32_t *x) { return H5T_NATIVE_INT32; }
static hid_t getHdfType(int64_t *x) { return H5T_NATIVE_INT64; }
static hid_t getHdfType(std::string *x) { return H5T_C_S1; }

#endif
#endif
