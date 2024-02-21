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

#define PARTHENON_ATTR_APPLY(...) \
  template void HDF5WriteAttribute<__VA_ARGS__>(const std::string &name, const __VA_ARGS__ &value, hid_t location)

PARTHENON_ATTR_FOREACH_VECTOR_TYPE(bool);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(int32_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(int64_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(uint32_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(uint64_t);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(float);
PARTHENON_ATTR_FOREACH_VECTOR_TYPE(double);

#undef PARTHENON_ATTR_APPLY

} // namespace HDF5
} // namespace parthenno

#endif // ENABLE_HDF5
