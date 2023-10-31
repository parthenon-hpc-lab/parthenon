//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_INSTRUMENT_HPP_
#define UTILS_INSTRUMENT_HPP_

#include <string>

#include <Kokkos_Core.hpp>

#define PARTHENON_INSTRUMENT KokkosTimer internal_inst_(__FILE__, __LINE__, __func__);
#define PARTHENON_INSTRUMENT_REGION(name)                                                \
  KokkosTimer internal_inst_reg____LINE__(__FILE__, __LINE__, name);

namespace parthenon {

struct KokkosTimer {
  KokkosTimer(const std::string &file, const int line, const std::string &name) {
    std::string label;
    auto npos = file.rfind('/');
    if (npos == std::string::npos) {
      label = file;
    } else {
      label = file.substr(npos + 1);
    }
    label += "::" + std::to_string(line) + "::" + name;
    Kokkos::Profiling::pushRegion(label);
  }
  ~KokkosTimer() { Kokkos::Profiling::popRegion(); }
};

} // namespace parthenon

#endif
