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

#define __UNIQUE_INST_VAR2(x, y) x##y
#define __UNIQUE_INST_VAR(x, y) __UNIQUE_INST_VAR2(x, y)
#define PARTHENON_INSTRUMENT                                                             \
  KokkosTimer __UNIQUE_INST_VAR(internal_inst, __LINE__)(__FILE__, __LINE__, __func__);
#define PARTHENON_INSTRUMENT_REGION(name)                                                \
  KokkosTimer __UNIQUE_INST_VAR(internal_inst_reg, __LINE__)(name);
#define PARTHENON_INSTRUMENT_REGION_PUSH                                                 \
  Kokkos::Profiling::pushRegion(build_auto_label(__FILE__, __LINE__, __func__));
#define PARTHENON_INSTRUMENT_REGION_POP Kokkos::Profiling::popRegion();
#define PARTHENON_AUTO_LABEL parthenon::build_auto_label(__FILE__, __LINE__, __func__)

namespace parthenon {

inline std::string build_auto_label(const std::string &fullpath, const int line,
                                    const std::string &name) {
  size_t pos = fullpath.find_last_of("/\\");
  std::string file = (pos != std::string::npos ? fullpath.substr(pos + 1) : fullpath);
  return file + "::" + std::to_string(line) + "::" + name;
}

struct KokkosTimer {
  KokkosTimer(const std::string &file, const int line, const std::string &name) {
    Push(build_auto_label(file, line, name));
  }
  explicit KokkosTimer(const std::string &name) { Push(name); }
  ~KokkosTimer() { Kokkos::Profiling::popRegion(); }

 private:
  void Push(const std::string &name) { Kokkos::Profiling::pushRegion(name); }
};

} // namespace parthenon

#endif // UTILS_INSTRUMENT_HPP_
