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

#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "../globals.hpp"
#include "config.hpp"

#define __UNIQUE_INST_VAR2(x, y) x##y
#define __UNIQUE_INST_VAR(x, y) __UNIQUE_INST_VAR2(x, y)
#define PARTHENON_INSTRUMENT                                                             \
  parthenon::KokkosTimer __UNIQUE_INST_VAR(internal_inst, __LINE__)(__FILE__, __LINE__,  \
                                                                    __func__);
#define PARTHENON_INSTRUMENT_REGION(name)                                                \
  parthenon::KokkosTimer __UNIQUE_INST_VAR(internal_inst_reg, __LINE__)(name);
#define PARTHENON_TRACE                                                                  \
  parthenon::Trace __UNIQUE_INST_VAR(internal_trace, __LINE__)(__FILE__, __LINE__,       \
                                                               __func__);
#define PARTHENON_TRACE_REGION(name)                                                     \
  parthenon::Trace __UNIQUE_INST_VAR(internal_trace_reg, __LINE__)(name);
#define PARTHENON_AUTO_LABEL parthenon::build_auto_label(__FILE__, __LINE__, __func__)

namespace parthenon {
namespace detail {
inline std::string strip_full_path(const std::string &full_path) {
  std::string label;
  auto npos = full_path.rfind('/');
  if (npos == std::string::npos) {
    label = full_path;
  } else {
    label = full_path.substr(npos + 1);
  }
  return label;
}
} // namespace detail

inline std::string build_auto_label(const std::string &file, const int line,
                                    const std::string &name) {
  return detail::strip_full_path(file) + "::" + std::to_string(line) + "::" + name;
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

class Trace {
 public:
#ifdef ENABLE_TRACE
  using clock = std::chrono::high_resolution_clock;
  using seconds = std::chrono::duration<double>;
  using tp_t = std::chrono::time_point<clock>;
  Trace(const std::string &file, const int line, const std::string &name)
      : label_(build_auto_label(file, line, name)) {
    time_[label_].push_back(clock::now());
  }
  explicit Trace(const std::string &name) : label_(name) {
    time_[label_].push_back(clock::now());
  }
  ~Trace() { time_[label_].push_back(clock::now()); }
  static void Initialize() { t0_ = clock::now(); }
  static void Report() {
    std::stringstream fname;
    fname << "trace_" << Globals::my_rank << ".txt";
    std::ofstream f(fname.str().c_str());
    for (const auto &pair : time_) {
      f << "Region: " << pair.first << std::endl;
      const auto &times = pair.second;
      for (int i = 0; i < times.size(); i += 2) {
        f << seconds(times[i] - t0_).count() << " " << seconds(times[i + 1] - t0_).count()
          << std::endl;
      }
    }
    f.close();
  }

 private:
  const std::string label_;
  static std::map<std::string, std::vector<tp_t>> time_;
  static tp_t t0_;
#else
  // stub out if not configured to do tracing
  template <class... Args>
  Trace(Args &&...args) {}
#endif
};

} // namespace parthenon

#endif // UTILS_INSTRUMENT_HPP_
