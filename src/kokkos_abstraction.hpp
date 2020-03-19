//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef KOKKOS_ABSTRACTION_HPP_
#define KOKKOS_ABSTRACTION_HPP_

#include <string> // string

// Kokkos headers
#include <Kokkos_Core.hpp>

namespace parthenon {

#ifdef KOKKOS_ENABLE_CUDA_UVM
typedef Kokkos::CudaUVMSpace DevSpace;
typedef Kokkos::CudaUVMSpace HostSpace;
#else
typedef Kokkos::DefaultExecutionSpace DevSpace;
typedef Kokkos::HostSpace HostSpace;
#endif

template <typename T>
using ParArray1D = Kokkos::View<T *, Kokkos::LayoutRight, DevSpace>;
template <typename T>
using ParArray2D = Kokkos::View<T **, Kokkos::LayoutRight, DevSpace>;
template <typename T>
using ParArray3D = Kokkos::View<T ***, Kokkos::LayoutRight, DevSpace>;
template <typename T>
using ParArray4D = Kokkos::View<T ****, Kokkos::LayoutRight, DevSpace>;
template <typename T>
using ParArray5D = Kokkos::View<T *****, Kokkos::LayoutRight, DevSpace>;

typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;

static struct LoopPatternSimdFor {
} loop_pattern_simdfor_tag;
static struct LoopPatternRange {
} loop_pattern_range_tag;
static struct LoopPatternMDRange {
} loop_pattern_mdrange_tag;
static struct LoopPatternTPTTR {
} loop_pattern_tpttr_tag;
static struct LoopPatternTPTVR {
} loop_pattern_tptvr_tag;
static struct LoopPatternTPTTRTVR {
} loop_pattern_tpttrtvr_tag;
static struct LoopPatternUndefined {
} loop_pattern_undefined_tag;

// TODO(pgrete) I don't like this and would prefer to make the default a
// parameter that is read from the parameter file rather than a compile time
// constant. Any suggestions on how to do this elegantly? One could use
// parthenon::Globals but then this unit here would get a dependcy on the global
// part whereas right now it's completely encapuslated.
// Alternatively, I could think of putting all this in the parthenon::wrapper
// namespace so that the default variable can live there.
// Again, I'm open for suggestions.
#ifdef MANUAL1D_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_range_tag
#elif defined SIMDFOR_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_simdfor_tag
#elif defined MDRANGE_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_mdrange_tag
#elif defined TP_TTR_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_tpttr_tag
#elif defined TP_TVR_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_tptvr_tag
#elif defined TPTTRTVR_LOOP
#define DEFAULT_LOOP_PATTERN loop_pattern_tpttrtvr_tag
#else
#define DEFAULT_LOOP_PATTERN loop_pattern_undefined_tag
#endif

// 1D default loop pattern
template <typename Function>
inline void par_for(const std::string &NAME, DevSpace exec_space, const int &IL,
                    const int &IU, const Function &function) {
  par_for(loop_pattern_mdrange_tag, NAME, exec_space, IL, IU, function);
}

// 2D default loop pattern
template <typename Function>
inline void par_for(const std::string &NAME, DevSpace exec_space, const int &JL,
                    const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  par_for(loop_pattern_mdrange_tag, NAME, exec_space, JL, JU, IL, IU, function);
}

// 3D default loop pattern
template <typename Function>
inline void par_for(const std::string &NAME, DevSpace exec_space, const int &KL,
                    const int &KU, const int &JL, const int &JU, const int &IL,
                    const int &IU, const Function &function) {
  par_for(DEFAULT_LOOP_PATTERN, NAME, exec_space, KL, KU, JL, JU, IL, IU,
          function);
}

// 4D default loop pattern
template <typename Function>
inline void par_for(const std::string &NAME, DevSpace exec_space, const int &NL,
                    const int &NU, const int &KL, const int &KU, const int &JL,
                    const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  par_for(DEFAULT_LOOP_PATTERN, NAME, exec_space, NL, NU, KL, KU, JL, JU, IL,
          IU, function);
}

// 1D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string &NAME,
                    DevSpace exec_space, const int &IL, const int &IU,
                    const Function &function) {
  Kokkos::parallel_for(NAME, Kokkos::Experimental::require(
                                 Kokkos::RangePolicy<>(exec_space, IL, IU + 1),
                                 Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       function);
}

// 2D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string &NAME,
                    DevSpace exec_space, const int &JL, const int &JU,
                    const int &IL, const int &IU, const Function &function) {
  Kokkos::parallel_for(NAME,
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                           exec_space, {JL, IL}, {JU + 1, IU + 1}),
                       function);
}

// 3D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(LoopPatternRange, const std::string &NAME,
                    DevSpace exec_space, const int &KL, const int &KU,
                    const int &JL, const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NI = IU - IL + 1;
  const int NKNJNI = NK * NJ * NI;
  const int NJNI = NJ * NI;
  Kokkos::parallel_for(
      NAME, Kokkos::RangePolicy<>(exec_space, 0, NKNJNI),
      KOKKOS_LAMBDA(const int &IDX) {
        int k = IDX / NJNI;
        int j = (IDX - k * NJNI) / NI;
        int i = IDX - k * NJNI - j * NI;
        k += KL;
        j += JL;
        i += IL;
        function(k, j, i);
      });
}

// 3D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string &NAME,
                    DevSpace exec_space, const int &KL, const int &KU,
                    const int &JL, const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  Kokkos::parallel_for(NAME, Kokkos::Experimental::require(
                       Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                           exec_space, {KL, JL, IL}, {KU + 1, JU + 1, IU + 1}),
                                 Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       function);
}

// 3D loop using TeamPolicy with single inner TeamThreadRange
template <typename Function>
inline void par_for(LoopPatternTPTTR, const std::string &NAME,
                    DevSpace exec_space, const int &KL, const int &KU,
                    const int &JL, const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NKNJ = NK * NJ;
  Kokkos::parallel_for(
      NAME, team_policy(exec_space, NKNJ, Kokkos::AUTO),
      KOKKOS_LAMBDA(member_type team_member) {
        const int k = team_member.league_rank() / NJ + KL;
        const int j = team_member.league_rank() % NJ + JL;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, IL, IU + 1),
                             [&](const int i) { function(k, j, i); });
      });
}

// 3D loop using TeamPolicy with single inner ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTVR, const std::string &NAME,
                    DevSpace exec_space, const int &KL, const int &KU,
                    const int &JL, const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  // TODO(pgrete) if exec space is Cuda,throw error
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NKNJ = NK * NJ;
  Kokkos::parallel_for(
      NAME, team_policy(exec_space, NKNJ, Kokkos::AUTO),
      KOKKOS_LAMBDA(member_type team_member) {
        const int k = team_member.league_rank() / NJ + KL;
        const int j = team_member.league_rank() % NJ + JL;
        Kokkos::parallel_for(Kokkos::TeamVectorRange<>(team_member, IL, IU + 1),
                             [&](const int i) { function(k, j, i); });
      });
}

// 3D loop using TeamPolicy with nested TeamThreadRange and ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTTRTVR, const std::string &NAME,
                    DevSpace exec_space, const int &KL, const int &KU,
                    const int &JL, const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  const int NK = KU - KL + 1;
  Kokkos::parallel_for(
      NAME, team_policy(exec_space, NK, Kokkos::AUTO),
      KOKKOS_LAMBDA(member_type team_member) {
        const int k = team_member.league_rank() + KL;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, JL, JU + 1),
            [&](const int j) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange<>(team_member, IL, IU + 1),
                  [&](const int i) { function(k, j, i); });
            });
      });
}

// 3D loop using SIMD FOR loops
template <typename Function>
inline void par_for(LoopPatternSimdFor, const std::string &NAME,
                    DevSpace exec_space, const int &KL, const int &KU,
                    const int &JL, const int &JU, const int &IL, const int &IU,
                    const Function &function) {
  Kokkos::Profiling::pushRegion(NAME);
  for (auto k = KL; k <= KU; k++)
    for (auto j = JL; j <= JU; j++)
#pragma omp simd
      for (auto i = IL; i <= IU; i++)
        function(k, j, i);
  Kokkos::Profiling::popRegion();
}

// 4D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(LoopPatternRange, const std::string &NAME,
                    DevSpace exec_space, const int NL, const int NU,
                    const int KL, const int KU, const int JL, const int JU,
                    const int IL, const int IU, const Function &function) {
  const int NN = (NU) - (NL) + 1;
  const int NK = (KU) - (KL) + 1;
  const int NJ = (JU) - (JL) + 1;
  const int NI = (IU) - (IL) + 1;
  const int NNNKNJNI = NN * NK * NJ * NI;
  const int NKNJNI = NK * NJ * NI;
  const int NJNI = NJ * NI;
  Kokkos::parallel_for(
      NAME, Kokkos::RangePolicy<>(exec_space, 0, NNNKNJNI),
      KOKKOS_LAMBDA(const int &IDX) {
        int n = IDX / NKNJNI;
        int k = (IDX - n * NKNJNI) / NJNI;
        int j = (IDX - n * NKNJNI - k * NJNI) / NI;
        int i = IDX - n * NKNJNI - k * NJNI - j * NI;
        n += (NL);
        k += (KL);
        j += (JL);
        i += (IL);
        function(n, k, j, i);
      });
}

// 4D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string &NAME,
                    DevSpace exec_space, const int NL, const int NU,
                    const int KL, const int KU, const int JL, const int JU,
                    const int IL, const int IU, const Function &function) {
  Kokkos::parallel_for(
      NAME,
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(exec_space, {NL, KL, JL, IL},
                                             {NU + 1, KU + 1, JU + 1, IU + 1}),
      function);
}

// 4D loop using TeamPolicy loop with inner TeamThreadRange
template <typename Function>
inline void par_for(LoopPatternTPTTR, const std::string &NAME,
                    DevSpace exec_space, const int NL, const int NU,
                    const int KL, const int KU, const int JL, const int JU,
                    const int IL, const int IU, const Function &function) {
  const int NN = NU - NL + 1;
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NKNJ = NK * NJ;
  const int NNNKNJ = NN * NK * NJ;
  Kokkos::parallel_for(
      NAME, team_policy(exec_space, NNNKNJ, Kokkos::AUTO),
      KOKKOS_LAMBDA(member_type team_member) {
        int n = team_member.league_rank() / NKNJ;
        int k = (team_member.league_rank() - n * NKNJ) / NJ;
        int j = team_member.league_rank() - n * NKNJ - k * NJ + JL;
        n += NL;
        k += KL;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, IL, IU + 1),
                             [&](const int i) { function(n, k, j, i); });
      });
}

// 4D loop using TeamPolicy loop with inner ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTVR, const std::string &NAME,
                    DevSpace exec_space, const int NL, const int NU,
                    const int KL, const int KU, const int JL, const int JU,
                    const int IL, const int IU, const Function &function) {
  // TODO(pgrete) if exec space is Cuda,throw error
  const int NN = NU - NL + 1;
  const int NK = KU - KL + 1;
  const int NJ = JU - JL + 1;
  const int NKNJ = NK * NJ;
  const int NNNKNJ = NN * NK * NJ;
  Kokkos::parallel_for(
      NAME, team_policy(exec_space, NNNKNJ, Kokkos::AUTO),
      KOKKOS_LAMBDA(member_type team_member) {
        int n = team_member.league_rank() / NKNJ;
        int k = (team_member.league_rank() - n * NKNJ) / NJ;
        int j = team_member.league_rank() - n * NKNJ - k * NJ + JL;
        n += NL;
        k += KL;
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange<>(team_member, IL, IU + 1),
            [&](const int i) { function(n, k, j, i); });
      });
}

// 4D loop using TeamPolicy with nested TeamThreadRange and ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTTRTVR, const std::string &NAME,
                    DevSpace exec_space, const int NL, const int NU,
                    const int KL, const int KU, const int JL, const int JU,
                    const int IL, const int IU, const Function &function) {
  const int NN = NU - NL + 1;
  const int NK = KU - KL + 1;
  const int NNNK = NN * NK;
  Kokkos::parallel_for(
      NAME, team_policy(exec_space, NNNK, Kokkos::AUTO),
      KOKKOS_LAMBDA(member_type team_member) {
        int n = team_member.league_rank() / NK + NL;
        int k = team_member.league_rank() % NK + KL;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, JL, JU + 1),
            [&](const int j) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange<>(team_member, IL, IU + 1),
                  [&](const int i) { function(n, k, j, i); });
            });
      });
}

// 4D loop using SIMD FOR loops
template <typename Function>
inline void par_for(LoopPatternSimdFor, const std::string &NAME,
                    DevSpace exec_space, const int NL, const int NU,
                    const int KL, const int KU, const int JL, const int JU,
                    const int IL, const int IU, const Function &function) {
  Kokkos::Profiling::pushRegion(NAME);
  for (auto n = NL; n <= NU; n++)
    for (auto k = KL; k <= KU; k++)
      for (auto j = JL; j <= JU; j++)
#pragma omp simd
        for (auto i = IL; i <= IU; i++)
          function(n, k, j, i);
  Kokkos::Profiling::popRegion();
}

// reused from kokoks/core/perf_test/PerfTest_ExecSpacePartitioning.cpp
// commit a0d011fb30022362c61b3bb000ae3de6906cb6a7
namespace {
template <class ExecSpace> struct SpaceInstance {
  static ExecSpace create() { return ExecSpace(); }
  static void destroy(ExecSpace &) {}
  static bool overlap() { return false; }
};

#ifdef KOKKOS_ENABLE_CUDA
template <> struct SpaceInstance<Kokkos::Cuda> {
  static Kokkos::Cuda create() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return Kokkos::Cuda(stream);
  }
  static void destroy(Kokkos::Cuda &space) {
    cudaStream_t stream = space.cuda_stream();
    cudaStreamDestroy(stream);
  }
  static bool overlap() {
    bool value = true;
    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (local_rank_str) {
      value = (std::atoi(local_rank_str) == 0);
    }
    return value;
  }
};
#endif
} // namespace
} // namespace parthenon

#endif // KOKKOS_ABSTRACTION_HPP_
