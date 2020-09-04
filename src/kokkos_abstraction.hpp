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

#include <string>

#include <Kokkos_Core.hpp>

namespace parthenon {

#ifdef KOKKOS_ENABLE_CUDA_UVM
using DevMemSpace = Kokkos::CudaUVMSpace;
using HostMemSpace = Kokkos::CudaUVMSpace;
using DevExecSpace = Kokkos::Cuda;
#else
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using DevExecSpace = Kokkos::DefaultExecutionSpace;
#endif
using ScratchMemSpace = DevExecSpace::scratch_memory_space;

using LayoutWrapper = Kokkos::LayoutRight;

template <typename T>
using ParArray1D = Kokkos::View<T *, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray2D = Kokkos::View<T **, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray3D = Kokkos::View<T ***, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray4D = Kokkos::View<T ****, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray5D = Kokkos::View<T *****, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray6D = Kokkos::View<T ******, LayoutWrapper, DevMemSpace>;

using team_policy = Kokkos::TeamPolicy<>;
using team_mbr_t = Kokkos::TeamPolicy<>::member_type;

template <typename T>
using ScratchPad1D = Kokkos::View<T *, LayoutWrapper, ScratchMemSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using ScratchPad2D = Kokkos::View<T **, LayoutWrapper, ScratchMemSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using ScratchPad3D = Kokkos::View<T ***, LayoutWrapper, ScratchMemSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using ScratchPad4D = Kokkos::View<T ****, LayoutWrapper, ScratchMemSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using ScratchPad5D = Kokkos::View<T *****, LayoutWrapper, ScratchMemSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <typename T>
using ScratchPad6D = Kokkos::View<T ******, LayoutWrapper, ScratchMemSpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// Defining tags to determine loop_patterns using a tag dispatch design pattern

// Translates a non-Kokkos standard C++ nested `for` loop where the innermost `for` is
// decorated with a #pragma omp simd
// IMPORTANT: This only works on CPUs
static struct LoopPatternSimdFor {
} loop_pattern_simdfor_tag;
// Translates to a Kokkos 1D range (Kokkos::RangePolicy) where the wrapper takes care
// of the (hidden) 1D index to `n`, `k`, `j`, `i indices conversion
static struct LoopPatternFlatRange {
} loop_pattern_flatrange_tag;
// Translates to a Kokkos multi dimensional  range (Kokkos::MDRangePolicy) with
// a 1:1 indices matching
static struct LoopPatternMDRange {
} loop_pattern_mdrange_tag;
// Translates to a Kokkos::TeamPolicy with a single inner Kokkos::TeamThreadRange
static struct LoopPatternTPTTR {
} loop_pattern_tpttr_tag;
// Translates to a Kokkos::TeamPolicy with a single inner Kokkos::ThreadVectorRange
static struct LoopPatternTPTVR {
} loop_pattern_tptvr_tag;
// Translates to a Kokkos::TeamPolicy with a middle Kokkos::TeamThreadRange and
// inner Kokkos::ThreadVectorRange
static struct LoopPatternTPTTRTVR {
} loop_pattern_tpttrtvr_tag;
// Used to catch undefined behavior as it results in throwing an error
static struct LoopPatternUndefined {
} loop_pattern_undefined_tag;

// Tags for Nested parallelism where the outermost layer supports 1, 2, or 3 indices

// Translates to outermost loop being a Kokkos::TeamPolicy
// Currently the only available option.
static struct OuterLoopPatternTeams {
} outer_loop_pattern_teams_tag;
// Translate to a Kokkos::TeamVectorRange as innermost loop (single index)
static struct InnerLoopPatternTVR {
} inner_loop_pattern_tvr_tag;
// Translate to a non-Kokkos plain C++ innermost loop (single index)
// decorated with #pragma omp simd
// IMPORTANT: currently only supported on CPUs
static struct InnerLoopPatternSimdFor {
} inner_loop_pattern_simdfor_tag;

// 1D loop using RangePolicy loops
template <typename Function>
inline void par_for(LoopPatternFlatRange, const std::string &name,
                    DevExecSpace exec_space, const int &il, const int &iu,
                    const Function &function) {
  Kokkos::parallel_for(name,
                       Kokkos::Experimental::require(
                           Kokkos::RangePolicy<>(exec_space, il, iu + 1),
                           Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       function);
}

// 2D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
                    const int &jl, const int &ju, const int &il, const int &iu,
                    const Function &function) {
  Kokkos::parallel_for(
      name,
      Kokkos::Experimental::require(
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>(exec_space, {jl, il}, {ju + 1, iu + 1}),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      function);
}

// 3D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(LoopPatternFlatRange, const std::string &name,
                    DevExecSpace exec_space, const int &kl, const int &ku, const int &jl,
                    const int &ju, const int &il, const int &iu,
                    const Function &function) {
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_for(
      name, Kokkos::RangePolicy<>(exec_space, 0, NkNjNi), KOKKOS_LAMBDA(const int &idx) {
        int k = idx / NjNi;
        int j = (idx - k * NjNi) / Ni;
        int i = idx - k * NjNi - j * Ni;
        k += kl;
        j += jl;
        i += il;
        function(k, j, i);
      });
}

// 3D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
                    const int &kl, const int &ku, const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function) {
  Kokkos::parallel_for(name,
                       Kokkos::Experimental::require(
                           Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                               exec_space, {kl, jl, il}, {ku + 1, ju + 1, iu + 1}),
                           Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       function);
}

// 3D loop using TeamPolicy with single inner TeamThreadRange
template <typename Function>
inline void par_for(LoopPatternTPTTR, const std::string &name, DevExecSpace exec_space,
                    const int &kl, const int &ku, const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function) {
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int NkNj = Nk * Nj;
  Kokkos::parallel_for(
      name, team_policy(exec_space, NkNj, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        const int k = team_member.league_rank() / Nj + kl;
        const int j = team_member.league_rank() % Nj + jl;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, il, iu + 1),
                             [&](const int i) { function(k, j, i); });
      });
}

// 3D loop using TeamPolicy with single inner ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTVR, const std::string &name, DevExecSpace exec_space,
                    const int &kl, const int &ku, const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function) {
  // TODO(pgrete) if exec space is Cuda,throw error
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int NkNj = Nk * Nj;
  Kokkos::parallel_for(
      name, team_policy(exec_space, NkNj, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        const int k = team_member.league_rank() / Nj + kl;
        const int j = team_member.league_rank() % Nj + jl;
        Kokkos::parallel_for(Kokkos::TeamVectorRange<>(team_member, il, iu + 1),
                             [&](const int i) { function(k, j, i); });
      });
}

// 3D loop using TeamPolicy with nested TeamThreadRange and ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTTRTVR, const std::string &name, DevExecSpace exec_space,
                    const int &kl, const int &ku, const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function) {
  const int Nk = ku - kl + 1;
  Kokkos::parallel_for(
      name, team_policy(exec_space, Nk, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        const int k = team_member.league_rank() + kl;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, jl, ju + 1), [&](const int j) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange<>(team_member, il, iu + 1),
                                   [&](const int i) { function(k, j, i); });
            });
      });
}

// 3D loop using SIMD FOR loops
template <typename Function>
inline void par_for(LoopPatternSimdFor, const std::string &name, DevExecSpace exec_space,
                    const int &kl, const int &ku, const int &jl, const int &ju,
                    const int &il, const int &iu, const Function &function) {
  Kokkos::Profiling::pushRegion(name);
  for (auto k = kl; k <= ku; k++)
    for (auto j = jl; j <= ju; j++)
#pragma omp simd
      for (auto i = il; i <= iu; i++)
        function(k, j, i);
  Kokkos::Profiling::popRegion();
}

// 4D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(LoopPatternFlatRange, const std::string &name,
                    DevExecSpace exec_space, const int nl, const int nu, const int kl,
                    const int ku, const int jl, const int ju, const int il, const int iu,
                    const Function &function) {
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NnNkNjNi = Nn * Nk * Nj * Ni;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_for(
      name, Kokkos::RangePolicy<>(exec_space, 0, NnNkNjNi),
      KOKKOS_LAMBDA(const int &idx) {
        int n = idx / NkNjNi;
        int k = (idx - n * NkNjNi) / NjNi;
        int j = (idx - n * NkNjNi - k * NjNi) / Ni;
        int i = idx - n * NkNjNi - k * NjNi - j * Ni;
        n += nl;
        k += kl;
        j += jl;
        i += il;
        function(n, k, j, i);
      });
}

// 4D loop using MDRange loops
template <typename Function>
inline void par_for(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
                    const int nl, const int nu, const int kl, const int ku, const int jl,
                    const int ju, const int il, const int iu, const Function &function) {
  Kokkos::parallel_for(
      name,
      Kokkos::Experimental::require(
          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(exec_space, {nl, kl, jl, il},
                                                 {nu + 1, ku + 1, ju + 1, iu + 1}),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      function);
}

// 4D loop using TeamPolicy loop with inner TeamThreadRange
template <typename Function>
inline void par_for(LoopPatternTPTTR, const std::string &name, DevExecSpace exec_space,
                    const int nl, const int nu, const int kl, const int ku, const int jl,
                    const int ju, const int il, const int iu, const Function &function) {
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int NkNj = Nk * Nj;
  const int NnNkNj = Nn * Nk * Nj;
  Kokkos::parallel_for(
      name, team_policy(exec_space, NnNkNj, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        int n = team_member.league_rank() / NkNj;
        int k = (team_member.league_rank() - n * NkNj) / Nj;
        int j = team_member.league_rank() - n * NkNj - k * Nj + jl;
        n += nl;
        k += kl;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, il, iu + 1),
                             [&](const int i) { function(n, k, j, i); });
      });
}

// 4D loop using TeamPolicy loop with inner ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTVR, const std::string &name, DevExecSpace exec_space,
                    const int nl, const int nu, const int kl, const int ku, const int jl,
                    const int ju, const int il, const int iu, const Function &function) {
  // TODO(pgrete) if exec space is Cuda,throw error
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int NkNj = Nk * Nj;
  const int NnNkNj = Nn * Nk * Nj;
  Kokkos::parallel_for(
      name, team_policy(exec_space, NnNkNj, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        int n = team_member.league_rank() / NkNj;
        int k = (team_member.league_rank() - n * NkNj) / Nj;
        int j = team_member.league_rank() - n * NkNj - k * Nj + jl;
        n += nl;
        k += kl;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange<>(team_member, il, iu + 1),
                             [&](const int i) { function(n, k, j, i); });
      });
}

// 4D loop using TeamPolicy with nested TeamThreadRange and ThreadVectorRange
template <typename Function>
inline void par_for(LoopPatternTPTTRTVR, const std::string &name, DevExecSpace exec_space,
                    const int nl, const int nu, const int kl, const int ku, const int jl,
                    const int ju, const int il, const int iu, const Function &function) {
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int NnNk = Nn * Nk;
  Kokkos::parallel_for(
      name, team_policy(exec_space, NnNk, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        int n = team_member.league_rank() / Nk + nl;
        int k = team_member.league_rank() % Nk + kl;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, jl, ju + 1), [&](const int j) {
              Kokkos::parallel_for(Kokkos::ThreadVectorRange<>(team_member, il, iu + 1),
                                   [&](const int i) { function(n, k, j, i); });
            });
      });
}

// 4D loop using SIMD FOR loops
template <typename Function>
inline void par_for(LoopPatternSimdFor, const std::string &name, DevExecSpace exec_space,
                    const int nl, const int nu, const int kl, const int ku, const int jl,
                    const int ju, const int il, const int iu, const Function &function) {
  Kokkos::Profiling::pushRegion(name);
  for (auto n = nl; n <= nu; n++)
    for (auto k = kl; k <= ku; k++)
      for (auto j = jl; j <= ju; j++)
#pragma omp simd
        for (auto i = il; i <= iu; i++)
          function(n, k, j, i);
  Kokkos::Profiling::popRegion();
}

// 5D loop using Kokkos 1D Range
template <typename Function>
inline void par_for(LoopPatternFlatRange, const std::string &name,
                    DevExecSpace exec_space, const int bl, const int bu, const int nl,
                    const int nu, const int kl, const int ku, const int jl, const int ju,
                    const int il, const int iu, const Function &function) {
  const int Nb = bu - bl + 1;
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NbNnNkNjNi = Nb * Nn * Nk * Nj * Ni;
  const int NnNkNjNi = Nn * Nk * Nj * Ni;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_for(
      name, Kokkos::RangePolicy<>(exec_space, 0, NbNnNkNjNi),
      KOKKOS_LAMBDA(const int &idx) {
        int b = idx / NnNkNjNi;
        int n = (idx - b * NnNkNjNi) / NkNjNi;
        int k = (idx - b * NnNkNjNi - n * NkNjNi) / NjNi;
        int j = (idx - b * NnNkNjNi - n * NkNjNi - k * NjNi) / Ni;
        int i = idx - b * NnNkNjNi - n * NkNjNi - k * NjNi - j * Ni;
        b += bl;
        n += nl;
        k += kl;
        j += jl;
        i += il;
        function(b, n, k, j, i);
      });
}

// 5D loop using SIMD FOR loops
template <typename Function>
inline void par_for(LoopPatternSimdFor, const std::string &name, DevExecSpace exec_space,
                    const int bl, const int bu, const int nl, const int nu, const int kl,
                    const int ku, const int jl, const int ju, const int il, const int iu,
                    const Function &function) {
  Kokkos::Profiling::pushRegion(name);
  for (auto b = bl; b <= bu; b++)
    for (auto n = nl; n <= nu; n++)
      for (auto k = kl; k <= ku; k++)
        for (auto j = jl; j <= ju; j++)
#pragma omp simd
          for (auto i = il; i <= iu; i++)
            function(b, n, k, j, i);
  Kokkos::Profiling::popRegion();
}

// 1D  outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(OuterLoopPatternTeams, const std::string &name,
                          DevExecSpace exec_space, size_t scratch_size_in_bytes,
                          const int scratch_level, const int kl, const int ku,
                          const Function &function) {
  const int Nk = ku + 1 - kl;

  team_policy policy(exec_space, Nk, Kokkos::AUTO);

  Kokkos::parallel_for(
      name,
      policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        const int k = team_member.league_rank() + kl;
        function(team_member, k);
      });
}

// 2D  outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(OuterLoopPatternTeams, const std::string &name,
                          DevExecSpace exec_space, size_t scratch_size_in_bytes,
                          const int scratch_level, const int kl, const int ku,
                          const int jl, const int ju, const Function &function) {
  const int Nk = ku + 1 - kl;
  const int Nj = ju + 1 - jl;
  const int NkNj = Nk * Nj;

  team_policy policy(exec_space, NkNj, Kokkos::AUTO);

  Kokkos::parallel_for(
      name,
      policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        const int k = team_member.league_rank() / Nj + kl;
        const int j = team_member.league_rank() % Nj + jl;
        function(team_member, k, j);
      });
}

// 3D  outer parallel loop using Kokkos Teams
template <typename Function>
inline void par_for_outer(OuterLoopPatternTeams, const std::string &name,
                          DevExecSpace exec_space, size_t scratch_size_in_bytes,
                          const int scratch_level, const int nl, const int nu,
                          const int kl, const int ku, const int jl, const int ju,
                          const Function &function) {
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int NkNj = Nk * Nj;
  const int NnNkNj = Nn * Nk * Nj;

  team_policy policy(exec_space, NnNkNj, Kokkos::AUTO);

  Kokkos::parallel_for(
      name,
      policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        int n = team_member.league_rank() / NkNj;
        int k = (team_member.league_rank() - n * NkNj) / Nj;
        const int j = team_member.league_rank() - n * NkNj - k * Nj + jl;
        n += nl;
        k += kl;
        function(team_member, n, k, j);
      });
}

// Inner parallel loop using TeamVectorRange
template <typename Function>
KOKKOS_INLINE_FUNCTION void par_for_inner(InnerLoopPatternTVR, team_mbr_t team_member,
                                          const int il, const int iu,
                                          const Function &function) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, il, iu + 1), function);
}

// Inner parallel loop using FOR SIMD
template <typename Function>
KOKKOS_INLINE_FUNCTION void par_for_inner(InnerLoopPatternSimdFor, team_mbr_t team_member,
                                          const int il, const int iu,
                                          const Function &function) {
#pragma omp simd
  for (int i = il; i <= iu; i++) {
    function(i);
  }
}

// reused from kokoks/core/perf_test/PerfTest_ExecSpacePartitioning.cpp
// commit a0d011fb30022362c61b3bb000ae3de6906cb6a7
template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create() { return ExecSpace(); }
  static void destroy(ExecSpace &) {}
  static bool overlap() { return false; }
};

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct SpaceInstance<Kokkos::Cuda> {
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

} // namespace parthenon

#endif // KOKKOS_ABSTRACTION_HPP_
