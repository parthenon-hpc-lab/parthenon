//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"
#include "utils/multi_pointer.hpp"
#include "utils/object_pool.hpp"

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

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using LayoutWrapper = Kokkos::LayoutRight;
using MemUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

#if defined(PARTHENON_ENABLE_HOST_COMM_BUFFERS)
#if defined(KOKKOS_ENABLE_CUDA)
using BufMemSpace = Kokkos::CudaHostPinnedSpace::memory_space;
#elif defined(KOKKOS_ENABLE_HIP)
using BufMemSpace = Kokkos::Experimental::HipHostPinnedSpace::memory_space;
#else
#error "Unknow comm buffer space for chose execution space."
#endif
#else
using BufMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
#endif

// MPI communication buffers
template <typename T>
using BufArray1D = Kokkos::View<T *, LayoutWrapper, BufMemSpace>;

// Structures for reusable memory pools and communication
template <typename T>
using buf_pool_t = ObjectPool<BufArray1D<T>>;

// Raw ParArrays (that directly map a view)
template <typename T>
using ParArray0DRaw = Kokkos::View<T, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray1DRaw = Kokkos::View<T *, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray2DRaw = Kokkos::View<T **, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray3DRaw = Kokkos::View<T ***, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray4DRaw = Kokkos::View<T ****, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray5DRaw = Kokkos::View<T *****, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray6DRaw = Kokkos::View<T ******, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray7DRaw = Kokkos::View<T *******, LayoutWrapper, DevMemSpace>;
template <typename T>
using ParArray8DRaw = Kokkos::View<T ********, LayoutWrapper, DevMemSpace>;

//  Standard ParArrays that wrap a raw view and a Stage
template <typename T, typename State = empty_state_t>
using ParArray0D = ParArrayGeneric<ParArray0DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray1D = ParArrayGeneric<ParArray1DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray2D = ParArrayGeneric<ParArray2DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray3D = ParArrayGeneric<ParArray3DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray4D = ParArrayGeneric<ParArray4DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray5D = ParArrayGeneric<ParArray5DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray6D = ParArrayGeneric<ParArray6DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray7D = ParArrayGeneric<ParArray7DRaw<T>, State>;
template <typename T, typename State = empty_state_t>
using ParArray8D = ParArrayGeneric<ParArray8DRaw<T>, State>;

// Host mirrors
template <typename T>
using HostArray0D = typename ParArray0D<T>::HostMirror;
template <typename T>
using HostArray1D = typename ParArray1D<T>::HostMirror;
template <typename T>
using HostArray2D = typename ParArray2D<T>::HostMirror;
template <typename T>
using HostArray3D = typename ParArray3D<T>::HostMirror;
template <typename T>
using HostArray4D = typename ParArray4D<T>::HostMirror;
template <typename T>
using HostArray5D = typename ParArray5D<T>::HostMirror;
template <typename T>
using HostArray6D = typename ParArray6D<T>::HostMirror;
template <typename T>
using HostArray7D = typename ParArray7D<T>::HostMirror;

using team_policy = Kokkos::TeamPolicy<>;
using team_mbr_t = Kokkos::TeamPolicy<>::member_type;

template <typename T>
using ScratchPad1D = Kokkos::View<T *, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad2D = Kokkos::View<T **, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad3D = Kokkos::View<T ***, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad4D = Kokkos::View<T ****, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad5D = Kokkos::View<T *****, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad6D = Kokkos::View<T ******, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;

// Used for ParArrayND
// TODO(JMM): Should all of parthenon_arrays.hpp
// be moved here? Or should all of the above stuff be moved to
// parthenon_arrays.hpp?
inline constexpr std::size_t MAX_VARIABLE_DIMENSION = 7;
template <typename T, typename Layout = LayoutWrapper>
using device_view_t =
    Kokkos::View<multi_pointer_t<T, MAX_VARIABLE_DIMENSION>, Layout, DevMemSpace>;
template <typename T, typename Layout = LayoutWrapper>
using host_view_t = typename device_view_t<T, Layout>::HostMirror;

// Defining tags to determine loop_patterns using a tag dispatch design pattern

// Translates a non-Kokkos standard C++ nested `for` loop where the innermost
// `for` is decorated with a #pragma omp simd IMPORTANT: This only works on CPUs
static struct LoopPatternSimdFor {
} loop_pattern_simdfor_tag;
// Translates to a Kokkos 1D range (Kokkos::RangePolicy) where the wrapper takes
// care of the (hidden) 1D index to `n`, `k`, `j`, `i indices conversion
static struct LoopPatternFlatRange {
} loop_pattern_flatrange_tag;
// Translates to a Kokkos multi dimensional  range (Kokkos::MDRangePolicy) with
// a 1:1 indices matching
static struct LoopPatternMDRange {
} loop_pattern_mdrange_tag;
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::TeamThreadRange
static struct LoopPatternTPTTR {
} loop_pattern_tpttr_tag;
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::ThreadVectorRange
static struct LoopPatternTPTVR {
} loop_pattern_tptvr_tag;
// Translates to a Kokkos::TeamPolicy with a middle Kokkos::TeamThreadRange and
// inner Kokkos::ThreadVectorRange
static struct LoopPatternTPTTRTVR {
} loop_pattern_tpttrtvr_tag;
// Used to catch undefined behavior as it results in throwing an error
static struct LoopPatternUndefined {
} loop_pattern_undefined_tag;

// Tags for Nested parallelism where the outermost layer supports 1, 2, or 3
// indices

// Translates to outermost loop being a Kokkos::TeamPolicy
// Currently the only available option.
static struct OuterLoopPatternTeams {
} outer_loop_pattern_teams_tag;
// Inner loop pattern tags must be constexpr so they're available on device
// Translate to a Kokkos::TeamVectorRange as innermost loop (single index)
struct InnerLoopPatternTVR {};
constexpr InnerLoopPatternTVR inner_loop_pattern_tvr_tag;
// Translates to a Kokkos::TeamThreadRange as innermost loop
struct InnerLoopPatternTTR {};
constexpr InnerLoopPatternTTR inner_loop_pattern_ttr_tag;
// Translate to a non-Kokkos plain C++ innermost loop (single index)
// decorated with #pragma omp simd
// IMPORTANT: currently only supported on CPUs
struct InnerLoopPatternSimdFor {};
constexpr InnerLoopPatternSimdFor inner_loop_pattern_simdfor_tag;

namespace dispatch_impl {
static struct ParallelForDispatch {
} parallel_for_dispatch_tag;
static struct ParallelReduceDispatch {
} parallel_reduce_dispatch_tag;
static struct ParallelScanDispatch {
} parallel_scan_dispatch_tag;

template <class... Args>
inline void kokkos_dispatch(ParallelForDispatch, Args &&...args) {
  Kokkos::parallel_for(std::forward<Args>(args)...);
}
template <class... Args>
inline void kokkos_dispatch(ParallelReduceDispatch, Args &&...args) {
  Kokkos::parallel_reduce(std::forward<Args>(args)...);
}
template <class... Args>
inline void kokkos_dispatch(ParallelScanDispatch, Args &&...args) {
  Kokkos::parallel_scan(std::forward<Args>(args)...);
}
} // namespace dispatch_impl

// 1D loop using RangePolicy loops
template <typename Tag, typename Pattern, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(Pattern, const std::string &name, DevExecSpace exec_space, const int &il,
             const int &iu, const Function &function, Args &&...args) {
  PARTHENON_INSTRUMENT_REGION(name)
  if constexpr (std::is_same<Pattern, LoopPatternSimdFor>::value &&
                std::is_same<Tag, dispatch_impl::ParallelForDispatch>::value) {
#pragma omp simd
    for (auto i = il; i <= iu; i++) {
      function(i);
    }
  } else {
    Tag tag;
    kokkos_dispatch(tag, name,
                    Kokkos::Experimental::require(
                        Kokkos::RangePolicy<>(exec_space, il, iu + 1),
                        Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                    function, std::forward<Args>(args)...);
  }
}

template <typename Tag, typename Pattern, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(Pattern p, const std::string &name, DevExecSpace exec_space,
             const IndexRange &r, const Function &function, Args &&...args) {
  par_dispatch<Tag>(p, name, exec_space, r.s, r.e, function, std::forward<Args>(args)...);
}

// 2D loop using MDRange loops
template <typename Tag, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
             const int jl, const int ju, const int il, const int iu,
             const Function &function, Args &&...args) {
  Tag tag;
  kokkos_dispatch(tag, name,
                  Kokkos::Experimental::require(
                      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                          exec_space, {jl, il}, {ju + 1, iu + 1}, {1, iu + 1 - il}),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  function, std::forward<Args>(args)...);
}

template <typename, typename>
class FlatFunctor;

template <typename F, typename... Args>
auto MakeFlatFunctor(F &function, Args... args) {
  return FlatFunctor<F, decltype(&F::operator())>(function, std::forward<Args>(args)...);
}

template <typename Function, typename R, typename T, typename Index, typename... FArgs>
class FlatFunctor<Function, R (T::*)(Index, Index, Index, FArgs...) const> {
  int NjNi, Ni, kl, jl, il;
  Function function;

 public:
  FlatFunctor(const Function _function, const int _NjNi, const int _Ni, const int _kl,
              const int _jl, const int _il)
      : function(_function), NjNi(_NjNi), Ni(_Ni), kl(_kl), jl(_jl), il(_il) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &idx, FArgs &&...fargs) const {
    int k = idx / NjNi;
    int j = (idx - k * NjNi) / Ni;
    int i = idx - k * NjNi - j * Ni;
    k += kl;
    j += jl;
    i += il;
    function(k, j, i, std::forward<FArgs>(fargs)...);
  }
};

// 3D loop using Kokkos 1D Range
template <typename Tag, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(LoopPatternFlatRange, const std::string &name, DevExecSpace exec_space,
             const int kl, const int ku, const int jl, const int ju, const int il,
             const int iu, const Function &function, Args &&...args) {
  Tag tag;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  kokkos_dispatch(tag, name, Kokkos::RangePolicy<>(exec_space, 0, NkNjNi),
                  MakeFlatFunctor(function, NjNi, Ni, kl, jl, il),
                  std::forward<Args>(args)...);
}

// 3D loop using MDRange loops
template <typename Tag, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
             const int &kl, const int &ku, const int &jl, const int &ju, const int &il,
             const int &iu, const Function &function, Args &&...args) {
  Tag tag;
  kokkos_dispatch(tag, name,
                  Kokkos::Experimental::require(
                      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(exec_space, {kl, jl, il},
                                                             {ku + 1, ju + 1, iu + 1},
                                                             {1, 1, iu + 1 - il}),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  function, std::forward<Args>(args)...);
}

// 3D loop using TeamPolicy with single inner TeamThreadRange
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternTPTTR, const std::string &name,
                         DevExecSpace exec_space, const int &kl, const int &ku,
                         const int &jl, const int &ju, const int &il, const int &iu,
                         const Function &function) {
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
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternTPTVR, const std::string &name,
                         DevExecSpace exec_space, const int &kl, const int &ku,
                         const int &jl, const int &ju, const int &il, const int &iu,
                         const Function &function) {
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
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternTPTTRTVR, const std::string &name,
                         DevExecSpace exec_space, const int &kl, const int &ku,
                         const int &jl, const int &ju, const int &il, const int &iu,
                         const Function &function) {
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
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternSimdFor, const std::string &name,
                         DevExecSpace exec_space, const int &kl, const int &ku,
                         const int &jl, const int &ju, const int &il, const int &iu,
                         const Function &function) {
  PARTHENON_INSTRUMENT_REGION(name)
  for (auto k = kl; k <= ku; k++)
    for (auto j = jl; j <= ju; j++)
#pragma omp simd
      for (auto i = il; i <= iu; i++)
        function(k, j, i);
}

template <typename Function, typename R, typename T, typename Index, typename... FArgs>
class FlatFunctor<Function, R (T::*)(Index, Index, Index, Index, FArgs...) const> {
  int NkNjNi, NjNi, Ni, nl, kl, jl, il;
  Function function;

 public:
  FlatFunctor(const Function _function, const int _NkNjNi, const int _NjNi, const int _Ni,
              const int _nl, const int _kl, const int _jl, const int _il)
      : function(_function), NkNjNi(_NkNjNi), NjNi(_NjNi), Ni(_Ni), nl(_nl), kl(_kl),
        jl(_jl), il(_il) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int &idx, FArgs &&...fargs) const {
    int n = idx / NkNjNi;
    int k = (idx - n * NkNjNi) / NjNi;
    int j = (idx - n * NkNjNi - k * NjNi) / Ni;
    int i = idx - n * NkNjNi - k * NjNi - j * Ni;
    n += nl;
    k += kl;
    j += jl;
    i += il;
    function(n, k, j, i, std::forward<FArgs>(fargs)...);
  }
};

// 4D loop using Kokkos 1D Range
template <typename Tag, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(LoopPatternFlatRange, const std::string &name, DevExecSpace exec_space,
             const int nl, const int nu, const int kl, const int ku, const int jl,
             const int ju, const int il, const int iu, const Function &function,
             Args &&...args) {
  Tag tag;
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NnNkNjNi = Nn * Nk * Nj * Ni;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  kokkos_dispatch(tag, name, Kokkos::RangePolicy<>(exec_space, 0, NnNkNjNi),
                  MakeFlatFunctor(function, NkNjNi, NjNi, Ni, nl, kl, jl, il),
                  std::forward<Args>(args)...);
}

// 4D loop using MDRange loops
template <typename Tag, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
             const int nl, const int nu, const int kl, const int ku, const int jl,
             const int ju, const int il, const int iu, const Function &function,
             Args &&...args) {
  Tag tag;
  kokkos_dispatch(tag, name,
                  Kokkos::Experimental::require(
                      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
                          exec_space, {nl, kl, jl, il}, {nu + 1, ku + 1, ju + 1, iu + 1},
                          {1, 1, 1, iu + 1 - il}),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  function, std::forward<Args>(args)...);
}

// 4D loop using TeamPolicy loop with inner TeamThreadRange
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternTPTTR, const std::string &name,
                         DevExecSpace exec_space, const int nl, const int nu,
                         const int kl, const int ku, const int jl, const int ju,
                         const int il, const int iu, const Function &function) {
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
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternTPTVR, const std::string &name,
                         DevExecSpace exec_space, const int nl, const int nu,
                         const int kl, const int ku, const int jl, const int ju,
                         const int il, const int iu, const Function &function) {
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
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternTPTTRTVR, const std::string &name,
                         DevExecSpace exec_space, const int nl, const int nu,
                         const int kl, const int ku, const int jl, const int ju,
                         const int il, const int iu, const Function &function) {
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
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternSimdFor, const std::string &name,
                         DevExecSpace exec_space, const int nl, const int nu,
                         const int kl, const int ku, const int jl, const int ju,
                         const int il, const int iu, const Function &function) {
  PARTHENON_INSTRUMENT_REGION(name)
  for (auto n = nl; n <= nu; n++)
    for (auto k = kl; k <= ku; k++)
      for (auto j = jl; j <= ju; j++)
#pragma omp simd
        for (auto i = il; i <= iu; i++)
          function(n, k, j, i);
}

// 5D loop using MDRange loops
template <typename Tag, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
             const int ml, const int mu, const int nl, const int nu, const int kl,
             const int ku, const int jl, const int ju, const int il, const int iu,
             const Function &function, Args &&...args) {
  Tag tag;
  kokkos_dispatch(
      tag, name,
      Kokkos::Experimental::require(
          Kokkos::MDRangePolicy<Kokkos::Rank<5>>(exec_space, {ml, nl, kl, jl, il},
                                                 {mu + 1, nu + 1, ku + 1, ju + 1, iu + 1},
                                                 {1, 1, 1, 1, iu + 1 - il}),
          Kokkos::Experimental::WorkItemProperty::HintLightWeight),
      function, std::forward<Args>(args)...);
}

// 5D loop using Kokkos 1D Range
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternFlatRange, const std::string &name,
                         DevExecSpace exec_space, const int bl, const int bu,
                         const int nl, const int nu, const int kl, const int ku,
                         const int jl, const int ju, const int il, const int iu,
                         const Function &function) {
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
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternSimdFor, const std::string &name,
                         DevExecSpace exec_space, const int bl, const int bu,
                         const int nl, const int nu, const int kl, const int ku,
                         const int jl, const int ju, const int il, const int iu,
                         const Function &function) {
  PARTHENON_INSTRUMENT_REGION(name)
  for (auto b = bl; b <= bu; b++)
    for (auto n = nl; n <= nu; n++)
      for (auto k = kl; k <= ku; k++)
        for (auto j = jl; j <= ju; j++)
#pragma omp simd
          for (auto i = il; i <= iu; i++)
            function(b, n, k, j, i);
}

// 6D loop using MDRange loops
template <typename Tag, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(LoopPatternMDRange, const std::string &name, DevExecSpace exec_space,
             const int ll, const int lu, const int ml, const int mu, const int nl,
             const int nu, const int kl, const int ku, const int jl, const int ju,
             const int il, const int iu, const Function &function, Args &&...args) {
  Tag tag;
  kokkos_dispatch(tag, name,
                  Kokkos::Experimental::require(
                      Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
                          exec_space, {ll, ml, nl, kl, jl, il},
                          {lu + 1, mu + 1, nu + 1, ku + 1, ju + 1, iu + 1},
                          {1, 1, 1, 1, 1, iu + 1 - il}),
                      Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                  function, std::forward<Args>(args)...);
}

// 6D loop using Kokkos 1D Range
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternFlatRange, const std::string &name,
                         DevExecSpace exec_space, const int ll, const int lu,
                         const int ml, const int mu, const int nl, const int nu,
                         const int kl, const int ku, const int jl, const int ju,
                         const int il, const int iu, const Function &function) {
  const int Nl = lu - ll + 1;
  const int Nm = mu - ml + 1;
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  const int NkNjNi = Nk * NjNi;
  const int NnNkNjNi = Nn * NkNjNi;
  const int NmNnNkNjNi = Nm * NnNkNjNi;
  const int NlNmNnNkNjNi = Nl * NmNnNkNjNi;
  Kokkos::parallel_for(
      name, Kokkos::RangePolicy<>(exec_space, 0, NlNmNnNkNjNi),
      KOKKOS_LAMBDA(const int &idx) {
        int l = idx / NmNnNkNjNi;
        int m = (idx - l * NmNnNkNjNi) / NnNkNjNi;
        int n = (idx - l * NmNnNkNjNi - m * NnNkNjNi) / NkNjNi;
        int k = (idx - l * NmNnNkNjNi - m * NnNkNjNi - n * NkNjNi) / NjNi;
        int j = (idx - l * NmNnNkNjNi - m * NnNkNjNi - n * NkNjNi - k * NjNi) / Ni;
        int i = idx - l * NmNnNkNjNi - m * NnNkNjNi - n * NkNjNi - k * NjNi - j * Ni;
        l += ll;
        m += ml;
        n += nl;
        k += kl;
        j += jl;
        i += il;
        function(l, m, n, k, j, i);
      });
}

// 6D loop using SIMD FOR loops
template <typename Tag, typename Function>
inline void par_dispatch(LoopPatternSimdFor, const std::string &name,
                         DevExecSpace exec_space, const int ll, const int lu,
                         const int ml, const int mu, const int nl, const int nu,
                         const int kl, const int ku, const int jl, const int ju,
                         const int il, const int iu, const Function &function) {
  PARTHENON_INSTRUMENT_REGION(name)
  for (auto l = ll; l <= lu; l++)
    for (auto m = ml; m <= mu; m++)
      for (auto n = nl; n <= nu; n++)
        for (auto k = kl; k <= ku; k++)
          for (auto j = jl; j <= ju; j++)
#pragma omp simd
            for (auto i = il; i <= iu; i++)
              function(l, m, n, k, j, i);
}

template <typename Tag, typename... Args>
inline void par_dispatch(const std::string &name, Args &&...args) {
  par_dispatch<Tag>(DEFAULT_LOOP_PATTERN, name, DevExecSpace(),
                    std::forward<Args>(args)...);
}

template <class... Args>
inline void par_for(Args &&...args) {
  par_dispatch<dispatch_impl::ParallelForDispatch>(std::forward<Args>(args)...);
}

template <class... Args>
inline void par_reduce(Args &&...args) {
  par_dispatch<dispatch_impl::ParallelReduceDispatch>(std::forward<Args>(args)...);
}

template <class... Args>
inline void par_scan(Args &&...args) {
  par_dispatch<dispatch_impl::ParallelScanDispatch>(std::forward<Args>(args)...);
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

template <typename... Args>
inline void par_for_outer(const std::string &name, Args &&...args) {
  par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, name, DevExecSpace(),
                std::forward<Args>(args)...);
}

// Inner parallel loop using TeamThreadRange
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int ll, const int lu,
              const int ml, const int mu, const int nl, const int nu, const int kl,
              const int ku, const int jl, const int ju, const int il, const int iu,
              const Function &function) {
  const int Nl = lu - ll + 1;
  const int Nm = mu - ml + 1;
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  const int NkNjNi = Nk * NjNi;
  const int NnNkNjNi = Nn * NkNjNi;
  const int NmNnNkNjNi = Nm * NnNkNjNi;
  const int NlNmNnNkNjNi = Nl * NmNnNkNjNi;
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, NlNmNnNkNjNi), [&](const int &idx) {
        int l = idx / NmNnNkNjNi;
        int m = (idx - l * NmNnNkNjNi) / NnNkNjNi;
        int n = (idx - l * NmNnNkNjNi - m * NnNkNjNi) / NkNjNi;
        int k = (idx - l * NmNnNkNjNi - m * NnNkNjNi - n * NkNjNi) / NjNi;
        int j = (idx - l * NmNnNkNjNi - m * NnNkNjNi - n * NkNjNi - k * NjNi) / Ni;
        int i = idx - l * NmNnNkNjNi - m * NnNkNjNi - n * NkNjNi - k * NjNi - j * Ni;
        l += nl;
        m += ml;
        n += nl;
        k += kl;
        j += jl;
        i += il;
        function(l, m, n, k, j, i);
      });
}
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int ml, const int mu,
              const int nl, const int nu, const int kl, const int ku, const int jl,
              const int ju, const int il, const int iu, const Function &function) {
  const int Nm = mu - ml + 1;
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  const int NkNjNi = Nk * NjNi;
  const int NnNkNjNi = Nn * NkNjNi;
  const int NmNnNkNjNi = Nm * NnNkNjNi;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, NmNnNkNjNi),
                       [&](const int &idx) {
                         int m = idx / NnNkNjNi;
                         int n = (idx - m * NnNkNjNi) / NkNjNi;
                         int k = (idx - m * NnNkNjNi - n * NkNjNi) / NjNi;
                         int j = (idx - m * NnNkNjNi - n * NkNjNi - k * NjNi) / Ni;
                         int i = idx - m * NnNkNjNi - n * NkNjNi - k * NjNi - j * Ni;
                         m += ml;
                         n += nl;
                         k += kl;
                         j += jl;
                         i += il;
                         function(m, n, k, j, i);
                       });
}
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int nl, const int nu,
              const int kl, const int ku, const int jl, const int ju, const int il,
              const int iu, const Function &function) {
  const int Nn = nu - nl + 1;
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  const int NkNjNi = Nk * NjNi;
  const int NnNkNjNi = Nn * NkNjNi;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, NnNkNjNi),
                       [&](const int &idx) {
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
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int kl, const int ku,
              const int jl, const int ju, const int il, const int iu,
              const Function &function) {
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, NkNjNi), [&](const int &idx) {
    int k = idx / NjNi;
    int j = (idx - k * NjNi) / Ni;
    int i = idx - k * NjNi - j * Ni;
    k += kl;
    j += jl;
    i += il;
    function(k, j, i);
  });
}
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int jl, const int ju,
              const int il, const int iu, const Function &function) {
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, NjNi), [&](const int &idx) {
    int j = idx / Ni + jl;
    int i = idx % Ni + il;
    function(j, i);
  });
}
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(InnerLoopPatternTTR,
                                               team_mbr_t team_member, const int il,
                                               const int iu, const Function &function) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, il, iu + 1), function);
}
// Inner parallel loop using TeamVectorRange
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(InnerLoopPatternTVR,
                                               team_mbr_t team_member, const int il,
                                               const int iu, const Function &function) {
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, il, iu + 1), function);
}

// Inner parallel loop using FOR SIMD
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternSimdFor, team_mbr_t team_member, const int nl, const int nu,
              const int kl, const int ku, const int jl, const int ju, const int il,
              const int iu, const Function &function) {
  for (int n = nl; n <= nu; ++n) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
#pragma omp simd
        for (int i = il; i <= iu; i++) {
          function(k, j, i);
        }
      }
    }
  }
}
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternSimdFor, team_mbr_t team_member, const int kl, const int ku,
              const int jl, const int ju, const int il, const int iu,
              const Function &function) {
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
#pragma omp simd
      for (int i = il; i <= iu; i++) {
        function(k, j, i);
      }
    }
  }
}
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void
par_for_inner(InnerLoopPatternSimdFor, team_mbr_t team_member, const int jl, const int ju,
              const int il, const int iu, const Function &function) {
  for (int j = jl; j <= ju; ++j) {
#pragma omp simd
    for (int i = il; i <= iu; i++) {
      function(j, i);
    }
  }
}
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(InnerLoopPatternSimdFor,
                                               team_mbr_t team_member, const int il,
                                               const int iu, const Function &function) {
#pragma omp simd
  for (int i = il; i <= iu; i++) {
    function(i);
  }
}

template <typename Tag, typename Function>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(const Tag &t, team_mbr_t member,
                                               const IndexRange r,
                                               const Function &function) {
  par_for_inner(t, member, r.s, r.e, function);
}

template <typename... Args>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(team_mbr_t team_member, Args &&...args) {
  par_for_inner(DEFAULT_INNER_LOOP_PATTERN, team_member, std::forward<Args>(args)...);
}

// Inner reduction loops
template <typename Function, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
par_reduce_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int kl, const int ku,
                 const int jl, const int ju, const int il, const int iu,
                 const Function &function, T reduction) {
  const int Nk = ku - kl + 1;
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NkNjNi = Nk * Nj * Ni;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, NkNjNi),
      [&](const int &idx, typename T::value_type &lreduce) {
        int k = idx / NjNi;
        int j = (idx - k * NjNi) / Ni;
        int i = idx - k * NjNi - j * Ni;
        k += kl;
        j += jl;
        i += il;
        function(k, j, i, lreduce);
      },
      reduction);
}

template <typename Function, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
par_reduce_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int jl, const int ju,
                 const int il, const int iu, const Function &function, T reduction) {
  const int Nj = ju - jl + 1;
  const int Ni = iu - il + 1;
  const int NjNi = Nj * Ni;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, NjNi),
      [&](const int &idx, typename T::value_type &lreduce) {
        int j = idx / Ni;
        int i = idx - j * Ni;
        j += jl;
        i += il;
        function(j, i, lreduce);
      },
      reduction);
}

template <typename Function, typename T>
KOKKOS_FORCEINLINE_FUNCTION void
par_reduce_inner(InnerLoopPatternTTR, team_mbr_t team_member, const int il, const int iu,
                 const Function &function, T reduction) {
  const int Ni = iu - il + 1;
  Kokkos::parallel_reduce(
      Kokkos::TeamThreadRange(team_member, Ni),
      [&](const int &idx, typename T::value_type &lreduce) {
        int i = idx;
        i += il;
        function(i, lreduce);
      },
      reduction);
}

// For ViewOfView we need to call the destructor of the inner views on
// the host and not on the device (which would happen by default).
// Thus, we need to pass `SquentialHostInit` as allocator, but only if the ViewOfView is
// on the host. If the ViewOfViews in on the device, then `SequentialHostInit` should be
// passed when calling `create_mirror_view`, which is automatically handled by the
// create_view_of_view_mirror() function below.
template <typename T = DevMemSpace>
auto ViewOfViewAlloc(const std::string &label) {
  if constexpr (std::is_same_v<T, HostMemSpace>) {
    return Kokkos::view_alloc(Kokkos::SequentialHostInit, label);
  } else {
    return Kokkos::view_alloc(label);
  }
}

// Returns a host mirror view with the `SequentialHostInit` property for proper
// deallocations of inner views in views of views.
template <typename T>
auto create_view_of_view_mirror(T view) {
  return Kokkos::create_mirror_view(Kokkos::view_alloc(Kokkos::SequentialHostInit), view);
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
