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

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"
#include "utils/indexer.hpp"
#include "utils/instrument.hpp"
#include "utils/multi_pointer.hpp"
#include "utils/object_pool.hpp"
#include "utils/type_list.hpp"

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

template <typename T, typename State = empty_state_t>
using ParArray0D = ParArrayGeneric<Kokkos::View<T, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray1D = ParArrayGeneric<Kokkos::View<T *, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray2D = ParArrayGeneric<Kokkos::View<T **, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray3D =
    ParArrayGeneric<Kokkos::View<T ***, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray4D =
    ParArrayGeneric<Kokkos::View<T ****, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray5D =
    ParArrayGeneric<Kokkos::View<T *****, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray6D =
    ParArrayGeneric<Kokkos::View<T ******, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray7D =
    ParArrayGeneric<Kokkos::View<T *******, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray8D =
    ParArrayGeneric<Kokkos::View<T ********, LayoutWrapper, DevMemSpace>, State>;

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
// Used as generic catch all for LoopPatternTeam<>
static struct LoopPatternTeamGeneric {
} loop_pattern_team_generic_tag;
// Used to catch undefined behavior as it results in throwing an error
static struct LoopPatternUndefined {
} loop_pattern_undefined_tag;
// Translates to a Kokkos::TeamPolicy that collapse Nteams outer loops
// with Nthread & Nvector inner loop collapses
template <size_t Nteam, std::size_t Nthread, std::size_t Nvector>
struct LoopPatternCollapse {};

// trait to track if pattern requests any type of hierarchial parallelism
template <typename Pattern, typename, typename T = void>
struct LoopPatternTeam : std::false_type {
  static constexpr std::size_t Nvector = 0;
  static constexpr std::size_t Nthread = 0;
};

// This pattern needs to determine the team and thread/vector count at compile time
// By contrast the others specify the thread/vector count at compile time and the
// outer team policy collapses all remaining loops
template <size_t team, std::size_t thread, std::size_t vector>
struct LoopPatternTeam<LoopPatternCollapse<team, thread, vector>,
                       std::integral_constant<size_t, team + thread + vector>, void>
    : std::true_type {
  static constexpr std::size_t Nvector = vector;
  static constexpr std::size_t Nthread = thread;
  static constexpr std::size_t Nteam = team;
  using LoopPattern = LoopPatternCollapse<team, thread, vector>;
};

// Patterns with an outer team pattern that collapses all
// remaining loops
template <typename Pattern, typename Rank>
struct LoopPatternTeam<
    Pattern, Rank,
    typename std::enable_if<std::is_same<Pattern, LoopPatternTPTTR>::value ||
                            std::is_same<Pattern, LoopPatternTPTVR>::value ||
                            std::is_same<Pattern, LoopPatternTPTTRTVR>::value>::type>
    : std::true_type {

  static constexpr bool IsTPTTR =
      std::is_same<Pattern, LoopPatternTPTTR>::value; // inner TeamThreadRange
  static constexpr bool IsTPTVR =
      std::is_same<Pattern, LoopPatternTPTVR>::value; // inner ThreadVectorRange
  static constexpr bool IsTPTTRTVR = std::is_same<Pattern, LoopPatternTPTTRTVR>::value;

  static constexpr std::size_t Nvector = IsTPTVR || IsTPTTRTVR;
  static constexpr std::size_t Nthread = IsTPTTR || IsTPTTRTVR;
  static constexpr std::size_t Nteam = Rank::value - Nthread - Nvector;
  using LoopPattern = LoopPatternCollapse<Nteam, Nthread, Nvector>;
  using OuterPattern = Pattern;
};

// Tags for Nested parallelism

// Translates to outermost loop being a Kokkos::TeamPolicy for par_for_outer like loops
static struct OuterLoopPatternTeams {
} outer_loop_pattern_teams_tag;
template <size_t Rank>
struct LoopPatternTeam<OuterLoopPatternTeams, std::integral_constant<size_t, Rank>, void>
    : std::true_type {
  static constexpr std::size_t Nvector = 0;
  static constexpr std::size_t Nthread = 0;
  static constexpr std::size_t Nteam = Rank;
  using LoopPattern = LoopPatternCollapse<Rank, 0, 0>;
  using OuterPattern = OuterLoopPatternTeams;
};

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

// Patterns for par_for_inner
template <typename Pattern, std::size_t Rank>
struct LoopPatternTeam<
    Pattern, std::integral_constant<size_t, Rank>,
    typename std::enable_if<std::is_same<Pattern, InnerLoopPatternTTR>::value ||
                            std::is_same<Pattern, InnerLoopPatternTVR>::value>::type>
    : std::true_type {

  static constexpr bool IsTTR = std::is_same<Pattern, InnerLoopPatternTTR>::value;
  static constexpr bool IsTVR = std::is_same<Pattern, InnerLoopPatternTVR>::value;

  static constexpr std::size_t Nvector = IsTVR ? Rank : 0;
  static constexpr std::size_t Nthread = IsTTR ? Rank : 0;
  using LoopPattern = LoopPatternCollapse<0, Nthread, Nvector>;
};

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

template <typename>
struct DispatchSignature {};

template <typename Index, typename... AllArgs>
struct DispatchSignature<TypeList<Index, AllArgs...>> {
 private:
  using TL = TypeList<Index, AllArgs...>;
  static constexpr std::size_t func_idx = FirstFuncIdx<TL>();

 public:
  using LaunchBounds = typename TL::template continuous_sublist<0, func_idx - 1>;
  static constexpr std::size_t rank = GetNumBounds(LaunchBounds()) / 2;
  using Rank = std::integral_constant<size_t, rank>;
  using Function = typename TL::template type<func_idx>;
  using Args = typename TL::template continuous_sublist<func_idx + 1>;
};

template <typename Tag, typename Pattern, typename... Bounds>
struct DispatchType {

  static constexpr std::size_t Rank = GetNumBounds(TypeList<Bounds...>()) / 2;

  static constexpr bool is_ParFor =
      std::is_same<Tag, dispatch_impl::ParallelForDispatch>::value;
  static constexpr bool is_ParScan =
      std::is_same<Tag, dispatch_impl::ParallelScanDispatch()>::value;

  static constexpr bool IsFlatRange = std::is_same<Pattern, LoopPatternFlatRange>::value;
  static constexpr bool IsMDRange = std::is_same<Pattern, LoopPatternMDRange>::value;
  static constexpr bool IsSimdFor = std::is_same<Pattern, LoopPatternSimdFor>::value;
  using TeamPattern =
      LoopPatternTeam<Pattern,
                      std::integral_constant<size_t, Rank>>; // false_type unless we use
                                                             // an outer team policy

  static constexpr auto GetTag() {
    // fallback simd par_reduce to flat range and force par_scan to flat range
    if constexpr (IsFlatRange || (IsSimdFor && !is_ParFor))
      return loop_pattern_flatrange_tag;
    if constexpr (IsSimdFor && is_ParFor) return loop_pattern_simdfor_tag;
    if constexpr (IsMDRange && !is_ParScan) return loop_pattern_mdrange_tag;
    if constexpr (TeamPattern::value) return loop_pattern_team_generic_tag;
  }
};

// Struct for translating between loop bounds given in terms of IndexRanges and loop
// bounds given in terms of raw integers
template <class... Bound_ts>
struct BoundTranslator {
 private:
  template <std::size_t Nx, typename... Bounds>
  static void GetIndexRanges_impl(const int idx, std::array<IndexRange, Nx> &out,
                                  const int s, const int e, Bounds &&...bounds) {
    out[idx].s = s;
    out[idx].e = e;
    if constexpr (sizeof...(Bounds) > 0) {
      GetIndexRanges_impl(idx + 1, out, std::forward<Bounds>(bounds)...);
    }
  }
  template <std::size_t Nx, typename... Bounds>
  static void GetIndexRanges_impl(const int idx, std::array<IndexRange, Nx> &out,
                                  const IndexRange ir, Bounds &&...bounds) {
    out[idx] = ir;
    if constexpr (sizeof...(Bounds) > 0) {
      GetIndexRanges_impl(idx + 1, out, std::forward<Bounds>(bounds)...);
    }
  }

 public:
  using Bound_tl = TypeList<Bound_ts...>;
  static constexpr std::size_t rank = GetNumBounds(Bound_tl()) / 2;
  static std::array<IndexRange, rank> GetIndexRanges(Bound_ts &&...bounds) {
    std::array<IndexRange, rank> out;
    GetIndexRanges_impl(0, out, std::forward<Bound_ts>(bounds)...);
    return out;
  }
};

template <class... Bound_ts>
struct BoundTranslator<TypeList<Bound_ts...>> : public BoundTranslator<Bound_ts...> {};

// don't know why my LSP can't find this in type_list.hpp
template <typename T>
using base_type = typename std::remove_cv_t<typename std::remove_reference_t<T>>;
template <size_t Rank, typename F>
using function_signature = FunctionSignature<Rank, decltype(&base_type<F>::operator())>;

template <typename, typename, typename>
struct InnerFunctor {};

template <typename Function, typename... Index, std::size_t... Iteam>
struct InnerFunctor<Function, TypeList<Index...>,
                    std::integer_sequence<size_t, Iteam...>> {
  static constexpr std::size_t Nteam = sizeof...(Iteam);
  Function function;
  Kokkos::Array<int, Nteam> inds_team;

  KOKKOS_INLINE_FUNCTION
  InnerFunctor(Kokkos::Array<int, Nteam> _inds_team, Function _function)
      : inds_team(_inds_team), function(_function) {}

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(Index... inds) const {
    function(inds_team[Iteam]..., std::forward<Index>(inds)...);
  }
};

template <typename, typename, typename, typename, bool Outer = false>
class CollapseFunctor {};

template <typename Function, std::size_t... Iteam, std::size_t... Ithread,
          std::size_t... Ivector, bool ParForOuter>
class CollapseFunctor<std::integer_sequence<size_t, Iteam...>,
                      std::integer_sequence<size_t, Ithread...>,
                      std::integer_sequence<size_t, Ivector...>, Function, ParForOuter> {
  static constexpr std::size_t Nteam = sizeof...(Iteam);
  static constexpr std::size_t Nthread = sizeof...(Ithread);
  static constexpr std::size_t Nvector = sizeof...(Ivector);
  static constexpr std::size_t Rank = Nteam + Nthread + Nvector;

  Kokkos::Array<IndexRange, Rank> ranges;
  Kokkos::Array<int, Rank> strides;
  Function function;

 public:
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION CollapseFunctor(const Function _function, IndexRange idr,
                                         Args... args)
      : function(_function), ranges({{idr, args...}}) {
    Initialize();
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION CollapseFunctor(const Function _function, Args... args)
      : function(_function) {
    std::array<int, 2 * Rank> indices{{static_cast<int>(args)...}};
    for (int i = 0; i < Rank; i++) {
      ranges[i] = {indices[2 * i], indices[2 * i + 1]};
    }
    Initialize();
  }

  KOKKOS_INLINE_FUNCTION
  void Initialize() {
    if constexpr (Rank > 1) {
      for (int ri = 0; ri < Nteam - 1; ri++) {
        const int N = ranges[ri + 1].e - ranges[ri + 1].s + 1;
        strides[ri] = N;
        for (int rj = 0; rj < ri; rj++) {
          strides[rj] *= N;
        }
      }
      for (int ri = Nteam; ri < Nteam + Nthread - 1; ri++) {
        const int N = ranges[ri + 1].e - ranges[ri + 1].s + 1;
        strides[ri] = N;
        for (int rj = Nteam; rj < ri; rj++) {
          strides[rj] *= N;
        }
      }
      for (int ri = Nteam + Nthread; ri < Rank - 1; ri++) {
        const int N = ranges[ri + 1].e - ranges[ri + 1].s + 1;
        strides[ri] = N;
        for (int rj = Nteam + Nthread; rj < ri; rj++) {
          strides[rj] *= N;
        }
      }
    }
  }

  template <size_t N, std::size_t start>
  KOKKOS_INLINE_FUNCTION void recoverIndex(Kokkos::Array<int, N> &inds, int idx) const {
    inds[0] = idx;
    for (int i = 1; i < N; i++) {
      inds[i] = idx;
      inds[i - 1] /= strides[i - 1 + start];
      for (int j = 0; j < i; j++) {
        inds[i] -= inds[j] * strides[j + start];
      }
    }
    for (int i = 0; i < N; i++) {
      inds[i] += ranges[i + start].s;
    }
  }

  KOKKOS_INLINE_FUNCTION
  int FlattenLaunchBound(int start, int end) const {
    int rangeNx = 1;
    for (int i = start; i < end; i++) {
      rangeNx *= ranges[i].e - ranges[i].s + 1;
    }
    return rangeNx;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(team_mbr_t team_member) const {
    Kokkos::Array<int, Nteam> inds_team;
    recoverIndex<Nteam, 0>(inds_team, team_member.league_rank());
    using signature = function_signature<Rank, Function>;
    using ThreadVectorInds =
        typename signature::IndexND::template continuous_sublist<Nteam>;

    if constexpr (ParForOuter) {
      function(team_member, inds_team[Iteam]...);
    } else {
      collapse_inner(
          team_member,
          InnerFunctor<Function, ThreadVectorInds, std::make_index_sequence<Nteam>>(
              inds_team, function));
    }
  }

  template <typename InnerFunction>
  KOKKOS_INLINE_FUNCTION void collapse_inner(team_mbr_t team_member,
                                             InnerFunction inner_function) const {
    if constexpr (Nthread > 0) {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member, 0,
                                    FlattenLaunchBound(Nteam, Nteam + Nthread)),
          [&](const int idThread) {
            Kokkos::Array<int, Nthread> inds_thread;
            recoverIndex<Nthread, Nteam>(inds_thread, idThread);
            if constexpr (Nvector > 0) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, 0,
                                            FlattenLaunchBound(Nteam + Nthread, Rank)),
                  [&](const int idVector) {
                    Kokkos::Array<int, Nvector> inds_vector;
                    recoverIndex<Nvector, Nteam + Nthread>(inds_vector, idVector);
                    inner_function(inds_thread[Ithread]..., inds_vector[Ivector]...);
                  });
            } else {
              inner_function(inds_thread[Ithread]...);
            }
          });
    } else {
      Kokkos::parallel_for(Kokkos::TeamVectorRange(
                               team_member, 0, FlattenLaunchBound(Nteam + Nthread, Rank)),
                           [&](const int idVector) {
                             Kokkos::Array<int, Nvector> inds_vector;
                             recoverIndex<Nvector, Nteam + Nthread>(inds_vector,
                                                                    idVector);
                             inner_function(inds_vector[Ivector]...);
                           });
    }
  }
};

template <bool ParForOuter = false, std::size_t Nteam, std::size_t Nthread,
          std::size_t Nvector, typename F, typename... Bounds>
KOKKOS_INLINE_FUNCTION auto
MakeCollapseFunctor(LoopPatternCollapse<Nteam, Nthread, Nvector>, F &function,
                    Bounds &&...bounds) {
  constexpr std::size_t Rank = Nteam + Nthread + Nvector;
  using signature = function_signature<Rank, F>;
  using IndexND = typename signature::IndexND;

  return CollapseFunctor<std::make_index_sequence<Nteam>,
                         std::make_index_sequence<Nthread>,
                         std::make_index_sequence<Nvector>, F, ParForOuter>(
      function, std::forward<Bounds>(bounds)...);
}

template <typename, std::size_t, typename, typename>
struct par_dispatch_inner {};

template <typename Pattern, std::size_t Rank, typename Function, typename... Bounds>
struct par_dispatch_inner<Pattern, Rank, Function, TypeList<Bounds...>> {
  using signature = function_signature<Rank, Function>;
  using LPT = LoopPatternTeam<Pattern, std::integral_constant<size_t, Rank>>;

  static_assert(LPT::value, "unsupported inner loop pattern");
  KOKKOS_FORCEINLINE_FUNCTION
  void dispatch(team_mbr_t team_member, Bounds &&...bounds, Function function) const {
    MakeCollapseFunctor(typename LPT::LoopPattern(), function,
                        std::forward<Bounds>(bounds)...)
        .collapse_inner(team_member, function);
  }
};

template <std::size_t Rank, std::size_t... OuterIs, typename Function>
void SimdFor(std::index_sequence<OuterIs...>, Function function,
             std::array<IndexRange, Rank> bounds) {
  if constexpr (Rank == 1) {
#pragma omp simd
    for (int i = bounds[0].s; i <= bounds[0].e; i++) {
      function(i);
    }
  } else {
    auto idxer =
        MakeIndexer(std::pair<int, int>(bounds[OuterIs].s, bounds[OuterIs].e)...);
    for (int idx = 0; idx < idxer.size(); idx++) {
      auto indices = std::tuple_cat(idxer(idx), std::tuple<int>({0}));
#pragma omp simd
      for (int i = bounds[0].s; i <= bounds[0].e; i++) {
        int &j = std::get<decltype(idxer)::rank>(indices);
        j = i;
        std::apply(function, indices);
      }
    }
  }
}

template <typename Tag, typename Pattern, typename TL_bounds, typename Function,
          typename TL_args, typename TL_functorArgs>
struct par_dispatch_functor {};

template <typename Tag, typename Pattern, typename... Bounds, typename Function,
          typename... Args, typename... FunctorArgs>
struct par_dispatch_functor<Tag, Pattern, TypeList<Bounds...>, Function,
                            TypeList<Args...>, TypeList<FunctorArgs...>> {
  using dispatch_type = DispatchType<Tag, Pattern, Bounds...>;
  void operator()(Pattern pattern, Bounds &&...bounds, Function function,
                  Args &&...args) {
    constexpr std::size_t Rank = dispatch_type::Rank;
    static_assert(!(dispatch_type::is_MDRange && Rank < 2),
                  "Can not launch MDRange with Rank < 2");
  }
};

template <typename, typename, typename, typename, typename, typename>
struct par_dispatch_impl {};

template <typename Tag, typename Pattern, typename Function, typename... Bounds,
          typename... Args, typename... ExtraFuncArgs>
struct par_dispatch_impl<Tag, Pattern, Function, TypeList<Bounds...>, TypeList<Args...>,
                         TypeList<ExtraFuncArgs...>> {
  using dispatch_type = DispatchType<Tag, Pattern, Bounds...>;
  using bound_translator = BoundTranslator<Bounds...>;
  static constexpr std::size_t Rank = bound_translator::rank;

  template <typename ExecSpace>
  static inline void dispatch(std::string name, ExecSpace exec_space, Bounds &&...bounds,
                              Function function, Args &&...args,
                              const int scratch_level = 0,
                              const std::size_t scratch_size_in_bytes = 0) {
    PARTHENON_INSTRUMENT_REGION(name)
    constexpr std::size_t Ninner =
        dispatch_type::TeamPattern::Nvector + dispatch_type::TeamPattern::Nthread;
    auto bound_arr = bound_translator::GetIndexRanges(std::forward<Bounds>(bounds)...);
    constexpr auto tag = dispatch_type::GetTag();
    if constexpr (std::is_same_v<LoopPatternSimdFor, base_type<decltype(tag)>>) {
      SimdFor(std::make_index_sequence<Rank - 1>(), function, bound_arr);
    } else {
      dispatch(tag, std::make_index_sequence<Rank - Ninner>(),
               std::make_index_sequence<Ninner>(), name, exec_space, bound_arr, function,
               std::forward<Args>(args)...);
    }
  }

 private:
  template <std::size_t... Is>
  using sequence = std::integer_sequence<std::size_t, Is...>;

  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  static inline void
  dispatch(LoopPatternFlatRange, sequence<OuterIs...>, sequence<InnerIs...>,
           std::string name, ExecSpace exec_space, std::array<IndexRange, Rank> bound_arr,
           Function function, Args &&...args) {
    auto idxer = MakeIndexer(bound_arr);
    kokkos_dispatch(
        Tag(), name, Kokkos::RangePolicy<>(exec_space, 0, idxer.size()),
        KOKKOS_LAMBDA(const int idx, ExtraFuncArgs... fargs) {
          auto idx_tuple = idxer(idx);
          function(std::get<OuterIs>(idx_tuple)...,
                   std::forward<ExtraFuncArgs>(fargs)...);
        },
        std::forward<Args>(args)...);
  }

  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  static inline void
  dispatch(LoopPatternMDRange, sequence<OuterIs...>, sequence<InnerIs...>,
           std::string name, ExecSpace exec_space, std::array<IndexRange, Rank> bound_arr,
           Function function, Args &&...args) {
    auto idxer = MakeIndexer(bound_arr);
    kokkos_dispatch(
        Tag(), name, Kokkos::RangePolicy<>(exec_space, 0, idxer.size()),
        KOKKOS_LAMBDA(const int idx, ExtraFuncArgs... fargs) {
          auto idx_tuple = idxer(idx);
          function(std::get<OuterIs>(idx_tuple)...,
                   std::forward<ExtraFuncArgs>(fargs)...);
        },
        std::forward<Args>(args)...);
  }

  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  static inline void
  dispatch(LoopPatternTeamGeneric, sequence<OuterIs...>, sequence<InnerIs...>,
           std::string name, ExecSpace exec_space, std::array<IndexRange, Rank> bound_arr,
           Function function, Args &&...args) {
    auto idxer =
        MakeIndexer(std::array<IndexRange, sizeof...(OuterIs)>{bound_arr[OuterIs]...});
    constexpr bool ParForOuter = std::is_same_v<OuterLoopPatternTeams, Pattern>;
    if constexpr (ParForOuter) {
      kokkos_dispatch(
          Tag(), name, team_policy(exec_space, idxer.size(), Kokkos::AUTO),
          KOKKOS_LAMBDA(team_mbr_t team_member, ExtraFuncArgs... fargs) {
            auto idx_tuple = idxer(team_member.league_rank());
            function(team_member, std::get<OuterIs>(idx_tuple)...,
                     std::forward<ExtraFuncArgs>(fargs)...);
          },
          std::forward<Args>(args)...);
    } else {
      constexpr std::size_t Nouter = Rank - dispatch_type::TeamPattern::Nvector -
                                     dispatch_type::TeamPattern::Nthread;
      kokkos_dispatch(Tag(), name, team_policy(exec_space, idxer.size(), Kokkos::AUTO),
                      MakeCollapseFunctor<ParForOuter>(
                          typename dispatch_type::TeamPattern::LoopPattern(), function,
                          bound_arr[OuterIs]..., bound_arr[Nouter + InnerIs]...),
                      std::forward<Args>(args)...);
    }
  }
};

template <typename Tag, typename Pattern, typename... AllArgs>
inline void par_dispatch(Pattern, std::string name, DevExecSpace exec_space,
                         AllArgs &&...args) {
  using dispatchsig = DispatchSignature<TypeList<AllArgs...>>;
  constexpr std::size_t Rank = dispatchsig::Rank::value;
  using Function = typename dispatchsig::Function;
  using LaunchBounds = typename dispatchsig::LaunchBounds;
  using Args = typename dispatchsig::Args;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;

  if constexpr (Rank > 1 && std::is_same_v<dispatch_impl::ParallelScanDispatch, Tag>) {
    static_assert(always_false<Tag>, "par_scan only for 1D loops");
  }
  par_dispatch_impl<Tag, Pattern, Function, LaunchBounds, Args, ExtraFuncArgs>::dispatch(
      name, exec_space, std::forward<AllArgs>(args)...);
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

template <typename Pattern, typename... AllArgs>
inline std::enable_if_t<std::is_same<Pattern, OuterLoopPatternTeams>::value, void>
par_for_outer(Pattern, const std::string &name, DevExecSpace exec_space,
              std::size_t scratch_size_in_bytes, const int scratch_level,
              AllArgs &&...args) {
  using dispatchsig = DispatchSignature<TypeList<AllArgs...>>;
  static constexpr std::size_t Rank = dispatchsig::Rank::value;
  using Function = typename dispatchsig::Function;
  using LaunchBounds = typename dispatchsig::LaunchBounds;
  using Args = typename dispatchsig::Args;
  using Tag = dispatch_impl::ParallelForDispatch;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;

  par_dispatch_impl<Tag, Pattern, Function, LaunchBounds, Args, ExtraFuncArgs>::dispatch(
      name, exec_space, std::forward<AllArgs>(args)..., scratch_level,
      scratch_size_in_bytes);
}

template <typename... Args>
inline void par_for_outer(const std::string &name, Args &&...args) {
  par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, name, DevExecSpace(),
                std::forward<Args>(args)...);
}

template <typename Pattern, typename... AllArgs>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(Pattern, team_mbr_t team_member,
                                               AllArgs &&...args) {
  using DispatchSig = DispatchSignature<TypeList<AllArgs...>>;
  constexpr std::size_t Rank = DispatchSig::Rank::value;
  using Function = typename DispatchSig::Function;
  using LaunchBounds = typename DispatchSig::LaunchBounds;

  if constexpr (std::is_same_v<Pattern, InnerLoopPatternSimdFor>) {
    using Args = typename DispatchSig::Args;
    using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;
    par_dispatch_impl<dispatch_impl::ParallelForDispatch, LoopPatternSimdFor, Function,
                      LaunchBounds, Args, ExtraFuncArgs>()
        .dispatch("simd", HostExecSpace(), std::forward<AllArgs>(args)...);
  } else {
    par_dispatch_inner<Pattern, Rank, Function, LaunchBounds>().dispatch(
        team_member, std::forward<AllArgs>(args)...);
  }
}

template <typename... Args>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(team_mbr_t team_member, Args &&...args) {
  par_for_inner(DEFAULT_INNER_LOOP_PATTERN, team_member, std::forward<Args>(args)...);
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
