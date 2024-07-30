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

#include <Kokkos_Core.hpp>
#include <variant>

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "config.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/concepts_lite.hpp"
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
// Used to catch undefined behavior as it results in throwing an error
static struct LoopPatternUndefined {
} loop_pattern_undefined_tag;
// Translates to a Kokkos::TeamPolicy that collapse Nteams outer loops
// with Nthread & Nvector inner loop collapses
template <size_t Nteam, size_t Nthread, size_t Nvector>
struct LoopPatternCollapse {};

template <typename, typename, typename T = void>
struct LoopPatternTeam : std::false_type {};

template <size_t team, size_t thread, size_t vector>
struct LoopPatternTeam<LoopPatternCollapse<team, thread, vector>,
                       std::integral_constant<size_t, team + thread + vector>, void>
    : std::true_type {
  static constexpr size_t Nvector = vector;
  static constexpr size_t Nthread = thread;
  static constexpr size_t Nteam = team;
  using LoopPattern = LoopPatternCollapse<team, thread, vector>;
};

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

  static constexpr size_t Nvector = IsTPTVR || IsTPTTRTVR;
  static constexpr size_t Nthread = IsTPTTR || IsTPTTRTVR;
  static constexpr size_t Nteam = Rank::value - Nthread - Nvector;
  using LoopPattern = LoopPatternCollapse<Nteam, Nthread, Nvector>;
  using OuterPattern = Pattern;
};

// Tags for Nested parallelism where the outermost layer supports 1, 2, or 3
// indices

// Translates to outermost loop being a Kokkos::TeamPolicy
// Currently the only available option.
static struct OuterLoopPatternTeams {
} outer_loop_pattern_teams_tag;
template <size_t Rank>
struct LoopPatternTeam<OuterLoopPatternTeams, std::integral_constant<size_t, Rank>, void>
    : std::true_type {
  static constexpr size_t Nvector = 0;
  static constexpr size_t Nthread = 0;
  static constexpr size_t Nteam = Rank;
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

template <typename Pattern, size_t Rank>
struct LoopPatternTeam<
    Pattern, std::integral_constant<size_t, Rank>,
    typename std::enable_if<std::is_same<Pattern, InnerLoopPatternTTR>::value ||
                            std::is_same<Pattern, InnerLoopPatternTVR>::value>::type>
    : std::true_type {

  static constexpr bool IsTTR = std::is_same<Pattern, InnerLoopPatternTTR>::value;
  static constexpr bool IsTVR = std::is_same<Pattern, InnerLoopPatternTVR>::value;

  static constexpr size_t Nvector = IsTVR ? Rank : 0;
  static constexpr size_t Nthread = IsTTR ? Rank : 0;
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

namespace meta {

// c++-20 has std:remove_cvref_t that does this same thing
template <typename T>
using base_type = typename std::remove_cv_t<typename std::remove_reference_t<T>>;

template <typename... Ts>
struct TypeList {};

template <typename... Ts>
constexpr int SizeOfList(TypeList<Ts...>) {
  return sizeof...(Ts);
}

template <size_t N, typename>
struct PopList {};

template <typename T, typename... Ts>
struct PopList<1, TypeList<T, Ts...>> {
  using type = T;
  using value = TypeList<Ts...>;
};

template <size_t N, typename T, typename... Ts>
struct PopList<N, TypeList<T, Ts...>> {
  static_assert(N >= 1, "PopList requires N>=1");

 private:
  using pop = PopList<N - 1, TypeList<Ts...>>;

 public:
  using type = typename pop::type;
  using value = typename pop::value;
};

template <typename, typename>
struct AppendList {};

template <typename T, typename... Ts>
struct AppendList<T, TypeList<Ts...>> {
  using value = TypeList<Ts..., T>;
};

template <typename, typename>
struct PrependList {};

template <typename T, typename... Ts>
struct PrependList<T, TypeList<Ts...>> {
  using value = TypeList<T, Ts...>;
};

template <typename, typename>
struct MergeLists {};

template <typename... Ts>
struct MergeLists<TypeList<Ts...>, TypeList<>> {
  using value = TypeList<Ts...>;
};

template <typename... Ts, typename F, typename... Fs>
struct MergeLists<TypeList<Ts...>, TypeList<F, Fs...>> {
  using value = typename MergeLists<TypeList<Ts..., F>, TypeList<Fs...>>::value;
};

template <size_t, typename>
struct SplitList {};

template <typename T, typename... Ts>
struct SplitList<1, TypeList<T, Ts...>> {
  using Left = TypeList<T>;
  using Right = TypeList<Ts...>;
};

template <size_t N, typename T, typename... Ts>
struct SplitList<N, TypeList<T, Ts...>> {
  static_assert(sizeof...(Ts) + 1 >= N, "size of list must be > N");

 private:
  using split = SplitList<N - 1, TypeList<Ts...>>;

 public:
  using Left = typename PrependList<T, typename split::Left>::value;
  using Right = typename split::Right;
};

template <size_t N, typename T>
struct ListOfType {
  using value = typename PrependList<T, typename ListOfType<N - 1, T>::value>::value;
};

template <typename T>
struct ListOfType<1, T> {
  using value = TypeList<T>;
};

template <size_t, size_t, typename>
struct SequenceOfOnes {};

template <size_t VAL, size_t... ones>
struct SequenceOfOnes<0, VAL, std::integer_sequence<size_t, ones...>> {
  using value = typename std::integer_sequence<size_t, ones...>;
};

template <size_t N, size_t VAL, size_t... ones>
struct SequenceOfOnes<N, VAL, std::integer_sequence<size_t, ones...>> {
  using value =
      typename SequenceOfOnes<N - 1, VAL,
                              std::integer_sequence<size_t, VAL, ones...>>::value;
};

template <size_t N, size_t VAL = 1>
using sequence_of_ones = SequenceOfOnes<N - 1, VAL, std::integer_sequence<size_t, VAL>>;

} // namespace meta

namespace meta {

template <size_t, typename>
struct FunctionSignature {};

template <size_t Rank, typename R, typename T, typename Index, typename... Args>
struct FunctionSignature<Rank, R (T::*)(Index, Args...) const> {
 private:
  static constexpr bool team_mbr = std::is_same_v<team_mbr_t, base_type<Index>>;
  using split = SplitList<Rank + team_mbr, TypeList<Index, Args...>>;

 public:
  using IndexND = typename split::Left;
  using FArgs = typename split::Right;
};

template <size_t Rank, typename F>
using function_signature = FunctionSignature<Rank, decltype(&base_type<F>::operator())>;

template <typename>
struct GetLaunchBounds {};

template <>
struct GetLaunchBounds<TypeList<>> {
  using value = TypeList<>;
  using NumInds = std::integral_constant<size_t, 0>;
};

template <typename T, typename... Args>
struct GetLaunchBounds<TypeList<T, Args...>> {
 private:
  template <typename V>
  static constexpr bool is_BoundType() {
    return std::numeric_limits<V>::is_integer || std::is_same_v<V, IndexRange>;
  }

  template <typename V>
  static constexpr size_t NumBnds() {
    if constexpr (!is_BoundType<V>()) {
      return 0;
    }
    return std::is_same_v<V, IndexRange> ? 2 : 1;
  }

  template <size_t N>
  using Rank_t = std::integral_constant<size_t, N>;

  using bound_variants = std::variant<IndexRange, IndexRange &>;
  using bound = base_type<T>;
  using LaunchBounds = GetLaunchBounds<TypeList<Args...>>;

 public:
  using value = typename std::conditional<
      is_BoundType<bound>(), typename PrependList<T, typename LaunchBounds::value>::value,
      TypeList<>>::type;
  using NumInds =
      std::conditional_t<is_BoundType<bound>(),
                         Rank_t<NumBnds<bound>() + LaunchBounds::NumInds::value>,
                         Rank_t<NumBnds<bound>()>>;
};

template <typename>
struct DispatchSignature {};

template <typename Index, typename... AllArgs>
struct DispatchSignature<TypeList<Index, AllArgs...>> {
 private:
  using LB = GetLaunchBounds<TypeList<Index, AllArgs...>>;
  using pop = PopList<SizeOfList(typename LB::value()) + 1, TypeList<Index, AllArgs...>>;

 public:
  using LaunchBounds = typename LB::value;
  using Rank = std::integral_constant<size_t, LB::NumInds::value / 2>;
  using Function = typename pop::type;
  using Args = typename pop::value;
};

template <typename Tag, typename Pattern, size_t Rank, typename... Bounds>
struct DispatchType {
  using BoundType = typename PopList<1, TypeList<Bounds...>>::type;
  static constexpr bool is_IndexRangeBounds =
      std::is_same<IndexRange, meta::base_type<BoundType>>::value;
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

  // fallback simd par_reduce to flat range and force par_scan to flat range
  static constexpr bool is_FlatRange =
      (IsFlatRange || (IsSimdFor && !is_ParFor)) || is_ParScan;
  static constexpr bool is_SimdFor = (IsSimdFor && is_ParFor);
  static constexpr bool is_MDRange = (IsMDRange && !is_ParScan);
  static constexpr bool is_Collapse = TeamPattern::value;
};

} // namespace meta

template <typename, typename, typename>
class FlatFunctor {};

template <typename Function, size_t... Is, typename... FArgs>
class FlatFunctor<Function, std::integer_sequence<size_t, Is...>,
                  meta::TypeList<FArgs...>> {

  static constexpr size_t Rank = sizeof...(Is);
  Kokkos::Array<IndexRange, Rank> ranges;
  Kokkos::Array<int, Rank - 1> strides;
  Function function;

 public:
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION FlatFunctor(const Function _function, IndexRange idr,
                                     Args... args)
      : function(_function), ranges({{idr, args...}}) {
    Initialize();
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION FlatFunctor(const Function _function, Args... args)
      : function(_function) {
    std::array<int, 2 * Rank> indices{{static_cast<int>(args)...}};
    for (int i = 0; i < Rank; i++) {
      ranges[i] = {indices[2 * i], indices[2 * i + 1]};
    }
    Initialize();
  }

  KOKKOS_INLINE_FUNCTION
  void Initialize() {
    for (int ri = 1; ri < Rank; ri++) {
      const int N = ranges[ri].e - ranges[ri].s + 1;
      strides[ri - 1] = N;
      for (int rj = 0; rj < ri - 1; rj++) {
        strides[rj] *= N;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &idx, FArgs... fargs) const {
    int inds[Rank];
    inds[0] = idx;
    for (int i = 1; i < Rank; i++) {
      inds[i] = idx;
      inds[i - 1] /= strides[i - 1];
      for (int j = 0; j < i; j++) {
        inds[i] -= inds[j] * strides[j];
      }
    }
    for (int i = 0; i < Rank; i++) {
      inds[i] += ranges[i].s;
    }

    function(inds[Is]..., std::forward<FArgs>(fargs)...);
  }
};

template <size_t Rank, typename F, typename... Bounds>
KOKKOS_INLINE_FUNCTION auto MakeFlatFunctor(F &function, Bounds &&...bounds) {
  using signature = meta::function_signature<Rank, F>;
  using IndexND = typename signature::IndexND;
  return FlatFunctor<F, std::make_index_sequence<Rank>, typename signature::FArgs>(
      function, std::forward<Bounds>(bounds)...);
}

template <typename, typename, typename>
struct InnerFunctor {};

template <typename Function, typename... Index, size_t... Iteam>
struct InnerFunctor<Function, meta::TypeList<Index...>,
                    std::integer_sequence<size_t, Iteam...>> {
  static constexpr size_t Nteam = sizeof...(Iteam);
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

template <typename Function, size_t... Iteam, size_t... Ithread, size_t... Ivector,
          bool ParForOuter>
class CollapseFunctor<std::integer_sequence<size_t, Iteam...>,
                      std::integer_sequence<size_t, Ithread...>,
                      std::integer_sequence<size_t, Ivector...>, Function, ParForOuter> {

  static constexpr size_t Nteam = sizeof...(Iteam);
  static constexpr size_t Nthread = sizeof...(Ithread);
  static constexpr size_t Nvector = sizeof...(Ivector);
  static constexpr size_t Rank = Nteam + Nthread + Nvector;

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

  template <size_t N, size_t start>
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
    using signature = meta::function_signature<Rank, Function>;
    using ThreadVectorInds =
        typename meta::PopList<Nteam, typename signature::IndexND>::value;

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
                  Kokkos::TeamVectorRange(team_member, 0,
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

template <bool ParForOuter = false, size_t Nteam, size_t Nthread, size_t Nvector,
          typename F, typename... Bounds>
KOKKOS_INLINE_FUNCTION auto
MakeCollapseFunctor(LoopPatternCollapse<Nteam, Nthread, Nvector>, F &function,
                    Bounds &&...bounds) {
  constexpr size_t Rank = Nteam + Nthread + Nvector;
  using signature = meta::function_signature<Rank, F>;
  using IndexND = typename signature::IndexND;

  return CollapseFunctor<std::make_index_sequence<Nteam>,
                         std::make_index_sequence<Nthread>,
                         std::make_index_sequence<Nvector>, F, ParForOuter>(
      function, std::forward<Bounds>(bounds)...);
}

template <typename, size_t, typename, typename>
struct par_dispatch_inner {};

template <typename Pattern, size_t Rank, typename Function, typename... Bounds>
struct par_dispatch_inner<Pattern, Rank, Function, meta::TypeList<Bounds...>> {
  using signature = meta::function_signature<Rank, Function>;
  using LoopPattern =
      typename LoopPatternTeam<Pattern,
                               std::integral_constant<size_t, Rank>>::LoopPattern;

  KOKKOS_FORCEINLINE_FUNCTION
  void dispatch(team_mbr_t team_member, Bounds &&...bounds, Function function) const {
    MakeCollapseFunctor(LoopPattern(), function, std::forward<Bounds>(bounds)...)
        .collapse_inner(team_member, function);
  }
};

template <size_t Rank>
class MDRange {
 public:
  Kokkos::Array<size_t, Rank> lower, upper;

  template <typename... Args>
  MDRange(IndexRange idr, Args... args) {
    std::array<IndexRange, Rank> ranges{{idr, args...}};
    for (int i = 0; i < Rank; i++) {
      lower[i] = ranges[i].s;
      upper[i] = ranges[i].e;
    }
  }

  template <typename... Args>
  MDRange(Args... args) {
    std::array<size_t, 2 * Rank> indices{{static_cast<size_t>(args)...}};
    for (int i = 0; i < Rank; i++) {
      lower[i] = indices[2 * i];
      upper[i] = indices[2 * i + 1];
    }
  }

  template <size_t... Is, size_t... ones>
  auto policy(std::integer_sequence<size_t, Is...>,
              std::integer_sequence<size_t, ones...>, DevExecSpace exec_space) {
    return Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(
        exec_space, {lower[Is]...}, {1 + upper[Is]...},
        {ones..., upper[Rank - 1] + 1 - lower[Rank - 1]});
  }
};

template <size_t Rank, typename... Args>
inline auto MakeMDRange(Args &&...args) {
  return MDRange<Rank>(std::forward<Args>(args)...);
}

template <size_t Rank, typename... Args>
inline auto MakeMDRangePolicy(DevExecSpace exec_space, Args &&...args) {
  using Indices = typename std::make_index_sequence<Rank>;
  using Ones = typename meta::sequence_of_ones<Rank - 1>::value;
  return MakeMDRange<Rank>(std::forward<Args>(args)...)
      .policy(Indices(), Ones(), exec_space);
}

template <size_t Rank>
struct SimdFor {
  template <size_t N>
  using Sequence = std::make_index_sequence<N>;

  std::array<int, Rank - 1> indices;
  MDRange<Rank> mdrange;

  template <typename... Args>
  SimdFor(Args &&...args) : mdrange(std::forward<Args>(args)...) {}

  template <typename Function>
  inline void dispatch(Function &function) {
    dispatch_impl<1>(function);
  }

 private:
  template <typename Function, size_t... Is>
  inline void dispatch_simd(std::integer_sequence<size_t, Is...>, Function &function) {
    for (int i = mdrange.lower[Rank - 1]; i <= mdrange.upper[Rank - 1]; i++) {
#pragma omp simd
      function(indices[Is]..., i);
    }
  }

  template <size_t LoopCount, typename Function>
  inline void dispatch_impl(Function &function) {
    if constexpr (LoopCount < Rank) {
      for (int i = mdrange.lower[LoopCount - 1]; i <= mdrange.upper[LoopCount - 1]; i++) {
        indices[LoopCount - 1] = i;
        dispatch_impl<LoopCount + 1>(function);
      }
    } else {
      dispatch_simd(Sequence<Rank - 1>(), function);
    }
  }
};

template <typename, typename, size_t, typename, typename, typename>
struct par_dispatch_impl {};

template <typename Tag, typename Pattern, size_t Rank, typename Function,
          typename... Bounds, typename... Args>
struct par_dispatch_impl<Tag, Pattern, Rank, Function, meta::TypeList<Bounds...>,
                         meta::TypeList<Args...>> {

  using DType = meta::DispatchType<Tag, Pattern, Rank, Bounds...>;

  static inline void dispatch(std::string name, DevExecSpace exec_space, Bounds &&...ids,
                              Function function, Args &&...args,
                              const int scratch_level = 0,
                              const size_t scratch_size_in_bytes = 0) {

    static_assert(!(DType::is_MDRange && Rank < 2),
                  "Can not launch MDRange with Rank < 2");
    Tag tag;
    PARTHENON_INSTRUMENT_REGION(name)
    if constexpr (DType::is_SimdFor) {
      SimdFor<Rank>(std::forward<Bounds>(ids)...).dispatch(function);
    } else {
      kokkos_dispatch(tag, name,
                      policy(exec_space, std::forward<Bounds>(ids)..., scratch_level,
                             scratch_size_in_bytes),
                      functor(function, std::forward<Bounds>(ids)...),
                      std::forward<Args>(args)...);
    }
  };

  static inline auto policy(DevExecSpace exec_space, Bounds &&...ids,
                            const int scratch_level = 0,
                            const size_t scratch_size_in_bytes = 0) {

    if constexpr (DType::is_FlatRange) {
      int rangeNx = FlattenLaunchBound<Rank>(std::forward<Bounds>(ids)...);
      return Kokkos::RangePolicy<>(exec_space, 0, rangeNx);

    } else if constexpr (DType::is_MDRange) {
      return MakeMDRangePolicy<Rank>(exec_space, std::forward<Bounds>(ids)...);

    } else if constexpr (DType::is_SimdFor) {
      return loop_pattern_simdfor_tag;

    } else if constexpr (DType::is_Collapse) {
      int rangeNx =
          FlattenLaunchBound<DType::TeamPattern::Nteam>(std::forward<Bounds>(ids)...);
      return team_policy(exec_space, rangeNx, Kokkos::AUTO)
          .set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes));
    }
  };

  static inline auto functor(Function function, Bounds &&...ids) {
    if constexpr (DType::is_FlatRange) {
      return MakeFlatFunctor<Rank>(function, std::forward<Bounds>(ids)...);
    } else if constexpr (DType::is_MDRange || DType::is_SimdFor) {
      return function;
    } else if constexpr (DType::is_Collapse) {
      constexpr bool ParForOuter = std::is_same_v<OuterLoopPatternTeams, Pattern>;
      return MakeCollapseFunctor<ParForOuter>(typename DType::TeamPattern::LoopPattern(),
                                              function, std::forward<Bounds>(ids)...);
    }
  }

 private:
  template <size_t NCollapse>
  static inline int FlattenLaunchBound(Bounds &&...ids) {
    static_assert(NCollapse <= Rank, "Can't flatten more loops than rank");
    int rangeNx = 1;
    if constexpr (DType::is_IndexRangeBounds) {
      std::array<IndexRange, Rank> ranges{{ids...}};
      for (int i = 0; i < NCollapse; i++) {
        rangeNx *= ranges[i].e - ranges[i].s + 1;
      }
    } else {
      int indices[sizeof...(Bounds)] = {static_cast<int>(ids)...};
      for (int i = 0; i < 2 * NCollapse; i += 2) {
        rangeNx *= indices[i + 1] - indices[i] + 1;
      }
    }
    return rangeNx;
  }
};

template <typename Tag, typename Pattern, typename... AllArgs>
inline typename std::enable_if<std::is_same<Pattern, LoopPatternFlatRange>::value ||
                                   std::is_same<Pattern, LoopPatternMDRange>::value ||
                                   std::is_same<Pattern, LoopPatternTPTTR>::value ||
                                   std::is_same<Pattern, LoopPatternTPTVR>::value ||
                                   std::is_same<Pattern, LoopPatternTPTTRTVR>::value ||
                                   std::is_same<Pattern, LoopPatternSimdFor>::value,
                               void>::type
par_dispatch(Pattern, std::string name, DevExecSpace exec_space, AllArgs &&...args) {
  using dispatchsig = meta::DispatchSignature<meta::TypeList<AllArgs...>>;
  static constexpr size_t Rank = dispatchsig::Rank::value;
  using Function = typename dispatchsig::Function;         // functor type
  using LaunchBounds = typename dispatchsig::LaunchBounds; // list of index types
  using Args = typename dispatchsig::Args;                 //

  if constexpr (Rank > 1 && std::is_same_v<dispatch_impl::ParallelScanDispatch, Tag>) {
    static_assert(always_false<Tag>, "par_scan only for 1D loops");
  }
  par_dispatch_impl<Tag, Pattern, Rank, Function, LaunchBounds, Args>::dispatch(
      name, exec_space, std::forward<AllArgs>(args)...);
}

template <typename Tag, typename Pattern, typename Function, class... Args>
inline typename std::enable_if<sizeof...(Args) <= 1, void>::type
par_dispatch(Pattern p, const std::string &name, DevExecSpace exec_space,
             const IndexRange &r, const Function &function, Args &&...args) {
  par_dispatch<Tag>(p, name, exec_space, r.s, r.e, function, std::forward<Args>(args)...);
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
              size_t scratch_size_in_bytes, const int scratch_level, AllArgs &&...args) {
  using dispatchsig = meta::DispatchSignature<meta::TypeList<AllArgs...>>;
  static constexpr size_t Rank = dispatchsig::Rank::value;
  using Function = typename dispatchsig::Function;         // functor type
  using LaunchBounds = typename dispatchsig::LaunchBounds; // list of index types
  using Args = typename dispatchsig::Args;                 //
  using LoopPattern = LoopPatternTeam<Pattern, std::integral_constant<size_t, Rank>>;
  using Tag = dispatch_impl::ParallelForDispatch;

  par_dispatch_impl<Tag, Pattern, Rank, Function, LaunchBounds, Args>::dispatch(
      name, exec_space, std::forward<AllArgs>(args)..., scratch_level,
      scratch_size_in_bytes);
}

template <typename... Args>
inline void par_for_outer(const std::string &name, Args &&...args) {
  par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, name, DevExecSpace(),
                std::forward<Args>(args)...);
}

template <typename Pattern, typename... AllArgs>
KOKKOS_FORCEINLINE_FUNCTION
    std::enable_if_t<std::is_same_v<Pattern, InnerLoopPatternTTR> ||
                         std::is_same_v<Pattern, InnerLoopPatternTVR> ||
                         std::is_same_v<Pattern, InnerLoopPatternSimdFor>,
                     void>
    par_for_inner(Pattern, team_mbr_t team_member, AllArgs &&...args) {

  using dispatchsig = meta::DispatchSignature<meta::TypeList<AllArgs...>>;
  constexpr size_t Rank = dispatchsig::Rank::value;
  using Function = typename dispatchsig::Function;
  using LaunchBounds = typename dispatchsig::LaunchBounds;
  if constexpr (std::is_same_v<Pattern, InnerLoopPatternSimdFor>) {
    using Args = typename dispatchsig::Args;
    par_dispatch_impl<dispatch_impl::ParallelForDispatch, LoopPatternSimdFor, Rank,
                      Function, LaunchBounds, Args>()
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
