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
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "parthenon_array_generic.hpp"
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
template <int nm, int ni>
struct PatternBase {
  static constexpr int nmiddle = nm;
  static constexpr int ninner = ni;
};

// Translates a non-Kokkos standard C++ nested `for` loop where the innermost
// `for` is decorated with a #pragma omp simd IMPORTANT: This only works on CPUs
static struct LoopPatternSimdFor : public PatternBase<0, 1> {
} loop_pattern_simdfor_tag;
// Translates to a Kokkos 1D range (Kokkos::RangePolicy) where the wrapper takes
// care of the (hidden) 1D index to `n`, `k`, `j`, `i indices conversion
static struct LoopPatternFlatRange : public PatternBase<0, 0> {
} loop_pattern_flatrange_tag;
// Translates to a Kokkos multi dimensional  range (Kokkos::MDRangePolicy) with
// a 1:1 indices matching
static struct LoopPatternMDRange : public PatternBase<0, 0> {
} loop_pattern_mdrange_tag;
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::TeamThreadRange
static struct LoopPatternTPTTR : public PatternBase<0, 1> {
} loop_pattern_tpttr_tag;
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::ThreadVectorRange
static struct LoopPatternTPTVR : public PatternBase<0, 1> {
} loop_pattern_tptvr_tag;
// Translates to a Kokkos::TeamPolicy with a middle Kokkos::TeamThreadRange and
// inner Kokkos::ThreadVectorRange
static struct LoopPatternTPTTRTVR : public PatternBase<1, 1> {
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

template <class BoundTranslator_t, class Function>
inline void RawFor(const BoundTranslator_t &bound_trans, const Function &function) {
  auto idxer = bound_trans.template GetIndexer<0, BoundTranslator_t::rank>();
  for (int idx = 0; idx < idxer.size(); ++idx) {
    std::apply(function, idxer(idx));
  }
}

template <class BoundTranslator_t, class Function>
inline void SimdFor(const BoundTranslator_t &bound_trans, const Function &function) {
  if constexpr (BoundTranslator_t::rank > 1) {
    constexpr int inner_start_rank = BoundTranslator_t::rank - 1;
    auto idxer = bound_trans.template GetIndexer<0, inner_start_rank>();
    const int istart = bound_trans[inner_start_rank].s;
    const int iend = bound_trans[inner_start_rank].e;
    // Loop over all outer indices using a flat indexer
    for (int idx = 0; idx < idxer.size(); ++idx) {
      auto idx_tuple = std::tuple_cat(idxer(idx), std::make_tuple(int{0}));
      int &i = std::get<inner_start_rank>(idx_tuple);
#pragma omp simd
      for (i = istart; i <= iend; ++i) {
        std::apply(function, idx_tuple);
      }
    }
  } else { // Easier to just explicitly specialize for 1D Simd loop
#pragma omp simd
    for (int i = bound_trans[0].s; i <= bound_trans[0].e; ++i)
      function(i);
  }
}
} // namespace dispatch_impl

// Struct for translating between loop bounds given in terms of IndexRanges and loop
// bounds given in terms of raw integers
template <class... Bound_ts>
struct BoundTranslator {
  using Bound_tl = TypeList<Bound_ts...>;
  static constexpr bool are_integers = std::is_integral_v<
      typename std::remove_reference<typename Bound_tl::template type<0>>::type>;
  static constexpr uint rank = sizeof...(Bound_ts) / (1 + are_integers);

  std::array<IndexRange, rank> bounds;

  KOKKOS_INLINE_FUNCTION
  IndexRange &operator[](int i) { return bounds[i]; }
  
  KOKKOS_INLINE_FUNCTION
  const IndexRange &operator[](int i) const { return bounds[i]; }
  
  KOKKOS_INLINE_FUNCTION
  explicit BoundTranslator(Bound_ts... bounds_in) {
    if constexpr (are_integers) {
      std::array<int64_t, 2 * rank> bounds_arr{static_cast<int64_t>(bounds_in)...};
      for (int r = 0; r < rank; ++r) {
        bounds[r].s = static_cast<int64_t>(bounds_arr[2 * r]);
        bounds[r].e = static_cast<int64_t>(bounds_arr[2 * r + 1]);
      }
    } else {
      bounds = std::array<IndexRange, rank>{bounds_in...};
    }
  }

  template <int RankStart, int RankStop>
  auto GetKokkosFlatRangePolicy(DevExecSpace exec_space) const {
    constexpr int ndim = RankStop - RankStart;
    static_assert(ndim > 0, "Need a valid range of ranks");
    static_assert(RankStart >= 0, "Need a valid range of ranks");
    static_assert(RankStop <= rank, "Need a valid range of ranks");
    int64_t npoints = 1;
    for (int d = RankStart; d < RankStop; ++d)
      npoints *= (bounds[d].e + 1 - bounds[d].s);
    return Kokkos::Experimental::require(
        Kokkos::RangePolicy<>(exec_space, 0, npoints),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  }

  template <int RankStart, int RankStop>
  auto GetKokkosMDRangePolicy(DevExecSpace exec_space) const {
    constexpr int ndim = RankStop - RankStart;
    static_assert(ndim > 1, "Need a valid range of ranks");
    static_assert(RankStart >= 0, "Need a valid range of ranks");
    static_assert(RankStop <= rank, "Need a valid range of ranks");
    Kokkos::Array<int64_t, ndim> start, end, tile;
    for (int d = 0; d < ndim; ++d) {
      start[d] = bounds[d + RankStart].s;
      end[d] = bounds[d + RankStart].e + 1;
      tile[d] = 1;
    }
    tile[ndim - 1] = end[ndim - 1] - start[ndim - 1];
    return Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<ndim>>(exec_space, start, end, tile),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  }

  template <int RankStart, std::size_t... Is>
  KOKKOS_INLINE_FUNCTION
  auto GetIndexer(std::index_sequence<Is...>) const {
    return MakeIndexer(
        std::pair<int, int>(bounds[Is + RankStart].s, bounds[Is + RankStart].e)...);
  }

  template <int RankStart, int RankStop>
  KOKKOS_INLINE_FUNCTION
  auto GetIndexer() const {
    constexpr int ndim = RankStop - RankStart;
    static_assert(ndim > 0, "Need a valid range of ranks");
    static_assert(RankStart >= 0, "Need a valid range of ranks");
    static_assert(RankStop <= rank, "Need a valid range of ranks");
    return GetIndexer<RankStart>(std::make_index_sequence<ndim>());
  }
};

template <class... Bound_ts>
struct BoundTranslator<TypeList<Bound_ts...>> : public BoundTranslator<Bound_ts...> {};

template <class Tag, class Pattern, class Bound_tl, class Function, class ExtArgs_tl,
          class FuncExtArgs_tl>
struct par_dispatch_funct {};

template <class Tag, class Pattern, class... Bound_ts, class Function,
          class... ExtraArgs_ts, class... FuncExtArgs_ts>
struct par_dispatch_funct<Tag, Pattern, TypeList<Bound_ts...>, Function,
                          TypeList<ExtraArgs_ts...>, TypeList<FuncExtArgs_ts...>> {
  template <class... Args>
  void operator()(Pattern pattern, Args &&...args) {
    using rt_t = BoundTranslator<Bound_ts...>;
    constexpr int total_rank = rt_t::rank;
    DoLoop(std::make_index_sequence<total_rank>(),
           std::make_index_sequence<total_rank - Pattern::nmiddle - Pattern::ninner>(),
           std::make_index_sequence<Pattern::nmiddle>(),
           std::make_index_sequence<Pattern::ninner>(), std::forward<Args>(args)...);
  }

  template <std::size_t... Is, std::size_t... OuterIs, std::size_t... MidIs,
            std::size_t... InnerIs>
  void DoLoop(std::index_sequence<Is...>, std::index_sequence<OuterIs...>,
              std::index_sequence<MidIs...>, std::index_sequence<InnerIs...>,
              const std::string &name, DevExecSpace exec_space, Bound_ts &&...bounds,
              const Function &function, ExtraArgs_ts &&...args) {
    using rt_t = BoundTranslator<Bound_ts...>;
    auto bound_trans = rt_t(std::forward<Bound_ts>(bounds)...);

    constexpr int total_rank = rt_t::rank;
    [[maybe_unused]] constexpr int inner_start_rank = total_rank - Pattern::ninner;
    [[maybe_unused]] constexpr int middle_start_rank = total_rank - Pattern::ninner - Pattern::nmiddle;

    constexpr bool SimdRequested = std::is_same_v<Pattern, LoopPatternSimdFor>;
    constexpr bool MDRangeRequested = std::is_same_v<Pattern, LoopPatternMDRange>;
    constexpr bool FlatRangeRequested = std::is_same_v<Pattern, LoopPatternFlatRange>;
    constexpr bool TPTTRRequested = std::is_same_v<Pattern, LoopPatternTPTTR>;
    constexpr bool TPTVRRequested = std::is_same_v<Pattern, LoopPatternTPTVR>;
    constexpr bool TPTTRTVRRequested = std::is_same_v<Pattern, LoopPatternTPTTRTVR>;

    constexpr bool doSimdFor = SimdRequested && sizeof...(args) == 0;
    constexpr bool doMDRange = MDRangeRequested && total_rank > 1;
    constexpr bool doTPTTRTV = TPTTRTVRRequested && total_rank > 2;
    constexpr bool doTPTTR = TPTTRRequested && total_rank > 1;
    constexpr bool doTPTVR = TPTVRRequested && total_rank > 1;

    constexpr bool doFlatRange =
        FlatRangeRequested || (SimdRequested && !doSimdFor) ||
        (MDRangeRequested && !doMDRange) || (TPTTRTVRRequested && !doTPTTRTV) ||
        (TPTTRRequested && !doTPTTR) || (TPTVRRequested && !doTPTVR);

    if constexpr (doSimdFor) {
      static_assert(std::is_same_v<dispatch_impl::ParallelForDispatch, Tag>,
                    "Only par_for is supported for simd_for pattern");
      dispatch_impl::SimdFor(bound_trans, function);
    } else if constexpr (doMDRange) {
      static_assert(total_rank > 1,
                    "MDRange pattern only works for multi-dimensional loops.");
      kokkos_dispatch(
          Tag(), name,
          bound_trans.template GetKokkosMDRangePolicy<0, total_rank>(exec_space),
          function, std::forward<ExtraArgs_ts>(args)...);
    } else if constexpr (doFlatRange) {
      auto idxer = bound_trans.template GetIndexer<0, total_rank>();
      kokkos_dispatch(
          Tag(), name,
          bound_trans.template GetKokkosFlatRangePolicy<0, total_rank>(exec_space),
          KOKKOS_LAMBDA(const int &idx, FuncExtArgs_ts &&...fargs) {
            auto idx_tuple = idxer(idx);
            function(std::get<Is>(idx_tuple)..., std::forward<FuncExtArgs_ts>(fargs)...);
          },
          std::forward<ExtraArgs_ts>(args)...);
    } else if constexpr (doTPTTR) {
      auto outer_idxer = bound_trans.template GetIndexer<0, inner_start_rank>();
      const int istart = bound_trans[inner_start_rank].s;
      const int iend = bound_trans[inner_start_rank].e;
      Kokkos::parallel_for(
          name, team_policy(exec_space, outer_idxer.size(), Kokkos::AUTO),
          KOKKOS_LAMBDA(team_mbr_t team_member) {
            const auto idx_tuple = outer_idxer(team_member.league_rank());
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team_member, istart, iend + 1),
                [&](const int i) {
                  function(std::get<OuterIs>(idx_tuple)...,
                           std::get<middle_start_rank + MidIs>(idx_tuple)..., i);
                });
          });
    } else if constexpr (doTPTVR) {
      auto outer_idxer = bound_trans.template GetIndexer<0, inner_start_rank>();
      const int istart = bound_trans[inner_start_rank].s;
      const int iend = bound_trans[inner_start_rank].e;
      Kokkos::parallel_for(
          name, team_policy(exec_space, outer_idxer.size(), Kokkos::AUTO),
          KOKKOS_LAMBDA(team_mbr_t team_member) {
            const auto idx_tuple = outer_idxer(team_member.league_rank());
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange<>(team_member, istart, iend + 1),
                [&](const int i) {
                  function(std::get<OuterIs>(idx_tuple)...,
                           std::get<middle_start_rank + MidIs>(idx_tuple)..., i);
                });
          });
    } else if constexpr (doTPTTRTV) {
      auto outer_idxer = bound_trans.template GetIndexer<0, middle_start_rank>();
      auto middle_idxer =
          bound_trans.template GetIndexer<middle_start_rank, inner_start_rank>();
      auto inner_idxer = bound_trans.template GetIndexer<inner_start_rank, total_rank>();
      Kokkos::parallel_for(
          name, team_policy(exec_space, outer_idxer.size(), Kokkos::AUTO),
          KOKKOS_LAMBDA(team_mbr_t team_member) {
            const auto idx_out = outer_idxer(team_member.league_rank());
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team_member, 0, middle_idxer.size()),
                [&](const int midx) {
                  const auto idx_mid = middle_idxer(midx);
                  Kokkos::parallel_for(
                      Kokkos::ThreadVectorRange<>(team_member, 0, inner_idxer.size()),
                      [&](const int iidx) {
                        const auto idx_in = inner_idxer(iidx);
                        function(std::get<OuterIs>(idx_out)...,
                                 std::get<MidIs>(idx_mid)...,
                                 std::get<InnerIs>(idx_in)...);
                      });
                });
          });
    } else {
      printf("Loop pattern unsupported.");
    }
  }
};

template <typename Tag, typename Pattern, class... Args>
void par_dispatch(Pattern pattern, const std::string &name, DevExecSpace exec_space,
                  Args &&...args) {
  using arg_tl = TypeList<Args...>;
  constexpr std::size_t func_idx = FirstFuncIdx<arg_tl>();
  static_assert(func_idx < arg_tl::n_types,
                "Apparently we didn't successfully find a function.");
  using func_t = typename arg_tl::template type<func_idx>;
  constexpr int loop_rank = BoundTranslator<
      typename arg_tl::template continuous_sublist<0, func_idx - 1>>::rank;

  using func_sig_tl = typename FuncSignature<func_t>::arg_types_tl;
  // The first loop_rank arguments of the function are just indices, the rest are extra
  // arguments for reductions, etc.
  using bounds_tl = typename arg_tl::template continuous_sublist<0, func_idx - 1>;
  using extra_arg_tl = typename arg_tl::template continuous_sublist<func_idx + 1>;
  using func_sig_extra_tl = typename func_sig_tl::template continuous_sublist<loop_rank>;
  par_dispatch_funct<Tag, Pattern, bounds_tl, func_t, extra_arg_tl, func_sig_extra_tl>
      loop_funct;
  loop_funct(pattern, name, exec_space, std::forward<Args>(args)...);
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

template <class TL, class TLFunction>
struct par_for_outer_funct_t;

template <class... Bound_ts, class Function>
struct par_for_outer_funct_t<TypeList<Bound_ts...>, TypeList<Function>> {
  template <class Pattern>
  void operator()(Pattern, const std::string &name, DevExecSpace exec_space,
                  size_t scratch_size_in_bytes, const int scratch_level,
                  Bound_ts &&...bounds, const Function &function) {
    using rt_t = BoundTranslator<Bound_ts...>;
    auto bound_trans = rt_t(std::forward<Bound_ts>(bounds)...);
    auto idxer = bound_trans.template GetIndexer<0, rt_t::rank>();

    if constexpr (std::is_same_v<OuterLoopPatternTeams, Pattern>) {
      team_policy policy(exec_space, idxer.size(), Kokkos::AUTO);

      Kokkos::parallel_for(
          name,
          policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),
          KOKKOS_LAMBDA(team_mbr_t team_member) {
            std::apply(function, std::tuple_cat(std::make_tuple(team_member),
                                                idxer(team_member.league_rank())));
          });
    } else {
      PARTHENON_FAIL("Unsupported par_for_outer loop pattern.");
    }
  }
};

template <class... Args>
inline void par_for_outer(OuterLoopPatternTeams, const std::string &name,
                          DevExecSpace exec_space, size_t scratch_size_in_bytes,
                          const int scratch_level, Args &&...args) {
  using arg_tl = TypeList<Args...>;
  constexpr int nm2 = arg_tl::n_types - 2;
  constexpr int nm1 = arg_tl::n_types - 1;
  using bound_tl = typename arg_tl::template continuous_sublist<0, nm2>;
  using function_tl = typename arg_tl::template continuous_sublist<nm1, nm1>;
  par_for_outer_funct_t<bound_tl, function_tl> par_for_outer_funct;
  par_for_outer_funct(OuterLoopPatternTeams(), name, exec_space, scratch_size_in_bytes,
                      scratch_level, std::forward<Args>(args)...);
}

template <typename... Args>
inline void par_for_outer(const std::string &name, Args &&...args) {
  par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, name, DevExecSpace(),
                std::forward<Args>(args)...);
}

template <class TL, class TLFunction>
struct par_for_inner_funct_t;

template <class... Bound_ts, class Function>
struct par_for_inner_funct_t<TypeList<Bound_ts...>, TypeList<Function>> {
  KOKKOS_DEFAULTED_FUNCTION
  par_for_inner_funct_t() = default;

  template <class Pattern>
  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(Pattern, team_mbr_t team_member, Bound_ts &&...bounds,
                  const Function &function) {
    using rt_t = BoundTranslator<Bound_ts...>;
    auto bound_trans = rt_t(std::forward<Bound_ts>(bounds)...);

    if constexpr (std::is_same_v<InnerLoopPatternSimdFor, Pattern>) {
      dispatch_impl::SimdFor(bound_trans, function);
    } else if constexpr (std::is_same_v<InnerLoopPatternTTR, Pattern>) {
      auto idxer = bound_trans.template GetIndexer<0, rt_t::rank>();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, idxer.size()),
                           [&](const int &idx) { std::apply(function, idxer(idx)); });
    } else if constexpr (std::is_same_v<InnerLoopPatternTVR, Pattern>) {
      auto idxer = bound_trans.template GetIndexer<0, rt_t::rank>();
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, idxer.size()),
                           [&](const int &idx) { std::apply(function, idxer(idx)); });
    } else {
      PARTHENON_FAIL("Unsupported par_for_outer loop pattern.");
    }
  }
};

template <class Pattern, class... Args>
KOKKOS_FORCEINLINE_FUNCTION
void par_for_inner(Pattern pattern, Args &&...args) {
  using arg_tl = TypeList<Args...>;
  constexpr int nm2 = arg_tl::n_types - 2;
  constexpr int nm1 = arg_tl::n_types - 1;
  // First Arg of Args is team_member
  using bound_tl = typename arg_tl::template continuous_sublist<1, nm2>;
  using function_tl = typename arg_tl::template continuous_sublist<nm1, nm1>;
  par_for_inner_funct_t<bound_tl, function_tl> par_for_inner_funct;
  par_for_inner_funct(pattern, std::forward<Args>(args)...);
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
