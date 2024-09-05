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

#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "impl/Kokkos_Tools_Generic.hpp"
#include "kokkos_types.hpp"
#include "loop_bounds.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/indexer.hpp"
#include "utils/instrument.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

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
// Translates to a Kokkos::TeamPolicy that collapse Nthread & Nvector inner loop collapses
template <std::size_t num_thread, std::size_t num_vector>
struct LoopPatternCollapse : std::true_type {
  static constexpr std::size_t Nthread = num_thread;
  static constexpr std::size_t Nvector = num_vector;
};
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::TeamThreadRange
using LoopPatternTPTTR = LoopPatternCollapse<1, 0>;
constexpr auto loop_pattern_tpttr_tag = LoopPatternTPTTR();
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::ThreadVectorRange
using LoopPatternTPTVR = LoopPatternCollapse<0, 1>;
constexpr auto loop_pattern_tptvr_tag = LoopPatternTPTVR();
// Translates to a Kokkos::TeamPolicy with a middle Kokkos::TeamThreadRange and
// inner Kokkos::ThreadVectorRange
using LoopPatternTPTTRTVR = LoopPatternCollapse<1, 1>;
constexpr auto loop_pattern_tpttrtvr_tag = LoopPatternTPTTRTVR();
// Translates to an outer team policy
using LoopPatternTeamOuter = LoopPatternCollapse<0, 0>;
constexpr auto loop_pattern_team_outer_tag = LoopPatternTeamOuter();
// Used to catch undefined behavior as it results in throwing an error
static struct LoopPatternUndefined {
} loop_pattern_undefined_tag;

// Tags for Nested parallelism

// Translates to outermost loop being a Kokkos::TeamPolicy for par_for_outer like loops
static struct OuterLoopPatternTeams {
} outer_loop_pattern_teams_tag;
// Inner loop pattern tags must be constexpr so they're available on device
// Translate to a Kokkos::TeamVectorRange as innermost loop (single index)
using InnerLoopPatternTVR = LoopPatternCollapse<0, 0>;
constexpr auto inner_loop_pattern_tvr_tag = InnerLoopPatternTVR();
// Translates to a Kokkos::TeamThreadRange as innermost loop
using InnerLoopPatternTTR = LoopPatternCollapse<0, 0>;
constexpr auto inner_loop_pattern_ttr_tag = InnerLoopPatternTTR();
// Translate to a non-Kokkos plain C++ innermost loop (single index)
// decorated with #pragma omp simd
// IMPORTANT: currently only supported on CPUs
using InnerLoopPatternSimdFor = LoopPatternCollapse<0, 0>;
constexpr auto inner_loop_pattern_simdfor_tag = InnerLoopPatternSimdFor();

// trait to track if pattern requests any type of hierarchial parallelism
template <typename Pattern, typename T = void>
struct UsesHierarchialPar : std::false_type {
  static constexpr std::size_t Nvector = 0;
  static constexpr std::size_t Nthread = 0;
};

template <std::size_t num_thread, std::size_t num_vector>
struct UsesHierarchialPar<LoopPatternCollapse<num_thread, num_vector>> : std::true_type {
  static constexpr std::size_t Nthread = num_thread;
  static constexpr std::size_t Nvector = num_vector;
};
template <>
struct UsesHierarchialPar<OuterLoopPatternTeams> : std::true_type {
  static constexpr std::size_t Nvector = 0;
  static constexpr std::size_t Nthread = 0;
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

template <typename... AllArgs>
struct DispatchSignature<TypeList<AllArgs...>> {
 private:
  using TL = TypeList<AllArgs...>;
  static constexpr std::size_t func_idx = FirstFuncIdx<TL>();
  static_assert(sizeof...(AllArgs) > func_idx,
                "Couldn't determine functor index from dispatch args");

 public:
  using LoopBounds = typename TL::template continuous_sublist<0, func_idx - 1>;
  using Translator = LoopBoundTranslator<LoopBounds>;
  static constexpr std::size_t Rank = Translator::Rank;
  using Function = typename TL::template type<func_idx>;
  using Args = typename TL::template continuous_sublist<func_idx + 1>;
};

enum class LoopPattern { flat, md, simd, outer, collapse, undef };

template <LoopPattern Pattern>
struct LoopPatternTag {};

template <typename Tag, typename Pattern, typename... Bounds>
struct DispatchType {
  using Translator = LoopBoundTranslator<Bounds...>;
  static constexpr std::size_t Rank = Translator::Rank;

  using TeamPattern = UsesHierarchialPar<Pattern>; // false_type unless we use
                                                   // an outer team policy
  static constexpr bool is_ParFor =
      std::is_same<Tag, dispatch_impl::ParallelForDispatch>::value;
  static constexpr bool is_ParScan =
      std::is_same<Tag, dispatch_impl::ParallelScanDispatch()>::value;

  static constexpr bool IsFlatRange = std::is_same<Pattern, LoopPatternFlatRange>::value;
  static constexpr bool IsMDRange = std::is_same<Pattern, LoopPatternMDRange>::value;
  static constexpr bool IsSimdFor = std::is_same<Pattern, LoopPatternSimdFor>::value;

  // check any confilcts with the requested pattern
  // and return the actual one we use
  static constexpr LoopPattern GetPatternTag() {
    using LP = LoopPattern;

    if constexpr (is_ParScan) {
      return LP::flat;
    } else if constexpr (IsFlatRange) {
      return LP::flat;
    } else if constexpr (IsSimdFor) {
      return is_ParFor ? LP::simd : LP::flat;
    } else if constexpr (IsMDRange) {
      return LP::md;
    } else if constexpr (std::is_same_v<Pattern, OuterLoopPatternTeams>) {
      return LP::outer;
    } else if constexpr (TeamPattern::value) {
      return LP::collapse;
    }

    return LP::undef;
  }
};

template <std::size_t Rank, typename IdxTeam, std::size_t Nteam, std::size_t Nthread,
          std::size_t Nvector, typename Function, typename... ExtraFuncArgs>
struct dispatch_collapse {
  IdxTeam idxer_team;
  Kokkos::Array<IndexRange, Rank> bound_arr;
  Function function;

  KOKKOS_FORCEINLINE_FUNCTION
  dispatch_collapse(IdxTeam idxer, Kokkos::Array<IndexRange, Rank> bounds, Function func)
      : idxer_team(idxer), bound_arr(bounds), function(func) {}

  template <std::size_t... TeamIs, std::size_t... ThreadIs, std::size_t... VectorIs,
            typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION void
  execute(std::integer_sequence<std::size_t, TeamIs...>,
          std::integer_sequence<std::size_t, ThreadIs...>,
          std::integer_sequence<std::size_t, VectorIs...>, team_mbr_t team_member,
          Args &&...args) const {
    auto inds_team = idxer_team.GetIdxArray(team_member.league_rank());
    if constexpr (Nthread > 0) {
      const auto idxer_thread =
          MakeIndexer(Kokkos::Array<IndexRange, Nthread>{bound_arr[ThreadIs]...});
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange<>(team_member, 0, idxer_thread.size()),
          [&](const int idThread, ExtraFuncArgs... fargs) {
            const auto inds_thread = idxer_thread.GetIdxArray(idThread);
            if constexpr (Nvector > 0) {
              static_assert(Nvector * Nthread == 0 || sizeof...(Args) == 0,
                            "thread + vector range pattern only supported for par_for ");
              const auto idxer_vector = MakeIndexer(
                  Kokkos::Array<IndexRange, Nvector>{bound_arr[Nthread + VectorIs]...});
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team_member, 0, idxer_vector.size()),
                  [&](const int idVector) {
                    const auto inds_vector = idxer_vector.GetIdxArray(idVector);
                    function(inds_team[TeamIs]..., inds_thread[ThreadIs]...,
                             inds_vector[VectorIs]...,
                             std::forward<ExtraFuncArgs>(fargs)...);
                  });
            } else {
              function(inds_team[TeamIs]..., inds_thread[ThreadIs]...,
                       std::forward<ExtraFuncArgs>(fargs)...);
            }
          },
          std::forward<Args>(args)...);
    } else {
      const auto idxer_vector = MakeIndexer(
          Kokkos::Array<IndexRange, Nvector>{bound_arr[Nthread + VectorIs]...});
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(team_member, 0, idxer_vector.size()),
          [&](const int idVector, ExtraFuncArgs... fargs) {
            const auto inds_vector = idxer_vector.GetIdxArray(idVector);
            function(inds_team[TeamIs]..., inds_vector[VectorIs]...,
                     std::forward<ExtraFuncArgs>(fargs)...);
          },
          std::forward<Args>(args)...);
    }
  }

  template <std::size_t N>
  using sequence = std::make_index_sequence<N>;
  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(team_mbr_t team_member) const {
    execute(sequence<Nteam>(), sequence<Nthread>(), sequence<Nvector>(), team_member);
  }
};

template <std::size_t Rank, std::size_t Nteam, std::size_t Nthread, std::size_t Nvector,
          typename IdxTeam, typename Function, typename... ExtraFuncArgs>
KOKKOS_FORCEINLINE_FUNCTION auto
MakeCollapse(IdxTeam idxer, Kokkos::Array<IndexRange, Rank> bounds, Function func) {
  return dispatch_collapse<Rank, IdxTeam, Nteam, Nthread, Nvector, Function,
                           ExtraFuncArgs...>(idxer, bounds, func);
}

template <std::size_t Rank, std::size_t... OuterIs, typename Function>
KOKKOS_INLINE_FUNCTION void SimdFor(std::index_sequence<OuterIs...>, Function function,
                                    Kokkos::Array<IndexRange, Rank> bounds) {
  if constexpr (Rank == 1) {
#pragma omp simd
    for (int i = bounds[0].s; i <= bounds[0].e; i++) {
      function(i);
    }
  } else {
    const auto idxer =
        MakeIndexer(std::pair<int, int>(bounds[OuterIs].s, bounds[OuterIs].e)...);
    for (int idx = 0; idx < idxer.size(); idx++) {
      const auto indices = idxer.GetIdxArray(idx);
#pragma omp simd
      for (int i = bounds[Rank - 1].s; i <= bounds[Rank - 1].e; i++) {
        function(indices[OuterIs]..., i);
      }
    }
  }
}

template <typename, typename, typename, typename, typename>
struct par_disp_inner_impl {};

template <typename Pattern, typename Function, typename... Bounds, typename... Args,
          typename... ExtraFuncArgs>
struct par_disp_inner_impl<Pattern, Function, TypeList<Bounds...>, TypeList<Args...>,
                           TypeList<ExtraFuncArgs...>> {
  using bound_translator = LoopBoundTranslator<Bounds...>;
  static constexpr std::size_t Rank = bound_translator::Rank;
  using TeamPattern =
      UsesHierarchialPar<Pattern, std::integral_constant<std::size_t, Rank>>;
  template <std::size_t N>
  using sequence = std::make_index_sequence<N>;

  KOKKOS_FORCEINLINE_FUNCTION void execute(team_mbr_t team_member, Bounds &&...bounds,
                                           Function function, Args &&...args) {
    auto bound_arr = bound_translator().GetIndexRanges(std::forward<Bounds>(bounds)...);
    constexpr bool isSimdFor = std::is_same_v<InnerLoopPatternSimdFor, Pattern>;
    if constexpr (isSimdFor) {
      static_assert(!isSimdFor ||
                        (isSimdFor && std::is_same_v<DevExecSpace, HostExecSpace>),
                    "par_inner simd for pattern only supported on HostExecSpace");
      SimdFor(std::make_index_sequence<Rank - 1>(), function, bound_arr);
    } else {
      auto idxer = Indexer<>();
      constexpr std::size_t Nthread = TeamPattern::Nthread;
      constexpr std::size_t Nvector = TeamPattern::Nvector;
      MakeCollapse<Rank, 0, Nthread, Nvector, ExtraFuncArgs...>(idxer, bound_arr,
                                                                function)
          .execute(sequence<0>(), sequence<Nthread>(), sequence<Nvector>(), team_member,
                   std::forward<Args>(args)...);
    }
  }
};

template <typename Pattern, typename... AllArgs>
KOKKOS_FORCEINLINE_FUNCTION void par_disp_inner(Pattern, team_mbr_t team_member,
                                                AllArgs &&...args) {
  using dispatchsig = DispatchSignature<TypeList<AllArgs...>>;
  constexpr std::size_t Rank = dispatchsig::Rank;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;
  using Args = typename dispatchsig::Args;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;
  par_disp_inner_impl<Pattern, Function, LoopBounds, Args, ExtraFuncArgs>().execute(
      team_member, std::forward<AllArgs>(args)...);
}

template <typename, typename, typename, typename, typename, typename>
struct par_dispatch_impl {};

template <typename Tag, typename Pattern, typename Function, typename... Bounds,
          typename... Args, typename... ExtraFuncArgs>
struct par_dispatch_impl<Tag, Pattern, Function, TypeList<Bounds...>, TypeList<Args...>,
                         TypeList<ExtraFuncArgs...>> {
  using LP = LoopPattern;
  using dispatch_type = DispatchType<Tag, Pattern, Bounds...>;
  using bound_translator = LoopBoundTranslator<Bounds...>;
  static constexpr std::size_t Rank = bound_translator::Rank;

  template <typename ExecSpace>
  inline void dispatch(std::string name, ExecSpace exec_space, Bounds &&...bounds,
                       Function function, Args &&...args, const int scratch_level = 0,
                       const std::size_t scratch_size_in_bytes = 0) {
    constexpr std::size_t Ninner =
        dispatch_type::TeamPattern::Nvector + dispatch_type::TeamPattern::Nthread;
    constexpr auto pattern_tag = LoopPatternTag<dispatch_type::GetPatternTag()>();
    static_assert(
        !std::is_same_v<decltype(pattern_tag), LoopPatternTag<LP::undef>> &&
            !always_false<Tag, Pattern>,
        "Loop pattern & tag combination not recognized in DispatchType::GetPatternTag");

    constexpr bool isSimdFor = std::is_same_v<LoopPatternTag<LoopPattern::simd>,
                                              base_type<decltype(pattern_tag)>>;
    auto bound_arr = bound_translator().GetIndexRanges(std::forward<Bounds>(bounds)...);
    if constexpr (isSimdFor) {
      static_assert(!isSimdFor || (isSimdFor && std::is_same_v<ExecSpace, HostExecSpace>),
                    "SimdFor pattern only supported in HostExecSpace");
      SimdFor(std::make_index_sequence<Rank - 1>(), function, bound_arr);
    } else {
      dispatch_impl(pattern_tag, std::make_index_sequence<Rank - Ninner>(),
                    std::make_index_sequence<Ninner>(), name, exec_space, bound_arr,
                    function, std::forward<Args>(args)..., scratch_level,
                    scratch_size_in_bytes);
    }
  }

  template <std::size_t... Is>
  using sequence = std::integer_sequence<std::size_t, Is...>;

  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void dispatch_impl(LoopPatternTag<LP::flat>, sequence<OuterIs...>,
                            sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                            Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                            Args &&...args, const int scratch_level,
                            const std::size_t scratch_size_in_bytes) {
    static_assert(sizeof...(InnerIs) == 0);
    const auto idxer = MakeIndexer(bound_arr);
    kokkos_dispatch(
        Tag(), name, Kokkos::RangePolicy<>(exec_space, 0, idxer.size()),
        KOKKOS_LAMBDA(const int idx, ExtraFuncArgs... fargs) {
          const auto idx_arr = idxer.GetIdxArray(idx);
          function(idx_arr[OuterIs]..., std::forward<ExtraFuncArgs>(fargs)...);
        },
        std::forward<Args>(args)...);
  }

  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void dispatch_impl(LoopPatternTag<LP::md>, sequence<OuterIs...>,
                            sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                            Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                            Args &&...args, const int scratch_level,
                            const std::size_t scratch_size_in_bytes) {
    static_assert(sizeof...(InnerIs) == 0);
    kokkos_dispatch(
        Tag(), name,
        Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(exec_space, {bound_arr[OuterIs].s...},
                                                  {(1 + bound_arr[OuterIs].e)...}),
        function, std::forward<Args>(args)...);
  }

  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void dispatch_impl(LoopPatternTag<LP::outer>, sequence<OuterIs...>,
                            sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                            Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                            Args &&...args, const int scratch_level,
                            const std::size_t scratch_size_in_bytes) {
    const auto idxer =
        MakeIndexer(Kokkos::Array<IndexRange, sizeof...(OuterIs)>{bound_arr[OuterIs]...});
    kokkos_dispatch(
        Tag(), name,
        team_policy(exec_space, idxer.size(), Kokkos::AUTO)
            .set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),
        KOKKOS_LAMBDA(team_mbr_t team_member, ExtraFuncArgs... fargs) {
          const auto idx_arr = idxer.GetIdxArray(team_member.league_rank());
          function(team_member, idx_arr[OuterIs]...,
                   std::forward<ExtraFuncArgs>(fargs)...);
        },
        std::forward<Args>(args)...);
  }

  template <typename ExecSpace, std::size_t... OuterIs, std::size_t... InnerIs>
  inline void dispatch_impl(LoopPatternTag<LP::collapse>, sequence<OuterIs...>,
                            sequence<InnerIs...>, std::string name, ExecSpace exec_space,
                            Kokkos::Array<IndexRange, Rank> bound_arr, Function function,
                            Args &&...args, const int scratch_level,
                            const std::size_t scratch_size_in_bytes) {
    const auto idxer =
        MakeIndexer(Kokkos::Array<IndexRange, sizeof...(OuterIs)>{bound_arr[OuterIs]...});
    using TeamPattern = typename dispatch_type::TeamPattern;
    constexpr std::size_t Nvector = TeamPattern::Nvector;
    constexpr std::size_t Nthread = TeamPattern::Nthread;
    constexpr std::size_t Nouter = Rank - Nvector - Nthread;
    kokkos_dispatch(
        Tag(), name,
        team_policy(exec_space, idxer.size(), Kokkos::AUTO)
            .set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),

        MakeCollapse<Rank, Nouter, Nthread, Nvector, ExtraFuncArgs...>(idxer, bound_arr,
                                                                       function),
        std::forward<Args>(args)...);
  }
};

template <typename Tag, typename Pattern, typename... AllArgs>
inline void par_dispatch(Pattern, std::string name, DevExecSpace exec_space,
                         AllArgs &&...args) {
  using dispatchsig = DispatchSignature<TypeList<AllArgs...>>;
  constexpr std::size_t Rank = dispatchsig::Rank;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;
  using Args = typename dispatchsig::Args;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;

  if constexpr (Rank > 1 && std::is_same_v<dispatch_impl::ParallelScanDispatch, Tag>) {
    static_assert(always_false<Tag>, "par_scan only for 1D loops");
  }
  par_dispatch_impl<Tag, Pattern, Function, LoopBounds, Args, ExtraFuncArgs>().dispatch(
      name, exec_space, std::forward<AllArgs>(args)...);
}

template <typename Tag, typename... Args>
inline void par_dispatch(const std::string &name, Args &&...args) {
  par_dispatch<Tag>(DEFAULT_LOOP_PATTERN, name, DevExecSpace(),
                    std::forward<Args>(args)...);
}

template <class, class>
struct seq_for_impl {};

template <class Function, class... Bounds>
struct seq_for_impl<Function, TypeList<Bounds...>> {
  KOKKOS_INLINE_FUNCTION void execute(Bounds &&...bounds, Function function) {
    using bound_translator = LoopBoundTranslator<Bounds...>;
    constexpr std::size_t Rank = bound_translator::Rank;
    const auto bound_arr =
        bound_translator().GetIndexRanges(std::forward<Bounds>(bounds)...);
    SimdFor(std::make_index_sequence<Rank - 1>(), function, bound_arr);
  }
};

template <class... Args>
KOKKOS_INLINE_FUNCTION void seq_for(Args &&...args) {
  using dispatchsig = DispatchSignature<TypeList<Args...>>;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;

  seq_for_impl<Function, LoopBounds>().execute(std::forward<Args>(args)...);
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
  static constexpr std::size_t Rank = dispatchsig::Rank;
  using Function = typename dispatchsig::Function;
  using LoopBounds = typename dispatchsig::LoopBounds;
  using Args = typename dispatchsig::Args;
  using Tag = dispatch_impl::ParallelForDispatch;
  using ExtraFuncArgs = typename function_signature<Rank, Function>::FArgs;

  par_dispatch_impl<Tag, Pattern, Function, LoopBounds, Args, ExtraFuncArgs>().dispatch(
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
  par_disp_inner(Pattern(), team_member, std::forward<AllArgs>(args)...);
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
