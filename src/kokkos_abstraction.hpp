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
#include "kokkos_types.hpp"
#include "loop_bound_translator.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/indexer.hpp"
#include "utils/instrument.hpp"
#include "utils/multi_pointer.hpp"
#include "utils/object_pool.hpp"
#include "utils/type_list.hpp"

namespace parthenon {
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
  KOKKOS_FORCEINLINE_FUNCTION
  static auto InnerRange(team_mbr_t &team_member, std::size_t size) {
    return Kokkos::TeamThreadRange<>(team_member, 0, size);
  }
} loop_pattern_tpttr_tag;
// Translates to a Kokkos::TeamPolicy with a single inner
// Kokkos::ThreadVectorRange
static struct LoopPatternTPTVR : public PatternBase<0, 1> {
  KOKKOS_FORCEINLINE_FUNCTION
  static auto InnerRange(team_mbr_t &team_member, std::size_t size) {
    return Kokkos::TeamVectorRange<>(team_member, 0, size);
  }
} loop_pattern_tptvr_tag;
// Translates to a Kokkos::TeamPolicy with a middle Kokkos::TeamThreadRange and
// inner Kokkos::ThreadVectorRange
static struct LoopPatternTPTTRTVR : public PatternBase<1, 1> {
  KOKKOS_FORCEINLINE_FUNCTION
  static auto MiddleRange(team_mbr_t &team_member, std::size_t size) {
    return Kokkos::TeamThreadRange<>(team_member, 0, size);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  static auto InnerRange(team_mbr_t &team_member, std::size_t size) {
    return Kokkos::ThreadVectorRange<>(team_member, 0, size);
  }
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
struct InnerLoopPatternTVR {
  KOKKOS_FORCEINLINE_FUNCTION
  static auto Range(team_mbr_t &team_member, std::size_t size) {
    return Kokkos::TeamVectorRange<>(team_member, 0, size);
  }
};
constexpr InnerLoopPatternTVR inner_loop_pattern_tvr_tag;
// Translates to a Kokkos::TeamThreadRange as innermost loop
struct InnerLoopPatternTTR {
  KOKKOS_FORCEINLINE_FUNCTION
  static auto Range(team_mbr_t &team_member, std::size_t size) {
    return Kokkos::TeamThreadRange<>(team_member, 0, size);
  }
};
constexpr InnerLoopPatternTTR inner_loop_pattern_ttr_tag;
// Translates to a Kokkos::ThreadVectorRange as innermost loop
struct InnerLoopPatternThreadVR {
  KOKKOS_FORCEINLINE_FUNCTION
  static auto Range(team_mbr_t &team_member, std::size_t size) {
    return Kokkos::ThreadVectorRange<>(team_member, 0, size);
  }
};
constexpr InnerLoopPatternThreadVR inner_loop_pattern_threadvr_tag;
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

template <class Pattern, class BT_t, class Func, std::size_t... Is>
void KokkosTwoLevelFor(DevExecSpace exec_space, const std::string &name,
                       const BT_t &bound_trans, const Func &function,
                       std::index_sequence<Is...>) {
  constexpr int nouter = sizeof...(Is) - Pattern::ninner;
  const auto outer_idxer = bound_trans.template GetIndexer<0, nouter>();
  const auto inner_idxer =
      bound_trans.template GetIndexer<nouter, Pattern::ninner + nouter>();
  Kokkos::parallel_for(
      name, team_policy(exec_space, outer_idxer.size(), Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        // WARNING/QUESTION(LFR): Is this array defined per thread and is it safe
        // update it at different levels of the parallel hierarchy? Could call
        // all of the indexers at the innermost level, but that results in more
        // index calculations (which are unlikely to matter though).
        int indices[sizeof...(Is)];
        outer_idxer.GetIdxCArray(team_member.league_rank(), indices);
        Kokkos::parallel_for(Pattern::InnerRange(team_member, inner_idxer.size()),
                             [&](const int idx) {
                               inner_idxer.GetIdxCArray(idx, &indices[nouter]);
                               function(indices[Is]...);
                             });
      });
}

template <class Pattern, class BT_t, class Func, std::size_t... Is>
void KokkosThreeLevelFor(DevExecSpace exec_space, const std::string &name,
                         const BT_t &bound_trans, const Func &function,
                         std::index_sequence<Is...>) {
  constexpr int nouter = sizeof...(Is) - Pattern::ninner - Pattern::nmiddle;
  const auto outer_idxer = bound_trans.template GetIndexer<0, nouter>();
  const auto middle_idxer =
      bound_trans.template GetIndexer<nouter, Pattern::nmiddle + nouter>();
  const auto inner_idxer =
      bound_trans.template GetIndexer<Pattern::nmiddle + nouter,
                                      Pattern::ninner + Pattern::nmiddle + nouter>();
  Kokkos::parallel_for(
      name, team_policy(exec_space, outer_idxer.size(), Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team_member) {
        // WARNING/QUESTION(LFR): Is this array defined per thread and is it safe
        // update it at different levels of the parallel hierarchy? Could call
        // all of the indexers at the innermost level, but that results in more
        // index calculations (which are unlikely to matter though).
        int indices[sizeof...(Is)];
        outer_idxer.GetIdxCArray(team_member.league_rank(), indices);
        Kokkos::parallel_for(
            Pattern::MiddleRange(team_member, inner_idxer.size()), [&](const int idx) {
              middle_idxer.GetIdxCArray(idx, &indices[nouter]);
              Kokkos::parallel_for(
                  Pattern::InnerRange(team_member, inner_idxer.size()), [&](const int i) {
                    inner_idxer.GetIdxCArray(i, &indices[nouter + Pattern::nmiddle]);
                    function(indices[Is]...);
                  });
            });
      });
}

template <class LoopBoundTranslator_t, class Function>
KOKKOS_INLINE_FUNCTION void RawFor(const LoopBoundTranslator_t &bound_trans,
                                   const Function &function) {
  auto idxer = bound_trans.template GetIndexer<0, LoopBoundTranslator_t::rank>();
  for (int idx = 0; idx < idxer.size(); ++idx) {
    std::apply(function, idxer(idx));
  }
}

template <class LoopBoundTranslator_t, class Function, std::size_t... Is>
KOKKOS_INLINE_FUNCTION void SimdFor(const LoopBoundTranslator_t &bound_trans,
                                    const Function &function,
                                    std::index_sequence<Is...>) {
  if constexpr (LoopBoundTranslator_t::rank > 1) {
    constexpr int inner_start_rank = LoopBoundTranslator_t::rank - 1;
    auto idxer = bound_trans.template GetIndexer<0, inner_start_rank>();
    const int istart = bound_trans[inner_start_rank].s;
    const int iend = bound_trans[inner_start_rank].e;
    // Loop over all outer indices using a flat indexer
    for (int idx = 0; idx < idxer.size(); ++idx) {
      const auto indices = idxer.GetIdxArray(idx);
#pragma omp simd
      for (int i = istart; i <= iend; ++i) {
        function(indices[Is]..., i);
      }
    }
  } else { // Easier to just explicitly specialize for 1D Simd loop
#pragma omp simd
    for (int i = bound_trans[0].s; i <= bound_trans[0].e; ++i)
      function(i);
  }
}
template <class LoopBoundTranslator_t, class Function, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION void SimdFor(const LoopBoundTranslator_t &bound_trans,
                                         const Function &function) {
  SimdFor(bound_trans, function,
          std::make_index_sequence<LoopBoundTranslator_t::rank - 1>());
}

} // namespace dispatch_impl

template <class Tag, class Pattern, class Bound_tl, class Function, class ExtArgs_tl,
          class FuncExtArgs_tl>
struct par_dispatch_funct {};

template <class Tag, class Pattern, class... Bound_ts, class Function,
          class... ExtraArgs_ts, class... FuncExtArgs_ts>
struct par_dispatch_funct<Tag, Pattern, TypeList<Bound_ts...>, Function,
                          TypeList<ExtraArgs_ts...>, TypeList<FuncExtArgs_ts...>> {
  template <std::size_t... Is>
  void operator()(std::index_sequence<Is...>, const std::string &name,
                  DevExecSpace exec_space, Bound_ts &&...bounds, const Function &function,
                  ExtraArgs_ts &&...args) {
    using bt_t = LoopBoundTranslator<Bound_ts...>;
    auto bound_trans = bt_t(std::forward<Bound_ts>(bounds)...);

    constexpr int total_rank = bt_t::rank;
    [[maybe_unused]] constexpr int kTotalRank = bt_t::rank;
    [[maybe_unused]] constexpr int inner_start_rank = total_rank - Pattern::ninner;
    [[maybe_unused]] constexpr int middle_start_rank =
        total_rank - Pattern::ninner - Pattern::nmiddle;

    constexpr bool SimdRequested = std::is_same_v<Pattern, LoopPatternSimdFor>;
    constexpr bool MDRangeRequested = std::is_same_v<Pattern, LoopPatternMDRange>;
    constexpr bool FlatRangeRequested = std::is_same_v<Pattern, LoopPatternFlatRange>;
    constexpr bool TPTTRRequested = std::is_same_v<Pattern, LoopPatternTPTTR>;
    constexpr bool TPTVRRequested = std::is_same_v<Pattern, LoopPatternTPTVR>;
    constexpr bool TPTTRTVRRequested = std::is_same_v<Pattern, LoopPatternTPTTRTVR>;

    constexpr bool isParFor = std::is_same_v<dispatch_impl::ParallelForDispatch, Tag>;
    constexpr bool doSimdFor = SimdRequested && isParFor;
    constexpr bool doMDRange = MDRangeRequested && (total_rank > 1);
    constexpr bool doTPTTRTV = TPTTRTVRRequested &&
                               (total_rank > Pattern::ninner + Pattern::nmiddle) &&
                               isParFor;
    constexpr bool doTPTTR = TPTTRRequested && (total_rank > Pattern::ninner) && isParFor;
    constexpr bool doTPTVR = TPTVRRequested && (total_rank > Pattern::ninner) && isParFor;

    [[maybe_unused]] constexpr bool doFlatRange =
        FlatRangeRequested || (SimdRequested && !doSimdFor) ||
        (MDRangeRequested && !doMDRange) || (TPTTRTVRRequested && !doTPTTRTV) ||
        (TPTTRRequested && !doTPTTR) || (TPTVRRequested && !doTPTVR);

    if constexpr (doSimdFor) {
      dispatch_impl::SimdFor(bound_trans, function);
    } else if constexpr (doMDRange) {
      static_assert(total_rank > 1,
                    "MDRange pattern only works for multi-dimensional loops.");
      auto policy =
          bound_trans.template GetKokkosMDRangePolicy<0, total_rank>(exec_space);
      dispatch_impl::kokkos_dispatch(Tag(), name, policy, function,
                                     std::forward<ExtraArgs_ts>(args)...);
    } else if constexpr (doFlatRange) {
      auto idxer = bound_trans.template GetIndexer<0, total_rank>();
      auto policy =
          bound_trans.template GetKokkosFlatRangePolicy<0, total_rank>(exec_space);
      auto lam = KOKKOS_LAMBDA(const int &idx, FuncExtArgs_ts... fargs) {
        const auto indices = idxer.GetIdxArray(idx);
        function(indices[Is]..., fargs...);
      };
      dispatch_impl::kokkos_dispatch(Tag(), name, policy, lam,
                                     std::forward<ExtraArgs_ts>(args)...);
    } else if constexpr (doTPTTR || doTPTVR) {
      static_assert(Pattern::nmiddle == 0, "Two-level paralellism chosen.");
      dispatch_impl::KokkosTwoLevelFor<Pattern>(exec_space, name, bound_trans, function,
                                                std::make_index_sequence<total_rank>());
    } else if constexpr (doTPTTRTV) {
      dispatch_impl::KokkosThreeLevelFor<Pattern>(exec_space, name, bound_trans, function,
                                                  std::make_index_sequence<total_rank>());
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
  constexpr int loop_rank = LoopBoundTranslator<
      typename arg_tl::template continuous_sublist<0, func_idx - 1>>::rank;

  using func_sig_tl = typename FuncSignature<func_t>::arg_types_tl;
  // The first loop_rank arguments of the function are just indices, the rest are extra
  // arguments for reductions, etc.
  using bounds_tl = typename arg_tl::template continuous_sublist<0, func_idx - 1>;
  using extra_arg_tl = typename arg_tl::template continuous_sublist<func_idx + 1>;
  using func_sig_extra_tl = typename func_sig_tl::template continuous_sublist<loop_rank>;
  par_dispatch_funct<Tag, Pattern, bounds_tl, func_t, extra_arg_tl, func_sig_extra_tl>
      loop_funct;
  loop_funct(std::make_index_sequence<loop_rank>(), name, exec_space,
             std::forward<Args>(args)...);
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
  template <class Pattern, std::size_t... Is>
  void operator()(std::index_sequence<Is...>, Pattern, const std::string &name,
                  DevExecSpace exec_space, size_t scratch_size_in_bytes,
                  const int scratch_level, Bound_ts &&...bounds,
                  const Function &function) {
    using bt_t = LoopBoundTranslator<Bound_ts...>;
    auto bound_trans = bt_t(std::forward<Bound_ts>(bounds)...);
    auto idxer = bound_trans.template GetIndexer<0, bt_t::rank>();
    constexpr int kTotalRank = bt_t::rank;

    if constexpr (std::is_same_v<OuterLoopPatternTeams, Pattern>) {
      team_policy policy(exec_space, idxer.size(), Kokkos::AUTO);
      Kokkos::parallel_for(
          name,
          policy.set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_size_in_bytes)),
          KOKKOS_LAMBDA(team_mbr_t team_member) {
            const auto indices = idxer.GetIdxArray(team_member.league_rank());
            function(team_member, indices[Is]...);
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
  using bt_t = LoopBoundTranslator<bound_tl>;
  par_for_outer_funct(std::make_index_sequence<bt_t::rank>(), OuterLoopPatternTeams(),
                      name, exec_space, scratch_size_in_bytes, scratch_level,
                      std::forward<Args>(args)...);
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

  template <class Pattern, std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION void
  operator()(std::index_sequence<Is...>, Pattern, team_mbr_t team_member,
             Bound_ts &&...bounds, const Function &function) {
    using bt_t = LoopBoundTranslator<Bound_ts...>;
    auto bound_trans = bt_t(std::forward<Bound_ts>(bounds)...);
    [[maybe_unused]] constexpr int kTotalRank = bt_t::rank;
    constexpr bool doTTR = std::is_same_v<InnerLoopPatternTTR, Pattern>;
    constexpr bool doTVR = std::is_same_v<InnerLoopPatternTVR, Pattern>;
    constexpr bool doThreadVR = std::is_same_v<InnerLoopPatternThreadVR, Pattern>;
    constexpr bool doSimd = std::is_same_v<InnerLoopPatternSimdFor, Pattern>;
    if constexpr (doSimd) {
      dispatch_impl::SimdFor(bound_trans, function);
    } else if constexpr (doTTR || doTVR || doThreadVR) {
      const auto idxer = bound_trans.template GetIndexer<0, bt_t::rank>();
      Kokkos::parallel_for(Pattern::Range(team_member, idxer.size()),
                           [&](const int &idx) {
                             const auto indices = idxer.GetIdxArray(idx);
                             function(indices[Is]...);
                           });
    } else {
      PARTHENON_FAIL("Unsupported par_for_inner loop pattern.");
    }
  }
};

template <class Pattern, class... Args>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(Pattern pattern, Args &&...args) {
  using arg_tl = TypeList<Args...>;
  constexpr int nm2 = arg_tl::n_types - 2;
  constexpr int nm1 = arg_tl::n_types - 1;
  // First Arg of Args is team_member
  using bound_tl = typename arg_tl::template continuous_sublist<1, nm2>;
  using function_tl = typename arg_tl::template continuous_sublist<nm1, nm1>;
  par_for_inner_funct_t<bound_tl, function_tl> par_for_inner_funct;
  using bt_t = LoopBoundTranslator<bound_tl>;
  par_for_inner_funct(std::make_index_sequence<bt_t::rank>(), pattern,
                      std::forward<Args>(args)...);
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
