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
#include <variant>

#include "Kokkos_Macros.hpp"
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

namespace meta {

template <typename T>
using base_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <typename... Ts>
struct PackList {};

template <typename... Ts>
constexpr int PackLength(PackList<Ts...>) {
  return sizeof...(Ts);
}

template <size_t N, typename>
struct PopList {};

template <typename T, typename... Ts>
struct PopList<1, PackList<T, Ts...>> {
  using type = T;
  using value = PackList<Ts...>;
};

template <size_t N, typename T, typename... Ts>
struct PopList<N, PackList<T, Ts...>> {
  static_assert(N > 1, "PopList requires N>=1");

 private:
  using pop = PopList<N - 1, PackList<Ts...>>;

 public:
  using type = typename pop::type;
  using value = typename pop::value;
};

template <typename, typename>
struct AppendList {};

template <typename T, typename... Ts>
struct AppendList<T, PackList<Ts...>> {
   using value = PackList<Ts..., T>;
};

template <typename, typename>
struct PrependList {};

template <typename T, typename... Ts>
struct PrependList<T, PackList<Ts...>> {
   using value = PackList<T, Ts...>;
};

template <typename, typename>
struct MergeLists {};

template <typename... Ts>
struct MergeLists<PackList<Ts...>, PackList<>> {
  using value = PackList<Ts...>;
};

template <typename... Ts, typename F, typename... Fs>
struct MergeLists<PackList<Ts...>, PackList<F, Fs...>> {
  using value = typename MergeLists<PackList<Ts..., F>, PackList<Fs...>>::value;
};

template <typename IndexList, typename ArgList>
struct PackSameType {};

template <typename... Is>
struct PackSameType<PackList<Is...>, PackList<>> {
  using value = PackList<Is...>;
};

template <typename Index, typename... Is, typename T, typename... Args>
struct PackSameType<PackList<Index, Is...>, PackList<T, Args...>> {
  using value = typename std::conditional<
      std::is_convertible<Index, T>::value,
      typename PackSameType<PackList<Index, Is..., T>, PackList<Args...>>::value,
      PackList<Index, Is...>>::type;
};

template<size_t, typename>
struct SequenceOfOnes {};

template<size_t... ones>
struct SequenceOfOnes<0, std::integer_sequence<size_t, ones...>>{
   using value = typename std::integer_sequence<size_t, ones...>;
};

template<size_t N, size_t... ones>
struct SequenceOfOnes<N, std::integer_sequence<size_t, ones...>> {
   using value = typename SequenceOfOnes<N-1, std::integer_sequence<size_t, 1>>::value;
};

template<size_t N>
struct SequenceOfOnes<N, void> {
   static_assert(N > 0, "N must be positive");
   using value = typename SequenceOfOnes<N-1, std::integer_sequence<size_t, 1>>::value;
};

} // namespace meta

namespace meta {

template <typename, typename>
struct PackIntegralType {};

template <typename... Is>
struct PackIntegralType<PackList<Is...>, PackList<>> {
  using value = PackList<Is...>;
};

template <typename... Is, typename T, typename... Ts>
struct PackIntegralType<PackList<Is...>, PackList<T, Ts...>> {
  using value = std::conditional<
      std::is_integral<T>::value,
      typename PackIntegralType<PackList<Is..., T>, PackList<Ts...>>::value,
      PackList<Is...>>;
};

template <typename>
struct FunctionSignature {};

template <typename R, typename T, typename Index, typename... Args>
struct FunctionSignature<R (T::*)(Index, Args...) const> {
  using IndexND = typename PackSameType<PackList<Index>, PackList<Args...>>::value;
  using FArgs = PopList<PackLength(IndexND()), PackList<Index, Args...>>;
};

template<typename F>
using function_signature = FunctionSignature<decltype(&F::operator())>;

template<typename>
struct GetLaunchBounds {};

template <>
struct GetLaunchBounds<PackList<>> {
   using value = PackList<>;
};

template <typename T, typename... Args>
struct GetLaunchBounds<PackList<T,Args...>> {

   template <typename V>
   static constexpr bool is_BoundType() {
      return std::numeric_limits<V>::is_integer || std::is_same_v<V, IndexRange> ;
   }

   using bound_variants = std::variant<IndexRange, IndexRange&>;
   using bound =std::remove_cv_t<std::remove_reference_t<T>>;
   using LaunchBounds = GetLaunchBounds<PackList<Args...>>;
   using value = typename std::conditional < is_BoundType<bound>(),
         typename PrependList<T, typename GetLaunchBounds<PackList<Args...>>::value>::value, PackList<>>::type;
};

template <typename>
struct DispatchSignature {};

template <typename Index, typename... AllArgs>
struct DispatchSignature<PackList<Index, AllArgs...>> {
  using LaunchBounds = typename GetLaunchBounds<PackList<Index, AllArgs...>>::value;
  using Function =
      typename PopList<PackLength(LaunchBounds()) + 1, PackList<Index, AllArgs...>>::type;
  using Args =
      typename PopList<PackLength(LaunchBounds()) + 1, PackList<Index, AllArgs...>>::value;

};

} // namespace meta

template <typename, typename, typename>
class FlatFunctor {};

template <typename Function, size_t... Is, typename... FArgs>
class FlatFunctor<Function, std::integer_sequence<size_t, Is...>,
                  meta::PackList<FArgs...>> {
  Kokkos::Array<IndexRange, sizeof...(Is)> ranges;
  Kokkos::Array<int, sizeof...(Is) - 1> strides;
  Function function;

 public:
  template <typename... Args>
  FlatFunctor(const Function _function, IndexRange idr, Args... args)
      : function(_function), ranges({{idr, args...}}) {
    Initialize();
  }

  template <typename... Args>
  FlatFunctor(const Function _function, Args... args) : function(_function) {
    std::array<int, 2 * sizeof...(Is)> indices{{args...}};
    for (int i = 0; i < sizeof...(Is); i++) {
      ranges[i] = {indices[2 * i], indices[2 * i + 1]};
    }
    Initialize();
  }

  inline void Initialize() {
    for (int ri = 1; ri < sizeof...(Is); ri++) {
      const int N = ranges[ri].e - ranges[ri].s + 1;
      strides[ri - 1] = N;
      for (int rj = 0; rj < ri - 1; rj++) {
        strides[rj] *= N;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &idx, FArgs... fargs) const {
    constexpr int ND = sizeof...(Is);
    int inds[ND];
    inds[0] = idx;
    for (int i = 1; i < ND; i++) {
      inds[i] = idx;
      inds[i - 1] /= strides[i - 1];
      for (int j = 0; j < i; j++) {
        inds[i] -= inds[j] * strides[j];
      }
    }
    for (int i = 0; i < ND; i++) {
      inds[i] += ranges[i].s;
    }

    function(inds[Is]..., std::forward<FArgs>(fargs)...);
  }
};

template <typename F, typename... Args>
auto MakeFlatFunctor(F &function, Args &&...args) {
  using signature = meta::FunctionSignature<decltype(&F::operator())>;
  using IndexND = typename signature::IndexND;
  return FlatFunctor<F, std::make_index_sequence<meta::PackLength(IndexND())>,
                     typename signature::FArgs::value>(function,
                                                       std::forward<Args>(args)...);
}

template <typename, typename>
class MDRange {};

template <size_t... Is, size_t... ones>
class MDRange<std::integer_sequence<size_t, Is...>, std::integer_sequence<size_t, ones...>> {
   Kokkos::Array<int, sizeof...(Is)> lower, upper;

   public:
  template <typename... Args>
  MDRange(IndexRange idr, Args... args) {
    std::array<IndexRange, sizeof...(Is)> ranges{{idr, args...}};
    for (int i = 0; i < sizeof...(Is); i++) {
       lower[i] = ranges[i].s;
       upper[i] = ranges[i].e;
    }
      }

  template <typename... Args>
  MDRange(Args... args) {
    std::array<int, 2 * sizeof...(Is)> indices{{args...}};
    for (int i = 0; i < sizeof...(Is); i++) {
       lower[i] = indices[2*i];
       upper[i] = indices[2*i+1];
    }
  }

  auto policy(DevExecSpace exec_space) {
     constexpr int ND = sizeof...(Is);
      return Kokkos::MDRangePolicy<Kokkos::Rank<ND>>(exec_space, 
            {lower[Is]...}, {1+upper[Is]...}, {ones..., upper[ND-1] + 1 - lower[ND-1]});
  }
};

template<typename F, typename... Args>
auto MakeMDRangePolicy(DevExecSpace exec_space, Args &&...args) {
  using signature = meta::FunctionSignature<decltype(&F::operator())>;
  using IndexND = typename signature::IndexND;
  static_assert(sizeof...(Args) % meta::PackLength(IndexND()) == 0, 
                "Launch Bounds don't match functor signature");
  return MDRange<std::make_index_sequence<meta::PackLength(IndexND())>,
                 typename meta::SequenceOfOnes<meta::PackLength(IndexND())-1,void>::value >(std::forward<Args>(args)...).policy(exec_space);
}

template <typename, typename, typename, typename>
struct par_dispatch_impl {};

template <typename Tag, typename Function, typename... Bounds, typename... Args>
struct par_dispatch_impl<Tag, Function, meta::PackList<Bounds...>,
                         meta::PackList<Args...>> {

  using BoundType = typename meta::PopList<1, meta::PackList<Bounds...>>::type;

  template <typename Pattern>
  inline void dispatch(Pattern, std::string name, DevExecSpace exec_space, Bounds &&...ids,
                       Function function, Args &&...args) {

    Tag tag;

      kokkos_dispatch(tag, name,
                      policy(Pattern(), exec_space, std::forward<Bounds>(ids)...),
                      functor(Pattern(), function, std::forward<Bounds>(ids)...),
                      std::forward<Args>(args)...);
  };

  template <typename Pattern>
  KOKKOS_INLINE_FUNCTION
  auto policy(Pattern, DevExecSpace exec_space, Bounds &&...ids) const {
     constexpr bool is_FlatRange = std::is_same<Pattern, LoopPatternFlatRange>::value;
     constexpr bool is_MDRange = std::is_same<Pattern, LoopPatternMDRange>::value;

     if constexpr (is_FlatRange) {
      int rangeNx = 1;
      if constexpr (std::is_same<BoundType &, IndexRange &>::value) {
         for (auto &irange : {ids...}) {
            rangeNx *= irange.e - irange.s + 1;
         }
      } else {
         int indices[sizeof...(Bounds)] = {ids...};
         for (int i = 0; i < sizeof...(Bounds); i += 2) {
            rangeNx *= indices[i + 1] - indices[i] + 1;
         }
      }
      return Kokkos::RangePolicy<>(exec_space, 0, rangeNx);

     } else if constexpr (is_MDRange) {
        return MakeMDRangePolicy<Function>(exec_space, std::forward<Bounds>(ids)...);
     } else {
     }
  };

  template<typename Pattern>
  KOKKOS_INLINE_FUNCTION
  auto functor(Pattern, Function function, Bounds &&...ids) const {
     constexpr bool is_FlatRange = std::is_same<Pattern, LoopPatternFlatRange>::value;
     constexpr bool is_MDRange = std::is_same<Pattern, LoopPatternMDRange>::value;
     if constexpr (is_FlatRange) {
        return MakeFlatFunctor(function, std::forward<Bounds>(ids)...);
     } else if constexpr(is_MDRange) {
        return function;
     } else {
     }

  }
};

template <typename Tag, typename Pattern, typename... AllArgs>
inline typename std::enable_if<std::is_same<Pattern, LoopPatternFlatRange>::value ||
                               std::is_same<Pattern, LoopPatternMDRange>::value,
                               void>::type
par_dispatch(Pattern, std::string name, DevExecSpace exec_space, AllArgs &&...args) {
  using dispatchsig = meta::DispatchSignature<meta::PackList<AllArgs...>>;
  using Function = typename dispatchsig::Function; // functor type
  using LaunchBounds = typename dispatchsig::LaunchBounds;   // list of index types
  using Args = typename dispatchsig::Args;         //
  par_dispatch_impl<Tag, Function, LaunchBounds, Args>().dispatch(
      Pattern(), name, exec_space, std::forward<AllArgs>(args)...);
}

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
