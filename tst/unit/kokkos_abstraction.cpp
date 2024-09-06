//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
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

#include <iostream>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/indexer.hpp"
#include "utils/type_list.hpp"

using parthenon::DevExecSpace;
using parthenon::ParArray1D;
using parthenon::ParArray2D;
using parthenon::ParArray3D;
using parthenon::ParArray4D;
using Real = double;

template <std::size_t Ni>
using Sequence = std::make_index_sequence<Ni>;

template <class... Args>
void capture(Args... args) {}

template <std::size_t>
struct ParArrayND_impl {};
template <>
struct ParArrayND_impl<0> {
  template <typename T>
  using type = parthenon::ParArray0D<T>;
};
template <>
struct ParArrayND_impl<1> {
  template <typename T>
  using type = parthenon::ParArray1D<T>;
};
template <>
struct ParArrayND_impl<2> {
  template <typename T>
  using type = parthenon::ParArray2D<T>;
};
template <>
struct ParArrayND_impl<3> {
  template <typename T>
  using type = parthenon::ParArray3D<T>;
};
template <>
struct ParArrayND_impl<4> {
  template <typename T>
  using type = parthenon::ParArray4D<T>;
};
template <>
struct ParArrayND_impl<5> {
  template <typename T>
  using type = parthenon::ParArray5D<T>;
};
template <>
struct ParArrayND_impl<6> {
  template <typename T>
  using type = parthenon::ParArray6D<T>;
};
template <>
struct ParArrayND_impl<7> {
  template <typename T>
  using type = parthenon::ParArray7D<T>;
};
template <>
struct ParArrayND_impl<8> {
  template <typename T>
  using type = parthenon::ParArray8D<T>;
};
template <std::size_t>
struct HostArrayND_impl {};
template <>
struct HostArrayND_impl<0> {
  template <typename T>
  using type = parthenon::HostArray0D<T>;
};
template <>
struct HostArrayND_impl<1> {
  template <typename T>
  using type = parthenon::HostArray1D<T>;
};
template <>
struct HostArrayND_impl<2> {
  template <typename T>
  using type = parthenon::HostArray2D<T>;
};
template <>
struct HostArrayND_impl<3> {
  template <typename T>
  using type = parthenon::HostArray3D<T>;
};
template <>
struct HostArrayND_impl<4> {
  template <typename T>
  using type = parthenon::HostArray4D<T>;
};
template <>
struct HostArrayND_impl<5> {
  template <typename T>
  using type = parthenon::HostArray5D<T>;
};
template <>
struct HostArrayND_impl<6> {
  template <typename T>
  using type = parthenon::HostArray6D<T>;
};
template <>
struct HostArrayND_impl<7> {
  template <typename T>
  using type = parthenon::HostArray7D<T>;
};

template <std::size_t>
struct ScratchPadND_impl {};
template <>
struct ScratchPadND_impl<1> {
  template <typename T>
  using type = parthenon::ScratchPad1D<T>;
};
template <>
struct ScratchPadND_impl<2> {
  template <typename T>
  using type = parthenon::ScratchPad2D<T>;
};

template <std::size_t ND, typename T, typename... Args>
auto ParArrayND(Args &&...args) {
  static_assert(ND <= 8, "ParArrayND supoorted up to ND=8");
  return typename ParArrayND_impl<ND>::template type<T>(std::forward<Args>(args)...);
}
template <std::size_t ND, typename T, typename... Args>
auto HostArrayND(Args &&...args) {
  static_assert(ND <= 7, "HostArrayND supoorted up to ND=7");
  return typename HostArrayND_impl<ND>::template type<T>(std::forward<Args>(args)...);
}

template <std::size_t ND, typename T, typename... Args>
auto ScratchPadND(Args &&...args) {
  static_assert(ND <= 2, "ScratchPadND supported up to ND=2");
  return typename ScratchPadND_impl<ND>::template type<T>(std::forward<Args>(args)...);
}

template <std::size_t, std::size_t, typename>
struct SequenceOfInt {};

template <std::size_t VAL, std::size_t... ones>
struct SequenceOfInt<0, VAL, std::integer_sequence<std::size_t, ones...>> {
  using value = typename std::integer_sequence<std::size_t, ones...>;
};

template <std::size_t N, std::size_t VAL, std::size_t... ones>
struct SequenceOfInt<N, VAL, std::integer_sequence<std::size_t, ones...>> {
  using value =
      typename SequenceOfInt<N - 1, VAL,
                             std::integer_sequence<std::size_t, VAL, ones...>>::value;
};

template <std::size_t N, std::size_t VAL = 1>
using sequence_of_int_v =
    typename SequenceOfInt<N - 1, VAL, std::integer_sequence<std::size_t, VAL>>::value;

template <std::size_t Rank, class... Args>
auto GetArray_impl(Args... Ns) {
  static_assert(sizeof...(Args) == Rank);
  return ParArrayND<Rank, Real>("device", Ns...);
}

enum class lbounds { integer, indexrange };

template <std::size_t Rank, std::size_t N>
struct test_wrapper_nd_impl {
  int int_bounds[2 * Rank];
  parthenon::IndexRange bounds[Rank];
  decltype(ParArrayND<Rank, Real>()) arr_dev;
  decltype(HostArrayND<Rank, Real>()) arr_host_orig, arr_host_mod;

  test_wrapper_nd_impl() {
    arr_dev = GetArray(sequence_of_int_v<Rank, 1>());
    arr_host_orig = Kokkos::create_mirror(arr_dev);
    arr_host_mod = Kokkos::create_mirror(arr_dev);
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<Real> dis(-1.0, 1.0);
    par_for_init(std::make_index_sequence<Rank>(), gen, dis);
  }

  template <std::size_t... Is>
  auto GetArray(std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) == Rank);
    return GetArray_impl<Rank>(N * Is...);
  }

  template <std::size_t... Is>
  void par_for_init(std::index_sequence<Is...>, std::mt19937 &gen,
                    std::uniform_real_distribution<Real> &dis) {
    for (int id = 0; id < Rank; id++) {
      bounds[id].s = 0;
      bounds[id].e = N - 1;
      int_bounds[2 * id] = 0;
      int_bounds[2 * id + 1] = N - 1;
    }
    const auto idxer =
        parthenon::MakeIndexer(std::pair<int, int>(bounds[Is].s, bounds[Is].e)...);
    for (int idx = 0; idx < idxer.size(); idx++) {
      const auto indices = idxer.GetIdxArray(idx);
      arr_host_orig(indices[Is]...) = dis(gen);
    }
  }

  template <typename... KJI>
  KOKKOS_INLINE_FUNCTION static Real increment_data(KJI... kji) {
    static_assert(Rank == sizeof...(KJI), "number of indices matches Rank");
    int inc = 0;
    int inds[sizeof...(KJI)]{kji...};
    for (int i = 0; i < Rank; i++) {
      inc += N * inds[i];
    }
    return static_cast<Real>(inc);
  }

  template <std::size_t... Is>
  bool par_for_comp(std::index_sequence<Is...>) {
    bool all_same = true;
    const auto idxer =
        parthenon::MakeIndexer(std::pair<int, int>(bounds[Is].s, bounds[Is].e)...);
    for (int idx = 0; idx < idxer.size(); idx++) {
      const auto indices = idxer.GetIdxArray(idx);
      if (arr_host_orig(indices[Is]...) + increment_data(indices[Is]...) !=
          arr_host_mod(indices[Is]...)) {
        all_same = false;
      }
    }
    return all_same;
  }

  template <typename, lbounds, typename, typename>
  struct dispatch {};

  template <typename Pattern, lbounds bound_type, std::size_t... Ids, typename... Ts>
  struct dispatch<Pattern, bound_type, std::index_sequence<Ids...>,
                  parthenon::TypeList<Ts...>> {
    template <typename view_t>
    void execute(DevExecSpace exec_space, view_t &dev, int *int_bounds,
                 parthenon::IndexRange *bounds) {
      const auto functor = KOKKOS_CLASS_LAMBDA(Ts... args) {
        dev(std::forward<decltype(args)>(args)...) +=
            increment_data(std::forward<decltype(args)>(args)...);
      };
      if constexpr (bound_type == lbounds::integer) {
        parthenon::par_for(Pattern(), "unit test ND integer bounds", exec_space,
                           int_bounds[Ids]..., functor);
      } else {
        parthenon::par_for(Pattern(), "unit test ND IndexRange bounds", exec_space,
                           bounds[Ids]..., functor);
      }
    }
  };

  template <typename T>
  void test(T loop_pattern, DevExecSpace exec_space) {
    Kokkos::deep_copy(arr_dev, arr_host_orig);
    SECTION("integer launch bounds") {
      dispatch<T, lbounds::integer, Sequence<2 * Rank>,
               parthenon::list_of_type_t<Rank, const int>>()
          .execute(exec_space, arr_dev, int_bounds, bounds);
      Kokkos::deep_copy(arr_host_mod, arr_dev);
      REQUIRE(par_for_comp(Sequence<Rank>()) == true);
    }
    SECTION("IndexRange launch bounds") {
      dispatch<T, lbounds::indexrange, Sequence<Rank>,
               parthenon::list_of_type_t<Rank, const int>>()
          .execute(exec_space, arr_dev, int_bounds, bounds);
      Kokkos::deep_copy(arr_host_mod, arr_dev);
      REQUIRE(par_for_comp(Sequence<Rank>()) == true);
    }
  }

  template <typename OuterPattern, typename InnerPattern>
  void test_nest(OuterPattern outer_patter, InnerPattern inner_pattern) {}
};

template <std::size_t Rank, std::size_t N>
void test_wrapper_nd(DevExecSpace exec_space) {
  auto wrappernd = test_wrapper_nd_impl<Rank, N>();
  SECTION("LoopPatternFlatRange") {
    wrappernd.test(parthenon::loop_pattern_flatrange_tag, exec_space);
  }
  if constexpr (Rank > 1 && Rank < 7) {
    SECTION("LoopPatternMDRange") {
      wrappernd.test(parthenon::loop_pattern_mdrange_tag, exec_space);
    }
  }
  if constexpr (Rank > 2) {
    SECTION("LoopPatternTPTTRTVR") {
      wrappernd.test(parthenon::loop_pattern_tpttrtvr_tag, exec_space);
    }
    SECTION("LoopPatternTPTTR") {
      wrappernd.test(parthenon::loop_pattern_tpttr_tag, exec_space);
    }
  }
  if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                             Kokkos::DefaultHostExecutionSpace>::value) {
    if constexpr (Rank > 2) {
      SECTION("LoopPatternTPTVR") {
        wrappernd.test(parthenon::loop_pattern_tptvr_tag, exec_space);
      }
    }
    SECTION("LoopPatternSimdFor") {
      wrappernd.test(parthenon::loop_pattern_simdfor_tag, exec_space);
    }
  }
}

TEST_CASE("par_for loops", "[wrapper]") {
  auto default_exec_space = DevExecSpace();

  SECTION("1D loops") { test_wrapper_nd<1, 32>(default_exec_space); }
  SECTION("2D loops") { test_wrapper_nd<2, 32>(default_exec_space); }
  SECTION("3D loops") { test_wrapper_nd<3, 32>(default_exec_space); }
  SECTION("4D loops") { test_wrapper_nd<4, 32>(default_exec_space); }
  SECTION("5D loops") { test_wrapper_nd<5, 10>(default_exec_space); }
  SECTION("6D loops") { test_wrapper_nd<6, 10>(default_exec_space); }
  SECTION("7D loops") { test_wrapper_nd<7, 10>(default_exec_space); }
}

template <std::size_t Rank, std::size_t N>
struct test_wrapper_nested_nd_impl {
  Kokkos::Array<parthenon::IndexRange, Rank> bounds;
  decltype(ParArrayND<Rank, Real>()) dev_u, dev_du;
  decltype(HostArrayND<Rank, Real>()) host_u, host_du;

  test_wrapper_nested_nd_impl() {
    dev_u = GetArray(sequence_of_int_v<Rank, 1>());
    dev_du = GetArray(sequence_of_int_v<Rank - 1, 1>(), N - 2);
    host_u = Kokkos::create_mirror(dev_u);
    host_du = Kokkos::create_mirror(dev_du);
    init(std::make_index_sequence<Rank>());
  }

  template <std::size_t... Is, class... Args>
  auto GetArray(std::index_sequence<Is...>, Args... Ns) {
    return GetArray_impl<Rank>(Is * N..., Ns...);
  }

  template <std::size_t... Is>
  void init(std::index_sequence<Is...>) {
    for (int id = 0; id < Rank; id++) {
      bounds[id].s = 0;
      bounds[id].e = N - 1;
    }
    const auto idxer =
        parthenon::MakeIndexer(std::pair<int, int>(bounds[Is].s, bounds[Is].e)...);
    for (int idx = 0; idx < idxer.size(); idx++) {
      const auto indices = idxer.GetIdxArray(idx);
      // initialize with i^2 * j^2 * k^2
      host_u(indices[Is]...) = pow((1 * ... * (indices[Is] + 1 + Is)), 2.0);
    }
    // Copy host array content to device
    Kokkos::deep_copy(dev_u, host_u);
  }

  template <typename, typename, typename, typename, typename, typename>
  struct dispatch {};

  template <typename OuterPattern, std::size_t... OuterIs, typename... OuterArgs,
            typename InnerPattern, std::size_t... InnerIs, typename... InnerArgs>
  struct dispatch<OuterPattern, std::index_sequence<OuterIs...>,
                  parthenon::TypeList<OuterArgs...>, InnerPattern,
                  std::index_sequence<InnerIs...>, parthenon::TypeList<InnerArgs...>> {
    using team_mbr_t = parthenon::team_mbr_t;
    static constexpr std::size_t Nouter = sizeof...(OuterIs);
    static constexpr std::size_t Ninner = Rank - Nouter - 1;

    template <typename view_t>
    void execute(DevExecSpace exec_space, view_t &dev_u, view_t &dev_du,
                 Kokkos::Array<parthenon::IndexRange, Rank> bounds) {
      // Compute the scratch memory needs
      const int scratch_level = 0;
      std::size_t scratch_size_in_bytes =
          parthenon::ScratchPad1D<Real>::shmem_size(pow(N, Ninner));

      parthenon::par_for(
          OuterPattern(), "unit test ND nested", exec_space, bounds[OuterIs]...,
          KOKKOS_CLASS_LAMBDA(team_mbr_t team_member, OuterArgs... outer_args) {
            auto scratch_u = GetScratchPad(std::make_index_sequence<Ninner + 1>(),
                                           team_member, scratch_level);

            parthenon::par_for_inner(
                InnerPattern(), team_member, bounds[Nouter + InnerIs]...,
                bounds[Rank - 1], [&](InnerArgs... inner_args, const int i) {
                  scratch_u(inner_args..., i) = dev_u(outer_args..., inner_args..., i);
                });
            // Sync all threads in the team so that scratch memory is consistent
            team_member.team_barrier();

            // Compute the derivative from scratch memory
            parthenon::par_for_inner(InnerPattern(), team_member,
                                     bounds[Nouter + InnerIs]..., 1, N - 2,
                                     [&](InnerArgs... inner_args, const int i) {
                                       dev_du(outer_args..., inner_args..., i) =
                                           (scratch_u(inner_args..., i + 1) -
                                            scratch_u(inner_args..., i - 1)) /
                                           2.;
                                     });
          });
    }

    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION auto GetScratchPad(std::index_sequence<Is...>,
                                              team_mbr_t team_member,
                                              const int &scratch_level) const {
      return ScratchPadND<Ninner + 1, Real>(team_member.team_scratch(scratch_level),
                                            N * Is...);
    }
  };

  template <std::size_t... Is>
  bool test_comp(std::index_sequence<Is...>) {
    // Copy array back from device to host
    Kokkos::deep_copy(host_du, dev_du);

    Real max_rel_err = -1;
    const Real rel_tol = std::numeric_limits<Real>::epsilon();

    auto idxer =
        parthenon::MakeIndexer(std::pair<int, int>(bounds[Is].s, bounds[Is].e)...);
    for (int idx = 0; idx < idxer.size(); idx++) {
      auto indices = idxer.GetIdxArray(idx);
      for (int i = 1 + bounds[Rank - 1].s; i < bounds[Rank - 1].e - 1; i++) {
        const Real analytic =
            2.0 * (i + Rank) * pow((1 * ... * (indices[Is] + 1 + Is)), 2.0);
        const Real err = host_du(indices[Is]..., i) - analytic;
        max_rel_err = fmax(fabs(err / analytic), max_rel_err);
      }
    }

    return max_rel_err < rel_tol;
  }

  template <std::size_t Ninner, class OuterPattern, class InnerPattern>
  bool test(OuterPattern, InnerPattern, DevExecSpace exec_space) {
    Kokkos::deep_copy(dev_du, -11111.0);
    constexpr std::size_t Nouter = Rank - Ninner;
    dispatch<OuterPattern, std::make_index_sequence<Nouter>,
             parthenon::list_of_type_t<Nouter, const int>, InnerPattern,
             std::make_index_sequence<Ninner - 1>,
             parthenon::list_of_type_t<Ninner - 1, const int>>()
        .execute(exec_space, dev_u, dev_du, bounds);

    Kokkos::fence();
    return test_comp(std::make_index_sequence<Rank - 1>());
  }
};

template <std::size_t Rank, std::size_t N>
void test_nested_nd() {
  auto default_exec_space = DevExecSpace();
  auto test_nested_ND = test_wrapper_nested_nd_impl<Rank, N>();
  SECTION("Inner collaspe 1") {
    REQUIRE(test_nested_ND.template test<1>(parthenon::outer_loop_pattern_teams_tag,
                                            parthenon::inner_loop_pattern_tvr_tag,
                                            default_exec_space) == true);
    REQUIRE(test_nested_ND.template test<1>(parthenon::outer_loop_pattern_teams_tag,
                                            parthenon::inner_loop_pattern_ttr_tag,
                                            default_exec_space) == true);
    if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>::value) {
      REQUIRE(test_nested_ND.template test<1>(parthenon::outer_loop_pattern_teams_tag,

                                              parthenon::inner_loop_pattern_simdfor_tag,

                                              default_exec_space) == true);
    }
  }
  SECTION("Inner collaspe 2") {
    REQUIRE(test_nested_ND.template test<2>(parthenon::outer_loop_pattern_teams_tag,
                                            parthenon::inner_loop_pattern_tvr_tag,
                                            default_exec_space) == true);
    REQUIRE(test_nested_ND.template test<2>(parthenon::outer_loop_pattern_teams_tag,
                                            parthenon::inner_loop_pattern_ttr_tag,
                                            default_exec_space) == true);
    if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>::value) {
      REQUIRE(test_nested_ND.template test<2>(parthenon::outer_loop_pattern_teams_tag,

                                              parthenon::inner_loop_pattern_simdfor_tag,

                                              default_exec_space) == true);
    }
  }
}

TEST_CASE("nested par_for loops", "[wrapper]") {
  auto default_exec_space = DevExecSpace();

  SECTION("3D nested loops") { test_nested_nd<3, 32>(); }
  SECTION("4D nested loops") { test_nested_nd<4, 32>(); }
  SECTION("5D nested loops") { test_nested_nd<5, 10>(); }
  SECTION("6D nested loops") { test_nested_nd<6, 10>(); }
  SECTION("7D nested loops") { test_nested_nd<7, 10>(); }
}

template <class T>
bool test_wrapper_scan_1d(T loop_pattern, DevExecSpace exec_space) {
  const int N = 10;
  parthenon::ParArray1D<int> buffer("Testing buffer", N);
  // Initialize data
  parthenon::par_for(
      loop_pattern, "Initialize parallel scan array", exec_space, 0, N - 1,
      KOKKOS_LAMBDA(const int i) { buffer(i) = i; });

  parthenon::ParArray1D<int> scanned("Result of scan", N);
  int result;
  parthenon::par_scan(
      loop_pattern, "Parallel scan", exec_space, 0, N - 1,
      KOKKOS_LAMBDA(const int i, int &partial_sum, bool is_final) {
        if (is_final) {
          scanned(i) = partial_sum;
        }
        partial_sum += buffer(i);
      },
      result);

  // compare data on the host
  bool all_same = true;
  auto scanned_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scanned);
  for (int i = 0; i < N; i++) {
    int ans = 0;
    for (int j = 0; j < i; j++) {
      ans += j;
    }
    if (scanned_h(i) != ans) {
      all_same = false;
    }
  }

  return all_same;
}

TEST_CASE("Parallel scan", "[par_scan]") {
  auto default_exec_space = DevExecSpace();

  SECTION("1D loops") {
    REQUIRE(test_wrapper_scan_1d(parthenon::loop_pattern_flatrange_tag,
                                 default_exec_space) == true);
  }
}

template <std::size_t Rank, std::size_t N>
struct test_wrapper_reduce_nd_impl {
  int indices[Rank - 1], int_bounds[2 * Rank];
  parthenon::IndexRange bounds[Rank];
  int h_sum;

  test_wrapper_reduce_nd_impl() {
    h_sum = 0;
    par_red_init(std::make_index_sequence<Rank>(), h_sum);
  }

  template <std::size_t... Is>
  auto GetArray(std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) == Rank);
    return ParArrayND<Rank, Real>("device", N * Is...);
  }

  template <std::size_t... Is>
  void par_red_init(std::index_sequence<Is...>, int &sum) {
    for (int id = 0; id < Rank; id++) {
      bounds[id].s = 0;
      bounds[id].e = N - 1;
      int_bounds[2 * id] = 0;
      int_bounds[2 * id + 1] = N - 1;
    }
    const auto idxer =
        parthenon::MakeIndexer(std::pair<int, int>(bounds[Is].s, bounds[Is].e)...);
    for (int idx = 0; idx < idxer.size(); idx++) {
      const auto indices = idxer.GetIdxArray(idx);
      sum += (0 + ... + indices[Is]);
    }
  }

  template <typename, lbounds, typename, typename>
  struct dispatch {};

  template <typename Pattern, lbounds bound_type, std::size_t... Ids, typename... Ts>
  struct dispatch<Pattern, bound_type, std::index_sequence<Ids...>,
                  parthenon::TypeList<Ts...>> {
    bool execute(DevExecSpace exec_space, const int h_sum, int *int_bounds,
                 parthenon::IndexRange *bounds) {
      int test_sum = 0;
      if constexpr (bound_type == lbounds::integer) {
        parthenon::par_reduce(
            Pattern(), "sum via par_reduce integer bounds", exec_space,
            int_bounds[Ids]...,

            KOKKOS_CLASS_LAMBDA(Ts... args, int &sum) { sum += (args + ...); },
            Kokkos::Sum<int>(test_sum));
      } else {
        parthenon::par_reduce(
            Pattern(), "sum via par_reduce IndexRange bounds", exec_space, bounds[Ids]...,

            KOKKOS_CLASS_LAMBDA(Ts... args, int &sum) { sum += (args + ...); },
            Kokkos::Sum<int>(test_sum));
      }
      return h_sum == test_sum;
    }
  };

  template <typename T>
  void test(T loop_pattern, DevExecSpace exec_space) {
    SECTION("integer launch bounds") {
      REQUIRE(dispatch<T, lbounds::integer, Sequence<2 * Rank>,
                       parthenon::list_of_type_t<Rank, const int>>()
                  .execute(exec_space, h_sum, int_bounds, bounds) == true);
    }
    SECTION("IndexRange launch bounds") {
      REQUIRE(dispatch<T, lbounds::integer, Sequence<2 * Rank>,
                       parthenon::list_of_type_t<Rank, const int>>()
                  .execute(exec_space, h_sum, int_bounds, bounds) == true);
    }
  }

  template <typename OuterPattern, typename InnerPattern>
  void test_nest(OuterPattern outer_patter, InnerPattern inner_pattern) {}
};

template <std::size_t Rank, std::size_t N>
void test_wrapper_reduce_nd(DevExecSpace exec_space) {
  auto wrappernd = test_wrapper_reduce_nd_impl<Rank, N>();
  SECTION("LoopPatternFlatRange") {
    wrappernd.test(parthenon::loop_pattern_flatrange_tag, exec_space);
  }
  if constexpr (Rank > 1 && Rank < 7) {
    SECTION("LoopPatternMDRange") {
      wrappernd.test(parthenon::loop_pattern_mdrange_tag, exec_space);
    }
  }
  if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                             Kokkos::DefaultHostExecutionSpace>::value) {
    // this should fall-back to LoopPatternFlatRange
    SECTION("LoopPatternSimdFor") {
      wrappernd.test(parthenon::loop_pattern_simdfor_tag, exec_space);
    }
  }
}

TEST_CASE("Parallel reduce", "[par_reduce]") {
  auto default_exec_space = DevExecSpace();
  SECTION("1D loops") { test_wrapper_reduce_nd<1, 10>(default_exec_space); }
  SECTION("2D loops") { test_wrapper_reduce_nd<2, 10>(default_exec_space); }
  SECTION("3D loops") { test_wrapper_reduce_nd<3, 10>(default_exec_space); }
  SECTION("4D loops") { test_wrapper_reduce_nd<4, 10>(default_exec_space); }
  SECTION("5D loops") { test_wrapper_reduce_nd<5, 10>(default_exec_space); }
  SECTION("6D loops") { test_wrapper_reduce_nd<6, 10>(default_exec_space); }
  SECTION("7D loops") { test_wrapper_reduce_nd<7, 10>(default_exec_space); }
}
