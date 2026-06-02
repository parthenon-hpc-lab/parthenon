//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include <array>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "tensors/tt_operations.hpp"
#include "tensors/tt_types.hpp"

using namespace parthenon;
using namespace parthenon::tensor2;

namespace {

template <class TTraits>
TensorTrainT<TTraits>
MakeSparseDeltaTrain3D(const std::array<int, 3> &dims,
                       const std::vector<std::array<int, 3>> &entries,
                       const std::vector<typename TTraits::real_t> &values) {
  using real_t = typename TTraits::real_t;

  PARTHENON_REQUIRE(entries.size() == values.size(),
                    "MakeSparseDeltaTrain3D: entries and values must have the same size.");
  PARTHENON_REQUIRE(dims[0] > 0 && dims[1] > 0 && dims[2] > 0,
                    "MakeSparseDeltaTrain3D: physical dimensions must be positive.");

  const int nterms = static_cast<int>(entries.size());

  if (nterms == 0) {
    TensorTrainT<TTraits> train({dims[0], dims[1], dims[2]}, {1, 1});
    std::vector<TensorTrainT<TTraits>> trains{train};
    TensorPackT<TTraits> pack(trains);
    SetTTPackToValue(pack, real_t(0));
    return train;
  }

  for (int m = 0; m < nterms; ++m) {
    const auto &e = entries[m];
    PARTHENON_REQUIRE(0 <= e[0] && e[0] < dims[0],
                      "MakeSparseDeltaTrain3D: first index out of bounds.");
    PARTHENON_REQUIRE(0 <= e[1] && e[1] < dims[1],
                      "MakeSparseDeltaTrain3D: second index out of bounds.");
    PARTHENON_REQUIRE(0 <= e[2] && e[2] < dims[2],
                      "MakeSparseDeltaTrain3D: third index out of bounds.");
  }

  TensorTrainT<TTraits> train({dims[0], dims[1], dims[2]}, {nterms, nterms});
  std::vector<TensorTrainT<TTraits>> trains{train};
  TensorPackT<TTraits> pack(trains);

  SetTTPackToValue(pack, real_t(0));

  using entries_view_t = typename TTraits::template view_t<int*[3], ManagedTag>;
  using values_view_t = typename TTraits::template view_t<real_t*, ManagedTag>;

  entries_view_t entries_d("delta_entries", nterms);
  values_view_t values_d("delta_values", nterms);

  auto entries_h = Kokkos::create_mirror_view(entries_d);
  auto values_h = Kokkos::create_mirror_view(values_d);

  for (int m = 0; m < nterms; ++m) {
    entries_h(m, 0) = entries[m][0];
    entries_h(m, 1) = entries[m][1];
    entries_h(m, 2) = entries[m][2];
    values_h(m) = values[m];
  }

  Kokkos::deep_copy(entries_d, entries_h);
  Kokkos::deep_copy(values_d, values_h);

  parthenon::par_for(
      "MakeSparseDeltaTrain3D",
      0, nterms - 1,
      KOKKOS_LAMBDA(const int m) {
        auto &core0 = pack(0, 0, 0);
        auto &core1 = pack(0, 0, 1);
        auto &core2 = pack(0, 0, 2);

        const int i0 = entries_d(m, 0);
        const int i1 = entries_d(m, 1);
        const int i2 = entries_d(m, 2);
        const real_t a = values_d(m);

        core0(0, i0, m) = real_t(1);
        core1(m, i1, m) = real_t(1);
        core2(m, i2, 0) = a;
      });

  return train;
}

template <class TTraits>
KOKKOS_INLINE_FUNCTION
typename TTraits::real_t ReconstructDenseValue3D(const TensorPackT<TTraits> &pack,
                                                 int b, int i1, int i2, int i3) {
  auto &core0 = pack(b, 0, 0);
  auto &core1 = pack(b, 0, 1);
  auto &core2 = pack(b, 0, 2);

  typename TTraits::real_t val{0};
  for (int r1 = 0; r1 < core0.RR(); ++r1) {
    for (int r2 = 0; r2 < core1.RR(); ++r2) {
      val += core0(0, i1, r1) * core1(r1, i2, r2) * core2(r2, i3, 0);
    }
  }
  return val;
}

template <class TTraits, class... Packs>
void CheckCompatibleDense3D(const TensorPackT<TTraits> &pack0, const Packs &...packs) {
  PARTHENON_REQUIRE(pack0.GetNCores() == 3, "Only works for three-core tensor trains.");

  const int nblocks = pack0.GetNBlocks();
  const int d0 = pack0.GetPhysicalDimension(0);
  const int d1 = pack0.GetPhysicalDimension(1);
  const int d2 = pack0.GetPhysicalDimension(2);

  auto check_one = [&](const auto &pack) {
    PARTHENON_REQUIRE(pack.GetNCores() == 3, "Only works for three-core tensor trains.");
    PARTHENON_REQUIRE(pack.GetNBlocks() == nblocks,
                      "All packs must have the same number of blocks.");
    PARTHENON_REQUIRE(pack.GetPhysicalDimension(0) == d0,
                      "All packs must have the same first physical dimension.");
    PARTHENON_REQUIRE(pack.GetPhysicalDimension(1) == d1,
                      "All packs must have the same second physical dimension.");
    PARTHENON_REQUIRE(pack.GetPhysicalDimension(2) == d2,
                      "All packs must have the same third physical dimension.");
  };

  (check_one(packs), ...);
}

template <class TTraits, class CheckFunctor, class... Packs>
int CountDenseMismatches3D(CheckFunctor check, const TensorPackT<TTraits> &pack0,
                           const Packs &...packs) {
  static_assert(sizeof...(packs) >= 0,
                "CountDenseMismatches3D requires at least one pack.");

  CheckCompatibleDense3D(pack0, packs...);

  const int n0 = pack0.GetPhysicalDimension(0);
  const int n1 = pack0.GetPhysicalDimension(1);
  const int n2 = pack0.GetPhysicalDimension(2);

  int nwrong{0};
  par_reduce(loop_pattern_mdrange_tag, "Check TT", DevExecSpace(),
             0, pack0.GetNBlocks() - 1,
             0, n0 - 1,
             0, n1 - 1,
             0, n2 - 1,
             KOKKOS_LAMBDA(int b, int i1, int i2, int i3, int &lnwrong) {
               lnwrong += check(
                   b, i1, i2, i3,
                   ReconstructDenseValue3D(pack0, b, i1, i2, i3),
                   ReconstructDenseValue3D(packs, b, i1, i2, i3)...);
             },
             nwrong);

  return nwrong;
}
} // namespace

SCENARIO("tensor2 single-core train basic structure", "[tensor2]") {
  TensorTrain train({4}, {});
  TensorTrain train_copy = train;

  REQUIRE(train.NCores() == 1);
  REQUIRE(train(0).LR() == 1);
  REQUIRE(train(0).DD() == 4);
  REQUIRE(train(0).RR() == 1);

  REQUIRE(train_copy.NCores() == 1);
  REQUIRE(train_copy(0).LR() == 1);
  REQUIRE(train_copy(0).DD() == 4);
  REQUIRE(train_copy(0).RR() == 1);

  std::vector<TensorTrain> trains{train, train_copy};
  TensorPack pack(trains);

  REQUIRE(pack.GetNBlocks() == 2);
  REQUIRE(pack.GetNCores() == 1);
  REQUIRE(pack.GetPhysicalDimension(0) == 4);
  REQUIRE(pack.GetPhysicalDimensions() == std::vector<int>{4});

  SetTTPackToValue(pack, 2.0);
  Kokkos::fence();

  int nwrong{0};
  par_reduce(loop_pattern_mdrange_tag, "Check single-core TT", DevExecSpace(),
             0, pack.GetNBlocks() - 1,
             0, pack.GetPhysicalDimension(0) - 1,
             KOKKOS_LAMBDA(int b, int i, int &lnwrong) {
               auto &core = pack(b, 0, 0);
               lnwrong += (core(0, i, 0) != 2.0);
             },
             nwrong);

  REQUIRE(nwrong == 0);
}

SCENARIO("tensor2 train construction and pack metadata", "[tensor2]") {
  TensorTrain train({2, 3, 4}, {5, 6});
  std::vector<TensorTrain> trains{train};

  REQUIRE(train.NCores() == 3);
  REQUIRE(train(0).LR() == 1);
  REQUIRE(train(0).DD() == 2);
  REQUIRE(train(0).RR() == 5);
  REQUIRE(train(1).LR() == 5);
  REQUIRE(train(1).DD() == 3);
  REQUIRE(train(1).RR() == 6);
  REQUIRE(train(2).LR() == 6);
  REQUIRE(train(2).DD() == 4);
  REQUIRE(train(2).RR() == 1);

  TensorPack pack(trains);
  REQUIRE(pack.GetNBlocks() == 1);
  REQUIRE(pack.GetNCores() == 3);
  REQUIRE(pack.GetPhysicalDimension(0) == 2);
  REQUIRE(pack.GetPhysicalDimension(1) == 3);
  REQUIRE(pack.GetPhysicalDimension(2) == 4);
  REQUIRE(pack.GetPhysicalDimensions() == std::vector<int>{2, 3, 4});
  // Check zero initialized
  REQUIRE(CountDenseMismatches3D(
              KOKKOS_LAMBDA(int, int, int, int, Real value) {
                return value != 0.0;
              }, pack) == 0);
}

SCENARIO("tensor2 train copy and move preserve packable storage", "[tensor2]") {
  TensorTrain original({2, 3, 2}, {2, 2});
  std::vector<TensorTrain> originals{original};
  TensorPack original_pack(originals);
  SetTTPackToValue(original_pack, 1.5);
  Kokkos::fence();

  TensorTrain copy_constructed = original;
  TensorTrain copy_assigned({2, 3, 2}, {1, 1});
  copy_assigned = original;

  std::vector<TensorTrain> copied_trains;
  copied_trains.emplace_back(original);
  copied_trains.emplace_back(copy_constructed);
  copied_trains.emplace_back(copy_assigned);

  TensorPack copied_pack(copied_trains);
  REQUIRE(copied_pack.GetNBlocks() == 3);
  REQUIRE(copied_pack.GetNCores() == 3);

  constexpr Real core_value = 1.5;
  constexpr Real expected_dense_value = 4.0 * core_value * core_value * core_value;
  REQUIRE(CountDenseMismatches3D(
              KOKKOS_LAMBDA(int, int, int, int, Real value) {
                return value != expected_dense_value;
              }, copied_pack) == 0);

  TensorTrain move_constructed = std::move(copy_constructed);
  TensorTrain move_assigned({2, 3, 2}, {1, 1});
  move_assigned = std::move(copy_assigned);

  std::vector<TensorTrain> trains;
  trains.emplace_back(original);
  trains.emplace_back(move_constructed);
  trains.emplace_back(move_assigned);

  TensorPack pack(trains);
  REQUIRE(pack.GetNBlocks() == 3);
  REQUIRE(pack.GetNCores() == 3);

  REQUIRE(CountDenseMismatches3D(
              KOKKOS_LAMBDA(int, int, int, int, Real value) {
                return value != expected_dense_value;
              }, pack) == 0);
}

SCENARIO("tensor2 train vector push_back preserves packable storage", "[tensor2]") {
  TensorTrain train_a({2, 3, 2}, {2, 2});
  TensorTrain train_b({2, 3, 2}, {2, 2});

  std::vector<TensorTrain> one_train{train_a};
  TensorPack pack_a(one_train);
  SetTTPackToValue(pack_a, 2.5);

  one_train[0] = train_b;
  TensorPack pack_b(one_train);
  SetTTPackToValue(pack_b, 4.5);
  Kokkos::fence();

  std::vector<TensorTrain> trains;
  trains.push_back(train_a);
  trains.push_back(train_b);
  trains.push_back(train_a);

  TensorPack pack(trains);
  REQUIRE(pack.GetNBlocks() == 3);
  REQUIRE(pack.GetNCores() == 3);
  
  constexpr Real expected_a_dense_value = 4.0 * 2.5 * 2.5 * 2.5;
  constexpr Real expected_b_dense_value = 4.0 * 4.5 * 4.5 * 4.5;

  REQUIRE(CountDenseMismatches3D(
              KOKKOS_LAMBDA(int b, int, int, int, Real value) {
                if (b == 1) return value != expected_b_dense_value;  
                return value != expected_a_dense_value;  
              }, pack) == 0);
}

SCENARIO("tensor2 sparse delta train reconstructs to the correct dense values", "[tensor2]") {
  using real_t = typename DefaultTTraits::real_t;
  using entry_view_t = DefaultTTraits::template view_t<int*[3], ManagedTag>;
  using value_view_t = DefaultTTraits::template view_t<real_t*, ManagedTag>;

  const std::array<int, 3> dims{3, 4, 5};
  const std::vector<std::array<int, 3>> entries_h{
      {1, 2, 3},
      {0, 1, 4},
      {2, 0, 1}
  };
  const std::vector<real_t> values_h{7.5, -2.0, 3.25};

  TensorTrain train = MakeSparseDeltaTrain3D<DefaultTTraits>(dims, entries_h, values_h);
  std::vector<TensorTrain> trains{train};
  TensorPack pack(trains);

  REQUIRE(pack.GetNBlocks() == 1);
  REQUIRE(pack.GetNCores() == 3);
  REQUIRE(pack.GetPhysicalDimension(0) == dims[0]);
  REQUIRE(pack.GetPhysicalDimension(1) == dims[1]);
  REQUIRE(pack.GetPhysicalDimension(2) == dims[2]);

  const int nentries = static_cast<int>(entries_h.size());
  entry_view_t entries_d("entries_d", nentries);
  value_view_t values_d("values_d", nentries);

  auto entries_m = Kokkos::create_mirror_view(entries_d);
  auto values_m = Kokkos::create_mirror_view(values_d);

  for (int n = 0; n < nentries; ++n) {
    entries_m(n, 0) = entries_h[n][0];
    entries_m(n, 1) = entries_h[n][1];
    entries_m(n, 2) = entries_h[n][2];
    values_m(n) = values_h[n];
  }

  Kokkos::deep_copy(entries_d, entries_m);
  Kokkos::deep_copy(values_d, values_m);

  REQUIRE(CountDenseMismatches3D(
              KOKKOS_LAMBDA(int, int i1, int i2, int i3, real_t dense_val) {
                real_t expected = real_t(0);
                for (int n = 0; n < nentries; ++n) {
                  if (i1 == entries_d(n, 0) &&
                      i2 == entries_d(n, 1) &&
                      i3 == entries_d(n, 2)) {
                    expected += values_d(n);
                  }
                }
                return dense_val != expected;
              }, pack) == 0);
}

SCENARIO("tensor2 ReduceSize preserves retained core data", "[tensor2]") {
  using real_t = typename DefaultTTraits::real_t;

  TensorTrain train({3, 4, 2}, {3, 4});

  std::vector<TensorTrain> trains{train};
  TensorPack pack(trains);

  // Fill every entry with a value that uniquely identifies its location.
  parthenon::par_for(
      "FillTensorTrainForReduceSizeTest",
      0, pack.GetNBlocks() - 1,
      0, pack.GetNCores() - 1,
      KOKKOS_LAMBDA(const int b, const int c) {
        auto &core = pack(b, 0, c);
        for (int l = 0; l < core.LR(); ++l) {
          for (int r = 0; r < core.RR(); ++r) {
            auto *f = &core(l, 0, r);
            for (int j = 0; j < core.DD(); ++j) {
              f[j] = 1000 * c + 100 * l + 10 * r + j;
            }
          }
        }
      });
  Kokkos::fence();

  // Shrink the first core's right rank and the second core's left/right ranks
  // consistently with the train structure:
  //   core 0: (1,3,3) -> (1,3,2)
  //   core 1: (3,4,4) -> (2,4,2)
  //   core 2: (4,2,1) -> (2,2,1)
  train(0).ReduceSize(1, 2);
  train(1).ReduceSize(2, 2);
  train(2).ReduceSize(2, 1);

  TensorPack shrunk_pack(trains);

  REQUIRE(shrunk_pack.GetNBlocks() == 1);
  REQUIRE(shrunk_pack.GetNCores() == 3);

  REQUIRE(train(0).LR() == 1);
  REQUIRE(train(0).RR() == 2);
  REQUIRE(train(1).LR() == 2);
  REQUIRE(train(1).RR() == 2);
  REQUIRE(train(2).LR() == 2);
  REQUIRE(train(2).RR() == 1);

  int nwrong{0};
  par_reduce(
      loop_pattern_mdrange_tag, "CheckReduceSizeRetainedEntries", DevExecSpace(),
      0, shrunk_pack.GetNCores() - 1,
      0, 1,           // block index always 0
      0, 2,           // maximum retained left rank range we need to inspect
      0, 4,           // maximum physical dimension in this test
      0, 2,           // maximum retained right rank range we need to inspect
      KOKKOS_LAMBDA(int c, int b_dummy, int l, int j, int r, int &lnwrong) {
        auto &core = shrunk_pack(0, 0, c);

        if (l < core.LR() && j < core.DD() && r < core.RR()) {
          const real_t expected = 1000 * c + 100 * l + 10 * r + j;
          lnwrong += (core(l, j, r) != expected);
        }
      },
      nwrong);

  REQUIRE(nwrong == 0);
}

SCENARIO("tensor2 non-destructive sum of constant trains reconstructs correctly", "[tensor2]") {
  using real_t = typename DefaultTTraits::real_t;

  TensorTrain train_a({2, 5, 4}, {2, 1});
  TensorTrain train_b({2, 5, 4}, {2, 3});

  std::vector<TensorTrain> trains_a{train_a};
  std::vector<TensorTrain> trains_b{train_b};

  TensorPack pack_a(trains_a);
  TensorPack pack_b(trains_b);

  constexpr real_t a = 1.5;
  constexpr real_t b = -0.25;

  SetTTPackToValue(pack_a, a);
  SetTTPackToValue(pack_b, b);
  Kokkos::fence();

  auto trains_c = NonDestructiveSum(trains_a, trains_b);
  TensorPack pack_c(trains_c);

  REQUIRE(pack_c.GetNBlocks() == 1);
  REQUIRE(pack_c.GetNCores() == 3);
  REQUIRE(pack_c.GetPhysicalDimensions() == std::vector<int>{2, 5, 4});

  REQUIRE(CountDenseMismatches3D(
              KOKKOS_LAMBDA(int, int, int, int, real_t va, real_t vb, real_t vc) {
                return vc != va + vb;
              }, pack_a, pack_b, pack_c) == 0);
}