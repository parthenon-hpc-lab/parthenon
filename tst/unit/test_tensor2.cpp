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
int CountRegionMismatches(TensorPackT<TTraits> &pack, int b, int c, int l0,
                          int l1, int r0, int r1, typename TTraits::real_t expected) {
  Kokkos::View<int> mismatches("tensor2_mismatches");
  Kokkos::deep_copy(mismatches, 0);

  constexpr int scratch_size = 0;
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, b, b, c, c,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int bb, const int cc) {
        auto core = pack(bb, 0, cc);
        const int lr = core.LR();
        const int rr = core.RR();
        int local_mismatches = 0;
        for (int l = 0; l < lr; ++l) {
          for (int r = 0; r < rr; ++r) {
            for (int j = 0; j < core.DD(); ++j) {
              if (l0 <= l && l < l1 && r0 <= r && r < r1 && core(l, j, r) != expected) {
                ++local_mismatches;
              }
            }
          }
        }
        Kokkos::atomic_add(&mismatches(), local_mismatches);
      });

  auto mismatches_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mismatches);
  return mismatches_h();
}

template <class TTraits>
void SetRank1TrainFromFactors(std::vector<TensorTrainT<TTraits>> &trains, int t,
                              const std::array<std::vector<typename TTraits::real_t>, 3>
                                  &core_factors) {
  TensorPackT<TTraits> pack(trains);
  auto &core0 = pack(t, 0, 0);
  auto &core1 = pack(t, 0, 1);
  auto &core2 = pack(t, 0, 2);

  PARTHENON_REQUIRE(core0.LR() == 1 && core0.RR() == 1,
                    "Rank-1 initializers require rank-1 boundary cores.");
  PARTHENON_REQUIRE(core1.LR() == 1 && core1.RR() == 1,
                    "Rank-1 initializers require rank-1 middle cores.");
  PARTHENON_REQUIRE(core2.LR() == 1 && core2.RR() == 1,
                    "Rank-1 initializers require rank-1 boundary cores.");

  for (int j = 0; j < core0.DD(); ++j) {
    core0(0, j, 0) = core_factors[0][j];
  }
  for (int j = 0; j < core1.DD(); ++j) {
    core1(0, j, 0) = core_factors[1][j];
  }
  for (int j = 0; j < core2.DD(); ++j) {
    core2(0, j, 0) = core_factors[2][j];
  }
}

template <class TTraits>
typename TTraits::real_t DenseValue3D(const std::vector<TensorTrainT<TTraits>> &trains,
                                      int t, int i0, int i1, int i2) {
  TensorPackT<TTraits> pack(trains);
  const auto core0 = pack(t, 0, 0);
  const auto core1 = pack(t, 0, 1);
  const auto core2 = pack(t, 0, 2);

  typename TTraits::real_t value{0};
  for (int r01 = 0; r01 < core0.RR(); ++r01) {
    for (int r12 = 0; r12 < core1.RR(); ++r12) {
      value += core0(0, i0, r01) * core1(r01, i1, r12) * core2(r12, i2, 0);
    }
  }
  return value;
}

} // namespace

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
}

SCENARIO("tensor2 train copy and move preserve packable storage", "[tensor2]") {
  TensorTrain original({2, 3, 2}, {2, 2});
  std::vector<TensorTrain> originals{original};
  TensorPack original_pack(originals);
  SetTTPackToValue(original_pack, 1.5);
  Kokkos::fence();

  TensorTrain copy_constructed = original;
  TensorTrain move_constructed = std::move(copy_constructed);

  TensorTrain copy_assigned({2, 3, 2}, {1, 1});
  copy_assigned = original;

  TensorTrain move_assigned({2, 3, 2}, {1, 1});
  move_assigned = std::move(copy_assigned);

  std::vector<TensorTrain> trains;
  trains.emplace_back(original);
  trains.emplace_back(move_constructed);
  trains.emplace_back(move_assigned);

  TensorPack pack(trains);
  REQUIRE(pack.GetNBlocks() == 3);
  REQUIRE(pack.GetNCores() == 3);

  for (int b = 0; b < pack.GetNBlocks(); ++b) {
    REQUIRE(CountRegionMismatches(pack, b, 0, 0, 1, 0, 2, 1.5) == 0);
    REQUIRE(CountRegionMismatches(pack, b, 1, 0, 2, 0, 2, 1.5) == 0);
    REQUIRE(CountRegionMismatches(pack, b, 2, 0, 2, 0, 1, 1.5) == 0);
  }
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

  REQUIRE(CountRegionMismatches(pack, 0, 0, 0, 1, 0, 2, 2.5) == 0);
  REQUIRE(CountRegionMismatches(pack, 0, 1, 0, 2, 0, 2, 2.5) == 0);
  REQUIRE(CountRegionMismatches(pack, 0, 2, 0, 2, 0, 1, 2.5) == 0);

  REQUIRE(CountRegionMismatches(pack, 1, 0, 0, 1, 0, 2, 4.5) == 0);
  REQUIRE(CountRegionMismatches(pack, 1, 1, 0, 2, 0, 2, 4.5) == 0);
  REQUIRE(CountRegionMismatches(pack, 1, 2, 0, 2, 0, 1, 4.5) == 0);

  REQUIRE(CountRegionMismatches(pack, 2, 0, 0, 1, 0, 2, 2.5) == 0);
  REQUIRE(CountRegionMismatches(pack, 2, 1, 0, 2, 0, 2, 2.5) == 0);
  REQUIRE(CountRegionMismatches(pack, 2, 2, 0, 2, 0, 1, 2.5) == 0);
}

SCENARIO("tensor2 pack fill sets every entry", "[tensor2]") {
  TensorTrain train_a({2, 3, 4}, {2, 3});
  TensorTrain train_b({2, 3, 4}, {1, 5});
  std::vector<TensorTrain> trains{train_a, train_b};
  TensorPack pack(trains);

  SetTTPackToValue(pack, 3.25);
  Kokkos::fence();

  REQUIRE(CountRegionMismatches(pack, 0, 0, 0, 1, 0, 2, 3.25) == 0);
  REQUIRE(CountRegionMismatches(pack, 0, 1, 0, 2, 0, 3, 3.25) == 0);
  REQUIRE(CountRegionMismatches(pack, 0, 2, 0, 3, 0, 4, 3.25) == 0);

  REQUIRE(CountRegionMismatches(pack, 1, 0, 0, 1, 0, 1, 3.25) == 0);
  REQUIRE(CountRegionMismatches(pack, 1, 1, 0, 1, 0, 5, 3.25) == 0);
  REQUIRE(CountRegionMismatches(pack, 1, 2, 0, 5, 0, 4, 3.25) == 0);
}

SCENARIO("tensor2 non-destructive sum preserves block structure", "[tensor2]") {
  TensorTrain train_a({2, 3, 4}, {2, 3});
  TensorTrain train_b({2, 3, 4}, {5, 7});
  std::vector<TensorTrain> trains_a{train_a};
  std::vector<TensorTrain> trains_b{train_b};

  TensorPack pack_a(trains_a);
  TensorPack pack_b(trains_b);
  SetTTPackToValue(pack_a, 1.0);
  SetTTPackToValue(pack_b, 2.0);
  Kokkos::fence();

  auto trains_c = NonDestructiveSum(trains_a, trains_b);
  REQUIRE(trains_c.size() == 1);
  REQUIRE(trains_c[0].NCores() == 3);
  REQUIRE(trains_c[0](0).LR() == 1);
  REQUIRE(trains_c[0](0).RR() == 7);
  REQUIRE(trains_c[0](1).LR() == 7);
  REQUIRE(trains_c[0](1).RR() == 10);
  REQUIRE(trains_c[0](2).LR() == 10);
  REQUIRE(trains_c[0](2).RR() == 1);

  TensorPack pack_c(trains_c);
  REQUIRE(CountRegionMismatches(pack_c, 0, 0, 0, 1, 0, 2, 1.0) == 0);
  REQUIRE(CountRegionMismatches(pack_c, 0, 0, 0, 1, 2, 7, 2.0) == 0);

  REQUIRE(CountRegionMismatches(pack_c, 0, 1, 0, 2, 0, 3, 1.0) == 0);
  REQUIRE(CountRegionMismatches(pack_c, 0, 1, 2, 7, 3, 10, 2.0) == 0);
  REQUIRE(CountRegionMismatches(pack_c, 0, 1, 0, 2, 3, 10, 0.0) == 0);
  REQUIRE(CountRegionMismatches(pack_c, 0, 1, 2, 7, 0, 3, 0.0) == 0);

  REQUIRE(CountRegionMismatches(pack_c, 0, 2, 0, 3, 0, 1, 1.0) == 0);
  REQUIRE(CountRegionMismatches(pack_c, 0, 2, 3, 10, 0, 1, 2.0) == 0);
}

SCENARIO("tensor2 hadamard product multiplies values and ranks", "[tensor2]") {
  TensorTrain train_a({2, 3, 4}, {2, 3});
  TensorTrain train_b({2, 3, 4}, {5, 7});
  std::vector<TensorTrain> trains_a{train_a};
  std::vector<TensorTrain> trains_b{train_b};

  TensorPack pack_a(trains_a);
  TensorPack pack_b(trains_b);
  SetTTPackToValue(pack_a, 3.0);
  SetTTPackToValue(pack_b, 4.0);
  Kokkos::fence();

  auto trains_c = HadamardProduct(trains_a, trains_b);
  REQUIRE(trains_c.size() == 1);
  REQUIRE(trains_c[0].NCores() == 3);
  REQUIRE(trains_c[0](0).RR() == 10);
  REQUIRE(trains_c[0](1).RR() == 21);
  REQUIRE(trains_c[0](2).RR() == 1);

  TensorPack pack_c(trains_c);
  REQUIRE(CountRegionMismatches(pack_c, 0, 0, 0, 1, 0, 10, 12.0) == 0);
  REQUIRE(CountRegionMismatches(pack_c, 0, 1, 0, 10, 0, 21, 12.0) == 0);
  REQUIRE(CountRegionMismatches(pack_c, 0, 2, 0, 21, 0, 1, 12.0) == 0);
}

SCENARIO("tensor2 dense element evaluation on rank-1 data", "[tensor2][dense]") {
  std::vector<TensorTrain> trains{
      TensorTrain({2, 3, 2}, {1, 1}),
  };

  SetRank1TrainFromFactors(trains, 0,
                           std::array<std::vector<Real>, 3>{
                               std::vector<Real>{2.0, 3.0},
                               std::vector<Real>{5.0, 7.0, 11.0},
                               std::vector<Real>{13.0, 17.0},
                           });

  REQUIRE(DenseValue3D(trains, 0, 0, 0, 0) == Approx(2.0 * 5.0 * 13.0));
  REQUIRE(DenseValue3D(trains, 0, 1, 2, 1) == Approx(3.0 * 11.0 * 17.0));
  REQUIRE(DenseValue3D(trains, 0, 1, 1, 0) == Approx(3.0 * 7.0 * 13.0));
}

SCENARIO("tensor2 dense arithmetic matches sum and hadamard", "[tensor2][dense]") {
  std::vector<TensorTrain> trains_a{
      TensorTrain({2, 3, 2}, {1, 1}),
  };
  std::vector<TensorTrain> trains_b{
      TensorTrain({2, 3, 2}, {1, 1}),
  };

  SetRank1TrainFromFactors(trains_a, 0,
                           std::array<std::vector<Real>, 3>{
                               std::vector<Real>{2.0, 3.0},
                               std::vector<Real>{5.0, 7.0, 11.0},
                               std::vector<Real>{13.0, 17.0},
                           });
  SetRank1TrainFromFactors(trains_b, 0,
                           std::array<std::vector<Real>, 3>{
                               std::vector<Real>{1.0, 4.0},
                               std::vector<Real>{6.0, 8.0, 10.0},
                               std::vector<Real>{9.0, 12.0},
                           });

  auto trains_sum = NonDestructiveSum(trains_a, trains_b);
  auto trains_hadamard = HadamardProduct(trains_a, trains_b);

  REQUIRE(trains_sum.size() == 1);
  REQUIRE(trains_hadamard.size() == 1);

  for (int i0 = 0; i0 < 2; ++i0) {
    for (int i1 = 0; i1 < 3; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        const Real a = DenseValue3D(trains_a, 0, i0, i1, i2);
        const Real b = DenseValue3D(trains_b, 0, i0, i1, i2);
        const Real sum_expected = a + b;
        const Real had_expected = a * b;
        REQUIRE(DenseValue3D(trains_sum, 0, i0, i1, i2) == Approx(sum_expected));
        REQUIRE(DenseValue3D(trains_hadamard, 0, i0, i1, i2) == Approx(had_expected));
      }
    }
  }
}
