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
int CountDenseMismatches3D(const TensorPackT<TTraits> &pack0,
                           const Packs &...packs,
                           CheckFunctor check) {
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
  REQUIRE(CountDenseMismatches3D(pack,
              KOKKOS_LAMBDA(int, int, int, int, Real value) {
                return value != 0.0;
              }) == 0);
}

SCENARIO("tensor2 train copy and move preserve packable storage", "[tensor2]") {
  // TEMP(LFR): This storage-regression test is considered solidified. Some later
  // tensor2 tests are still scaffolding around temporary test utilities.
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
              copied_pack, KOKKOS_LAMBDA(int, int, int, int, Real value) {
                return value != expected_dense_value;
              }) == 0);

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
              pack, KOKKOS_LAMBDA(int, int, int, int, Real value) {
                return value != expected_dense_value;
              }) == 0);
}

SCENARIO("tensor2 train vector push_back preserves packable storage", "[tensor2]") {
  // TEMP(LFR): This storage-regression test is considered solidified. Some later
  // tensor2 tests are still scaffolding around temporary test utilities.
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

  REQUIRE(CountDenseMismatches3D(pack, 
              KOKKOS_LAMBDA(int b, int, int, int, Real value) {
                if (b == 1) return value != expected_b_dense_value;  
                return value != expected_a_dense_value;  
              }) == 0);
}
