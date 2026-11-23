//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
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

// This file was made in part with generative AI.

// Round-trip tests for tensor-train serialization (tensors/tt_serialize.hpp), the
// payload format used to ship trains between ranks.

#include <cstddef>
#include <vector>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "tensors/tt_serialize.hpp"
#include "tensors/tt_types.hpp"

using parthenon::BufArray1D;
using parthenon::Real;
using parthenon::tensor::DeserializeTrains;
using parthenon::tensor::ReadSerializedBondRanks;
using parthenon::tensor::SerializedHeaderSize;
using parthenon::tensor::SerializedSize;
using parthenon::tensor::SerializeTrains;
using parthenon::tensor::SerialTarget;
using parthenon::tensor::TensorTrain;
using train_t = TensorTrain;
using real_t = train_t::traits::real_t;
using buf_t = BufArray1D<real_t>;

namespace {

// A distinct value for every entry of every train, exactly representable in real_t.
real_t Fingerprint(int t, int c, int l, int j, int r) {
  return real_t(1 + t * 1000 + c * 100 + l * 31 + j * 7 + r * 3);
}

void Fill(train_t &train, int t) {
  for (int c = 0; c < train.NCores(); ++c) {
    const auto &core = train(c);
    for (int l = 0; l < core.LR(); ++l) {
      for (int r = 0; r < core.RR(); ++r) {
        auto fiber = core.GetFiber(l, r);
        auto fiber_h = Kokkos::create_mirror_view(fiber);
        for (int j = 0; j < core.DD(); ++j)
          fiber_h(j) = Fingerprint(t, c, l, j, r);
        Kokkos::deep_copy(fiber, fiber_h);
      }
    }
  }
}

// Number of entries of train that differ from the fingerprint of train t (or whose
// shape differs from ref).
int CountMismatches(const train_t &train, const train_t &ref, int t) {
  if (train.NCores() != ref.NCores()) return 1;
  int nwrong = 0;
  for (int c = 0; c < train.NCores(); ++c) {
    const auto &core = train(c);
    const auto &rcore = ref(c);
    if (core.LR() != rcore.LR() || core.DD() != rcore.DD() || core.RR() != rcore.RR())
      return 1;
    for (int l = 0; l < core.LR(); ++l) {
      for (int r = 0; r < core.RR(); ++r) {
        auto fiber_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), core.GetFiber(l, r));
        for (int j = 0; j < core.DD(); ++j)
          nwrong += (fiber_h(j) != Fingerprint(t, c, l, j, r));
      }
    }
  }
  return nwrong;
}

} // namespace

TEST_CASE("Tensor trains round-trip through a serialized buffer", "[tensor][serialize]") {
  // Closed multi-core trains with mixed ranks, an open single-core train (dangling
  // bonds, like an AMR coarse buffer), and a rank-one train.
  std::vector<train_t> trains;
  trains.emplace_back(std::vector<int>{6, 3, 5}, std::vector<int>{2, 4});
  trains.emplace_back(std::vector<int>{7}, std::vector<int>{}, 3, 2);
  trains.emplace_back(std::vector<int>{4, 2}, std::vector<int>{1});
  trains.emplace_back(std::vector<int>{5, 4, 3, 2}, std::vector<int>{3, 1, 2});
  const int ntrains = static_cast<int>(trains.size());
  for (int t = 0; t < ntrains; ++t)
    Fill(trains[t], t);

  THEN("SerializedSize counts the header and every core entry") {
    REQUIRE(SerializedSize(trains[0]) ==
            SerializedHeaderSize(3) + 6 * 2 + 2 * 3 * 4 + 4 * 5);
    REQUIRE(SerializedSize(trains[1]) == SerializedHeaderSize(1) + 3 * 7 * 2);
  }

  // Pack all trains back to back into one buffer with slack at the end, as a coalesced
  // message would, to exercise non-zero offsets and an oversized buffer.
  std::vector<std::size_t> offsets;
  std::size_t total = 0;
  for (const auto &train : trains) {
    offsets.push_back(total);
    total += SerializedSize(train);
  }
  constexpr std::size_t slack = 17;
  buf_t buf("tt_serial_test", total + slack);
  Kokkos::deep_copy(buf, real_t(-1));

  std::vector<const train_t *> srcs;
  std::vector<SerialTarget<buf_t>> targets;
  for (int t = 0; t < ntrains; ++t) {
    srcs.push_back(&trains[t]);
    targets.push_back({buf, offsets[t]});
  }
  SerializeTrains(srcs, targets);

  THEN("The headers carry each train's core count and bond ranks") {
    std::vector<int> ncores;
    for (const auto &train : trains)
      ncores.push_back(train.NCores());
    const auto ranks = ReadSerializedBondRanks<train_t::traits>(targets, ncores);
    REQUIRE(ranks[0] == std::vector<int>{1, 2, 4, 1});
    REQUIRE(ranks[1] == std::vector<int>{3, 2});
    REQUIRE(ranks[2] == std::vector<int>{1, 1, 1});
    REQUIRE(ranks[3] == std::vector<int>{1, 3, 1, 2, 1});
  }

  THEN("Deserializing onto the same indexers reproduces every train exactly") {
    std::vector<train_t> out(ntrains);
    std::vector<train_t *> outs;
    std::vector<std::vector<parthenon::Indexer6D>> indexers;
    for (int t = 0; t < ntrains; ++t) {
      outs.push_back(&out[t]);
      indexers.push_back(trains[t].GetCoreIndexers());
    }
    DeserializeTrains(targets, indexers, outs);
    for (int t = 0; t < ntrains; ++t) {
      INFO("train " << t);
      REQUIRE(CountMismatches(out[t], trains[t], t) == 0);
    }
  }

  THEN("The slack past the last train is untouched") {
    auto buf_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buf);
    for (std::size_t i = total; i < total + slack; ++i)
      REQUIRE(buf_h(i) == real_t(-1));
  }
}
