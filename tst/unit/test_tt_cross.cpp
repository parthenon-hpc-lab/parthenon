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

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch.hpp>

#include "config.hpp"
#include "kokkos_abstraction.hpp"
#include "tensors/tt_cross.hpp"
#include "tensors/tt_pack.hpp"
#include "tensors/tt_types.hpp"

// TTCross requires double-precision trains (it uses the double batched linear
// algebra), so there is nothing to test in a single-precision build.
#if !SINGLE_PRECISION_ENABLED

using namespace parthenon;         // NOLINT(build/namespaces)
using namespace parthenon::tensor; // NOLINT(build/namespaces)

namespace tt_cross_test {

constexpr int kMaxRank = 16;
constexpr int kMaxCores = 8;

// Value of tensor b of a pack at the multi-index mi, contracted left to right.
template <class Pack, class MI>
KOKKOS_INLINE_FUNCTION Real EvaluateTrain(const Pack &pack, const int b, const MI &mi) {
  Real v[kMaxRank];
  Real w[kMaxRank];
  v[0] = 1.0;
  for (int c = 0; c < pack.GetNCores(); ++c) {
    const auto &core = pack(b, 0, c);
    const int j = mi(c);
    for (int r = 0; r < core.RR(); ++r) {
      Real s = 0.0;
      for (int l = 0; l < core.LR(); ++l)
        s += v[l] * core(l, j, r);
      w[r] = s;
    }
    for (int r = 0; r < core.RR(); ++r)
      v[r] = w[r];
  }
  return v[0];
}

// A multi-index stored as an explicit array, for the dense checks.
struct flat_multi_index {
  int idx[kMaxCores];
  KOKKOS_INLINE_FUNCTION int operator()(const int c) const { return idx[c]; }
};

// A deterministic hash of the integers mapped to [-1, 1).
KOKKOS_INLINE_FUNCTION Real HashToReal(std::uint32_t key) {
  key ^= key >> 16;
  key *= 0x7feb352dU;
  key ^= key >> 15;
  key *= 0x846ca68bU;
  key ^= key >> 16;
  return Real(key) / Real(2147483648.0) - Real(1);
}

// prod_c g_c(i_c), different for each batch entry: TT rank one.
struct SeparableProduct {
  int ncores;
  template <class MI>
  KOKKOS_INLINE_FUNCTION Real operator()(const int b, const MI &mi) const {
    Real p = 1.0;
    for (int c = 0; c < ncores; ++c)
      p *= 1.5 + std::cos(0.7 * mi(c) + 0.3 * c + 0.9 * b);
    return p;
  }
};

// 1 / (1 + (b + 1) sum_c x_c) on the uniform grid x_c = i_c / n.
struct InverseSum {
  int ncores, n;
  template <class MI>
  KOKKOS_INLINE_FUNCTION Real operator()(const int b, const MI &mi) const {
    Real s = 0.0;
    for (int c = 0; c < ncores; ++c)
      s += Real(mi(c)) / n;
    return 1.0 / (1.0 + (b + 1) * s);
  }
};

// An unstructured tensor: every entry an independent hash. Its TT ranks are
// the full ranks of its unfoldings.
struct HashTensor {
  int ncores;
  template <class MI>
  KOKKOS_INLINE_FUNCTION Real operator()(const int b, const MI &mi) const {
    std::uint32_t key = 977U * (b + 1);
    for (int c = 0; c < ncores; ++c)
      key = 131U * key + mi(c) + 1;
    return HashToReal(key);
  }
};

// The tensors held by a pack of reference trains.
template <class TTraits>
struct TrainTensor {
  TensorPackT<TTraits> pack;
  template <class MI>
  KOKKOS_INLINE_FUNCTION Real operator()(const int b, const MI &mi) const {
    return EvaluateTrain(pack, b, mi);
  }
};

// Trains with the given dims and ranks, filled with hashed values.
template <class TTraits>
std::vector<TensorTrainT<TTraits>> RandomTrains(const int nbatch,
                                                const std::vector<int> &dims,
                                                const std::vector<int> &ranks) {
  std::vector<TensorTrainT<TTraits>> trains;
  for (int b = 0; b < nbatch; ++b)
    trains.emplace_back(dims, ranks);
  TensorPackT<TTraits> pack(trains);
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 1, 0, pack.GetNBlocks() - 1, 0, pack.GetNCores() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int b, const int c) {
        auto &core = pack(b, 0, c);
        parthenon::par_for_inner(tm, 0, core.LR() - 1, 0, core.RR() - 1, 0, core.DD() - 1,
                                 [&](const int l, const int r, const int j) {
                                   const std::uint32_t key =
                                       ((((b + 1) * 61U + c) * 59U + l) * 53U + r) * 47U +
                                       j;
                                   core(l, j, r) = HashToReal(key);
                                 });
      });
  Kokkos::fence();
  return trains;
}

// Empty trains carrying only the core structure for TTCross to fill.
template <class TTraits>
std::vector<TensorTrainT<TTraits>> ShapeTrains(const int nbatch,
                                               const std::vector<int> &dims) {
  std::vector<TensorTrainT<TTraits>> trains;
  for (int b = 0; b < nbatch; ++b)
    trains.emplace_back(dims, std::vector<int>(dims.size() - 1, 1));
  return trains;
}

// Largest |train_b(i) - g(b, i)| over every batch entry and every multi-index.
template <class TTraits, class G>
Real MaxDenseError(const std::vector<TensorTrainT<TTraits>> &trains, const G &g) {
  TensorPackT<TTraits> pack(trains);
  const int nbatch = pack.GetNBlocks();
  const int ncores = pack.GetNCores();
  PARTHENON_REQUIRE(ncores <= kMaxCores, "Too many cores for the dense check.");
  std::vector<int> dims = pack.GetPhysicalDimensions();
  int total = 1;
  for (const int n : dims)
    total *= n;
  Kokkos::View<int *, DevMemSpace> dims_d("dims", ncores);
  auto dims_h = Kokkos::create_mirror_view(dims_d);
  for (int c = 0; c < ncores; ++c)
    dims_h(c) = dims[c];
  Kokkos::deep_copy(dims_d, dims_h);

  Real err{0.0};
  Kokkos::parallel_reduce(
      "TT cross dense error",
      Kokkos::MDRangePolicy<DevExecSpace, Kokkos::Rank<2>>({0, 0}, {nbatch, total}),
      KOKKOS_LAMBDA(const int b, const int flat, Real &lmax) {
        flat_multi_index mi;
        int rem = flat;
        for (int c = ncores - 1; c >= 0; --c) {
          mi.idx[c] = rem % dims_d(c);
          rem /= dims_d(c);
        }
        const Real e = std::abs(EvaluateTrain(pack, b, mi) - g(b, mi));
        lmax = (e > lmax) ? e : lmax;
      },
      Kokkos::Max<Real>(err));
  return err;
}

template <class TTraits>
std::vector<int> InternalRanks(const TensorTrainT<TTraits> &train) {
  std::vector<int> out;
  for (int c = 0; c + 1 < static_cast<int>(train.NCores()); ++c)
    out.push_back(train(c).RR());
  return out;
}

} // namespace tt_cross_test

using namespace tt_cross_test; // NOLINT(build/namespaces)

TEMPLATE_TEST_CASE("TT cross reproduces a separable tensor at rank one",
                   "[tensor][cross]", FiberTTraits, ContiguousTTraits) {
  using TTraits = TestType;
  const std::vector<int> dims{5, 4, 6, 3};
  auto trains = ShapeTrains<TTraits>(3, dims);
  const SeparableProduct g{static_cast<int>(dims.size())};

  TTCross(trains, g, {1, 1, 1});
  REQUIRE(InternalRanks(trains[0]) == std::vector<int>{1, 1, 1});
  REQUIRE(MaxDenseError(trains, g) < 1.0e-12);
}

TEMPLATE_TEST_CASE("TT cross of a single-core train is the fiber itself",
                   "[tensor][cross]", FiberTTraits, ContiguousTTraits) {
  using TTraits = TestType;
  auto trains = ShapeTrains<TTraits>(2, {7});
  const HashTensor g{1};

  TTCross(trains, g, {});
  REQUIRE(MaxDenseError(trains, g) == 0.0);
}

TEMPLATE_TEST_CASE("TT cross recovers random low-rank trains exactly", "[tensor][cross]",
                   FiberTTraits, ContiguousTTraits) {
  using TTraits = TestType;
  const std::vector<int> dims{4, 5, 3, 6};
  const std::vector<int> ranks{3, 4, 3};
  const auto reference = RandomTrains<TTraits>(3, dims, ranks);
  const TrainTensor<TTraits> g{TensorPackT<TTraits>(reference)};

  auto trains = ShapeTrains<TTraits>(3, dims);
  TTCross(trains, g, ranks);
  REQUIRE(InternalRanks(trains[0]) == ranks);
  REQUIRE(MaxDenseError(trains, g) < 1.0e-10);
}

TEMPLATE_TEST_CASE("TT cross error decreases with rank on a smooth function",
                   "[tensor][cross]", FiberTTraits, ContiguousTTraits) {
  using TTraits = TestType;
  const int n = 8;
  const std::vector<int> dims(5, n);
  const InverseSum g{static_cast<int>(dims.size()), n};

  Real prev_err = 1.0e300;
  for (const int r : {1, 2, 4, 6}) {
    auto trains = ShapeTrains<TTraits>(2, dims);
    TTCross(trains, g, std::vector<int>(dims.size() - 1, r));
    const Real err = MaxDenseError(trains, g);
    INFO("rank " << r << " error " << err);
    REQUIRE(err < prev_err);
    prev_err = err;
  }
  REQUIRE(prev_err < 1.0e-6);
}

TEMPLATE_TEST_CASE("TT cross clips oversized ranks and is then exact", "[tensor][cross]",
                   FiberTTraits, ContiguousTTraits) {
  using TTraits = TestType;
  const std::vector<int> dims{2, 3, 2};
  const HashTensor g{static_cast<int>(dims.size())};

  auto trains = ShapeTrains<TTraits>(2, dims);
  TTCross(trains, g, {10, 10});
  REQUIRE(InternalRanks(trains[0]) == std::vector<int>{2, 2});
  REQUIRE(MaxDenseError(trains, g) < 1.0e-12);

  REQUIRE(parthenon::tensor::impl::ClipCrossRanks({3, 4, 5, 2}, {100, 100, 100}) ==
          std::vector<int>{3, 10, 2});
}

TEMPLATE_TEST_CASE("TT cross warm-starts from persistent index sets", "[tensor][cross]",
                   FiberTTraits, ContiguousTTraits) {
  using TTraits = TestType;
  const std::vector<int> dims{4, 5, 3, 6};
  const std::vector<int> ranks{3, 4, 3};
  const auto reference = RandomTrains<TTraits>(2, dims, ranks);
  const TrainTensor<TTraits> g{TensorPackT<TTraits>(reference)};

  TTCrossIndexSetsT<TTraits> sets;
  auto first = ShapeTrains<TTraits>(2, dims);
  const int nfirst = TTCross(first, g, ranks, 10, &sets);
  REQUIRE(sets.converged);
  REQUIRE(nfirst < 10);

  auto second = ShapeTrains<TTraits>(2, dims);
  REQUIRE(TTCross(second, g, ranks, 10, &sets) == 1);
  REQUIRE(sets.converged);
  REQUIRE(MaxDenseError(second, g) < 1.0e-10);

  // Different ranks need different sets, so the warm start is discarded.
  auto third = ShapeTrains<TTraits>(2, dims);
  TTCross(third, g, {2, 2, 2}, 1, &sets);
  REQUIRE(!sets.converged);
  REQUIRE(sets.Ranks() == std::vector<int>{2, 2, 2});
}

#endif // !SINGLE_PRECISION_ENABLED
