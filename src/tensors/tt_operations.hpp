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

#ifndef TENSORS_TT_OPERATIONS_HPP
#define TENSORS_TT_OPERATIONS_HPP

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "tt_traits.hpp"
#include "tt_types.hpp"

namespace parthenon {
namespace impl {
template <class TTraits>
KOKKOS_INLINE_FUNCTION
void CopyCoreBlock(parthenon::team_mbr_t member,
                   const TensorCoreDevice<TTraits> &src,
                   TensorCoreDevice<TTraits> &dst,
                   int loffset = 0, int roffset = 0) {
  for (int l = 0; l < src.LR(); ++l) {
    for (int r = 0; r < src.RR(); ++r) {
      auto const * const fs = &src(l, 0, r);
      auto *fd = &dst(l + loffset, 0, r + roffset);
      parthenon::par_for_inner(member, 0, src.DD() - 1,
                               [&](const int j) { fd[j] = fs[j]; });
    }
  }
}

template <class TTraits>
KOKKOS_INLINE_FUNCTION
void SetCoreBlock(parthenon::team_mbr_t member,
                  TensorCoreDevice<TTraits> &dst,
                  typename TTraits::real_t value,
                  std::pair<int, int> lrange, 
                  std::pair<int, int> rrange) {
  for (int l = lrange.first; l < lrange.second; ++l) {
    for (int r = rrange.first; r < rrange.second; ++r) {
      auto *fd = &dst(l, 0, r);
      parthenon::par_for_inner(member, 0, dst.DD() - 1,
                               [&](const int j) { fd[j] = value; });
    }
  }
}

template <class TTraits>
KOKKOS_INLINE_FUNCTION
void HadamardCoreBlocks(parthenon::team_mbr_t member,
                        const TensorCoreDevice<TTraits> &core_a,
                        const TensorCoreDevice<TTraits> &core_b,
                        TensorCoreDevice<TTraits> &core_c) {
  for (int la = 0; la < core_a.LR(); ++la) {
    for (int lb = 0; lb < core_b.LR(); ++lb) {
      const int lc = la * core_b.LR() + lb;

      for (int ra = 0; ra < core_a.RR(); ++ra) {
        for (int rb = 0; rb < core_b.RR(); ++rb) {
          const int rc = ra * core_b.RR() + rb;

          auto const * const fa = &core_a(la, 0, ra);
          auto const * const fb = &core_b(lb, 0, rb);
          auto *fc = &core_c(lc, 0, rc);

          parthenon::par_for_inner(member, 0, core_c.DD() - 1,
                                   [&](const int j) { fc[j] = fa[j] * fb[j]; });
        }
      }
    }
  }
}
} // namespace impl

template <class TTraits>
void SetTTPackToValue(TensorPack<TTraits> &pack, Real value) {
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
    PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
    0, pack.GetNBlocks() - 1, 0, pack.GetNcores() - 1,
    KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
      auto &core = pack.(b, 0, c);
      impl::SetCoreBlock(member, core, value, {0, core.LR()}, {0, core.RR()});
    });
}

template <class TTraits>
std::vector<TensorTrain<TTraits>>
NonDestructiveSum(std::vector<TensorTrain<TTraits>> &TrainsA,
                  std::vector<TensorTrain<TTraits>> &TrainsB) {
  PARTHENON_REQUIRE(TrainsA.size() == TrainsB.size(), "Must be adding the same number of TTs.");

  // First create the memory to store the new train
  std::vector<TensorTrain<TTraits>> TrainsC;
  TrainsC.reserve(TrainsA.size());

  for (int t = 0; t < TrainsA.size(); ++t) {
    const auto &train_A = TrainsA[t];
    const auto &train_B = TrainsB[t];
    std::vector<int> phys_dims, target_ranks;
    PARTHENON_REQUIRE(train_A.NCores() == train_B.NCores(), "Added trains must have the same number of cores.");
    for (int c = 0; c < train_A.NCores(); ++c) { 
      PARTHENON_REQUIRE(train_A(c).DD() == train_B(c).DD(), "Must have equivalent physical dims.");
      phys_dims.push_back(train_A(c).DD());
    }
    for (int c = 0; c < train_A.NCores() - 1; ++c)
      target_ranks.push_back(train_A(c).RR() + train_B(c).RR());
    TrainsC.emplace_back(phys_dims, target_ranks);
  }
  
  // Now make the packs, eventually this may be just one pack
  TensorPack<TTraits> pack_a(TrainsA);
  TensorPack<TTraits> pack_b(TrainsB);
  TensorPack<TTraits> pack_c(TrainsC);
  
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
    PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
    0, pack_a.GetNBlocks() - 1, 0, pack_a.GetNcores() - 1,
    KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
      auto &core_c = pack_c(b, 0, c);

      // First add the a contribution      
      auto &core_a = pack_a(b, 0, c);
      impl::CopyCoreBlock(member, core_a, core_c, 0, 0);

      // Then add the b contribution      
      auto &core_b = pack_b(b, 0, c);
      const int loffset = (c > 0) * core_a.LR(); // Should be zero if the first core
      const int roffset = (c != (pack_a.GetNcores() - 1)) * core_a.RR(); // Should be zero if the last core
      impl::CopyCoreBlock(member, core_b, core_c, loffset, roffset);

      // Zero the off diagonals, probably lots of room to optimize here (e.g. fill with null fibers)
      if (loffset && roffset) {
        impl::SetCoreBlock(member, core_c, 0.0, {0, core_a.LR()}, {core_a.RR(), core_a.RR() + core_b.RR()});
        impl::SetCoreBlock(member, core_c, 0.0, {core_a.LR(), core_a.LR() + core_b.LR()}, {0, core_a.RR()});
      }
    });
  return TrainsC; 
}

template <class TTraits>
std::vector<TensorTrain<TTraits>>
HadamardProduct(std::vector<TensorTrain<TTraits>> &TrainsA,
                std::vector<TensorTrain<TTraits>> &TrainsB) {
  PARTHENON_REQUIRE(TrainsA.size() == TrainsB.size(),
                    "Must be taking the Hadamard product of the same number of TTs.");

  std::vector<TensorTrain<TTraits>> TrainsC;
  TrainsC.reserve(TrainsA.size());

  for (int t = 0; t < TrainsA.size(); ++t) {
    const auto &train_A = TrainsA[t];
    const auto &train_B = TrainsB[t];

    PARTHENON_REQUIRE(train_A.NCores() == train_B.NCores(),
                      "Hadamard product requires the same number of cores.");

    std::vector<int> phys_dims, target_ranks;
    for (int c = 0; c < train_A.NCores(); ++c) {
      PARTHENON_REQUIRE(train_A(c).DD() == train_B(c).DD(),
                        "Hadamard product requires matching physical dimensions.");
      phys_dims.push_back(train_A(c).DD());
    }
    for (int c = 0; c < train_A.NCores() - 1; ++c) {
      target_ranks.push_back(train_A(c).RR() * train_B(c).RR());
    }

    TrainsC.emplace_back(phys_dims, target_ranks);
  }

  TensorPack<TTraits> pack_a(TrainsA);
  TensorPack<TTraits> pack_b(TrainsB);
  TensorPack<TTraits> pack_c(TrainsC);

  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
      0, pack_a.GetNBlocks() - 1, 0, pack_a.GetNcores() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
        auto &core_a = pack_a(b, 0, c);
        auto &core_b = pack_b(b, 0, c);
        auto &core_c = pack_c(b, 0, c);
        impl::HadamardCoreBlocks(member, core_a, core_b, core_c);
      });

  return TrainsC;
}
} // namespace parthenon

#endif // TENSOR_TT_OPERATIONS_HPP