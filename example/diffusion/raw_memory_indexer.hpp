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
#ifndef EXAMPLE_DIFFUSION_RAW_MEMORY_INDEXER_HPP_
#define EXAMPLE_DIFFUSION_RAW_MEMORY_INDEXER_HPP_

#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

struct RawMemoryIndexer {
  using ID = parthenon::IndexDomain;
  using TE = parthenon::TopologicalElement;
  RawMemoryIndexer(int inner_length, ID domain, int halo, parthenon::MeshData<Real> *md,
                   TE domain_te, TE memory_te = TE::CC)
      : inner_length(inner_length), idxer_entire(md->GetBoundsK(ID::entire, memory_te),
                                                 md->GetBoundsJ(ID::entire, memory_te),
                                                 md->GetBoundsI(ID::entire, memory_te)) {
    auto pmesh = md->GetMeshPointer();
    const int ndim = pmesh->ndim;
    ib = md->GetBoundsI(domain, domain_te);
    jb = md->GetBoundsJ(domain, domain_te);
    kb = md->GetBoundsK(domain, domain_te);
    if (ndim > 0) {
      ib.s -= halo;
      ib.e += halo;
    }
    if (ndim > 1) {
      jb.s -= halo;
      jb.e += halo;
    }
    if (ndim > 2) {
      kb.s -= halo;
      kb.e += halo;
    }
    idxer = parthenon::Indexer3D({kb.s, kb.e}, {jb.s, jb.e}, {ib.s, ib.e});
    PARTHENON_REQUIRE(memory_te == TE::CC || memory_te == TE::NN,
                      "Only two kinds of memory layouts for topological elements.");
  }

  static RawMemoryIndexer IJ(ID domain, parthenon::MeshData<Real> *md, TE domain_te,
                             TE memory_te = TE::CC) {
    return RawMemoryIndexer::IJ(domain, 0, md, domain_te, memory_te);
  }

  static RawMemoryIndexer IJ(ID domain, int halo, parthenon::MeshData<Real> *md,
                             TE domain_te, TE memory_te = TE::CC) {
    RawMemoryIndexer idxer(0, domain, halo, md, domain_te, memory_te);
    const int ni = idxer.ib.e - idxer.ib.s + 1;
    const int nj = idxer.jb.e - idxer.jb.s + 1;
    idxer.inner_length = ni * nj;
    return idxer;
  }

  KOKKOS_INLINE_FUNCTION
  auto GetStartIndices(int outer_idx) const { return idxer(outer_idx * inner_length); }

  KOKKOS_INLINE_FUNCTION
  int GetNinnerRaw(int outer_idx) const {
    auto [ks, js, is] = idxer(outer_idx * inner_length);
    auto [ke, je, ie] = idxer(
        std::min((outer_idx + 1) * inner_length - 1, static_cast<int>(idxer.size()) - 1));
    return idxer_entire.GetFlatIdx(ke, je, ie) - idxer_entire.GetFlatIdx(ks, js, is) + 1;
  }

  KOKKOS_INLINE_FUNCTION
  int GetNouter() const {
    return idxer.size() / inner_length + (idxer.size() % inner_length > 0);
  }

  int GetMaxNinnerRaw() const {
    int max_ninner_raw{0};
    for (int i = 0; i < GetNouter(); ++i) {
      max_ninner_raw = std::max(max_ninner_raw, GetNinnerRaw(i));
    }
    return max_ninner_raw;
  }

  KOKKOS_INLINE_FUNCTION
  int GetStartingRawFlatIdx(int outer_idx) const {
    auto [ks, js, is] = idxer(outer_idx * inner_length);
    return idxer_entire.GetFlatIdx(ks, js, is);
  }

  KOKKOS_INLINE_FUNCTION
  auto GetCurrentIndices(int starting_raw_flat_idx, int inner_idx) const {
    return idxer_entire(starting_raw_flat_idx + inner_idx);
  }

  int inner_length;
  parthenon::IndexRange ib, jb, kb;
  parthenon::Indexer3D idxer_entire;
  parthenon::Indexer3D idxer;
};

#endif // EXAMPLE_DIFFUSION_RAW_MEMORY_INDEXER_HPP_
