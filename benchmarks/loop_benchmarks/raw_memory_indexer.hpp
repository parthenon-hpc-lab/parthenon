// Derived from Parthenon's diffusion example RawMemoryIndexer on
// lroberts36/update-diffusion-example, adapted to remove MeshData dependence.
#pragma once

#include <algorithm>

#include <Kokkos_Core.hpp>

#include "utils/indexer.hpp"
namespace plb {

struct RawMemoryIndexer {
  RawMemoryIndexer() = default;

  RawMemoryIndexer(int inner_length_in, int ndim_in, parthenon::IndexRange domain_k_in,
                   parthenon::IndexRange domain_j_in, parthenon::IndexRange domain_i_in,
                   parthenon::IndexRange memory_k_in, parthenon::IndexRange memory_j_in,
                   parthenon::IndexRange memory_i_in, int halo = 0)
      : inner_length(inner_length_in),
        ndim(ndim_in),
        ib(domain_i_in),
        jb(domain_j_in),
        kb(domain_k_in),
        idxer_entire(memory_k_in, memory_j_in, memory_i_in) {
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
  }

  static RawMemoryIndexer IJ(int ndim, parthenon::IndexRange domain_k,
                             parthenon::IndexRange domain_j,
                             parthenon::IndexRange domain_i,
                             parthenon::IndexRange memory_k,
                             parthenon::IndexRange memory_j,
                             parthenon::IndexRange memory_i, int halo = 0) {
    RawMemoryIndexer idxer(0, ndim, domain_k, domain_j, domain_i, memory_k, memory_j,
                           memory_i, halo);
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
    int max_ninner_raw = 0;
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
  int ndim;
  parthenon::IndexRange ib, jb, kb;
  parthenon::Indexer3D idxer_entire;
  parthenon::Indexer3D idxer;
};

}  // namespace plb
