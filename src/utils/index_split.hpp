//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_INDEX_SPLIT_HPP_
#define UTILS_INDEX_SPLIT_HPP_

#include <algorithm>
#include <tuple>

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/domain.hpp"

namespace parthenon {

// forward declarations
template <typename T>
class MeshData;

class IndexSplit {
  using TE = TopologicalElement;

 public:
  static constexpr int all_outer = -100;
  static constexpr int no_outer = -200;
  IndexSplit(MeshData<Real> *md, const IndexRange &kb, const IndexRange &jb,
             const IndexRange &ib, const int nk_tiles, const int nj_tiles,
             TE te_mem = TE::CC);
  IndexSplit(MeshData<Real> *md, IndexDomain domain, const int nk_tiles,
             const int nj_tiles, TE te = TE::CC, TE te_mem = TE::CC);

  // Provides the same functionality as the raw memory indexer did when initialized with
  // the IJ factory
  static IndexSplit RawMemIJ(IndexDomain domain, int halo, MeshData<Real> *md,
                             TE logical_te, TE memory_te = TE::CC);
  static IndexSplit RawMemIJ(IndexDomain domain, MeshData<Real> *md, TE logical_te,
                             TE memory_te = TE::CC) {
    return RawMemIJ(domain, 0, md, logical_te, memory_te);
  }

  // Get the total number of kj-tiles
  int outer_size() const { return nk_tiles_ * nj_tiles_; }

  // Temporary backward compatibility with RawMemoryIndexer
  KOKKOS_INLINE_FUNCTION
  auto GetStartIndices(int outer_idx) const {
    PARTHENON_REQUIRE(nj_tiles_ == 1 && nk_tiles_ == logical_.Extent<KDIM>(),
                      "Only works for this case.");
    auto kb = GetBoundsK(outer_idx);
    auto jb = GetBoundsJ(outer_idx);
    return std::tuple<int, int, int>{kb.s, jb.s, logical_.StartIdx<IDIM>()};
  }

  KOKKOS_INLINE_FUNCTION
  int GetNinnerRaw(int outer_idx) const {
    PARTHENON_REQUIRE(nj_tiles_ == 1 && nk_tiles_ == logical_.Extent<KDIM>(),
                      "Only works for this case.");
    auto [ks, js, is] = GetStartIndices(outer_idx);
    int ke = ks;                      // Enforce fixed k by hand
    int je = logical_.EndIdx<JDIM>(); // Enforce end of j-range by hand
    int ie = logical_.EndIdx<IDIM>(); // Enforce end of i-range by hand
    return inner_size({ks, ke}, {js, je}, {is, ie});
  }

  KOKKOS_INLINE_FUNCTION
  int GetNouter() const { return outer_size(); }

  int GetMaxNinnerRaw() const {
    int max_ninner{0};
    for (int p = 0; p < outer_size(); ++p)
      max_ninner = std::max(max_ninner, GetNinnerRaw(p));
    return max_ninner;
  }

  KOKKOS_INLINE_FUNCTION
  int GetStartingRawFlatIdx(int outer_idx) const {
    auto [ks, js, is] = GetStartIndices(outer_idx);
    return memory_.GetFlatIdx(ks, js, is);
  }

  KOKKOS_INLINE_FUNCTION
  auto GetCurrentIndices(int starting_raw_flat_idx, int inner_idx) const {
    return memory_(starting_raw_flat_idx + inner_idx);
  }

  // Get the k-bounds of kj-tile indexed by p
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsK(const int p) const {
    const auto k_tile = p / nj_tiles_;
    const int ks = logical_.StartIdx<KDIM>();
    const int nk = logical_.Extent<KDIM>();
    const int start = ks + (k_tile * nk) / nk_tiles_;
    const int stop = ks + ((k_tile + 1) * nk) / nk_tiles_ - 1;
    return {start, stop};
  }

  // Get the j-bounds of kj-tile indexed by p
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsJ(const int p) const {
    const auto j_tile = p % nj_tiles_;
    const int js = logical_.StartIdx<JDIM>();
    const int nj = logical_.Extent<JDIM>();
    const int start = js + (j_tile * nj) / nj_tiles_;
    const int stop = js + ((j_tile + 1) * nj) / nj_tiles_ - 1;
    return {start, stop};
  }

  template <class F>
  KOKKOS_INLINE_FUNCTION void middle_for(int p, F &&f) const {
    // TODO(LFR): This could be generalized to allow for switching to including part of
    // k-space in the flattening.
    const auto [kb, jb, ib] = GetBoundsKJI(p);
    for (int k = kb.s; k <= kb.e; ++k)
      f(k, jb.s, ib.s, inner_size(kb, jb, ib));
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsI() const {
    return {logical_.StartIdx<IDIM>(), logical_.EndIdx<IDIM>()};
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsI(const int p) const { return GetBoundsI(); }

  KOKKOS_INLINE_FUNCTION
  auto GetBoundsKJI(const int p) const {
    const auto kb = GetBoundsK(p);
    const auto jb = GetBoundsJ(p);
    const auto ib = GetBoundsI(p);
    return std::make_tuple(kb, jb, ib);
  }

  KOKKOS_INLINE_FUNCTION int inner_size(const IndexRange &kb, const IndexRange &jb,
                                        const IndexRange &ib) const {
    return memory_.GetFlatIdx(kb.e, jb.e, ib.e) - memory_.GetFlatIdx(kb.s, jb.s, ib.s) +
           1;
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetInnerBounds(const IndexRange &jb) const {
    const int ibs = logical_.StartIdx<IDIM>();
    const int ibe = logical_.EndIdx<IDIM>();
    const int kbs = logical_.StartIdx<KDIM>();
    return {ibs, ibs + inner_size({kbs, kbs}, jb, {ibs, ibe}) - 1};
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetInnerBounds(const IndexRange &jb, const IndexRange &ib) const {
    const int kbs = logical_.StartIdx<KDIM>();
    return {ib.s, ib.s + inner_size({kbs, kbs}, jb, ib) - 1};
  }

  KOKKOS_INLINE_FUNCTION
  int GetMemoryIdx(int ks, int js, int is) const {
    return memory_.GetFlatIdx(ks, js, is);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int get_i(const int idx) const { return idx % memory_.Extent<IDIM>(); }

  KOKKOS_FORCEINLINE_FUNCTION
  int get_deltaj(const int idx) const { return idx / memory_.Extent<IDIM>(); }

  KOKKOS_INLINE_FUNCTION
  bool is_i_ghost(const int idx) const {
    const int i = get_i(idx);
    return !logical_.IdxInRange<IDIM>(i);
  }

  KOKKOS_INLINE_FUNCTION
  bool is_j_ghost(const int outer_idx, const int idx) const {
    const int j = GetBoundsJ(outer_idx).s + idx / memory_.Extent<IDIM>();
    return !logical_.IdxInRange<JDIM>(j);
  }

  KOKKOS_INLINE_FUNCTION
  bool is_k_ghost(const int k) const { return !logical_.IdxInRange<KDIM>(k); }

  KOKKOS_INLINE_FUNCTION
  bool is_ghost(const int outer_idx, const int k, const int idx) const {
    return is_k_ghost(k) || is_j_ghost(outer_idx, idx) || is_i_ghost(idx);
  }
  KOKKOS_INLINE_FUNCTION
  int get_max_ni() const { return memory_.Extent<IDIM>(); }
  // TODO(@jdolence) these overestimate max size...should probably fix
  KOKKOS_INLINE_FUNCTION
  int get_max_nj() const { return memory_.Extent<JDIM>() / nj_tiles_ + 1; }
  KOKKOS_INLINE_FUNCTION
  int get_max_nk() const { return memory_.Extent<KDIM>() / nk_tiles_ + 1; }
  KOKKOS_INLINE_FUNCTION
  int get_max_nij() const { return get_max_ni() * get_max_nj(); }

 private:
  // TODO(JMM): Replace this with a macro or something when available
  static constexpr int NSTREAMS_ = 1; // Change if we add streams back

  int nk_tiles_, nj_tiles_;
  Indexer3D logical_, memory_;

  static constexpr std::size_t IDIM{2};
  static constexpr std::size_t JDIM{1};
  static constexpr std::size_t KDIM{0};
};

} // namespace parthenon

#endif // UTILS_INDEX_SPLIT_HPP_
