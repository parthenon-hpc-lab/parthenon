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
             const IndexRange &ib, const int k_tiles, const int j_tiles,
             TE te_mem = TE::CC);
  IndexSplit(MeshData<Real> *md, IndexDomain domain, const int k_tiles, const int j_tiles,
             TE te = TE::CC, TE te_mem = TE::CC);
  
  // Provides the same functionality as the raw memory indexer did when initialized with the IJ factory
  static IndexSplit RawMemIJ(IndexDomain domain, int halo, MeshData<Real> *md, TE logical_te, TE memory_te = TE::CC);
  static IndexSplit RawMemIJ(IndexDomain domain, MeshData<Real> *md, TE logical_te, TE memory_te = TE::CC) {
    return RawMemIJ(domain, 0, md, logical_te, memory_te);
  } 

  // Get the total number of kj-tiles
  int outer_size() const { return k_tiles_ * j_tiles_; }
  
  // Get the k-bounds of kj-tile indexed by p
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsK(const int p) const {
    const auto k_tile = p / j_tiles_;
    const int ks = logical_.StartIdx<KDIM>();
    const int nk = logical_.Extent<KDIM>();
    const int start = ks + (k_tile * nk) / k_tiles_; 
    const int stop = ks + ((k_tile + 1) * nk) / k_tiles_ - 1; 
    return {start, stop};
  }

  // Get the j-bounds of kj-tile indexed by p
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsJ(const int p) const {
    const auto j_tile = p % j_tiles_;
    const int js = logical_.StartIdx<JDIM>();
    const int nj = logical_.Extent<JDIM>();
    const int start = js + (j_tile * nj) / j_tiles_; 
    const int stop = js + ((j_tile + 1) * nj) / j_tiles_ - 1; 
    return {start, stop};
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsI() const { return {logical_.StartIdx<IDIM>(), logical_.EndIdx<IDIM>()}; }
  
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsI(const int p) const { return GetBoundsI(); }

  KOKKOS_INLINE_FUNCTION
  auto GetBoundsKJI(const int p) const {
    const auto kb = GetBoundsK(p);
    const auto jb = GetBoundsJ(p);
    const auto ib = GetBoundsI(p);
    return std::make_tuple(kb, jb, ib);
  }
  
  KOKKOS_INLINE_FUNCTION int inner_size(const IndexRange &kb,
                                        const IndexRange &jb,
                                        const IndexRange &ib) const {
    return memory_.GetFlatIdx(kb.e, jb.e, ib.e)
         - memory_.GetFlatIdx(kb.s, jb.s, ib.s);
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetInnerBounds(const IndexRange &jb) const {
    const int ibs = logical_.StartIdx<IDIM>();
    const int ibe = logical_.EndIdx<IDIM>();
    const int kbs = logical_.StartIdx<KDIM>();
    return {ibs, ibs + inner_size({kbs, kbs}, jb, {ibs, ibe})};
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetInnerBounds(const IndexRange &jb, const IndexRange &ib) const {
    const int kbs = logical_.StartIdx<KDIM>();
    return {ib.s, ib.s + inner_size({kbs, kbs}, jb, ib)};
  }
  
  KOKKOS_INLINE_FUNCTION
  int GetMemoryIdx(int ks, int js, int is) const {
    return memory_.GetFlatIdx(ks, js, is);
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  auto GetCurrentIndices(int mem_idx_start, int inner_idx) const {
    return memory_(mem_idx_start + inner_idx); 
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
  bool is_k_ghost(const int k) const {
    return !logical_.IdxInRange<KDIM>(k);
  }

  KOKKOS_INLINE_FUNCTION
  bool is_ghost(const int outer_idx, const int k, const int idx) const {
    return is_k_ghost(k) || is_j_ghost(outer_idx, idx) || is_i_ghost(idx);
  }
  KOKKOS_INLINE_FUNCTION
  int get_max_ni() const { return memory_.Extent<IDIM>(); }
  // TODO(@jdolence) these overestimate max size...should probably fix
  KOKKOS_INLINE_FUNCTION
  int get_max_nj() const { return memory_.Extent<JDIM>() / j_tiles_ + 1; }
  KOKKOS_INLINE_FUNCTION
  int get_max_nk() const { return memory_.Extent<KDIM>() / k_tiles_ + 1; }
  KOKKOS_INLINE_FUNCTION
  int get_max_nij() const { return get_max_ni() * get_max_nj(); }

 private:
  // TODO(JMM): Replace this with a macro or something when available
  static constexpr int NSTREAMS_ = 1; // Change if we add streams back
  
  int k_tiles_, j_tiles_;
  Indexer3D logical_, memory_;

  static constexpr std::size_t IDIM{2};
  static constexpr std::size_t JDIM{1};
  static constexpr std::size_t KDIM{0};

};

} // namespace parthenon

#endif // UTILS_INDEX_SPLIT_HPP_
