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
 public:
  static constexpr int all_outer = -100;
  static constexpr int no_outer = -200;
  IndexSplit(MeshData<Real> *md, const IndexRange &kb, const IndexRange &jb,
             const IndexRange &ib, const int k_tiles, const int j_tiles);
  IndexSplit(MeshData<Real> *md, IndexDomain domain, const int k_tiles, const int j_tiles);
  
  // Get the total number of kj-tiles
  int outer_size() const { return k_tiles_ * j_tiles_; }
  
  // Get the k-bounds of kj-tile indexed by p
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsK(const int p) const {
    const auto kf = p / j_tiles_;
    const int start = kbs_ + (kf * logical_nk_) / k_tiles_; 
    const int stop = kbs_ + ((kf + 1) * logical_nk_) / k_tiles_ - 1; 
    return {start, stop};
  }

  // Get the j-bounds of kj-tile indexed by p
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsJ(const int p) const {
    const auto jf = p % j_tiles_;
    const int start = jbs_ + (jf * logical_nj_) / j_tiles_; 
    const int stop = jbs_ + ((jf + 1) * logical_nj_) / j_tiles_ - 1; 
    return {start, stop};
  }

  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsI() const { return {ibs_, ibe_}; }
  
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsI(const int p) const { return GetBoundsI(); }

  KOKKOS_INLINE_FUNCTION
  auto GetBoundsKJI(const int p) const {
    const auto kb = GetBoundsK(p);
    const auto jb = GetBoundsJ(p);
    const auto ib = GetBoundsI(p);
    return std::make_tuple(kb, jb, ib);
  }
  KOKKOS_INLINE_FUNCTION
  IndexRange GetInnerBounds(const IndexRange &jb) const {
    return {ibs_, ibs_ + inner_size({kbs_, kbs_}, jb, {ibs_, ibe_})};
  }
  KOKKOS_INLINE_FUNCTION
  IndexRange GetInnerBounds(const IndexRange &jb, const IndexRange &ib) const {
    return {ib.s, ib.s + inner_size({kbs_, kbs_}, jb, ib)};
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int get_i(const int idx) const { return idx % memory_ni_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int get_deltaj(const int idx) const { return idx / memory_ni_; }

  KOKKOS_INLINE_FUNCTION
  bool is_i_ghost(const int idx) const {
    const int ni = memory_ni_;
    const int i = idx % ni;
    const int i_inner_size = ni - 2 * nghost_;
    return (i < nghost_ || i - nghost_ >= i_inner_size);
  }

  KOKKOS_INLINE_FUNCTION
  bool is_j_ghost(const int outer_idx, const int idx) const {
    const int ni = memory_ni_;
    const int j = GetBoundsJ(outer_idx).s + idx / ni;
    const int j_inner_size = memory_nj_ - 2 * nghost_;
    return (ndim_ > 1 && (j < nghost_ || j - nghost_ >= j_inner_size));
  }

  KOKKOS_INLINE_FUNCTION
  bool is_k_ghost(const int k) const {
    const int k_inner_size = memory_nk_ - 2 * nghost_;
    return (ndim_ > 2 && (k < nghost_ || k - nghost_ >= k_inner_size));
  }
  KOKKOS_INLINE_FUNCTION
  bool is_ghost(const int outer_idx, const int k, const int idx) const {
    return is_k_ghost(k) || is_j_ghost(outer_idx, idx) || is_i_ghost(idx);
  }
  KOKKOS_INLINE_FUNCTION
  int get_max_ni() const { return memory_ni_; }
  // TODO(@jdolence) these overestimate max size...should probably fix
  KOKKOS_INLINE_FUNCTION
  int get_max_nj() const { return memory_nj_ / j_tiles_ + 1; }
  KOKKOS_INLINE_FUNCTION
  int get_max_nk() const { return memory_nk_ / k_tiles_ + 1; }
  KOKKOS_INLINE_FUNCTION
  int get_max_nij() const { return get_max_ni() * get_max_nj(); }

  KOKKOS_INLINE_FUNCTION int inner_size(const IndexRange &kb,
                                        const IndexRange &jb,
                                        const IndexRange &ib) const {
    return memory_idxer_.GetFlatIdx(kb.e, jb.e, ib.e)
         - memory_idxer_.GetFlatIdx(kb.s, jb.s, ib.s);
  }

 private:
  // TODO(JMM): Replace this with a macro or something when available
  static constexpr int NSTREAMS_ = 1; // Change if we add streams back
  int concurrency_;                   //  = NSMs = 132 for NVIDIA H100
  int nghost_, k_tiles_, j_tiles_, kbs_, jbs_, ibs_, ibe_;
  int ndim_;

  int logical_nk_, logical_nj_, logical_ni_;
  int memory_nk_, memory_nj_, memory_ni_;
  
  Indexer3D logical_idxer_, memory_idxer_;

  void Init(MeshData<Real> *md, const int kbe, const int jbe);
};

} // namespace parthenon

#endif // UTILS_INDEX_SPLIT_HPP_
