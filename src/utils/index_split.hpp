//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
             const IndexRange &ib, const int nkp, const int njp);
  IndexSplit(MeshData<Real> *md, IndexDomain domain, const int nkp, const int njp);

  int outer_size() const { return nkp_ * njp_; }
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsK(const int p) const {
    const auto kf = p / njp_;
    return {kbs_ + static_cast<int>(kf * target_k_),
            kbs_ + static_cast<int>((kf + 1) * target_k_) - 1};
  }
  KOKKOS_INLINE_FUNCTION
  IndexRange GetBoundsJ(const int p) const {
    const auto jf = p % njp_;
    return {jbs_ + static_cast<int>(jf * target_j_),
            jbs_ + static_cast<int>((jf + 1) * target_j_) - 1};
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
    return {ibs_, (ibe_entire_ + 1) * (jb.e - jb.s + 1) - (ibe_entire_ - ibe_) - 1};
  }
  KOKKOS_INLINE_FUNCTION
  IndexRange GetInnerBounds(const IndexRange &jb, const IndexRange &ib) const {
    return {ib.s, (ibe_entire_ + 1) * (jb.e - jb.s + 1) - (ibe_entire_ - ib.e) - 1};
  }
  KOKKOS_INLINE_FUNCTION
  bool is_i_ghost(const int idx) const {
    const int ni = ibe_entire_ + 1;
    const int i = idx % ni;
    const int i_inner_size = ni - 2 * nghost_;
    return (i < nghost_ || i - nghost_ >= i_inner_size);
  }
  KOKKOS_INLINE_FUNCTION
  bool is_j_ghost(const int outer_idx, const int idx) const {
    const int ni = ibe_entire_ + 1;
    const int j = GetBoundsJ(outer_idx).s + idx / ni;
    const int j_inner_size = jbe_entire_ + 1 - 2 * nghost_;
    return (ndim_ > 1 && (j < nghost_ || j - nghost_ >= j_inner_size));
  }
  KOKKOS_INLINE_FUNCTION
  bool is_k_ghost(const int k) const {
    const int k_inner_size = kbe_entire_ + 1 - 2 * nghost_;
    return (ndim_ > 2 && (k < nghost_ || k - nghost_ >= k_inner_size));
  }
  KOKKOS_INLINE_FUNCTION
  bool is_ghost(const int outer_idx, const int k, const int idx) const {
    return is_k_ghost(k) || is_j_ghost(outer_idx, idx) || is_i_ghost(idx);
  }
  KOKKOS_INLINE_FUNCTION
  int get_max_ni() const { return ibe_entire_ + 1; }
  // TODO(@jdolence) these overestimate max size...should probably fix
  int get_max_nj() const { return (jbe_entire_ + 1) / njp_ + 1; }
  int get_max_nk() const { return (kbe_entire_ + 1) / nkp_ + 1; }
  // inner_size could be used to find the bounds for a loop that is collapsed over
  // 1, 2, or 3 dimensions by providing the right starting and stopping indices
  template <typename V>
  KOKKOS_INLINE_FUNCTION int inner_size(const V &v, const IndexRange &kb,
                                        const IndexRange &jb,
                                        const IndexRange &ib) const {
    return &v(0, kb.e, jb.e, ib.e) - &v(0, kb.s, jb.s, ib.s);
  }

 private:
  int nghost_, nkp_, njp_, kbs_, jbs_, ibs_, ibe_;
  int kbe_entire_, jbe_entire_, ibe_entire_, ndim_;
  float target_k_, target_j_;

  void Init(MeshData<Real> *md, const int kbe, const int jbe);
};

} // namespace parthenon

#endif // UTILS_INDEX_SPLIT_HPP_
