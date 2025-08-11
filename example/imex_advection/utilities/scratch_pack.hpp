//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights
// reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================
#ifndef SRC_UTILITIES_SCRATCH_PACK_HPP_
#define SRC_UTILITIES_SCRATCH_PACK_HPP_

#include <parthenon/parthenon.hpp>

namespace parthenon {
namespace utils {

template <class pack_t>
struct ScratchPack {
  using scratch_pad_t = ScratchPad3D<Real>;

  template <class scratch_space_t>
  KOKKOS_INLINE_FUNCTION ScratchPack(scratch_space_t &&scratch_space, const pack_t *ppack, int b, int nj, int ni)
      : ppack{ppack}, block{b},
        data(std::forward<scratch_space_t>(scratch_space), ppack->GetUpperBound(b) + 1, nj, ni) {}

  template <class Tin>
  KOKKOS_FORCEINLINE_FUNCTION Real &operator()(const Tin &t, int j, int i) {
    const int vidx = ppack->GetLowerBound(block, t) + t.idx;
    return data(vidx, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real &operator()(int v, int j, int i) { return data(v, j, i); }

  static std::size_t get_size_in_bytes(const pack_t &pack, int nj, int ni) {
    const int nvar = pack.GetUpperBoundHost(0) + 1; // Assuming no sparse fields for the time being
    return scratch_pad_t::shmem_size(nvar, nj, ni);
  }

  const pack_t *ppack;
  int block;
  scratch_pad_t data;
};

template <class pack_t>
KOKKOS_INLINE_FUNCTION void swap(ScratchPack<pack_t> &a, ScratchPack<pack_t> &b) {
  auto *tmp = a.data.data();
  a.data.assign_data(b.data.data());
  b.data.assign_data(tmp);

  int t_block = a.block;
  a.block = b.block;
  b.block = t_block;

  auto *t_ppack = a.ppack;
  a.ppack = b.ppack;
  b.ppack = t_ppack;
}

} // namespace utils
} // namespace parthenon

#endif // SRC_UTILITIES_SCRATCH_PACK_HPP_
