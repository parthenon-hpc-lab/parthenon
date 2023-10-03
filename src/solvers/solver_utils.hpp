//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef SOLVERS_SOLVER_UTILS_HPP_
#define SOLVERS_SOLVER_UTILS_HPP_

#include <string>
#include <vector>

#include "kokkos_abstraction.hpp"

namespace parthenon {

namespace solvers {

struct SparseMatrixAccessor {
  ParArray1D<int> ioff, joff, koff;
  const int nstencil;
  int ndiag;

  SparseMatrixAccessor(const std::string &label, const int n,
                       std::vector<std::vector<int>> off)
      : ioff(label + "_ioff", n), joff(label + "_joff", n), koff(label + "_koff", n),
        nstencil(n) {
    PARTHENON_REQUIRE_THROWS(off.size() == 3,
                             "Offset array must have dimensions off[3][*]");
    PARTHENON_REQUIRE_THROWS(off[0].size() >= n, "Offset array off[0][*] too small");
    PARTHENON_REQUIRE_THROWS(off[1].size() >= n, "Offset array off[1][*] too small");
    PARTHENON_REQUIRE_THROWS(off[2].size() >= n, "Offset array off[2][*] too small");
    auto ioff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ioff);
    auto joff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), joff);
    auto koff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), koff);

    for (int i = 0; i < n; i++) {
      ioff_h(i) = off[0][i];
      joff_h(i) = off[1][i];
      koff_h(i) = off[2][i];
      if (off[0][i] == 0 && off[1][i] == 0 && off[2][i] == 0) {
        ndiag = i;
      }
    }

    Kokkos::deep_copy(ioff, ioff_h);
    Kokkos::deep_copy(joff, joff_h);
    Kokkos::deep_copy(koff, koff_h);
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real MatVec(const PackType &spmat, const int imat_lo,
                                     const int imat_hi, const PackType &v, const int iv,
                                     const int b, const int k, const int j,
                                     const int i) const {
    Real matvec = 0.0;
    for (int n = imat_lo; n <= imat_hi; n++) {
      const int m = n - imat_lo;
      matvec += spmat(b, n, k, j, i) * v(b, iv, k + koff(m), j + joff(m), i + ioff(m));
    }
    return matvec;
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real Jacobi(const PackType &spmat, const int imat_lo,
                                     const int imat_hi, const PackType &v, const int iv,
                                     const int b, const int k, const int j, const int i,
                                     const Real rhs) const {
    const Real matvec = MatVec(spmat, imat_lo, imat_hi, v, iv, b, k, j, i);
    return (rhs - matvec + spmat(b, imat_lo + ndiag, k, j, i) * v(b, iv, k, j, i)) /
           spmat(b, imat_lo + ndiag, k, j, i);
  }
};

template <typename T>
struct Stencil {
  ParArray1D<T> w;
  ParArray1D<int> ioff, joff, koff;
  const int nstencil;
  int ndiag;

  Stencil(const std::string &label, const int n, std::vector<T> wgt,
          std::vector<std::vector<int>> off)
      : w(label + "_w", n), ioff(label + "_ioff", n), joff(label + "_joff", n),
        koff(label + "_koff", n), nstencil(n) {
    PARTHENON_REQUIRE_THROWS(off.size() == 3,
                             "Offset array must have dimensions off[3][*]");
    PARTHENON_REQUIRE_THROWS(wgt.size() >= n, "Weight array too small");
    PARTHENON_REQUIRE_THROWS(off[0].size() >= n, "Offset array off[0][*] too small");
    PARTHENON_REQUIRE_THROWS(off[1].size() >= n, "Offset array off[1][*] too small");
    PARTHENON_REQUIRE_THROWS(off[2].size() >= n, "Offset array off[2][*] too small");
    auto w_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), w);
    auto ioff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), ioff);
    auto joff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), joff);
    auto koff_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), koff);

    for (int i = 0; i < n; i++) {
      w_h(i) = wgt[i];
      ioff_h(i) = off[0][i];
      joff_h(i) = off[1][i];
      koff_h(i) = off[2][i];
      if (off[0][i] == 0 && off[1][i] == 0 && off[2][i] == 0) {
        ndiag = i;
      }
    }

    Kokkos::deep_copy(w, w_h);
    Kokkos::deep_copy(ioff, ioff_h);
    Kokkos::deep_copy(joff, joff_h);
    Kokkos::deep_copy(koff, koff_h);
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real MatVec(const PackType &v, const int iv, const int b,
                                     const int k, const int j, const int i) const {
    Real matvec = 0.0;
    for (int n = 0; n < nstencil; n++) {
      matvec += w(n) * v(b, iv, k + koff(n), j + joff(n), i + ioff(n));
    }
    return matvec;
  }

  template <typename PackType>
  KOKKOS_INLINE_FUNCTION Real Jacobi(const PackType &v, const int iv, const int b,
                                     const int k, const int j, const int i,
                                     const Real rhs) const {
    const Real matvec = MatVec(v, iv, b, k, j, i);
    return (rhs - matvec + w(ndiag) * v(b, iv, k, j, i)) / w(ndiag);
  }
};

namespace impl { 
template <class in, class out, bool only_md_level = false>
TaskStatus CopyData(std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, te);

  int nblocks = md->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  if (only_md_level) {
    for (int b = 0; b < nblocks; ++b)
      include_block[b] =
          (md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level());
  }

  auto desc = parthenon::MakePackDescriptor<in, out>(md.get());
  auto pack = desc.GetPack(md.get(), include_block);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, out(), k, j, i) = pack(b, te, in(), k, j, i);
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t, class out, bool only_md_level = false>
TaskStatus AddFieldsAndStoreInteriorSelect(std::shared_ptr<MeshData<Real>> &md,
                                           Real wa = 1.0, Real wb = 1.0,
                                           bool only_interior = false) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire, te);

  int nblocks = md->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  if (only_interior) {
    // The neighbors array will only be set for a block if its a leaf block
    for (int b = 0; b < nblocks; ++b)
      include_block[b] = md->GetBlockData(b)->GetBlockPointer()->neighbors.size() == 0;
  }

  if (only_md_level) {
    for (int b = 0; b < nblocks; ++b)
      include_block[b] =
          include_block[b] &&
          (md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level());
  }

  auto desc = parthenon::MakePackDescriptor<a_t, b_t, out>(md.get());
  auto pack = desc.GetPack(md.get(), include_block);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, out(), k, j, i) =
            wa * pack(b, te, a_t(), k, j, i) + wb * pack(b, te, b_t(), k, j, i);
      });
  return TaskStatus::complete;
}

template <class var, bool only_md_level = false>
TaskStatus SetToZero(std::shared_ptr<MeshData<Real>> &md) {
  int nblocks = md->NumBlocks();
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  std::vector<bool> include_block(nblocks, true);
  if (only_md_level) {
    for (int b = 0; b < nblocks; ++b)
      include_block[b] =
          (md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level());
  }
  auto desc = parthenon::MakePackDescriptor<var>(md.get());
  auto pack = desc.GetPack(md.get(), include_block);
  const size_t scratch_size_in_bytes = 0;
  const int scratch_level = 1;
  const int ng = parthenon::Globals::nghost;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Print", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        auto cb = GetIndexShape(pack(b, te, 0), ng);
        const auto &coords = pack.GetCoordinates(b);
        IndexRange ib = cb.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = cb.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = cb.GetBoundsK(IndexDomain::interior, te);
        parthenon::par_for_inner(
            parthenon::inner_loop_pattern_simdfor_tag, member, kb.s, kb.e, jb.s, jb.e,
            ib.s, ib.e, [&](int k, int j, int i) { pack(b, te, var(), k, j, i) = 0.0; });
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t>
TaskStatus DotProductLocal(std::shared_ptr<MeshData<Real>> &md, Real *reduce_sum) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<a_t, b_t>(md.get());
  auto pack = desc.GetPack(md.get());
  Real gsum(0);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "DotProduct", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        lsum += pack(b, te, a_t(), k, j, i) * pack(b, te, b_t(), k, j, i);
      },
      Kokkos::Sum<Real>(gsum));
  *reduce_sum += gsum;
  return TaskStatus::complete;
}
} // namespace impl

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_UTILS_HPP_
