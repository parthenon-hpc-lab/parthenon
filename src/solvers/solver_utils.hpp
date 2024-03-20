//========================================================================================
// (C) (or copyright) 2021-2024. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kokkos_abstraction.hpp"

#define PARTHENON_INTERNALSOLVERVARIABLE(base, varname)                                  \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return base::name() + "." #varname; }                    \
  }

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

namespace utils {
template <class in_t, class out_t, bool only_fine_on_composite = true>
TaskStatus CopyData(const std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md->GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire, te);

  static auto desc = parthenon::MakePackDescriptor<in_t, out_t>(md.get());
  auto pack = desc.GetPack(md.get(), only_fine_on_composite);
  const int scratch_size = 0;
  const int scratch_level = 0;
  // Warning: This inner loop strategy only works because we are using IndexDomain::entire
  const int npoints_inner = (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "CopyData", DevExecSpace(), scratch_size, scratch_level,
      0, pack.GetNBlocks() - 1, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const int nvars =
            pack.GetUpperBound(b, in_t()) - pack.GetLowerBound(b, in_t()) + 1;
        for (int c = 0; c < nvars; ++c) {
          Real *in = &pack(b, te, in_t(c), kb.s, jb.s, ib.s);
          Real *out = &pack(b, te, out_t(c), kb.s, jb.s, ib.s);
          parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                                   npoints_inner - 1,
                                   [&](const int idx) { out[idx] = in[idx]; });
        }
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t, class out_t, bool only_fine_on_composite = true>
TaskStatus AddFieldsAndStoreInteriorSelect(const std::shared_ptr<MeshData<Real>> &md,
                                           Real wa = 1.0, Real wb = 1.0,
                                           bool only_interior_blocks = false) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md->GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire, te);

  int nblocks = md->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  if (only_interior_blocks) {
    // The neighbors array will only be set for a block if its a leaf block
    for (int b = 0; b < nblocks; ++b)
      include_block[b] = md->GetBlockData(b)->GetBlockPointer()->neighbors.size() == 0;
  }

  static auto desc = parthenon::MakePackDescriptor<a_t, b_t, out_t>(md.get());
  auto pack = desc.GetPack(md.get(), include_block, only_fine_on_composite);
  const int scratch_size = 0;
  const int scratch_level = 0;
  // Warning: This inner loop strategy only works because we are using IndexDomain::entire
  const int npoints_inner = (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "AddFieldsAndStore", DevExecSpace(), scratch_size,
      scratch_level, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const int nvars = pack.GetUpperBound(b, a_t()) - pack.GetLowerBound(b, a_t()) + 1;
        for (int c = 0; c < nvars; ++c) {
          Real *avar = &pack(b, te, a_t(c), kb.s, jb.s, ib.s);
          Real *bvar = &pack(b, te, b_t(c), kb.s, jb.s, ib.s);
          Real *out = &pack(b, te, out_t(c), kb.s, jb.s, ib.s);
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, 0, npoints_inner - 1,
              [&](const int idx) { out[idx] = wa * avar[idx] + wb * bvar[idx]; });
        }
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t, class out, bool only_fine_on_composite = true>
TaskStatus AddFieldsAndStore(const std::shared_ptr<MeshData<Real>> &md, Real wa = 1.0,
                             Real wb = 1.0) {
  return AddFieldsAndStoreInteriorSelect<a_t, b_t, out, only_fine_on_composite>(
      md, wa, wb, false);
}

template <class var, bool only_fine_on_composite = true>
TaskStatus SetToZero(const std::shared_ptr<MeshData<Real>> &md) {
  int nblocks = md->NumBlocks();
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  static auto desc = parthenon::MakePackDescriptor<var>(md.get());
  auto pack = desc.GetPack(md.get(), only_fine_on_composite);
  const size_t scratch_size_in_bytes = 0;
  const int scratch_level = 1;
  const int ng = parthenon::Globals::nghost;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "SetFieldsToZero", DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        auto cb = GetIndexShape(pack(b, te, 0), ng);
        const auto &coords = pack.GetCoordinates(b);
        IndexRange ib = cb.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = cb.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = cb.GetBoundsK(IndexDomain::interior, te);
        const int nvars = pack.GetUpperBound(b, var()) - pack.GetLowerBound(b, var()) + 1;
        for (int c = 0; c < nvars; ++c) {
          parthenon::par_for_inner(
              parthenon::inner_loop_pattern_simdfor_tag, member, kb.s, kb.e, jb.s, jb.e,
              ib.s, ib.e,
              [&](int k, int j, int i) { pack(b, te, var(c), k, j, i) = 0.0; });
        }
      });
  return TaskStatus::complete;
}

template <class a_t, class b_t>
TaskStatus DotProductLocal(const std::shared_ptr<MeshData<Real>> &md,
                           AllReduce<Real> *adotb) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  static auto desc = parthenon::MakePackDescriptor<a_t, b_t>(md.get());
  auto pack = desc.GetPack(md.get());
  Real gsum(0);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "DotProduct", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const int nvars = pack.GetUpperBound(b, a_t()) - pack.GetLowerBound(b, a_t()) + 1;
        // TODO(LFR): If this becomes a bottleneck, exploit hierarchical parallelism and
        //            pull the loop over vars outside of the innermost loop to promote
        //            vectorization.
        for (int c = 0; c < nvars; ++c)
          lsum += pack(b, te, a_t(c), k, j, i) * pack(b, te, b_t(c), k, j, i);
      },
      Kokkos::Sum<Real>(gsum));
  adotb->val += gsum;
  return TaskStatus::complete;
}

template <class a_t, class b_t>
TaskID DotProduct(TaskID dependency_in, TaskList &tl, AllReduce<Real> *adotb,
                  const std::shared_ptr<MeshData<Real>> &md) {
  using namespace impl;
  auto zero_adotb = tl.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, dependency_in,
      [](AllReduce<Real> *r) {
        r->val = 0.0;
        return TaskStatus::complete;
      },
      adotb);
  auto get_adotb = tl.AddTask(TaskQualifier::local_sync, zero_adotb,
                              DotProductLocal<a_t, b_t>, md, adotb);
  auto start_global_adotb = tl.AddTask(TaskQualifier::once_per_region, get_adotb,
                                       &AllReduce<Real>::StartReduce, adotb, MPI_SUM);
  auto finish_global_adotb =
      tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                 start_global_adotb, &AllReduce<Real>::CheckReduce, adotb);
  return finish_global_adotb;
}

template <class a_t>
TaskStatus GlobalMinLocal(const std::shared_ptr<MeshData<Real>> &md,
                          AllReduce<Real> *amin) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  static auto desc = parthenon::MakePackDescriptor<a_t>(md.get());
  auto pack = desc.GetPack(md.get());
  Real gmin(0);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "DotProduct", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmin) {
        const int nvars = pack.GetUpperBound(b, a_t()) - pack.GetLowerBound(b, a_t()) + 1;
        // TODO(LFR): If this becomes a bottleneck, exploit hierarchical parallelism and
        //            pull the loop over vars outside of the innermost loop to promote
        //            vectorization.
        for (int c = 0; c < nvars; ++c)
          lmin = std::min(lmin, pack(b, te, a_t(c), k, j, i));
      },
      Kokkos::Min<Real>(gmin));
  amin->val = std::min(gmin, amin->val);
  return TaskStatus::complete;
}

template <class a_t>
TaskID GlobalMin(TaskID dependency_in, TaskList &tl, AllReduce<Real> *amin,
                 const std::shared_ptr<MeshData<Real>> &md) {
  using namespace impl;
  auto max_amin = tl.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, dependency_in,
      [](AllReduce<Real> *r) {
        r->val = std::numeric_limits<Real>::max();
        return TaskStatus::complete;
      },
      amin);
  auto get_amin =
      tl.AddTask(TaskQualifier::local_sync, max_amin, GlobalMinLocal<a_t>, md, amin);
  auto start_global_amin = tl.AddTask(TaskQualifier::once_per_region, get_amin,
                                      &AllReduce<Real>::StartReduce, amin, MPI_MIN);
  return tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                    start_global_amin, &AllReduce<Real>::CheckReduce, amin);
}

} // namespace utils

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_UTILS_HPP_
