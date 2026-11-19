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
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"
#include "utils/reductions.hpp"
#include "utils/summable_array.hpp"

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

// The vector operations below support fields of any single topological type. The type
// is determined on the host and the kernels are selected at compile time, so that
// cell-centered fields use the same kernels as before. Face, edge and node fields launch
// one kernel per topological element and only count elements owned by a block in
// reductions, so that shared elements on block boundaries are counted once.
template <class desc_t>
TopologicalType FieldsTopologicalType(MeshData<Real> *md, const desc_t &desc) {
  const auto &pkgs = md->GetMeshPointer()->resolved_packages;
  bool first = true;
  TopologicalType tt = TopologicalType::Cell;
  for (const auto &group : desc.var_groups) {
    for (const auto &[vid, uid] : group) {
      const TopologicalType tv = GetTopologicalType(pkgs->GetFieldMetadata(uid));
      PARTHENON_REQUIRE(first || tv == tt,
                        "Solver fields must all have the same topological type.");
      tt = tv;
      first = false;
    }
  }
  return tt;
}

// Call f with std::integral_constant<TopologicalElement, el> for each element el of
// topological type tt
template <class F>
void ForEachElement(TopologicalType tt, const F &f) {
  using TE = TopologicalElement;
  if (tt == TopologicalType::Cell) {
    f(std::integral_constant<TE, TE::CC>());
  } else if (tt == TopologicalType::Face) {
    f(std::integral_constant<TE, TE::F1>());
    f(std::integral_constant<TE, TE::F2>());
    f(std::integral_constant<TE, TE::F3>());
  } else if (tt == TopologicalType::Edge) {
    f(std::integral_constant<TE, TE::E1>());
    f(std::integral_constant<TE, TE::E2>());
    f(std::integral_constant<TE, TE::E3>());
  } else {
    f(std::integral_constant<TE, TE::NN>());
  }
}

// Element whose entire index range covers the full allocation of fields with elements
// te, which is what the flat pointer loops below assume
constexpr TopologicalElement FlatBoundsElement(TopologicalElement te) {
  return te == TopologicalElement::CC ? TopologicalElement::CC : TopologicalElement::NN;
}

// Whether element (k, j, i) of type el, with interior bounds kb, jb, ib, is owned by
// block b of the pack. Only valid on the leaf grid.
template <TopologicalElement el, class pack_t>
KOKKOS_FORCEINLINE_FUNCTION bool
IsOwnedElement(const pack_t &pack, const int b, const IndexRange &kb,
               const IndexRange &jb, const IndexRange &ib, const int k, const int j,
               const int i) {
  constexpr bool offi = TopologicalOffsetI(el);
  constexpr bool offj = TopologicalOffsetJ(el);
  constexpr bool offk = TopologicalOffsetK(el);
  const int oi = offi ? (i == ib.e) - (i == ib.s) : 0;
  const int oj = offj ? (j == jb.e) - (j == jb.s) : 0;
  const int ok = offk ? (k == kb.e) - (k == kb.s) : 0;
  return pack.IsOwned(b, ok, oj, oi);
}

// Number of interior elements of type el on a block that the block owns, so that the sum
// over the blocks of the leaf grid counts every element once
inline std::int64_t OwnedElementCount(MeshBlock *pmb, TopologicalElement el) {
  const auto &cb = pmb->cellbounds;
  auto counts = [](const IndexRange &r, bool offset) -> std::array<std::int64_t, 3> {
    const std::int64_t n = r.e - r.s + 1;
    if (!offset || n == 1) return {0, n, 0};
    return {1, n - 2, 1};
  };
  const auto ni =
      counts(cb.GetBoundsI(IndexDomain::interior, el), TopologicalOffsetI(el));
  const auto nj =
      counts(cb.GetBoundsJ(IndexDomain::interior, el), TopologicalOffsetJ(el));
  const auto nk =
      counts(cb.GetBoundsK(IndexDomain::interior, el), TopologicalOffsetK(el));
  std::int64_t n = 0;
  for (int oi = -1; oi <= 1; ++oi) {
    for (int oj = -1; oj <= 1; ++oj) {
      for (int ok = -1; ok <= 1; ++ok) {
        const bool owned = !pmb->ownership.initialized || pmb->ownership(oi, oj, ok);
        if (owned) n += ni[oi + 1] * nj[oj + 1] * nk[ok + 1];
      }
    }
  }
  return n;
}

namespace between_fields {
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
} // namespace between_fields

template <TopologicalElement te, class pack_t>
void CopyDataElement(const std::shared_ptr<MeshData<Real>> &md_in, const pack_t &pack_in,
                     const pack_t &pack_out) {
  constexpr TopologicalElement tb = FlatBoundsElement(te);
  IndexRange ib = md_in->GetBoundsI(IndexDomain::entire, tb);
  IndexRange jb = md_in->GetBoundsJ(IndexDomain::entire, tb);
  IndexRange kb = md_in->GetBoundsK(IndexDomain::entire, tb);

  const int scratch_size = 0;
  const int scratch_level = 0;
  // Warning: This inner loop strategy only works because we are using IndexDomain::entire
  const int npoints_inner = (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "CopyData", DevExecSpace(), scratch_size, scratch_level,
      0, pack_in.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const int nvars = pack_in.GetUpperBound(b) - pack_in.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c) {
          Real *in = &pack_in(b, te, c, kb.s, jb.s, ib.s);
          Real *out = &pack_out(b, te, c, kb.s, jb.s, ib.s);
          parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                                   npoints_inner - 1,
                                   [&](const int idx) { out[idx] = in[idx]; });
        }
      });
}

template <class TL, bool only_fine_on_composite = true>
TaskStatus CopyData(const std::shared_ptr<MeshData<Real>> &md_in,
                    const std::shared_ptr<MeshData<Real>> &md_out) {
  static auto desc = parthenon::MakePackDescriptorFromTypeList<TL>(md_in.get());
  static const auto tt = FieldsTopologicalType(md_in.get(), desc);
  auto pack_in = desc.GetPack(md_in.get(), only_fine_on_composite);
  auto pack_out = desc.GetPack(md_out.get(), only_fine_on_composite);
  ForEachElement(tt, [&](auto el) {
    CopyDataElement<decltype(el)::value>(md_in, pack_in, pack_out);
  });
  return TaskStatus::complete;
}

template <TopologicalElement te, class pack_t>
void SetToZeroElement(const pack_t &pack) {
  const std::size_t scratch_size_in_bytes = 0;
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
        const int nvars = pack.GetUpperBound(b) - pack.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
              [&](int k, int j, int i) { pack(b, te, c, k, j, i) = 0.0; });
        }
      });
}

template <class TL, bool only_fine_on_composite = true>
TaskStatus SetToZero(const std::shared_ptr<MeshData<Real>> &md) {
  static auto desc = [&] {
    if constexpr (isTypeList<TL>::value) {
      return parthenon::MakePackDescriptorFromTypeList<TL>(md.get());
    } else {
      return parthenon::MakePackDescriptor<TL>(md.get());
    }
  }();
  static const auto tt = FieldsTopologicalType(md.get(), desc);
  auto pack = desc.GetPack(md.get(), only_fine_on_composite);
  ForEachElement(tt, [&](auto el) { SetToZeroElement<decltype(el)::value>(pack); });
  return TaskStatus::complete;
}

// Utilities functions in the between_fields namespace work on separate fields specified
// by types. Other utility functions work on the same list of fields across separate
// MeshData containers.
namespace between_fields {
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
      include_block[b] =
          md->GetBlockData(b)->GetBlockPointer()->GetNeighbors().size() == 0;
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
} // namespace between_fields

template <TopologicalElement te, class pack_t>
void AddFieldsAndStoreElement(const std::shared_ptr<MeshData<Real>> &md_a,
                              const pack_t &pack_a, const pack_t &pack_b,
                              const pack_t &pack_out, Real wa, Real wb) {
  constexpr TopologicalElement tb = FlatBoundsElement(te);
  IndexRange ib = md_a->GetBoundsI(IndexDomain::entire, tb);
  IndexRange jb = md_a->GetBoundsJ(IndexDomain::entire, tb);
  IndexRange kb = md_a->GetBoundsK(IndexDomain::entire, tb);

  const int scratch_size = 0;
  const int scratch_level = 0;
  // Warning: This inner loop strategy only works because we are using IndexDomain::entire
  const int npoints_inner = (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "AddFieldsAndStore", DevExecSpace(), scratch_size,
      scratch_level, 0, pack_a.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const int nvars = pack_a.GetUpperBound(b) - pack_a.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c) {
          Real *avar = &pack_a(b, te, c, kb.s, jb.s, ib.s);
          Real *bvar = &pack_b(b, te, c, kb.s, jb.s, ib.s);
          Real *out = &pack_out(b, te, c, kb.s, jb.s, ib.s);
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, 0, npoints_inner - 1,
              [&](const int idx) { out[idx] = wa * avar[idx] + wb * bvar[idx]; });
        }
      });
}

template <class TL, bool only_fine_on_composite = true>
TaskStatus AddFieldsAndStoreInteriorSelect(const std::shared_ptr<MeshData<Real>> &md_a,
                                           const std::shared_ptr<MeshData<Real>> &md_b,
                                           const std::shared_ptr<MeshData<Real>> &md_out,
                                           Real wa = 1.0, Real wb = 1.0,
                                           bool only_interior_blocks = false) {
  int nblocks = md_a->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  if (only_interior_blocks && (md_a->grid.type() == GridType::two_level_composite)) {
    for (int b = 0; b < nblocks; ++b) {
      auto pmb = md_a->GetBlockData(b)->GetBlockPointer();
      include_block[b] = !pmb->IsLeafLL() || (pmb->block_coarsenings != 0);
    }
  }

  static auto desc = parthenon::MakePackDescriptorFromTypeList<TL>(md_a.get());
  static const auto tt = FieldsTopologicalType(md_a.get(), desc);
  auto pack_a = desc.GetPack(md_a.get(), include_block, only_fine_on_composite);
  auto pack_b = desc.GetPack(md_b.get(), include_block, only_fine_on_composite);
  auto pack_out = desc.GetPack(md_out.get(), include_block, only_fine_on_composite);
  ForEachElement(tt, [&](auto el) {
    AddFieldsAndStoreElement<decltype(el)::value>(md_a, pack_a, pack_b, pack_out, wa, wb);
  });
  return TaskStatus::complete;
}

template <class TL, bool only_fine_on_composite = true>
TaskStatus AddFieldsAndStore(const std::shared_ptr<MeshData<Real>> &md_a,
                             const std::shared_ptr<MeshData<Real>> &md_b,
                             const std::shared_ptr<MeshData<Real>> &md_out, Real wa = 1.0,
                             Real wb = 1.0) {
  return AddFieldsAndStoreInteriorSelect<TL, only_fine_on_composite>(md_a, md_b, md_out,
                                                                     wa, wb, false);
}

namespace between_fields {
template <class a_t, class b_t, class out_t>
TaskStatus ADividedByB(const std::shared_ptr<MeshData<Real>> &md) {
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  static auto desc = parthenon::MakePackDescriptor<a_t, b_t, out_t>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DotProduct", DevExecSpace(), 0, pack.GetNBlocks() - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // printf("(%i, %i) diag_elem = %e\n", j, i, pack(b, b_t(), k, j, i));
        const int nvars = pack.GetUpperBound(b, a_t()) - pack.GetLowerBound(b, a_t()) + 1;
        for (int c = 0; c < nvars; ++c)
          pack(b, out_t(c), k, j, i) =
              pack(b, a_t(c), k, j, i) / pack(b, b_t(c), k, j, i);
      });
  return TaskStatus::complete;
}
} // namespace between_fields

template <TopologicalElement te, class pack_t>
void ADividedByBElement(const std::shared_ptr<MeshData<Real>> &md_a, const pack_t &pack_a,
                        const pack_t &pack_b, const pack_t &pack_out) {
  IndexRange ib = md_a->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md_a->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md_a->GetBoundsK(IndexDomain::interior, te);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DotProduct", DevExecSpace(), 0, pack_a.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int nvars = pack_a.GetUpperBound(b) - pack_a.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c)
          pack_out(b, te, c, k, j, i) =
              pack_a(b, te, c, k, j, i) / pack_b(b, te, c, k, j, i);
      });
}

template <class TL>
TaskStatus ADividedByB(const std::shared_ptr<MeshData<Real>> &md_a,
                       const std::shared_ptr<MeshData<Real>> &md_b,
                       const std::shared_ptr<MeshData<Real>> &md_out) {
  static auto desc = parthenon::MakePackDescriptorFromTypeList<TL>(md_a.get());
  static const auto tt = FieldsTopologicalType(md_a.get(), desc);
  auto pack_a = desc.GetPack(md_a.get());
  auto pack_b = desc.GetPack(md_b.get());
  auto pack_out = desc.GetPack(md_out.get());
  ForEachElement(tt, [&](auto el) {
    ADividedByBElement<decltype(el)::value>(md_a, pack_a, pack_b, pack_out);
  });
  return TaskStatus::complete;
}

namespace between_fields {
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
} // namespace between_fields

template <TopologicalElement te, class a_t, class pack_t>
Real GlobalMinElement(const std::shared_ptr<MeshData<Real>> &md, const pack_t &pack) {
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

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
  return gmin;
}

template <class a_t>
TaskStatus GlobalMinLocal(const std::shared_ptr<MeshData<Real>> &md,
                          AllReduce<Real> *amin) {
  static auto desc = parthenon::MakePackDescriptor<a_t>(md.get());
  static const auto tt = FieldsTopologicalType(md.get(), desc);
  auto pack = desc.GetPack(md.get());
  ForEachElement(tt, [&](auto el) {
    const Real gmin = GlobalMinElement<decltype(el)::value, a_t>(md, pack);
    amin->val = std::min(gmin, amin->val);
  });
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

template <TopologicalElement te, bool VolumeWeight, class pack_t>
Real DotProductElement(const std::shared_ptr<MeshData<Real>> &md_a, const pack_t &pack_a,
                       const pack_t &pack_b) {
  IndexRange ib = md_a->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md_a->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md_a->GetBoundsK(IndexDomain::interior, te);

  Real gsum(0);
  const int nvars = pack_a.GetUpperBoundHost(0) - pack_a.GetLowerBoundHost(0) + 1;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "DotProduct", DevExecSpace(), 0,
      pack_a.GetNBlocks() - 1, 0, nvars - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int c, const int k, const int j, const int i,
                    Real &lsum) {
        if constexpr (te != TopologicalElement::CC) {
          if (!IsOwnedElement<te>(pack_a, b, kb, jb, ib, k, j, i)) return;
        }
        const Real var_a = pack_a(b, te, c, k, j, i);
        const Real var_b = pack_b(b, te, c, k, j, i);
        if constexpr (VolumeWeight) {
          const auto vol = [&] {
            if constexpr (te == TopologicalElement::CC) {
              return pack_a.GetCoordinates(b).CellVolume(k, j, i);
            } else {
              return pack_a.GetCoordinates(b).template Volume<te>(k, j, i);
            }
          }();
          lsum += var_a * var_b * vol * vol;
        } else {
          lsum += var_a * var_b;
        }
      },
      Kokkos::Sum<Real>(gsum));
  return gsum;
}

template <class TL, bool VolumeWeight = false>
TaskStatus DotProductLocal(const std::shared_ptr<MeshData<Real>> &md_a,
                           const std::shared_ptr<MeshData<Real>> &md_b,
                           AllReduce<Real> *adotb) {
  static auto desc = parthenon::MakePackDescriptorFromTypeList<TL>(md_a.get());
  static const auto tt = FieldsTopologicalType(md_a.get(), desc);
  auto pack_a = desc.GetPack(md_a.get());
  auto pack_b = desc.GetPack(md_b.get());
  ForEachElement(tt, [&](auto el) {
    adotb->val +=
        DotProductElement<decltype(el)::value, VolumeWeight>(md_a, pack_a, pack_b);
  });
  return TaskStatus::complete;
}

template <class TL>
TaskID DotProduct(TaskID dependency_in, TaskList &tl, AllReduce<Real> *adotb,
                  const std::shared_ptr<MeshData<Real>> &md_a,
                  const std::shared_ptr<MeshData<Real>> &md_b,
                  bool volume_weight = false) {
  using namespace impl;
  auto zero_adotb = tl.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, dependency_in,
      [](AllReduce<Real> *r) {
        r->val = 0.0;
        return TaskStatus::complete;
      },
      adotb);
  auto get_adotb = zero_adotb;
  if (volume_weight) {
    get_adotb = tl.AddTask(TaskQualifier::local_sync, zero_adotb,
                           DotProductLocal<TL, true>, md_a, md_b, adotb);
  } else {
    get_adotb = tl.AddTask(TaskQualifier::local_sync, zero_adotb,
                           DotProductLocal<TL, false>, md_a, md_b, adotb);
  }
  auto start_global_adotb = tl.AddTask(TaskQualifier::once_per_region, get_adotb,
                                       &AllReduce<Real>::StartReduce, adotb, MPI_SUM);
  auto finish_global_adotb =
      tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                 start_global_adotb, &AllReduce<Real>::CheckReduce, adotb);
  return finish_global_adotb;
}

template <TopologicalElement te, bool VolumeWeight, class pack_t>
summable_array_t<Real, 2>
DoubleDotProductElement(const std::shared_ptr<MeshData<Real>> &md_a, const pack_t &pack_a,
                        const pack_t &pack_b) {
  IndexRange ib = md_a->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md_a->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md_a->GetBoundsK(IndexDomain::interior, te);

  summable_array_t<Real, 2> gsum;
  const int nvars = pack_a.GetUpperBoundHost(0) - pack_a.GetLowerBoundHost(0) + 1;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "DotProduct", DevExecSpace(), 0,
      pack_a.GetNBlocks() - 1, 0, nvars - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int c, const int k, const int j, const int i,
                    summable_array_t<Real, 2> &lsum) {
        if constexpr (te != TopologicalElement::CC) {
          if (!IsOwnedElement<te>(pack_a, b, kb, jb, ib, k, j, i)) return;
        }
        const Real var_a = pack_a(b, te, c, k, j, i);
        const Real var_b = pack_b(b, te, c, k, j, i);
        if constexpr (VolumeWeight) {
          const auto vol = [&] {
            if constexpr (te == TopologicalElement::CC) {
              return pack_a.GetCoordinates(b).CellVolume(k, j, i);
            } else {
              return pack_a.GetCoordinates(b).template Volume<te>(k, j, i);
            }
          }();
          lsum[0] += var_a * var_a * vol * vol;
          lsum[1] += var_a * var_b * vol * vol;
        } else {
          lsum[0] += var_a * var_a;
          lsum[1] += var_a * var_b;
        }
      },
      Kokkos::Sum<summable_array_t<Real, 2>>(gsum));
  return gsum;
}

template <class TL, bool VolumeWeight = false>
TaskStatus DoubleDotProductLocal(const std::shared_ptr<MeshData<Real>> &md_a,
                                 const std::shared_ptr<MeshData<Real>> &md_b,
                                 AllReduce<summable_array_t<Real, 2>> *adotb) {
  static auto desc = parthenon::MakePackDescriptorFromTypeList<TL>(md_a.get());
  static const auto tt = FieldsTopologicalType(md_a.get(), desc);
  auto pack_a = desc.GetPack(md_a.get());
  auto pack_b = desc.GetPack(md_b.get());
  ForEachElement(tt, [&](auto el) {
    adotb->val +=
        DoubleDotProductElement<decltype(el)::value, VolumeWeight>(md_a, pack_a, pack_b);
  });
  return TaskStatus::complete;
}

template <class TL>
TaskID DoubleDotProduct(TaskID dependency_in, TaskList &tl,
                        AllReduce<summable_array_t<Real, 2>> *adotb,
                        const std::shared_ptr<MeshData<Real>> &md_a,
                        const std::shared_ptr<MeshData<Real>> &md_b,
                        bool volume_weight = false) {
  using namespace impl;
  using value_t = summable_array_t<Real, 2>;

  auto zero_adotb = tl.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, dependency_in,
      [](AllReduce<value_t> *r) {
        r->val.init();
        return TaskStatus::complete;
      },
      adotb);
  auto get_adotb = zero_adotb;
  if (volume_weight) {
    get_adotb = tl.AddTask(TaskQualifier::local_sync, zero_adotb,
                           DoubleDotProductLocal<TL, true>, md_a, md_b, adotb);
  } else {
    get_adotb = tl.AddTask(TaskQualifier::local_sync, zero_adotb,
                           DoubleDotProductLocal<TL, false>, md_a, md_b, adotb);
  }
  auto start_global_adotb = tl.AddTask(TaskQualifier::once_per_region, get_adotb,
                                       &AllReduce<value_t>::StartReduce, adotb, MPI_SUM);
  auto finish_global_adotb =
      tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                 start_global_adotb, &AllReduce<value_t>::CheckReduce, adotb);
  return finish_global_adotb;
}

template <class TL>
TaskStatus OwnedElementCountLocal(const std::shared_ptr<MeshData<Real>> &md,
                                  AllReduce<Real> *n) {
  static auto desc = parthenon::MakePackDescriptorFromTypeList<TL>(md.get());
  static const auto tt = FieldsTopologicalType(md.get(), desc);
  std::int64_t nloc = 0;
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    for (const auto el : GetTopologicalElements(tt))
      nloc += OwnedElementCount(pmb, el);
  }
  n->val += static_cast<Real>(nloc);
  return TaskStatus::complete;
}

// Global number of mesh elements on which the fields in TL live, used to normalize rms
// residuals. This is the total number of cells for cell-centered fields, and otherwise
// counts each shared face, edge or node once. md should be on the leaf grid.
template <class TL>
TaskID GlobalElementCount(TaskID dependency_in, TaskList &tl, AllReduce<Real> *n,
                          const std::shared_ptr<MeshData<Real>> &md) {
  static auto desc = parthenon::MakePackDescriptorFromTypeList<TL>(md.get());
  static const auto tt = FieldsTopologicalType(md.get(), desc);
  if (tt == TopologicalType::Cell) {
    return tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync, dependency_in,
        [](AllReduce<Real> *r, Mesh *pmesh) {
          r->val = pmesh->GetTotalCells();
          return TaskStatus::complete;
        },
        n, md->GetMeshPointer());
  }
  auto zero_n = tl.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, dependency_in,
      [](AllReduce<Real> *r) {
        r->val = 0.0;
        return TaskStatus::complete;
      },
      n);
  auto get_n =
      tl.AddTask(TaskQualifier::local_sync, zero_n, OwnedElementCountLocal<TL>, md, n);
  auto start_global_n = tl.AddTask(TaskQualifier::once_per_region, get_n,
                                   &AllReduce<Real>::StartReduce, n, MPI_SUM);
  return tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                    start_global_n, &AllReduce<Real>::CheckReduce, n);
}

} // namespace utils

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_UTILS_HPP_
