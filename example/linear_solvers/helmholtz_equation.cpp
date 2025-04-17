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
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "helmholtz_equation.hpp"

namespace helmholtz_package {
using namespace parthenon::package::prelude;

parthenon::TaskStatus
HelmholtzEquation::AxImpl(std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                          std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
  using namespace parthenon;
  using TE = TopologicalElement;
  auto pkg = md_in->GetMeshPointer()->packages.Get("helmholtz_package");
  const auto alpha = pkg->Param<Real>("diagonal_alpha");

  const int ndim = md_in->GetMeshPointer()->ndim;
  IndexRange ib = md_in->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md_in->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md_in->GetBoundsK(IndexDomain::interior);

  auto desc = parthenon::MakePackDescriptorFromTypeList<IndependentVars>(md_in.get());
  auto pack_in = desc.GetPack(md_in.get());
  auto pack_out = desc.GetPack(md_out.get());

  const int ioff = ndim > 0;
  const int joff = ndim > 1;
  const int koff = ndim > 2;
  parthenon::par_for(
      "HelmholtzEquation::Ax", 0, pack_in.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack_in.GetCoordinates(b);
        const Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
        const Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
        const Real dx3 = coords.template Dxc<X3DIR>(k, j, i);

        Real Ax = -alpha * pack_in(b, TE::CC, vcc_t(), k, j, i);
        Ax -= (pack_in(b, TE::F1, vfc_t(), k, j, i + ioff) -
               pack_in(b, TE::F1, vfc_t(), k, j, i)) /
              dx1;
        Ax -= (pack_in(b, TE::F2, vfc_t(), k, j + joff, i) -
               pack_in(b, TE::F2, vfc_t(), k, j, i)) /
              dx2;
        Ax -= (pack_in(b, TE::F3, vfc_t(), k + koff, j, i) -
               pack_in(b, TE::F3, vfc_t(), k, j, i)) /
              dx3;

        pack_out(b, TE::CC, vcc_t(), k, j, i) = Ax;
      });
  std::vector<TE> tes{TE::F1};
  if (ndim > 1) tes.push_back(TE::F2);
  if (ndim > 2) tes.push_back(TE::F3);
  for (auto &&te : tes) {
    IndexRange ib = md_in->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_in->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_in->GetBoundsK(IndexDomain::interior, te);
    const int ioff = TopologicalOffsetI(te) * (ndim > 0);
    const int joff = TopologicalOffsetJ(te) * (ndim > 1);
    const int koff = TopologicalOffsetK(te) * (ndim > 2);
    parthenon::par_for(
        "HelmholtzEquation::Ax", 0, pack_in.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack_in.GetCoordinates(b);
          const Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          const Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
          const Real dx3 = coords.template Dxc<X3DIR>(k, j, i);

          Real Ax = alpha * pack_in(b, te, vfc_t(), k, j, i);
          Ax += (pack_in(b, TE::CC, vcc_t(), k, j, i) -
                 pack_in(b, TE::CC, vcc_t(), k - koff, j - joff, i - ioff)) /
                dx1;
          pack_out(b, te, vfc_t(), k, j, i) = Ax;
        });
  }
  return TaskStatus::complete;
}

parthenon::TaskStatus
HelmholtzEquation::SetBoundary(std::shared_ptr<parthenon::MeshData<Real>> &md,
                               bool coarse) {
  using namespace parthenon;

  using TE = TopologicalElement;
  const int ndim = md->GetMeshPointer()->ndim;

  std::set<PDOpt> opts{};
  if (coarse) opts.emplace(PDOpt::Coarse);
  auto desc = parthenon::MakePackDescriptor<vfc_t>(md.get(), {}, opts);
  auto pack = desc.GetPack(md.get(), GetBlockSelector::OnPhysicalBoundary());

  std::vector<TE> tes{TE::F1};
  if (ndim > 1) tes.push_back(TE::F2);
  if (ndim > 2) tes.push_back(TE::F3);
  for (auto &&te : tes) {
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    parthenon::par_for(
        "PoissonNodal::SetBoundary", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const int oi = TopologicalOffsetI(te) * ((ib.e == i) - (ib.s == i));
          const int oj = TopologicalOffsetJ(te) * ((jb.e == j) - (jb.s == j));
          const int ok = TopologicalOffsetK(te) * ((kb.e == k) - (kb.s == k));
          if (pack.IsPhysicalBoundary(b, ok, oj, oi)) pack(b, te, vfc_t(), k, j, i) = 0.0;
        });
  }
  return TaskStatus::complete;
}

parthenon::TaskStatus
HelmholtzEquation::SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> & /*md_mat*/,
                               std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
  using namespace parthenon;
  using TE = TopologicalElement;
  auto pkg = md_diag->GetMeshPointer()->packages.Get("helmholtz_package");
  const auto alpha = pkg->Param<Real>("diagonal_alpha");

  const int ndim = md_diag->GetMeshPointer()->ndim;
  IndexRange ib = md_diag->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md_diag->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md_diag->GetBoundsK(IndexDomain::interior);

  auto desc = parthenon::MakePackDescriptorFromTypeList<IndependentVars>(md_diag.get());
  auto pack_diag = desc.GetPack(md_diag.get());

  parthenon::par_for(
      "HelmholtzEquation::Ax", 0, pack_diag.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack_diag(b, TE::CC, vcc_t(), k, j, i) = -alpha;
      });

  std::vector<TE> tes{TE::F1};
  if (ndim > 1) tes.push_back(TE::F2);
  if (ndim > 2) tes.push_back(TE::F3);
  for (auto &&te : tes) {
    IndexRange ib = md_diag->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_diag->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_diag->GetBoundsK(IndexDomain::interior, te);
    parthenon::par_for(
        "HelmholtzEquation::Ax", 0, pack_diag.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack_diag(b, te, vfc_t(), k, j, i) = alpha;
        });
  }
  return TaskStatus::complete;
}

} // namespace helmholtz_package
