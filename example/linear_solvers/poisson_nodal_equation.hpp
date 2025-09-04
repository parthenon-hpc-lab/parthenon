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
#ifndef EXAMPLE_LINEAR_SOLVERS_POISSON_NODAL_EQUATION_HPP_
#define EXAMPLE_LINEAR_SOLVERS_POISSON_NODAL_EQUATION_HPP_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

namespace poisson_nodal_package {
using namespace parthenon::package::prelude;
// This class implement methods for calculating A.x = y and returning the diagonal of A,
// where A is the the matrix representing the discretized Poisson equation on the grid.
// Here we implement the Laplace operator in terms of a flux divergence to (potentially)
// consistently deal with coarse fine boundaries on the grid. Only the routines Ax and
// SetDiagonal need to be defined for interfacing this with solvers. The other methods
// are internal, but can't be marked private or protected because they launch kernels
// on device.
template <class var_t>
class PoissonEquation {
 public:
  using IndependentVars = parthenon::TypeList<var_t>;

  PoissonEquation(parthenon::ParameterInput *pin, const std::string &label) {}

  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> & /*md_mat*/,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    return tl.AddTask(depends_on, AxImpl, md_in, md_out);
  }

  static parthenon::TaskStatus
  AxImpl(std::shared_ptr<parthenon::MeshData<Real>> &md_in,
         std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using namespace parthenon;
    auto pkg = md_in->GetMeshPointer()->packages.Get("poisson_nodal_package");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    constexpr auto te = TopologicalElement::NN;
    const int ndim = md_in->GetMeshPointer()->ndim;
    IndexRange ib = md_in->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_in->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_in->GetBoundsK(IndexDomain::interior, te);

    auto desc = parthenon::MakePackDescriptor<var_t>(md_in.get());
    auto pack_in = desc.GetPack(md_in.get());
    auto pack_out = desc.GetPack(md_out.get());

    parthenon::par_for(
        "PoissonNodal::Ax", 0, pack_in.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack_in.GetCoordinates(b);
          const Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          const Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
          const Real dx3 = coords.template Dxc<X3DIR>(k, j, i);

          Real Ax = alpha * pack_in(b, te, var_t(), k, j, i);
          Ax += pack_in(b, te, var_t(), k, j, i) * 2.0 / (dx1 * dx1);
          Ax -= (pack_in(b, te, var_t(), k, j, i + 1) +
                 pack_in(b, te, var_t(), k, j, i - 1)) /
                (dx1 * dx1);
          if (ndim > 1) {
            Ax += pack_in(b, te, var_t(), k, j, i) * 2.0 / (dx2 * dx2);
            Ax -= (pack_in(b, te, var_t(), k, j + 1, i) +
                   pack_in(b, te, var_t(), k, j - 1, i)) /
                  (dx2 * dx2);
          }
          if (ndim > 2) {
            Ax += pack_in(b, te, var_t(), k, j, i) * 2.0 / (dx3 * dx3);
            Ax -= (pack_in(b, te, var_t(), k + 1, j, i) +
                   pack_in(b, te, var_t(), k - 1, j, i)) /
                  (dx3 * dx3);
          }
          pack_out(b, te, var_t(), k, j, i) = Ax;
        });
    return TaskStatus::complete;
  }

  static parthenon::TaskStatus SetBoundary(std::shared_ptr<parthenon::MeshData<Real>> &md,
                                           bool coarse) {
    using namespace parthenon;

    constexpr auto te = TopologicalElement::NN;
    const int ndim = md->GetMeshPointer()->ndim;
    CellLevel cl = coarse ? CellLevel::coarse : CellLevel::same;
    IndexRange ib = md->GetBoundsI(cl, IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(cl, IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(cl, IndexDomain::interior, te);

    std::set<PDOpt> opts{};
    if (coarse) opts.emplace(PDOpt::Coarse);
    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, opts);
    auto pack = desc.GetPack(md.get(), GetBlockSelector::OnPhysicalBoundary());

    parthenon::par_for(
        "PoissonNodal::SetBoundary", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const int oi = TopologicalOffsetI(te) * ((ib.e == i) - (ib.s == i));
          const int oj = TopologicalOffsetJ(te) * ((jb.e == j) - (jb.s == j));
          const int ok = TopologicalOffsetK(te) * ((kb.e == k) - (kb.s == k));
          if (pack.IsPhysicalBoundary(b, ok, oj, oi)) pack(b, te, var_t(), k, j, i) = 0.0;
        });
    return TaskStatus::complete;
  }

  parthenon::TaskStatus
  SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> & /*md_mat*/,
              std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    const int ndim = md_diag->GetMeshPointer()->ndim;
    constexpr auto te = TopologicalElement::NN;
    IndexRange ib = md_diag->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_diag->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_diag->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md_diag->GetMeshPointer()->packages.Get("poisson_nodal_package");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    auto desc = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack_diag = desc.GetPack(md_diag.get());
    parthenon::par_for(
        "StoreDiagonal", 0, pack_diag.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack_diag.GetCoordinates(b);
          // Build the diagonal of the matrix
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
          Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
          pack_diag(b, te, var_t(), k, j, i) = alpha + 2.0 / (dx1 * dx1) +
                                               (ndim > 1) * 2.0 / (dx2 * dx2) +
                                               (ndim > 2) * 2.0 / (dx3 * dx3);
        });
    return TaskStatus::complete;
  }
};

} // namespace poisson_nodal_package

#endif // EXAMPLE_LINEAR_SOLVERS_POISSON_NODAL_EQUATION_HPP_
