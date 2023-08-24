//========================================================================================
// (C) (or copyright) 2021-2023. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/solver_utils.hpp>

#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "poisson_package.hpp"

using namespace parthenon::package::prelude;
using parthenon::HostArray1D;
namespace poisson_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_package");

  int max_poisson_iterations = pin->GetOrAddInteger("poisson", "max_iterations", 10000);
  pkg->AddParam<>("max_iterations", max_poisson_iterations);

  int jacobi_iterations = pin->GetOrAddInteger("poisson", "jacobi_iterations", 4);
  pkg->AddParam<>("jacobi_iterations", jacobi_iterations);

  Real jacobi_damping = pin->GetOrAddReal("poisson", "jacobi_damping", 0.5);
  pkg->AddParam<>("jacobi_damping", jacobi_damping);

  int check_interval = pin->GetOrAddInteger("poisson", "check_interval", 100);
  pkg->AddParam<>("check_interval", check_interval);

  Real err_tol = pin->GetOrAddReal("poisson", "error_tolerance", 1.e-8);
  pkg->AddParam<>("error_tolerance", err_tol);

  bool fail_flag = pin->GetOrAddBoolean("poisson", "fail_without_convergence", false);
  pkg->AddParam<>("fail_without_convergence", fail_flag);

  bool warn_flag = pin->GetOrAddBoolean("poisson", "warn_without_convergence", true);
  pkg->AddParam<>("warn_without_convergence", warn_flag);

  // res_err enters a multigrid level as the residual from the previous level, which
  // is the rhs, and leaves as the solution for that level, which is the error for the
  // next finer level
  auto te_type = Metadata::Cell;
  if (GetTopologicalType(te) == parthenon::TopologicalType::Node) {
    te_type = Metadata::Node;
  } else if (GetTopologicalType(te) == parthenon::TopologicalType::Edge) {
    te_type = Metadata::Edge;
  } else if (GetTopologicalType(te) == parthenon::TopologicalType::Face) {
    te_type = Metadata::Face;
  }
  using namespace parthenon::refinement_ops;
  auto mres_err =
      Metadata({te_type, Metadata::Independent, Metadata::FillGhost, Metadata::GMG});
  mres_err.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField(res_err::name(), mres_err);

  auto mrhs = Metadata({te_type, Metadata::Independent, Metadata::FillGhost});
  mrhs.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  pkg->AddField(rhs::name(), mrhs);
  pkg->AddField(rhs_base::name(), mrhs);
  pkg->AddField(u::name(), mrhs);
  pkg->AddField(solution::name(), mrhs);
  pkg->AddField(temp::name(), mrhs);

  auto mAs =
      Metadata({te_type, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{3});
  auto mA = Metadata({te_type, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(Am::name(), mAs);
  pkg->AddField(Ac::name(), mA);
  pkg->AddField(Ap::name(), mAs);

  return pkg;
}

TaskStatus BuildMatrix(std::shared_ptr<MeshData<Real>> &md) {
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<Am, Ac, Ap>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "BuildMatrix", DevExecSpace(), 0, pack.GetNBlocks() - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real dx1 = coords.Dxc<1>(k, j, i);
        Real dx2 = coords.Dxc<2>(k, j, i);
        Real dx3 = coords.Dxc<3>(k, j, i);
        pack(b, te, Am(0), k, j, i) = -1.0 / (dx1 * dx1);
        pack(b, te, Ac(), k, j, i) = 2.0 / (dx1 * dx1);
        pack(b, te, Ap(0), k, j, i) = -1.0 / (dx1 * dx1);
        if (ndim > 1) {
          pack(b, te, Am(1), k, j, i) = -1.0 / (dx2 * dx2);
          pack(b, te, Ac(), k, j, i) += 2.0 / (dx2 * dx2);
          pack(b, te, Ap(1), k, j, i) = -1.0 / (dx2 * dx2);
        }
        if (ndim > 2) {
          pack(b, te, Am(2), k, j, i) = -1.0 / (dx3 * dx3);
          pack(b, te, Ac(), k, j, i) += 2.0 / (dx3 * dx3);
          pack(b, te, Ap(2), k, j, i) = -1.0 / (dx3 * dx3);
        }
      });
  return TaskStatus::complete;
}

TaskStatus CalculateResidual(std::shared_ptr<MeshData<Real>> &md) {
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<Ap, Ac, Am, u, rhs, res_err>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculateResidual", DevExecSpace(), 0, pack.GetNBlocks() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, te, res_err(), k, j, i) =
            pack(b, te, rhs(), k, j, i) -
            pack(b, te, Ac(), k, j, i) * pack(b, te, u(), k, j, i);

        pack(b, te, res_err(), k, j, i) -=
            pack(b, te, Am(0), k, j, i) * pack(b, te, u(), k, j, i - 1);
        pack(b, te, res_err(), k, j, i) -=
            pack(b, te, Ap(0), k, j, i) * pack(b, te, u(), k, j, i + 1);

        if (ndim > 1) {
          pack(b, te, res_err(), k, j, i) -=
              pack(b, te, Am(1), k, j, i) * pack(b, te, u(), k, j - 1, i);
          pack(b, te, res_err(), k, j, i) -=
              pack(b, te, Ap(1), k, j, i) * pack(b, te, u(), k, j + 1, i);
        }

        if (ndim > 2) {
          pack(b, te, res_err(), k, j, i) -=
              pack(b, te, Am(1), k, j, i) * pack(b, te, u(), k - 1, j, i);
          pack(b, te, res_err(), k, j, i) -=
              pack(b, te, Ap(1), k, j, i) * pack(b, te, u(), k + 1, j, i);
        }

        // printf("b = %i i = %i Am = %e Ac = %e Ap = %e rhs = %e res = %e u =%e\n", b, i,
        // pack(b, te, Am(), k, j, i), pack(b, te, Ac(), k, j, i), pack(b, te, Ap(), k, j,
        // i), pack(b, te, rhs(), k, j, i), pack(b, te, res_err(), k, j, i), pack(b, te,
        // u(), k, j, i));
      });
  // printf("\n");
  return TaskStatus::complete;
}

template <class x_t>
TaskStatus BlockLocalTriDiagX(std::shared_ptr<MeshData<Real>> &md) {
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<Am, Ac, Ap, x_t, rhs>(md.get());
  auto pack = desc.GetPack(md.get());

  int nx1 = ib.e - ib.s + 1;
  int nx2 = jb.e - jb.s + 1;
  size_t scratch_size = parthenon::ScratchPad2D<Real>::shmem_size(nx2, nx1);
  constexpr int scratch_level = 1;
  int upper_boundary_block = pack.GetNBlocks() - 1;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "tridiagonal solve x", DevExecSpace(), scratch_size,
      scratch_level, 0, pack.GetNBlocks() - 1, kb.s, kb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        // Solve T_x x = y for x
        const auto &A_im = pack(b, Am());
        const auto &A_diag = pack(b, Ac());
        const auto &A_ip = pack(b, Ap());
        const auto &x = pack(b, x_t());
        const auto &y = pack(b, rhs());
        int ie_block = ib.e;
        if (b != upper_boundary_block) ie_block--;
        parthenon::ScratchPad2D<Real> c(member.team_scratch(scratch_level), nx2, nx1);

        parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, jb.s, jb.e,
                                 [&](const int j) {
                                   const Real b = A_diag(k, j, ib.s);
                                   c(j, ib.s) = A_ip(k, j, ib.s) / b;
                                   x(k, j, ib.s) = y(k, j, ib.s) / b;
                                 });

        for (int i = ib.s + 1; i <= ie_block; ++i) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, jb.s, jb.e, [&](const int j) {
                const Real idenom = 1.0 / (A_diag(k, j, i) - A_im(k, j, i) * c(j, i - 1));
                c(j, i) = A_ip(k, j, i) * idenom;
                x(k, j, i) = (y(k, j, i) - A_im(k, j, i) * x(k, j, i - 1)) * idenom;
              });
        }

        member.team_barrier();
        for (int i = ie_block - 1; i >= ib.s; --i) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, jb.s, jb.e,
              [&](const int j) { x(k, j, i) = x(k, j, i) - c(j, i) * x(k, j, i + 1); });
        }
      });
  return TaskStatus::complete;
}

template TaskStatus BlockLocalTriDiagX<u>(std::shared_ptr<MeshData<Real>> &md);
template TaskStatus BlockLocalTriDiagX<res_err>(std::shared_ptr<MeshData<Real>> &md);

TaskStatus CorrectRHS(std::shared_ptr<MeshData<Real>> &md) {
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<Am, Ac, Ap, res_err, rhs>(md.get());
  auto pack = desc.GetPack(md.get());
  int upper_boundary_block = pack.GetNBlocks() - 1;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "update rhs", DevExecSpace(), 0, pack.GetNBlocks() - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        if (i == ib.s) {
          Real correction =
              pack(b, te, Am(), k, j, i) * pack(b, te, res_err(), k, j, i - 1);
          printf("Correction b=%i i=%i: %e\n", b, i, correction);
          pack(b, te, rhs(), k, j, i) -= correction;
        } else if (i == ib.e - (upper_boundary_block != b)) {
          Real correction =
              pack(b, te, Ap(), k, j, i) * pack(b, te, res_err(), k, j, i + 1);
          printf("Correction b=%i i=%i: %e\n", b, i, correction);
          pack(b, te, rhs(), k, j, i) -= correction;
        }
      });
  return TaskStatus::complete;
}

TaskStatus PrintValues(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor<res_err, u>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        double x = coords.X<1, te>(i);
        printf("block = %i %i: %e (%e)\n", b, i, pack(b, te, u(), k, j, i),
               pack(b, te, res_err(), k, j, i));
      });
  printf("Done with MeshData\n\n");
  return TaskStatus::complete;
}

} // namespace poisson_package
