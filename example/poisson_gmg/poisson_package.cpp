//========================================================================================
// (C) (or copyright) 2021-2022. Triad National Security, LLC. All rights reserved.
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

  int check_interval = pin->GetOrAddInteger("poisson", "check_interval", 100);
  pkg->AddParam<>("check_interval", check_interval);

  Real err_tol = pin->GetOrAddReal("poisson", "error_tolerance", 1.e-8);
  pkg->AddParam<>("error_tolerance", err_tol);

  bool fail_flag = pin->GetOrAddBoolean("poisson", "fail_without_convergence", false);
  pkg->AddParam<>("fail_without_convergence", fail_flag);

  bool warn_flag = pin->GetOrAddBoolean("poisson", "warn_without_convergence", true);
  pkg->AddParam<>("warn_without_convergence", warn_flag);

  auto mphi = Metadata(
      {Metadata::Node, Metadata::Independent, Metadata::FillGhost, Metadata::GMG});
  pkg->AddField(res_err::name(), mphi);

  auto mrhs = Metadata({Metadata::Node, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField(rhs::name(), mrhs);
  pkg->AddField(u::name(), mrhs);

  return pkg;
}

TaskStatus SetToZero(std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, TE::NN);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, TE::NN);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, TE::NN);

  auto desc = parthenon::MakePackDescriptor<res_err>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack(b, TE::NN, res_err(), k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

template <class Am, class Ac, class Ap, class x_t, class y_t>
TaskStatus BlockLocalTriDiagX(std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, TE::NN);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, TE::NN);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, TE::NN);

  auto desc = parthenon::MakePackDescriptor<Am, Ac, Ap, x_t, y_t>(md);
  auto pack = desc.GetPack(md);

  int nx1 = ib.e - ib.s + 1;
  int nx2 = jb.e - jb.s + 1;
  size_t scratch_size = parthenon::ScratchPad2D<Real>::shmem_size(nx2, nx1);
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "tridiagonal solve x", DevExecSpace(), scratch_size,
      scratch_level, 0, pack.GetNBlocks() - 1, kb.s, kb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        // Solve T_x x = y for x
        const auto &A_im = pack(b, Am());
        const auto &A_diag = pack(b, Ac());
        const auto &A_ip = pack(b, Ap());
        const auto &x = pack(b, x_t());
        const auto &y = v(b, y_t());

        parthenon::ScratchPad2D<Real> c(member.team_scratch(scratch_level), nx2, nx1);

        parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, jb.s, jb.e,
                                 [&](const int j) {
                                   const Real b = A_diag(k, j, ib.s);
                                   c(j, ib.s) = A_ip(k, j, ib.s) / b;
                                   x(k, j, ib.s) = y(k, j, ib.s) / b;
                                 });

        for (int i = ib.s + 1; i <= ib.e; ++i) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, jb.s, jb.e, [&](const int j) {
                const Real idenom = 1.0 / (A_diag(k, j, i) - A_im(k, j, i) * c(j, i - 1));
                c(j, i) = A_ip(k, j, i) * idenom;
                x(k, j, i) = (y(k, j, i) - A_im(k, j, i) * x(k, j, i - 1)) * idenom;
              });
        }

        member.team_barrier();
        for (int i = ib.e - 1; i >= ib.s; --i) {
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, jb.s, jb.e,
              [&](const int j) { x(k, j, i) = x(k, j, i) - c(j, i) * x(k, j, i + 1); });
        }
      });
  return TaskStatus::complete;
}

TaskStatus PrintValues(std::shared_ptr<MeshData<Real>> &md) {
  using TE = parthenon::TopologicalElement;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior, TE::NN);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior, TE::NN);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior, TE::NN);

  auto desc = parthenon::MakePackDescriptor<res_err>(md.get());
  auto pack = desc.GetPack(md.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetPotentialToZero", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        double x = coords.X<1, TE::NN>(i);
        printf("block = %i %i: %e (%e)\n", b, i, pack(b, TE::NN, res_err(), k, j, i), x);
      });
  printf("Done with MeshData\n\n");
  return TaskStatus::complete;
}

} // namespace poisson_package
