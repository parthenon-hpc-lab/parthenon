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

  auto mrho = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField("density", mrho);

  auto mphi = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField("potential", mphi);

  int ndim = 1 + (pin->GetInteger("parthenon/mesh", "nx2") > 1) +
             (pin->GetInteger("parthenon/mesh", "nx3") > 1);
  // set up the stencil object corresponding to the finite difference
  // discretization we adopt in this pacakge
  const int nstencil = 1 + 2 * ndim;
  std::vector<std::vector<int>> offsets(
      {{-1, 0, 1, 0, 0, 0, 0}, {0, 0, 0, -1, 1, 0, 0}, {0, 0, 0, 0, 0, -1, 1}});

  bool use_jacobi = pin->GetOrAddBoolean("poisson", "use_jacobi", true);
  pkg->AddParam<>("use_jacobi", use_jacobi);
  bool use_stencil = pin->GetOrAddBoolean("poisson", "use_stencil", true);
  pkg->AddParam<>("use_stencil", use_stencil);
  if (use_stencil) {
    std::vector<Real> wgts;
    if (use_jacobi) {
      wgts = std::vector<Real>({1.0, -2.0 * ndim, 1.0, 1.0, 1.0, 1.0, 1.0});
    } else {
      const Real w0 = 1.0 / (2.0 * ndim);
      wgts = std::vector<Real>({w0, -1.0, w0, w0, w0, w0, w0});
    }
    auto stencil = parthenon::solvers::Stencil<Real>("stencil", nstencil, wgts, offsets);
    pkg->AddParam<>("stencil", stencil);
  } else {
    // setup the sparse matrix
    Metadata msp = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                            std::vector<int>({nstencil}));
    pkg->AddField("poisson_sparse_matrix", msp);
    auto sp_accessor =
        parthenon::solvers::SparseMatrixAccessor("accessor", nstencil, offsets);
    pkg->AddParam("sparse_accessor", sp_accessor);
  }

  // ParArrays for reductions We stash the array, and AllReduce
  // object, which is per MPI rank, in the pkg object.
  // We must use a mutable Param here so that
  // the object can be modified by the driver.
  parthenon::AllReduce<HostArray1D<Real>> view_reduce;
  view_reduce.val = HostArray1D<Real>("Reduce me", 10);
  // First initialize to some values.
  // This first loop is actually unnecessary,
  // as Kokkos initializes to zero automatically.
  // We show it here just for illustration.
  for (int i = 0; i < view_reduce.val.size(); i++) {
    view_reduce.val(i) = 0;
  }
  for (int i = 0; i < view_reduce.val.size(); i++) {
    view_reduce.val(i) += i;
  }
  pkg->AddParam("view_reduce", view_reduce, true);

  return pkg;
}

template <typename T>
TaskStatus SetMatrixElements(T *u) {
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"poisson_sparse_matrix"});
  const auto &v = u->PackVariables(vars, imap);
  const int isp_lo = imap["poisson_sparse_matrix"].first;
  const int isp_hi = imap["poisson_sparse_matrix"].second;

  if (isp_hi < 0) { // must be using the stencil so return
    return TaskStatus::complete;
  }

  const int ndim = v.GetNdim();
  const Real w0 = -2.0 * ndim;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, v.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int n = isp_lo; n <= isp_hi; n++) {
          v(b, n, k, j, i) = 1;
        }
        v(b, isp_lo + 1, k, j, i) = w0;
      });

  return TaskStatus::complete;
}

auto &GetCoords(std::shared_ptr<MeshBlock> &pmb) { return pmb->coords; }
auto &GetCoords(Mesh *pm) { return pm->block_list[0]->coords; }

template <typename T>
TaskStatus SumMass(T *u, Real *reduce_sum) {
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"density"});
  const auto &v = u->PackVariables(vars, imap);
  const int irho = imap["density"].first;

  const parthenon::Coordinates_t &coords = GetCoords(pm);
  const int ndim = v.GetNdim();
  const Real dx = coords.Dxc<X1DIR>();
  for (int i = X2DIR; i <= ndim; i++) {
    const Real dy = coords.DxcFA(i);
    PARTHENON_REQUIRE_THROWS(dx == dy,
                             "SumMass requires that DX be equal in all directions.");
  }

  Real total;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &sum) {
        sum += v(b, irho, k, j, i) * std::pow(dx, ndim);
      },
      Kokkos::Sum<Real>(total));

  *reduce_sum += total;
  return TaskStatus::complete;
}

template <typename T>
TaskStatus SumDeltaPhi(T *du, Real *reduce_sum) {
  auto pm = du->GetParentPointer();

  IndexRange ib = du->GetBoundsI(IndexDomain::interior);
  IndexRange jb = du->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = du->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"potential"});
  const auto &dv = du->PackVariables(vars, imap);
  const int iphi = imap["potential"].first;

  Real total;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      dv.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &sum) {
        sum += std::pow(dv(b, iphi, k, j, i), 2);
      },
      Kokkos::Sum<Real>(total));

  *reduce_sum += total;
  return TaskStatus::complete;
}
template <typename T>
TaskStatus UpdatePhi(T *u, T *du) {
  using Stencil_t = parthenon::solvers::Stencil<Real>;
  PARTHENON_INSTRUMENT
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"poisson_sparse_matrix", "density", "potential"});
  const auto &v = u->PackVariables(vars, imap);
  const int isp_lo = imap["poisson_sparse_matrix"].first;
  const int isp_hi = imap["poisson_sparse_matrix"].second;
  const int irho = imap["density"].first;
  const int iphi = imap["potential"].first;
  const std::vector<std::string> phi_var({"potential"});
  PackIndexMap imap2;
  const auto &dv = du->PackVariables(phi_var, imap2);
  const int idphi = imap2["potential"].first;

  using PackType = decltype(v);

  const parthenon::Coordinates_t &coords = GetCoords(pm);
  const int ndim = v.GetNdim();
  const Real dx = coords.Dxc<X1DIR>();
  for (int i = X2DIR; i <= ndim; i++) {
    const Real dy = coords.DxcFA(i);
    PARTHENON_REQUIRE_THROWS(dx == dy,
                             "UpdatePhi requires that DX be equal in all directions.");
  }
  const Real dV = std::pow(dx, ndim);

  // Demonstrate the usage of the Stencil and SparseMatrixAccessor classes
  // which here represent the sparse matrix corresponding to a simple
  // second order finite difference discretization of the Poisson eq.
  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  if (isp_hi < 0) { // there is no sparse matrix, so we must be using the stencil
    const auto &stencil = pkg->Param<Stencil_t>("stencil");
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, v.GetDim(5) - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real rhs = dV * v(b, irho, k, j, i);
          const Real phi_new = stencil.Jacobi(v, iphi, b, k, j, i, rhs);
          dv(b, idphi, k, j, i) = phi_new - v(b, iphi, k, j, i);
        });

  } else {
    const auto &sp_accessor =
        pkg->Param<parthenon::solvers::SparseMatrixAccessor>("sparse_accessor");
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, v.GetDim(5) - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real rhs = dV * v(b, irho, k, j, i);
          const Real phi_new =
              sp_accessor.Jacobi(v, isp_lo, isp_hi, v, iphi, b, k, j, i, rhs);
          dv(b, idphi, k, j, i) = phi_new - v(b, iphi, k, j, i);
        });
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, dv.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        v(b, iphi, k, j, i) += dv(b, idphi, k, j, i);
      });

  return TaskStatus::complete;
}

template <typename T>
TaskStatus CheckConvergence(T *u, T *du) {
  PARTHENON_INSTRUMENT
  auto pm = u->GetParentPointer();

  IndexRange ib = u->GetBoundsI(IndexDomain::interior);
  IndexRange jb = u->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = u->GetBoundsK(IndexDomain::interior);

  const std::vector<std::string> vars({"potential"});
  PackIndexMap imap;
  const auto &v = u->PackVariables(vars, imap);
  const int iphi = imap["potential"].first;
  PackIndexMap imap2;
  const auto &dv = du->PackVariables(vars, imap2);
  const int idphi = imap2["potential"].first;

  Real max_err;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &eps) {
        Real reps = std::abs(dv(b, idphi, k, j, i) / v(b, iphi, k, j, i));
        Real aeps = std::abs(dv(b, idphi, k, j, i));
        eps = std::max(eps, std::min(reps, aeps));
      },
      Kokkos::Max<Real>(max_err));

  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  Real err_tol = pkg->Param<Real>("error_tolerance");

  auto status = (max_err < err_tol ? TaskStatus::complete : TaskStatus::iterate);

  return status;
}

TaskStatus PrintComplete() {
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Poisson solver complete!" << std::endl;
  }
  return TaskStatus::complete;
}

template TaskStatus CheckConvergence<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *);
template TaskStatus CheckConvergence<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                          MeshBlockData<Real> *);
template TaskStatus UpdatePhi<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *);
template TaskStatus UpdatePhi<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                   MeshBlockData<Real> *);
template TaskStatus SumDeltaPhi<MeshData<Real>>(MeshData<Real> *, Real *);
template TaskStatus SumDeltaPhi<MeshBlockData<Real>>(MeshBlockData<Real> *, Real *);
template TaskStatus SumMass<MeshData<Real>>(MeshData<Real> *, Real *);
template TaskStatus SumMass<MeshBlockData<Real>>(MeshBlockData<Real> *, Real *);
template TaskStatus SetMatrixElements<MeshData<Real>>(MeshData<Real> *);
template TaskStatus SetMatrixElements<MeshBlockData<Real>>(MeshBlockData<Real> *);

} // namespace poisson_package
