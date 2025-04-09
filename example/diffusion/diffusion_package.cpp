//========================================================================================
// (C) (or copyright) 2021-2025. Triad National Security, LLC. All rights reserved.
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
#include <utility>
#include <vector>

#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/solver_utils.hpp>

#include "defs.hpp"
#include "diffusion_equation.hpp"
#include "diffusion_package.hpp"
#include "kokkos_abstraction.hpp"

/**
 * @brief Time dependent implicit diffusion solver with non-constant coefficients
 *
 * Solve time dependent diffusion with a time implicit Backwards Euler method:
 *
 *   \frac{u^{n+1} - u^{n}}{\Delta t} = \nabla \cdot D^{n} \nabla u^{n+1}
 *
 * The initial profile is a Gaussian:
 *   phi(r,t) = sqrt(t0 / (t + t0)) exp(-r^2 / 4 * D * (t + t0))
 *   with the initial condition specified with t=0 and D=1.
 * For constant coefficient diffusion, the above is also the analytical solution.
 *
 * Note: for convenience, dt is absorbed into the diffusion coefficients
 *
 * @param solver the solver to use (CG, MG, BiCGSTAB)
 * @param constant_coefficient use constant diffusion coefficient D=1
 * @param t0 reference time for the initial Gaussian profile (close to 0)
 **/

using namespace parthenon::package::prelude;
using parthenon::HostArray1D;
namespace diffusion_package {

using namespace parthenon;
using namespace parthenon::BoundaryFunction;
// We need to register FixedFace boundary conditions by hand since they can't
// be chosen in the parameter input file. FixedFace boundary conditions assume
// Dirichlet booundary conditions on the face of the domain and linearly extrapolate
// into the ghosts to ensure the linear reconstruction on the block face obeys the
// chosen boundary condition. Just setting the ghost zones of CC variables to a fixed
// value results in poor MG convergence because the effective BC at the face
// changes with MG level.

// Build type that selects only variables within the diffusion namespace. Internal solver
// variables have the namespace of input variables prepended, so they will also be
// selected by this type.
struct any_diffusion : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_diffusion(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "diffusion[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_diffusion>(rc, coarse, 0.0);
  };
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("diffusion_package");

  // Set boundary conditions for Diffusion variables
  using BF = parthenon::BoundaryFace;
  pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBC<X1DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBC<X2DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBC<X3DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBC<X1DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBC<X2DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBC<X3DIR, BCSide::Outer>());

  // probably should stay 1.0
  Real diagonal_alpha = pin->GetOrAddReal("diffusion", "diagonal_alpha", 1.0);
  pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

  Real cfl = pin->GetOrAddReal("diffusion", "cfl", 1.0);
  pkg->AddParam<>("cfl", cfl);
  pkg->AddParam<>("dt", 1.0,
                  parthenon::Params::Mutability::Mutable); // hold timestep for controls

  Real t0 = pin->GetOrAddReal("diffusion", "t0", 0.001);
  pkg->AddParam<>("t0", t0);

  bool constant_coeff = pin->GetOrAddBoolean("diffusion", "constant_coefficient", true);
  pkg->AddParam<>("constant_coefficient", constant_coeff);

  std::string solver = pin->GetOrAddString("diffusion", "solver", "MG");
  pkg->AddParam<>("solver", solver);

  std::string prolong =
      pin->GetOrAddString("diffusion", "boundary_prolongation", "Linear");

  using PoissEq = diffusion_package::DiffusionEquation<u, D>;
  PoissEq eq(pin, "diffusion");
  pkg->AddParam<>("diffusion_equation", eq, parthenon::Params::Mutability::Mutable);

  std::shared_ptr<parthenon::solvers::SolverBase> psolver;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorDefault;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
  if (solver == "MG") {
    psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
        "base", "u", "rhs", pin, "diffusion/solver_params", PoissEq(pin, "diffusion"));
  } else if (solver == "CG") {
    psolver = std::make_shared<parthenon::solvers::CGSolver<PoissEq, preconditioner_t>>(
        "base", "u", "rhs", pin, "diffusion/solver_params", PoissEq(pin, "diffusion"));
  } else if (solver == "BiCGSTAB") {
    psolver =
        std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            "base", "u", "rhs", pin, "diffusion/solver_params",
            PoissEq(pin, "diffusion"));
  } else {
    PARTHENON_FAIL("Unknown solver type.");
  }
  pkg->AddParam<>("solver_pointer", psolver);

  using namespace parthenon::refinement_ops;
  auto mD = Metadata({Metadata::Independent, Metadata::OneCopy, Metadata::Face,
                      Metadata::GMGRestrict, Metadata::FillGhost});
  mD.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();

  // Holds the discretized version of D in \nabla \cdot D(\vec{x}) \nabla u = rhs. D = 1
  // for the standard Diffusion equation.
  pkg->AddField<D>(mD);

  std::vector<MetadataFlag> flags{Metadata::Cell,        Metadata::Independent,
                                  Metadata::FillGhost,   Metadata::WithFluxes,
                                  Metadata::GMGRestrict, Metadata::GMGProlongate};
  auto mflux_comm = Metadata(flags);
  if (prolong == "Linear") {
    mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  } else if (prolong == "Constant") {
    mflux_comm.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  } else {
    PARTHENON_FAIL("Unknown prolongation method for Diffusion boundaries.");
  }
  // u is the solution vector that starts with an initial guess and then gets updated
  // by the solver
  pkg->AddField<u>(mflux_comm);

  // timestep
  pkg->EstimateTimestepMesh = EstimateTimestep;

  return pkg;
}

// Maybe overkill, explicit timestep constraint. But stable with CFL > 1.0
// dt = CFL/2 * min(dx^2 / D)
Real EstimateTimestep(MeshData<Real> *md) {
  std::shared_ptr<StateDescriptor> pkg =
      md->GetMeshPointer()->packages.Get("diffusion_package");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &old_dt = pkg->Param<Real>("dt");

  auto desc = parthenon::MakePackDescriptor<diffusion_package::D>(md);
  auto pack = desc.GetPack(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  const int ndim = md->GetMeshPointer()->ndim;

  constexpr static Real ONE_FOURTH = 0.25;

  Real min_dt;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmin_dt) {
        auto &coords = pack.GetCoordinates(b);
        const Real dx1 = coords.Dxc<X1DIR>(k, j, i);
        const Real dx2 = coords.Dxc<X2DIR>(k, j, i);
        const Real dx3 = coords.Dxc<X3DIR>(k, j, i);

        // average face coefficients and divide by 2
        const Real d1 = (pack(b, TE::F1, diffusion_package::D(), k, j, i + 1) +
                         pack(b, TE::F1, diffusion_package::D(), k, j, i)) *
                        ONE_FOURTH;
        lmin_dt = std::min(lmin_dt, dx1 * dx1 / d1);
        if (ndim > 1) {
          const Real d2 = (pack(b, TE::F2, diffusion_package::D(), k, j + 1, i) +
                           pack(b, TE::F2, diffusion_package::D(), k, j, i)) *
                          ONE_FOURTH;
          lmin_dt = std::min(lmin_dt, dx2 * dx2 / d2);
        }
        if (ndim > 2) {
          const Real d3 = (pack(b, TE::F3, diffusion_package::D(), k + 1, j, i) +
                           pack(b, TE::F3, diffusion_package::D(), k, j, i)) *
                          ONE_FOURTH;
          lmin_dt = std::min(lmin_dt, dx3 * dx3 / d3);
        }
      },
      Kokkos::Min<Real>(min_dt));

  const Real new_dt = cfl * min_dt * old_dt; // need to scale by dt as D := D * dt
  pkg->UpdateParam<Real>("dt", new_dt);
  return new_dt;
} // EstimateTimestep

parthenon::TaskStatus SetRHS(std::shared_ptr<MeshData<Real>> md,
                             std::shared_ptr<MeshData<Real>> md_rhs) {
  auto pkg = md->GetMeshPointer()->packages.Get("diffusion_package");
  const auto alpha = pkg->Param<Real>("diagonal_alpha");
  auto desc = parthenon::MakePackDescriptor<diffusion_package::u>(md.get());
  auto pack = desc.GetPack(md.get());

  // holds rhs
  auto desc_rhs = parthenon::MakePackDescriptor<diffusion_package::u>(md_rhs.get());
  auto pack_rhs = desc_rhs.GetPack(md_rhs.get());

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      "SetRHS", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        pack_rhs(b, diffusion_package::u(), k, j, i) = // rhs
            -alpha * pack(b, diffusion_package::u(), k, j, i);
      });
  return TaskStatus::complete;
} // SetRHS

using TE = parthenon::TopologicalElement;
parthenon::TaskStatus SetDiffusionCoefficient(std::shared_ptr<MeshData<Real>> md,
                                              const Real dt) {
  const int ndim = md->GetMeshPointer()->ndim;
  auto pkg = md->GetMeshPointer()->packages.Get("diffusion_package");
  auto desc =
      parthenon::MakePackDescriptor<diffusion_package::u, diffusion_package::D>(md.get());
  auto pack = desc.GetPack(md.get());

  const bool constant_coeff = pkg->Param<bool>("constant_coefficient");

  for (auto te : {TE::F1, TE::F2, TE::F3}) {
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    int offset_x1 = te == TE::F1 ? 1 : 0;
    int offset_x2 = te == TE::F2 && (ndim > 1) ? 1 : 0;
    int offset_x3 = te == TE::F3 && (ndim > 2) ? 1 : 0;

    parthenon::par_for(
        "SetDiffusionCoefficient", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const Real u = 0.5 * (pack(b, TE::CC, diffusion_package::u(), k - offset_x3,
                                     j - offset_x2, i - offset_x1) +
                                pack(b, TE::CC, diffusion_package::u(), k, j, i));
          if (constant_coeff) {
            pack(b, te, diffusion_package::D(), k, j, i) = 1.0 * dt;
          } else {
            pack(b, te, diffusion_package::D(), k, j, i) =
                10.0 * std::sqrt(std::fabs(u)) * dt;
          }
        });
  }
  return TaskStatus::complete;
} // SetRHS

} // namespace diffusion_package
