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

#include <algorithm>
#include <cmath>
#include <cstdio>
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
#include "kokkos_abstraction.hpp"
#include "poisson_nodal_equation.hpp"
#include "poisson_nodal_package.hpp"

using namespace parthenon::package::prelude;
using parthenon::HostArray1D;
namespace poisson_nodal_package {

using namespace parthenon;
using namespace parthenon::BoundaryFunction;
// We need to register FixedFace boundary conditions by hand since they can't
// be chosen in the parameter input file. FixedFace boundary conditions assume
// Dirichlet booundary conditions on the face of the domain and linearly extrapolate
// into the ghosts to ensure the linear reconstruction on the block face obeys the
// chosen boundary condition. Just setting the ghost zones of CC variables to a fixed
// value results in poor MG convergence because the effective BC at the face
// changes with MG level.

// Build type that selects only variables within the poisson_nodal namespace. Internal
// solver variables have the namespace of input variables prepended, so they will also be
// selected by this type.
struct any_poisson_nodal : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_poisson_nodal(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "poisson_nodal[.].*"; }
};

template <CoordinateDirection DIR, BCSide SIDE>
auto GetBC() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, any_poisson_nodal>(rc, coarse, 0.0);
  };
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_nodal_package");

  // Set boundary conditions for Poisson variables
  using BF = parthenon::BoundaryFace;
  pkg->UserBoundaryFunctions[BF::inner_x1].push_back(GetBC<X1DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x2].push_back(GetBC<X2DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::inner_x3].push_back(GetBC<X3DIR, BCSide::Inner>());
  pkg->UserBoundaryFunctions[BF::outer_x1].push_back(GetBC<X1DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x2].push_back(GetBC<X2DIR, BCSide::Outer>());
  pkg->UserBoundaryFunctions[BF::outer_x3].push_back(GetBC<X3DIR, BCSide::Outer>());

  Real diagonal_alpha = pin->GetOrAddReal("poisson_nodal", "diagonal_alpha", 0.0);
  pkg->AddParam<>("diagonal_alpha", diagonal_alpha);

  std::string solver = pin->GetOrAddString("poisson_nodal", "solver", "MG");
  pkg->AddParam<>("solver", solver);

  bool use_exact_rhs = pin->GetOrAddBoolean("poisson_nodal", "use_exact_rhs", false);
  pkg->AddParam<>("use_exact_rhs", use_exact_rhs);

  std::string prolong =
      pin->GetOrAddString("poisson_nodal", "boundary_prolongation", "Linear");

  using PoissEq = poisson_nodal_package::PoissonEquation<u>;
  PoissEq eq(pin, "poisson_nodal");
  pkg->AddParam<>("poisson_nodal_equation", eq, parthenon::Params::Mutability::Mutable);

  std::shared_ptr<parthenon::solvers::SolverBase> psolver;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorDefault;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;

  const std::string base_label = "base";
  const std::string u_label = "nodal_u";
  const std::string rhs_label = "nodal_rhs";
  if (solver == "MG") {
    psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
        base_label, u_label, rhs_label, pin, "poisson_nodal/solver_params",
        PoissEq(pin, "poisson_nodal"));
  } else if (solver == "CG") {
    psolver = std::make_shared<parthenon::solvers::CGSolver<PoissEq, preconditioner_t>>(
        base_label, u_label, rhs_label, pin, "poisson_nodal/solver_params",
        PoissEq(pin, "poisson_nodal"));
  } else if (solver == "BiCGSTAB") {
    psolver =
        std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            base_label, u_label, rhs_label, pin, "poisson_nodal/solver_params",
            PoissEq(pin, "poisson_nodal"));
  } else {
    PARTHENON_FAIL("Unknown solver type.");
  }
  pkg->AddParam<>("solver_pointer", psolver);

  using namespace parthenon::refinement_ops;

  std::vector<MetadataFlag> flags{Metadata::Node,        Metadata::Independent,
                                  Metadata::FillGhost,   Metadata::WithFluxes,
                                  Metadata::GMGRestrict, Metadata::GMGProlongate};
  auto mflux_comm = Metadata(flags);
  if (prolong == "Linear") {
    mflux_comm.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
  } else if (prolong == "Constant") {
    mflux_comm.RegisterRefinementOps<ProlongatePiecewiseConstant, RestrictAverage>();
  } else {
    PARTHENON_FAIL("Unknown prolongation method for Poisson boundaries.");
  }
  // u is the solution vector that starts with an initial guess and then gets updated
  // by the solver
  pkg->AddField(u::name(), mflux_comm);

  auto m_no_ghost = Metadata({Metadata::Node, Metadata::Derived, Metadata::OneCopy});
  // rhs is the field that contains the desired rhs side
  pkg->AddField(rhs::name(), m_no_ghost);

  // Auxillary field for storing the exact solution when it is known
  pkg->AddField(exact::name(), m_no_ghost);

  return pkg;
}

parthenon::TaskStatus
SetVector(parthenon::ParameterInput *pin, bool use_exponential,
          std::shared_ptr<parthenon::MeshData<parthenon::Real>> md) {
  using namespace parthenon;
  Real x0 = pin->GetOrAddReal("poisson_nodal", "x0", 0.0);
  Real y0 = pin->GetOrAddReal("poisson_nodal", "y0", 0.0);
  Real z0 = pin->GetOrAddReal("poisson_nodal", "z0", 0.0);
  Real radius0 = pin->GetOrAddReal("poisson_nodal", "radius", 0.1);
  const int ndim = md->GetMeshPointer()->ndim;

  auto desc = MakePackDescriptor<u>(md.get());
  auto pack = desc.GetPack(md.get());

  using TE = parthenon::TopologicalElement;
  auto ib = md->GetBoundsI(IndexDomain::entire, TE::NN);
  auto jb = md->GetBoundsJ(IndexDomain::entire, TE::NN);
  auto kb = md->GetBoundsK(IndexDomain::entire, TE::NN);

  parthenon::par_for(
      "PoissonNodal::Ax", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        Real x1 = coords.X<1, TE::NN>(i);
        Real x2 = coords.X<2, TE::NN>(j);
        Real x3 = coords.X<2, TE::NN>(k);
        Real rad = (x1 - x0) * (x1 - x0);
        if (ndim > 1) rad += (x2 - y0) * (x2 - y0);
        if (ndim > 2) rad += (x3 - z0) * (x3 - z0);
        rad = std::sqrt(rad);

        pack(b, TE::NN, u(), k, j, i) = rad < radius0 ? 1.0 : 0.0;
        if (use_exponential) pack(b, TE::NN, u(), k, j, i) = -exp(-10.0 * rad * rad);
      });
  return TaskStatus::complete;
}

void AddTaskRegion(parthenon::TaskCollection &tc,
                   linear_solver_example::LinearSolverDriver *driver) {
  using namespace parthenon;
  using namespace poisson_nodal_package;
  auto pmesh = driver->pmesh;
  auto pinput = driver->pinput;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("poisson_nodal_package");
  auto use_exact_rhs = pkg->Param<bool>("use_exact_rhs");
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];
    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    auto &field_labels = psolver->GetFieldLabels();
    auto &md_u =
        pmesh->mesh_data.Add(psolver->GetSolutionContainerLabel(), md, field_labels);
    auto &md_rhs =
        pmesh->mesh_data.Add(psolver->GetRHSContainerLabel(), md, field_labels);
    auto &md_exact = pmesh->mesh_data.Add("exact_nodal", md, field_labels);

    // set the rhs
    auto set_rhs = tl.AddTask(none, SetVector, pinput, false, md_rhs);

    // Possibly set rhs <- A.u_exact for a given u_exact so that the exact solution is
    // known when we solve A.u = rhs
    if (use_exact_rhs) {
      auto set_exact = tl.AddTask(set_rhs, SetVector, pinput, true, md_exact);
      auto comm =
          AddBoundaryExchangeTasks<BoundaryType::any>(set_exact, tl, md_exact, true);
      set_rhs = psolver->Ax(tl, comm, md, md_exact, md_rhs);
    }

    // Set initial solution guess to zero
    auto zero_u = tl.AddTask(set_rhs, TF(solvers::utils::SetToZero<u>), md_u);
    auto setup = psolver->AddSetupTasks(tl, zero_u, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // If we are using a rhs to which we know the exact solution, compare our computed
    // solution to the exact solution
    if (use_exact_rhs) {
      auto diff = tl.AddTask(solve, solvers::utils::AddFieldsAndStore<TypeList<u>>,
                             md_exact, md_u, md_exact, 1.0, -1.0);
      auto get_err = solvers::utils::DotProduct<TypeList<u>>(diff, tl, &(driver->err),
                                                             md_exact, md_exact);
      tl.AddTask(
          get_err,
          [](linear_solver_example::LinearSolverDriver *driver, int partition,
             std::shared_ptr<parthenon::solvers::SolverBase> psolver) {
            if (partition != 0) return TaskStatus::complete;
            driver->final_rms_error["nodal_poisson"] =
                std::sqrt(driver->err.val / driver->pmesh->GetTotalCells());
            driver->final_rms_residual["nodal_poisson"] = psolver->GetFinalResidual();
            if (Globals::my_rank == 0)
              printf("Final residual: %e\n", driver->final_rms_residual["nodal_poisson"]);
            printf("Final rms error: %e\n", driver->final_rms_error["nodal_poisson"]);
            return TaskStatus::complete;
          },
          driver, i, psolver);
    } else {
      tl.AddTask(
          solve,
          [](linear_solver_example::LinearSolverDriver *driver, int partition,
             std::shared_ptr<parthenon::solvers::SolverBase> psolver) {
            if (partition != 0) return TaskStatus::complete;
            driver->final_rms_error["nodal_poisson"] = 0.0;
            driver->final_rms_residual["nodal_poisson"] = psolver->GetFinalResidual();
            if (Globals::my_rank == 0)
              printf("Final residual: %e\n", driver->final_rms_residual["nodal_poisson"]);
            return TaskStatus::complete;
          },
          driver, i, psolver);
    }
  }
}

} // namespace poisson_nodal_package
