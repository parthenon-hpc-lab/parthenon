//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// [This code was generated with the help of generative AI]

// Solves (1 - alpha Laplacian) u = rhs for a face centered vector field u with the
// linear solvers, where rhs = A.u_exact for a chosen u_exact. This exercises the solver
// vector operations on face fields, including counting faces shared between blocks once
// in dot products and residual norms.

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/cg_solver.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_utils.hpp>

#include "face_poisson_equation.hpp"
#include "parthenon_manager.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

namespace face_poisson {

struct u : public variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION u(Ts &&...args)
      : variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "face_poisson.u"; }
};
struct exact : public variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION exact(Ts &&...args)
      : variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "face_poisson.exact"; }
};

using FaceEq = FacePoissonEquation<u>;

template <class restrictor_t>
std::shared_ptr<solvers::SolverBase> MakeSolver(const std::string &solver,
                                                ParameterInput *pin, const FaceEq &eq) {
  using prolongator_t = solvers::ProlongationBlockInteriorDefault;
  using preconditioner_t = solvers::MGSolver<FaceEq, prolongator_t, restrictor_t>;
  const std::string params = "face_poisson/solver_params";
  if (solver == "MG") {
    return std::make_shared<preconditioner_t>("base", "u", "rhs", pin, params, eq);
  } else if (solver == "CG") {
    return std::make_shared<solvers::CGSolver<FaceEq, preconditioner_t>>(
        "base", "u", "rhs", pin, params, eq);
  } else if (solver == "BiCGSTAB") {
    return std::make_shared<solvers::BiCGSTABSolver<FaceEq, preconditioner_t>>(
        "base", "u", "rhs", pin, params, eq);
  }
  PARTHENON_FAIL("Unknown solver type " + solver + ".");
  return nullptr;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("face_poisson");

  std::string solver = pin->GetOrAddString("face_poisson", "solver", "MG");
  pkg->AddParam<>("solver", solver);

  FaceEq eq(pin, "face_poisson");
  pkg->AddParam<>("equation", eq, Params::Mutability::Mutable);

  // Multigrid restriction, either full weighting or the restriction operator registered
  // in the field metadata
  const std::string restriction =
      pin->GetOrAddString("face_poisson", "restriction", "FullWeighting");
  std::shared_ptr<solvers::SolverBase> psolver;
  if (restriction == "FullWeighting") {
    psolver = MakeSolver<solvers::RestrictionFullWeighting>(solver, pin, eq);
  } else if (restriction == "MetadataDefault") {
    psolver = MakeSolver<solvers::RestrictionDefault>(solver, pin, eq);
  } else {
    PARTHENON_FAIL("Unknown restriction " + restriction + ".");
  }
  pkg->AddParam<>("solver_pointer", psolver);

  using namespace refinement_ops;
  auto m = Metadata({Metadata::Face, Metadata::Independent, Metadata::FillGhost,
                     Metadata::GMGRestrict, Metadata::GMGProlongate});
  const std::string internal =
      pin->GetOrAddString("face_poisson", "internal_prolongation", "Average");
  if (internal == "Average") {
    m.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage,
                            ProlongateInternalAverage>();
  } else if (internal == "TothAndRoe") {
    m.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage,
                            ProlongateInternalTothAndRoe>();
  } else {
    PARTHENON_FAIL("Unknown internal prolongation " + internal + ".");
  }
  pkg->AddField<u>(m);
  pkg->AddField<exact>(Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy}));
  return pkg;
}

// Set u = u_exact, a smooth periodic field on [-1, 1]^ndim with different components
template <TopologicalElement te, class pack_t>
void SetExactElement(MeshData<Real> *md, const pack_t &pack, const Real phase) {
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
  parthenon::par_for(
      "FacePoisson::ProblemGenerator", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = pack.GetCoordinates(b);
        const Real x = coords.template X<1, te>(k, j, i);
        const Real y = coords.template X<2, te>(k, j, i);
        const Real z = coords.template X<3, te>(k, j, i);
        Real val = std::sin(M_PI * x + phase);
        if (ndim > 1) val *= std::cos(M_PI * y + 2.0 * phase) + 0.5;
        if (ndim > 2) val *= std::sin(2.0 * M_PI * z + 3.0 * phase) + 2.0;
        pack(b, te, exact(), k, j, i) = val;
        pack(b, te, u(), k, j, i) = val;
      });
}

void ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
  auto desc = MakePackDescriptor<u, exact>(md);
  auto pack = desc.GetPack(md);
  using TE = TopologicalElement;
  SetExactElement<TE::F1>(md, pack, 0.3);
  SetExactElement<TE::F2>(md, pack, 1.1);
  SetExactElement<TE::F3>(md, pack, 2.0);
}

// Store exact - u in exact on the interior faces
TaskStatus StoreError(std::shared_ptr<MeshData<Real>> &md) {
  using TE = TopologicalElement;
  auto desc = MakePackDescriptor<u, exact>(md.get());
  auto pack = desc.GetPack(md.get());
  for (const auto te : {TE::F1, TE::F2, TE::F3}) {
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
    parthenon::par_for(
        "FacePoisson::StoreError", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack(b, te, exact(), k, j, i) -= pack(b, te, u(), k, j, i);
        });
  }
  return TaskStatus::complete;
}

class FacePoissonDriver : public Driver {
 public:
  FacePoissonDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm) {}

  DriverStatus Execute() override {
    ConstructAndExecuteTaskLists<>(this);
    return DriverStatus::complete;
  }

  TaskCollection MakeTaskCollection(BlockList_t &blocks) {
    TaskCollection tc;
    TaskID none(0);

    auto pkg = pmesh->packages.Get("face_poisson");
    auto psolver = pkg->Param<std::shared_ptr<solvers::SolverBase>>("solver_pointer");
    auto *eq = pkg->MutableParam<FaceEq>("equation");
    using TL = TypeList<u>;

    auto partitions = pmesh->GetDefaultBlockPartitions();
    const int num_partitions = partitions.size();
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      TaskList &tl = region[i];
      auto &md = pmesh->mesh_data.Add("base", partitions[i]);
      auto &md_u = pmesh->mesh_data.Add("u", md, {u::name()});
      auto &md_rhs = pmesh->mesh_data.Add("rhs", md, {u::name()});

      // Set rhs = A.u_exact
      auto copy_u = tl.AddTask(none, TF(solvers::utils::CopyData<TL>), md, md_u);
      auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(copy_u, tl, md_u,
                                                              pmesh->multilevel);
      auto get_rhs = eq->Ax(tl, comm, md, md_u, md_rhs);

      // Solve A.u = rhs from u = 0
      auto zero_u = tl.AddTask(get_rhs, TF(solvers::utils::SetToZero<TL>), md_u);
      auto setup = psolver->AddSetupTasks(tl, zero_u, i, pmesh);
      auto solve = psolver->AddTasks(tl, setup, i, pmesh);

      // Compare to the exact solution
      auto copy_back = tl.AddTask(solve, TF(solvers::utils::CopyData<TL>), md_u, md);
      auto diff = tl.AddTask(copy_back, StoreError, md);
      auto count = solvers::utils::GlobalElementCount<TL>(diff, tl, &nelements, md);
      auto get_err = solvers::utils::DotProduct<TypeList<exact>>(count, tl, &err, md, md);
      tl.AddTask(
          get_err,
          [](FacePoissonDriver *driver, int partition) {
            if (partition != 0 || Globals::my_rank != 0) return TaskStatus::complete;
            printf("Owned faces: %.0f\n", driver->nelements.val);
            printf("Final rms error: %e\n",
                   std::sqrt(driver->err.val / driver->nelements.val));
            return TaskStatus::complete;
          },
          this, i);
    }
    return tc;
  }

 private:
  AllReduce<Real> err, nelements;
};

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages.Add(Initialize(pin.get()));
  return packages;
}

} // namespace face_poisson

int main(int argc, char *argv[]) {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = face_poisson::ProcessPackages;
  pman.app_input->MeshProblemGenerator = face_poisson::ProblemGenerator;

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  pman.ParthenonInitPackagesAndMesh();
  {
    face_poisson::FacePoissonDriver driver(pman.pinput.get(), pman.app_input.get(),
                                           pman.pmesh.get());
    driver.Execute();
  }
  pman.ParthenonFinalize();
  return 0;
}
