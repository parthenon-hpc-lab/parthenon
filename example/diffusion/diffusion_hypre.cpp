#include "diffusion_hypre.hpp"
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

namespace diffusion_package {
using Real = parthenon::Real;

HypreSolver::HypreSolver(parthenon::ParameterInput *pin) {
  // Solver type
  solver_type = pin->GetOrAddString("hypre", "solver_type", "pcg",
                                    "Hypre outer solver: pcg or bicgstab");
  tol = pin->GetOrAddReal("hypre", "tol", 1e-12, "Relative convergence tolerance");
  max_iter = pin->GetOrAddInteger("hypre", "max_iter", 50, "Maximum solver iterations");
  print_level = pin->GetOrAddInteger("hypre", "print_level", 1, "Solver print verbosity");

  // BoomerAMG preconditioner settings
  amg_coarsen_type =
      pin->GetOrAddInteger("hypre", "amg_coarsen_type", 10, "AMG coarsening type (HMIS)");
  amg_interp_type = pin->GetOrAddInteger("hypre", "amg_interp_type", 6,
                                         "AMG interpolation type (ext+i)");
  amg_relax_type = pin->GetOrAddInteger("hypre", "amg_relax_type", 6,
                                        "AMG relaxation type (symmetric GS)");
  amg_strong_threshold = pin->GetOrAddReal("hypre", "amg_strong_threshold", 0.25,
                                           "AMG strong threshold (0.25 for 2D)");
  amg_num_sweeps =
      pin->GetOrAddInteger("hypre", "amg_num_sweeps", 1, "AMG sweeps per level");

  // Cache problem parameters
  diagonal_alpha = pin->GetReal("diffusion", "diagonal_alpha");

  // Determine dimensionality
  const int nx3 = pin->GetInteger("parthenon/mesh", "nx3");
  ndim = (nx3 > 1) ? 3 : 2;
  nstencil = (ndim == 2) ? 5 : 7;
}

HypreSolver::~HypreSolver() {
  if (solver_handle) {
    if (solver_type == "pcg") {
      HYPRE_ParCSRPCGDestroy(solver_handle);
    } else {
      HYPRE_ParCSRBiCGSTABDestroy(solver_handle);
    }
    solver_handle = nullptr;
  }
  if (precond_handle) {
    HYPRE_BoomerAMGDestroy(precond_handle);
    precond_handle = nullptr;
  }
  if (A) {
    HYPRE_SStructMatrixDestroy(A);
    A = nullptr;
  }
  if (b) {
    HYPRE_SStructVectorDestroy(b);
    b = nullptr;
  }
  if (x) {
    HYPRE_SStructVectorDestroy(x);
    x = nullptr;
  }
  if (graph) {
    HYPRE_SStructGraphDestroy(graph);
    graph = nullptr;
  }
  if (stencil) {
    HYPRE_SStructStencilDestroy(stencil);
    stencil = nullptr;
  }
  if (grid) {
    HYPRE_SStructGridDestroy(grid);
    grid = nullptr;
  }
}

parthenon::TaskStatus HypreSolver::BuildMatrixVector(HypreSolver *solver, int b,
                                                     parthenon::MeshBlock *pmb,
                                                     const Real dt) {
  // also ad diagnoal term dt
  // uses simple finite difference formula for stencil coefficients
  // dx(D dx u) ~ ( D_i+1/2 *( u_+ - u_0) - D_i-1/2 * (u_0 - u_+) )* 1 / dx^2
  // for each dimension

  // at fine coarse boundaries we zero the stencils at the mesh block face and set the
  // graph entries
  // according to the 1/3 2/3 rule for the flux at the face, using D from the fine cells
  // note that when we are the coarser block should take this from the corresponding flux
  // that points to the fine cell we are adding the graph entry to
  //
  // This also sets the RHS and initial guess vectors. We will also need to check for
  // domain boundaries and appropriately adjust the matrix and rhs vector accordingly
  return parthenon::TaskStatus::complete;
}

parthenon::TaskStatus HypreSolver::Solve(HypreSolver *solver) {
  return parthenon::TaskStatus::complete;
}

parthenon::TaskStatus HypreSolver::UpdateSolution(HypreSolver *solver, int b,
                                                  parthenon::MeshBlock *pmb) {
  return parthenon::TaskStatus::complete;
}

void HypreSolver::SetupSolver() {}

void HypreSolver::SetupGrid(parthenon::Mesh *pmesh) {
  // need to add all of the meshblocks in our mesh to the hypre grid using its block
  // extents. also build the 5 point stencil in the dimensions of our problem also add
  // graphs between blocks at different refinement levels that share a face.
}

} // namespace diffusion_package
