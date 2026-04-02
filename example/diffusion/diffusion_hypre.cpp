#include "diffusion_hypre.hpp"
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

namespace diffusion_package {
using Real = parthenon::Real;

HypreSolver::HypreSolver(parthenon::ParameterInput *pin) {}

parthenon::TaskStatus HypreSolver::BuildMatrixVector(HypreSolver *solver,
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

parthenon::TaskStatus HypreSolver::UpdateSolution(HypreSolver *solver,
                                                  parthenon::MeshBlock *pmb) {
  return parthenon::TaskStatus::complete;
}

void HypreSolver::SetupGrid(parthenon::Mesh *pmesh) {
  // need to add all of the meshblocks in our mesh to the hypre grid using its block
  // extents. also build the 5 point stencil in the dimensions of our problem also add
  // graphs between blocks at different refinement levels that share a face.
}

} // namespace diffusion_package
