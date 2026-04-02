#ifndef EXAMPLE_DIFFUSION_DIFFUSION_HYPRE_
#define EXAMPLE_DIFFUSION_DIFFUSION_HYPRE_
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tasks/tasks.hpp"
#ifdef DIFFUSION_WITH_HYPRE

#include "parameter_input.hpp"

namespace diffusion_package {

struct HypreSolver {
  // class to hold information we need for the hypre solves
  // main hypre data members:
  //    * hypre solvers -- preconditioner and actual solver
  //    * hypre matrix  -- matrix we will build, A, for solving Ax=b
  //    * hypre vectors -- solution (x) and rhs (b)
  //    * hypre grid -- we will use the SStruct interface to map parthenon's tree mesh to
  //    hypre
  //    * hypre stencil -- used for same level couplings
  //
  // we will also require some cached information about the grid for mapping parthenon
  // meshblocks to thier sstruct hypre counterparts
  //
  // * Basic idea is to treat each AMR refinement level as a unique hypre part.
  // * We use parthenon's legacy logical location to map blocks to unique locations on the
  // part
  //   using a corner ID that maps the meshblock's lower left corner cell to the (k,j,i)
  //   index of that cell if the entire domain was at that meshblock's refinement level
  // * we use stencil couplings within a meshblock as well as between meshblocks at the
  // same level
  // * at F-C boundaries we create graph entries that map to our coarse/fine neighbors to
  // replace the stencil
  //   couplings that would have gone to the other meshblock
  //
  // per meshblock we will need
  //    * part number + (k,j,i) corner ID
  //    * neighbor information (relative refinement level of our face neighbors), can use
  //    the parthenon::CellLevel enum

  // initialize all our settings from the parameter input
  HypreSolver(parthenon::ParameterInput *pin);

  // adds all the meshblocks to the hypre grid via the sstruct interface
  void SetupGrid(parthenon::Mesh *pmesh);

  void SetupSolver();

  using Real = parthenon::Real;

  // calls hypre api to set matrix rows for a single mesh block
  // as well as the initial guess and rhs vectors
  static parthenon::TaskStatus
  BuildMatrixVector(HypreSolver *solver, parthenon::MeshBlock *pmb, const Real dt);
  // call the hypre solve (maybe we even setup the solvers here)
  static parthenon::TaskStatus Solve(HypreSolver *solver);
  static parthenon::TaskStatus UpdateSolution(HypreSolver *solver,
                                              parthenon::MeshBlock *pmb);
};

} // namespace diffusion_package
#endif // DIFFUSION_WITH_HYPRE
#endif // EXAMPLE_DIFFUSION_DIFFUSION_HYPRE_
