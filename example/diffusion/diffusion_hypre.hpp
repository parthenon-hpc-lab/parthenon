#ifndef EXAMPLE_DIFFUSION_DIFFUSION_HYPRE_
#define EXAMPLE_DIFFUSION_DIFFUSION_HYPRE_
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tasks/tasks.hpp"
#ifdef DIFFUSION_WITH_HYPRE

#include <array>
#include <string>
#include <vector>

#include "HYPRE_parcsr_ls.h"
#include "HYPRE_sstruct_ls.h"
#include "HYPRE_sstruct_mv.h"
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

  // ---------------------------------------------------------------------------
  // Per-block metadata cached at grid setup time (SoA layout)
  // Indexed by block list position (same order as pmesh->block_list)
  // ---------------------------------------------------------------------------
  std::vector<int> block_part;                  // Hypre part number
  std::vector<std::array<int, 3>> block_ilower; // lower corner ID (global cell index)
  std::vector<std::array<int, 3>> block_iupper; // upper corner ID (ilower + NX - 1)
  // Relative refinement level of each face neighbor per block:
  //   CellLevel::same (0), fine (+1), coarse (-1)
  //   Inner index: BoundaryFace (inner_x1=0 .. outer_x3=5)
  std::vector<std::array<parthenon::CellLevel, 6>> block_neighbor_level;
  // Whether each face is a physical (domain) boundary
  std::vector<std::array<bool, 6>> block_is_domain_boundary;

  // ---------------------------------------------------------------------------
  // Hypre object handles
  // ---------------------------------------------------------------------------
  HYPRE_SStructGrid grid = nullptr;
  HYPRE_SStructStencil stencil = nullptr;
  HYPRE_SStructGraph graph = nullptr;
  HYPRE_SStructMatrix A = nullptr;
  HYPRE_SStructVector b = nullptr; // RHS vector
  HYPRE_SStructVector x = nullptr; // solution vector
  HYPRE_Solver solver_handle = nullptr;
  HYPRE_Solver precond_handle = nullptr;

  // ---------------------------------------------------------------------------
  // Grid / solver state
  // ---------------------------------------------------------------------------
  bool grid_is_setup = false;
  bool solver_is_setup = false;
  bool needs_grid_setup = true;

  int ndim = 2;     // number of spatial dimensions (2 for this problem)
  int nparts = 0;   // number of distinct AMR levels with leaf blocks
  int nstencil = 5; // stencil size (5 for 2D, 7 for 3D)
  int min_active_level = -1;
  int max_active_level = -1;

  // Map from AMR refinement level -> Hypre part index (0-based)
  // Sized to max_level+1, indexed directly by level
  std::vector<int> level_to_part;

  // ---------------------------------------------------------------------------
  // Solver configuration (read from [hypre] input block)
  // ---------------------------------------------------------------------------
  std::string solver_type;    // "pcg" or "bicgstab"
  std::string preconditioner; // "amg" or "none"
  parthenon::Real tol;        // relative convergence tolerance
  int max_iter;               // maximum solver iterations
  int print_level;            // solver verbosity

  // BoomerAMG preconditioner settings
  int amg_coarsen_type;                 // HMIS coarsening
  int amg_interp_type;                  // ext+i interpolation
  int amg_relax_type;                   // symmetric Gauss-Seidel
  parthenon::Real amg_strong_threshold; // strong threshold (0.25 for 2D)
  int amg_num_sweeps;                   // sweeps per AMG level

  // Problem parameters cached from package
  parthenon::Real diagonal_alpha;
  std::array<parthenon::Real, 6> boundary_u{};

  // ---------------------------------------------------------------------------
  // Methods
  // ---------------------------------------------------------------------------

  // initialize all our settings from the parameter input
  HypreSolver(parthenon::ParameterInput *pin);

  // cleanup hypre objects
  ~HypreSolver();

  // adds all the meshblocks to the hypre grid via the sstruct interface
  void SetupGrid(parthenon::Mesh *pmesh);
  void DestroyGrid();
  void MarkGridDirty() { needs_grid_setup = true; }

  void SetupSolver();

  using Real = parthenon::Real;

  // calls hypre api to set matrix rows for a single mesh block
  // as well as the initial guess and rhs vectors
  static parthenon::TaskStatus
  BuildMatrixVector(HypreSolver *solver, int b, parthenon::MeshBlock *pmb, const Real dt);
  // call the hypre solve (maybe we even setup the solvers here)
  static parthenon::TaskStatus Solve(HypreSolver *solver);
  static parthenon::TaskStatus UpdateSolution(HypreSolver *solver, int b,
                                              parthenon::MeshBlock *pmb);
};

} // namespace diffusion_package
#endif // DIFFUSION_WITH_HYPRE
#endif // EXAMPLE_DIFFUSION_DIFFUSION_HYPRE_
