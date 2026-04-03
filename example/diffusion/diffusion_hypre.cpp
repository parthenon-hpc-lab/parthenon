#include "diffusion_hypre.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"

namespace diffusion_package {
using Real = parthenon::Real;

namespace {

int FaceFromOffsets(const parthenon::CellCentOffsets &ofs) {
  if (ofs(parthenon::X1DIR) == -1) return parthenon::BoundaryFace::inner_x1;
  if (ofs(parthenon::X1DIR) == 1) return parthenon::BoundaryFace::outer_x1;
  if (ofs(parthenon::X2DIR) == -1) return parthenon::BoundaryFace::inner_x2;
  if (ofs(parthenon::X2DIR) == 1) return parthenon::BoundaryFace::outer_x2;
  if (ofs(parthenon::X3DIR) == -1) return parthenon::BoundaryFace::inner_x3;
  if (ofs(parthenon::X3DIR) == 1) return parthenon::BoundaryFace::outer_x3;
  return parthenon::BoundaryFace::undef;
}

int FaceAxis(const int face) {
  if (face == parthenon::BoundaryFace::inner_x1 ||
      face == parthenon::BoundaryFace::outer_x1) {
    return 0;
  }
  if (face == parthenon::BoundaryFace::inner_x2 ||
      face == parthenon::BoundaryFace::outer_x2) {
    return 1;
  }
  if (face == parthenon::BoundaryFace::inner_x3 ||
      face == parthenon::BoundaryFace::outer_x3) {
    return 2;
  }
  return -1;
}

int FaceSide(const int face) {
  if (face == parthenon::BoundaryFace::inner_x1 ||
      face == parthenon::BoundaryFace::inner_x2 ||
      face == parthenon::BoundaryFace::inner_x3) {
    return -1;
  }
  if (face == parthenon::BoundaryFace::outer_x1 ||
      face == parthenon::BoundaryFace::outer_x2 ||
      face == parthenon::BoundaryFace::outer_x3) {
    return 1;
  }
  return 0;
}

} // namespace

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
  nstencil = (ndim == 1) ? 3 : (ndim == 2) ? 5 : 7;
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
  if (grid_is_setup) return;

  auto &blocks = pmesh->block_list;
  const int nblocks = static_cast<int>(blocks.size());

  block_part.resize(nblocks, -1);
  block_ilower.resize(nblocks);
  block_iupper.resize(nblocks);
  block_neighbor_level.resize(nblocks);
  block_is_domain_boundary.resize(nblocks);

  // Determine globally active refinement levels.
  int local_max_level = -1;
  int local_min_level = std::numeric_limits<int>::max();
  for (const auto &pmb : blocks) {
    const int lev = static_cast<int>(pmb->loc.level());
    local_max_level = std::max(local_max_level, lev);
    local_min_level = std::min(local_min_level, lev);
  }

  int max_level = -1;
  int min_level = std::numeric_limits<int>::max();
  MPI_Allreduce(&local_max_level, &max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_min_level, &min_level, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  std::vector<int> local_level_present(std::max(max_level + 1, 0), 0);
  for (const auto &pmb : blocks) {
    local_level_present[static_cast<int>(pmb->loc.level())] = 1;
  }

  std::vector<int> global_level_present(std::max(max_level + 1, 0), 0);
  if (max_level >= 0) {
    MPI_Allreduce(local_level_present.data(), global_level_present.data(), max_level + 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }

  // map from refinement level to hypre part
  level_to_part.assign(std::max(max_level + 1, 0), -1);
  std::vector<int> part_to_level;
  nparts = 0;
  min_active_level = -1;
  max_active_level = -1;
  for (int lev = 0; lev <= max_level; ++lev) {
    if (global_level_present[lev]) {
      level_to_part[lev] = nparts++;
      part_to_level.push_back(lev);
      if (min_active_level < 0) min_active_level = lev;
      max_active_level = lev;
    }
  }

  PARTHENON_REQUIRE(nparts > 0, "HYPRE SetupGrid found no active levels.");
  PARTHENON_REQUIRE(min_active_level == min_level,
                    "HYPRE SetupGrid min active level mismatch.");
  PARTHENON_REQUIRE(max_active_level == max_level,
                    "HYPRE SetupGrid max active level mismatch.");

  HYPRE_SStructGridCreate(MPI_COMM_WORLD, ndim, nparts, &grid);

  // Add block extents and cache per-block metadata.
  for (int b = 0; b < nblocks; ++b) {
    auto *pmb = blocks[b].get();
    const int lev = static_cast<int>(pmb->loc.level());
    const int part = level_to_part[lev];
    block_part[b] = part;

    const int nx1 = pmb->block_size.nx(parthenon::X1DIR);
    const int nx2 = pmb->block_size.nx(parthenon::X2DIR);
    const int nx3 = (ndim == 3) ? pmb->block_size.nx(parthenon::X3DIR) : 1;

    const int i0 = static_cast<int>(pmb->loc.lx1()) * nx1;
    const int j0 = static_cast<int>(pmb->loc.lx2()) * nx2;
    const int k0 = (ndim == 3) ? static_cast<int>(pmb->loc.lx3()) * nx3 : 0;

    block_ilower[b] = {i0, j0, k0};
    block_iupper[b] = {i0 + nx1 - 1, j0 + nx2 - 1, (ndim == 3) ? (k0 + nx3 - 1) : 0};

    HYPRE_SStructGridSetExtents(grid, part, block_ilower[b].data(),
                                block_iupper[b].data());

    std::array<parthenon::CellLevel, 6> nbr_level;
    nbr_level.fill(parthenon::CellLevel::same);
    block_neighbor_level[b] = nbr_level;

    std::array<bool, 6> is_domain;
    for (int face = 0; face < 6; ++face) {
      const auto bf = pmb->boundary_flag[face];
      const bool is_user = bf == parthenon::BoundaryFlag::user;
      const bool is_block = bf == parthenon::BoundaryFlag::block;
      const bool is_periodic = bf == parthenon::BoundaryFlag::periodic;
      PARTHENON_REQUIRE(is_user || is_block || is_periodic,
                        "HYPRE SetupGrid encountered unsupported BoundaryFlag.");
      // This solver currently treats user boundaries as physical Dirichlet boundaries.
      is_domain[face] = is_user;
    }
    block_is_domain_boundary[b] = is_domain;

    for (const auto &nb : pmb->GetNeighbors()) {
      const int ax = std::abs(nb.offsets(parthenon::X1DIR));
      const int ay = std::abs(nb.offsets(parthenon::X2DIR));
      const int az = std::abs(nb.offsets(parthenon::X3DIR));
      if (ax + ay + az != 1) continue;

      const int face = FaceFromOffsets(nb.offsets);
      if (face == parthenon::BoundaryFace::undef) continue;

      const int nlev = static_cast<int>(nb.loc.level());
      if (nlev > lev) {
        block_neighbor_level[b][face] = parthenon::CellLevel::fine;
      } else if (nlev < lev) {
        block_neighbor_level[b][face] = parthenon::CellLevel::coarse;
      } else {
        block_neighbor_level[b][face] = parthenon::CellLevel::same;
      }
    }
  }

  // Variables and periodicity for each part.
  HYPRE_SStructVariable cell_var = HYPRE_SSTRUCT_VARIABLE_CELL;
  for (int part = 0; part < nparts; ++part) {
    HYPRE_SStructGridSetVariables(grid, part, 1, &cell_var);

    std::array<int, 3> periodic{0, 0, 0};
    const int lev = part_to_level[part];
    if (pmesh->mesh_bcs[parthenon::BoundaryFace::inner_x1] ==
            parthenon::BoundaryFlag::periodic &&
        pmesh->mesh_bcs[parthenon::BoundaryFace::outer_x1] ==
            parthenon::BoundaryFlag::periodic) {
      periodic[0] = pmesh->mesh_size.nx(parthenon::X1DIR) * (1 << lev);
    }
    if (ndim > 1 &&
        pmesh->mesh_bcs[parthenon::BoundaryFace::inner_x2] ==
            parthenon::BoundaryFlag::periodic &&
        pmesh->mesh_bcs[parthenon::BoundaryFace::outer_x2] ==
            parthenon::BoundaryFlag::periodic) {
      periodic[1] = pmesh->mesh_size.nx(parthenon::X2DIR) * (1 << lev);
    }
    if (ndim > 2 &&
        pmesh->mesh_bcs[parthenon::BoundaryFace::inner_x3] ==
            parthenon::BoundaryFlag::periodic &&
        pmesh->mesh_bcs[parthenon::BoundaryFace::outer_x3] ==
            parthenon::BoundaryFlag::periodic) {
      periodic[2] = pmesh->mesh_size.nx(parthenon::X3DIR) * (1 << lev);
    }

    // Passing zeros is the Hypre convention for non-periodic directions.
    HYPRE_SStructGridSetPeriodic(grid, part, periodic.data());
  }

  HYPRE_SStructGridAssemble(grid);

  // 3-point (1D), 5-point (2D) or 7-point (3D) stencil.
  HYPRE_SStructStencilCreate(ndim, nstencil, &stencil);
  int var = 0;
  std::array<int, 3> off{0, 0, 0};
  HYPRE_SStructStencilSetEntry(stencil, 0, off.data(), var);
  off = {-1, 0, 0};
  HYPRE_SStructStencilSetEntry(stencil, 1, off.data(), var);
  off = {1, 0, 0};
  if (ndim > 1) {
    HYPRE_SStructStencilSetEntry(stencil, 2, off.data(), var);
    off = {0, -1, 0};
    HYPRE_SStructStencilSetEntry(stencil, 3, off.data(), var);
    off = {0, 1, 0};
    HYPRE_SStructStencilSetEntry(stencil, 4, off.data(), var);
  }
  if (ndim > 2) {
    off = {0, 0, -1};
    HYPRE_SStructStencilSetEntry(stencil, 5, off.data(), var);
    off = {0, 0, 1};
    HYPRE_SStructStencilSetEntry(stencil, 6, off.data(), var);
  }

  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &graph);
  HYPRE_SStructGraphSetObjectType(graph, HYPRE_PARCSR);
  for (int part = 0; part < nparts; ++part) {
    HYPRE_SStructGraphSetStencil(graph, part, 0, stencil);
  }

  // Add non-stencil graph entries across fine-coarse boundaries.
  for (int b = 0; b < nblocks; ++b) {
    const auto *pmb = blocks[b].get();
    const int lev = static_cast<int>(pmb->loc.level());
    const int part = block_part[b];
    const auto &lo = block_ilower[b];
    const auto &hi = block_iupper[b];

    auto add_entry = [&](const std::array<int, 3> &from, const std::array<int, 3> &to,
                         int to_part) {
      HYPRE_SStructGraphAddEntries(graph, part, const_cast<int *>(from.data()), 0,
                                   to_part, const_cast<int *>(to.data()), 0);
    };

    for (int face = 0; face < 2 * ndim; ++face) {
      const auto relative_nbr_level = block_neighbor_level[b][face];
      if (relative_nbr_level == parthenon::CellLevel::same) continue;

      const int to_level = lev + static_cast<int>(relative_nbr_level);
      if (to_level < 0 || to_level >= static_cast<int>(level_to_part.size())) {
        std::stringstream msg;
        msg << "Invalid neighbor level mapping in SetupGrid: block=" << b
            << " level=" << lev << " face=" << face
            << " rel=" << static_cast<int>(relative_nbr_level)
            << " to_level=" << to_level;
        PARTHENON_FAIL(msg);
      }
      const int to_part = level_to_part[to_level];
      if (to_part < 0) {
        std::stringstream msg;
        msg << "Invalid neighbor part mapping in SetupGrid: block=" << b
            << " level=" << lev << " face=" << face << " to_level=" << to_level;
        PARTHENON_FAIL(msg);
      }

      int is = lo[0], ie = hi[0];
      int js = lo[1], je = hi[1];
      int ks = lo[2], ke = hi[2];
      const int axis = FaceAxis(face);
      const int side = FaceSide(face);
      PARTHENON_REQUIRE(axis >= 0 && axis < ndim,
                        "Invalid face axis in SetupGrid graph construction.");
      PARTHENON_REQUIRE(side == -1 || side == 1,
                        "Invalid face side in SetupGrid graph construction.");

      if (axis == 0) {
        if (side < 0)
          ie = is;
        else
          is = ie;
      } else if (axis == 1) {
        if (side < 0)
          je = js;
        else
          js = je;
      } else {
        if (side < 0)
          ke = ks;
        else
          ks = ke;
      }

      const int ni = ie - is + 1;
      const int nj = je - js + 1;
      const int nk = ke - ks + 1;
      const int nface_cells = ni * nj * nk;
      std::vector<std::array<int, 3>> from_cells(static_cast<std::size_t>(nface_cells));

      // Build the per-face index list using Parthenon loop abstractions with SIMD tag.
      parthenon::seq_for(ks, ke, js, je, is, ie,
                         [&](const int k, const int j, const int i) {
                           const int lin = (k - ks) * nj * ni + (j - js) * ni + (i - is);
                           from_cells[lin] = {i, j, k};
                         });

      // we need to add graph entries from our row to the column holding our fine/coarse
      // neighbor(s)
      for (const auto &from : from_cells) {
        if (relative_nbr_level == parthenon::CellLevel::coarse) {
          std::array<int, 3> to{from[0] / 2, from[1] / 2, from[2] / 2};
          add_entry(from, to, to_part);
          continue;
        }

        std::array<int, 3> base = {2 * from[0], 2 * from[1], 2 * from[2]};
        base[axis] += (side < 0) ? -1 : 2;
        const int t0 = (axis + 1) % ndim;
        const int t1 = (axis + 2) % ndim;

        if (ndim == 2) {
          for (int s = 0; s < 2; ++s) {
            std::array<int, 3> to = base;
            to[t0] = 2 * from[t0] + s;
            add_entry(from, to, to_part);
          }
        } else {
          for (int s0 = 0; s0 < 2; ++s0) {
            for (int s1 = 0; s1 < 2; ++s1) {
              std::array<int, 3> to = base;
              to[t0] = 2 * from[t0] + s0;
              to[t1] = 2 * from[t1] + s1;
              add_entry(from, to, to_part);
            }
          }
        }
      }
    }
  }

  HYPRE_SStructGraphAssemble(graph);

  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, graph, &A);
  HYPRE_SStructMatrixSetObjectType(A, HYPRE_PARCSR);
  HYPRE_SStructMatrixInitialize(A);

  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &b);
  HYPRE_SStructVectorSetObjectType(b, HYPRE_PARCSR);
  HYPRE_SStructVectorInitialize(b);

  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &x);
  HYPRE_SStructVectorSetObjectType(x, HYPRE_PARCSR);
  HYPRE_SStructVectorInitialize(x);

  grid_is_setup = true;
  solver_is_setup = false;
}

} // namespace diffusion_package
