#include "diffusion_hypre.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "diffusion_package.hpp"
#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "pack/make_pack_descriptor.hpp"
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

int FaceFromAxisSide(const int axis, const int side) {
  if (axis == 0)
    return (side < 0) ? parthenon::BoundaryFace::inner_x1
                      : parthenon::BoundaryFace::outer_x1;
  if (axis == 1)
    return (side < 0) ? parthenon::BoundaryFace::inner_x2
                      : parthenon::BoundaryFace::outer_x2;
  if (axis == 2)
    return (side < 0) ? parthenon::BoundaryFace::inner_x3
                      : parthenon::BoundaryFace::outer_x3;
  return parthenon::BoundaryFace::undef;
}

int StencilEntryForFace(const int face) {
  if (face == parthenon::BoundaryFace::inner_x1) return 1;
  if (face == parthenon::BoundaryFace::outer_x1) return 2;
  if (face == parthenon::BoundaryFace::inner_x2) return 3;
  if (face == parthenon::BoundaryFace::outer_x2) return 4;
  if (face == parthenon::BoundaryFace::inner_x3) return 5;
  if (face == parthenon::BoundaryFace::outer_x3) return 6;
  return -1;
}

int DfcComponentFromGlobal(const int axis, const int gi, const int gj, const int gk,
                           const int ndim) {
  if (axis == 0) {
    const int comp1 = gj & 1;
    const int comp2 = (ndim > 2) ? (gk & 1) : 0;
    return comp1 + 2 * comp2;
  }
  if (axis == 1) {
    const int comp1 = (ndim > 2) ? (gk & 1) : 0;
    const int comp2 = gi & 1;
    return comp1 + 2 * comp2;
  }
  const int comp1 = gi & 1;
  const int comp2 = gj & 1;
  return comp1 + 2 * comp2;
}

void NeighborFaceBounds(const std::array<int, 3> &lo, const std::array<int, 3> &hi,
                        const int ndim, const parthenon::NeighborBlock &nb,
                        const bool neighbor_is_fine, int &is, int &ie, int &js, int &je,
                        int &ks, int &ke, int &axis, int &side) {
  const int face = FaceFromOffsets(nb.offsets);
  axis = FaceAxis(face);
  side = FaceSide(face);

  is = lo[0];
  ie = hi[0];
  js = lo[1];
  je = hi[1];
  ks = lo[2];
  ke = hi[2];

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

  // If neighbor is finer, this neighbor only covers a half-face (2D) or quarter-face
  // (3D). Restrict the tangential ranges using fi1/fi2.
  if (neighbor_is_fine) {
    auto split_half = [](int &s, int &e, int fi) {
      const int n = e - s + 1;
      const int h = n / 2;
      s += fi * h;
      e = s + h - 1;
    };

    if (axis == 0) {
      split_half(js, je, nb.fi1);
      if (ndim > 2) split_half(ks, ke, nb.fi2);
    } else if (axis == 1) {
      if (ndim > 2) {
        split_half(ks, ke, nb.fi1);
        split_half(is, ie, nb.fi2);
      } else {
        split_half(is, ie, nb.fi1);
      }
    } else {
      split_half(is, ie, nb.fi1);
      split_half(js, je, nb.fi2);
    }
  }
}

} // namespace

HypreSolver::HypreSolver(parthenon::ParameterInput *pin) {
  // Solver type
  solver_type = pin->GetOrAddString("hypre", "solver_type", "pcg",
                                    "Hypre outer solver: pcg or bicgstab");
  std::transform(solver_type.begin(), solver_type.end(), solver_type.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  PARTHENON_REQUIRE(solver_type == "pcg" || solver_type == "bicgstab",
                    "hypre/solver_type must be 'pcg' or 'bicgstab'.");
  preconditioner = pin->GetOrAddString("hypre", "preconditioner", "amg",
                                       "Hypre preconditioner: amg or none");
  std::transform(preconditioner.begin(), preconditioner.end(), preconditioner.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  PARTHENON_REQUIRE(preconditioner == "amg" || preconditioner == "none",
                    "hypre/preconditioner must be 'amg' or 'none'.");
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
  auto u_bounds =
      pin->GetOrAddVector<Real>("diffusion", "boundary_u", {0.0}, "Boundary us.");
  if (u_bounds.size() == 1) u_bounds = std::vector<Real>(6, u_bounds[0]);
  PARTHENON_REQUIRE(u_bounds.size() == 6,
                    "diffusion/boundary_u must have exactly 1 or 6 entries.");
  for (int f = 0; f < 6; ++f)
    boundary_u[f] = u_bounds[f];

  // Determine dimensionality
  const int nx3 = pin->GetInteger("parthenon/mesh", "nx3");
  ndim = (nx3 > 1) ? 3 : 2;
  nstencil = (ndim == 1) ? 3 : (ndim == 2) ? 5 : 7;
}

HypreSolver::~HypreSolver() { DestroyGrid(); }

void HypreSolver::DestroyGrid() {
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

  block_part.clear();
  block_ilower.clear();
  block_iupper.clear();
  block_neighbor_level.clear();
  block_is_domain_boundary.clear();
  level_to_part.clear();

  nparts = 0;
  min_active_level = -1;
  max_active_level = -1;
  grid_is_setup = false;
  solver_is_setup = false;
}

parthenon::TaskStatus HypreSolver::BuildMatrixVector(HypreSolver *solver, int b,
                                                     parthenon::MeshBlock *pmb,
                                                     const Real dt) {
  using namespace parthenon;
  using TE = parthenon::TopologicalElement;
  (void)dt;

  PARTHENON_REQUIRE(solver->grid_is_setup,
                    "BuildMatrixVector called before Hypre grid setup.");
  PARTHENON_REQUIRE(b >= 0 && b < static_cast<int>(solver->block_part.size()),
                    "BuildMatrixVector called with invalid block index.");

  auto *pmbd = pmb->meshblock_data.Get().get();
  auto desc = parthenon::MakePackDescriptor<diffusion_package::u, diffusion_package::D,
                                            diffusion_package::Dfc>(pmbd);
  auto pack = desc.GetPack(pmbd);
  PARTHENON_REQUIRE(pack.GetNBlocks() == 1,
                    "BuildMatrixVector expects exactly one block in pack.");
  constexpr int pb = 0;

  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int ni = ib.e - ib.s + 1;
  const int nj = jb.e - jb.s + 1;
  const int nk = kb.e - kb.s + 1;
  const int ncell = ni * nj * nk;

  const auto &il = solver->block_ilower[b];
  const auto &iu = solver->block_iupper[b];
  const int part = solver->block_part[b];
  const int legacy_root_level = pmb->pmy_mesh->GetLegacyTreeRootLevel();
  const auto legacy_loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);
  const int lev = static_cast<int>(legacy_loc.level()) - legacy_root_level;

  std::vector<int> stencil_entries(solver->nstencil);
  for (int e = 0; e < solver->nstencil; ++e)
    stencil_entries[e] = e;

  // Hypre SStruct matrix/vector APIs consume HYPRE_Complex typed arrays.
  std::vector<HYPRE_Complex> matvals(static_cast<std::size_t>(ncell * solver->nstencil),
                                     0.0);
  std::vector<Real> rhsvals(static_cast<std::size_t>(ncell), 0.0);
  std::vector<HYPRE_Complex> xvals(static_cast<std::size_t>(ncell), 0.0);

  auto lin_idx = [&](const int k, const int j, const int i) {
    return (k - kb.s) * nj * ni + (j - jb.s) * ni + (i - ib.s);
  };
  auto A = [&](const int lin, const int ent) -> HYPRE_Complex & {
    return matvals[static_cast<std::size_t>(lin * solver->nstencil + ent)];
  };
  auto local_from_global = [&](const int gk, const int gj, const int gi) {
    const int li = gi - il[0];
    const int lj = gj - il[1];
    const int lk = gk - il[2];
    PARTHENON_REQUIRE(li >= 0 && li < ni && lj >= 0 && lj < nj && lk >= 0 && lk < nk,
                      "Global-to-local mapping out of block bounds.");
    return std::array<int, 3>{ib.s + li, jb.s + lj, kb.s + lk};
  };

  auto face_conductance = [&](const int axis, const int side, const int k, const int j,
                              const int i) {
    if (axis == 0) {
      const Real d = pmb->coords.Dxc<X1DIR>(k, j, i);
      const Real area = (side < 0) ? pmb->coords.Volume<TE::F1>(k, j, i)
                                   : pmb->coords.Volume<TE::F1>(k, j, i + 1);
      const Real Dface = (side < 0)
                             ? pack(pb, TE::F1, diffusion_package::D(), k, j, i)
                             : pack(pb, TE::F1, diffusion_package::D(), k, j, i + 1);
      return Dface * area / d;
    } else if (axis == 1) {
      const Real d = pmb->coords.Dxc<X2DIR>(k, j, i);
      const Real area = (side < 0) ? pmb->coords.Volume<TE::F2>(k, j, i)
                                   : pmb->coords.Volume<TE::F2>(k, j + 1, i);
      const Real Dface = (side < 0)
                             ? pack(pb, TE::F2, diffusion_package::D(), k, j, i)
                             : pack(pb, TE::F2, diffusion_package::D(), k, j + 1, i);
      return Dface * area / d;
    }
    const Real d = pmb->coords.Dxc<X3DIR>(k, j, i);
    const Real area = (side < 0) ? pmb->coords.Volume<TE::F3>(k, j, i)
                                 : pmb->coords.Volume<TE::F3>(k + 1, j, i);
    const Real Dface = (side < 0) ? pack(pb, TE::F3, diffusion_package::D(), k, j, i)
                                  : pack(pb, TE::F3, diffusion_package::D(), k + 1, j, i);
    return Dface * area / d;
  };

  auto zero_face_stencil = [&](const int face, const int lin) {
    if (face == BoundaryFace::inner_x1)
      A(lin, 1) = 0.0;
    else if (face == BoundaryFace::outer_x1)
      A(lin, 2) = 0.0;
    else if (face == BoundaryFace::inner_x2)
      A(lin, 3) = 0.0;
    else if (face == BoundaryFace::outer_x2)
      A(lin, 4) = 0.0;
    else if (face == BoundaryFace::inner_x3)
      A(lin, 5) = 0.0;
    else if (face == BoundaryFace::outer_x3)
      A(lin, 6) = 0.0;
  };

  // Phase A: interior stencil + rhs (full-u solve form).
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "build_matrix_rows",
      parthenon::HostExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      [&](const int k, const int j, const int i) {
        const int lin = lin_idx(k, j, i);

        const Real cell_vol = pmb->coords.Volume<TE::CC>(k, j, i);
        const Real kxm = face_conductance(0, -1, k, j, i);
        const Real kxp = face_conductance(0, +1, k, j, i);
        const Real kym = (solver->ndim > 1) ? face_conductance(1, -1, k, j, i) : 0.0;
        const Real kyp = (solver->ndim > 1) ? face_conductance(1, +1, k, j, i) : 0.0;
        const Real kzm = (solver->ndim > 2) ? face_conductance(2, -1, k, j, i) : 0.0;
        const Real kzp = (solver->ndim > 2) ? face_conductance(2, +1, k, j, i) : 0.0;

        A(lin, 0) = solver->diagonal_alpha * cell_vol + kxm + kxp + kym + kyp + kzm + kzp;
        A(lin, 1) = -kxm;
        A(lin, 2) = -kxp;
        if (solver->ndim > 1) {
          A(lin, 3) = -kym;
          A(lin, 4) = -kyp;
        }
        if (solver->ndim > 2) {
          A(lin, 5) = -kzm;
          A(lin, 6) = -kzp;
        }

        const Real u0 = pack(pb, diffusion_package::u(), k, j, i);
        rhsvals[lin] = solver->diagonal_alpha * cell_vol * u0;
        xvals[lin] = 0.0;
      });

  // Phase B: physical Dirichlet boundary corrections.
  for (int face = 0; face < 2 * solver->ndim; ++face) {
    if (!solver->block_is_domain_boundary[b][face]) continue;
    const int axis = FaceAxis(face);
    const int side = FaceSide(face);
    int gis = il[0], gie = iu[0];
    int gjs = il[1], gje = iu[1];
    int gks = il[2], gke = iu[2];
    if (axis == 0) {
      if (side < 0)
        gie = gis;
      else
        gis = gie;
    } else if (axis == 1) {
      if (side < 0)
        gje = gjs;
      else
        gjs = gje;
    } else {
      if (side < 0)
        gke = gks;
      else
        gks = gke;
    }

    const Real bc = solver->boundary_u[face];
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "bc_fixup",
                       parthenon::HostExecSpace(), gks, gke, gjs, gje, gis, gie,
                       [&](const int gk, const int gj, const int gi) {
                         auto lidx = local_from_global(gk, gj, gi);
                         const int i = lidx[0], j = lidx[1], k = lidx[2];
                         const int lin = lin_idx(k, j, i);
                         const Real kface = face_conductance(axis, side, k, j, i);
                         A(lin, 0) += kface;
                         zero_face_stencil(face, lin);
                         rhsvals[lin] += 2.0 * kface * bc;
                       });
  }

  // Set non-stencil graph couplings at fine-coarse boundaries.
  std::vector<int> row_graph_count(static_cast<std::size_t>(ncell), 0);
  for (const auto &nb : pmb->GetNeighbors()) {
    const int ax = std::abs(nb.offsets(parthenon::X1DIR));
    const int ay = std::abs(nb.offsets(parthenon::X2DIR));
    const int az = std::abs(nb.offsets(parthenon::X3DIR));
    if (ax + ay + az != 1) continue;

    const int face = FaceFromOffsets(nb.offsets);
    if (face == parthenon::BoundaryFace::undef) continue;

    const auto nlegacy_loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(nb.origin_loc);
    const int nlev = static_cast<int>(nlegacy_loc.level()) - legacy_root_level;
    const int neighbor_level_relation = (nlev > lev) ? 1 : ((nlev < lev) ? -1 : 0);
    if (neighbor_level_relation == 0) continue;

    int is, ie, js, je, ks, ke, axis, side;
    NeighborFaceBounds(il, iu, solver->ndim, nb, lev, is, ie, js, je, ks, ke, axis, side);

    for (int gk = ks; gk <= ke; ++gk) {
      for (int gj = js; gj <= je; ++gj) {
        for (int gi = is; gi <= ie; ++gi) {
          const int lin = (gk - il[2]) * nj * ni + (gj - il[1]) * ni + (gi - il[0]);
          auto lidx = local_from_global(gk, gj, gi);
          const int i = lidx[0], j = lidx[1], k = lidx[2];

          auto set_graph_value = [&](const Real value) {
            int entry = solver->nstencil + row_graph_count[lin];
            int index[3] = {gi, gj, gk};
            HYPRE_Complex hval = value;
            HYPRE_SStructMatrixSetValues(solver->A, part, index, 0, 1, &entry, &hval);
            row_graph_count[lin] += 1;
          };

          const Real kface = face_conductance(axis, side, k, j, i);
          zero_face_stencil(face, lin);

          if (neighbor_level_relation < 0) {
            // Fine row coupled to coarse neighbor:
            // center contribution is +2/3 K and coarse coupling is -1/3 K.
            A(lin, 0) -= (1.0 / 3.0) * kface;
            set_graph_value(-(1.0 / 3.0) * kface);
          } else {
            // Coarse row coupled to fine neighbors:
            // each fine coupling is -2/3 K_sub and center gets +1/3 sum(K_sub).
            const int nsub = (solver->ndim == 2) ? 2 : 4;
            const Real face_area =
                (axis == 0)
                    ? ((side < 0) ? pmb->coords.Volume<TE::F1>(k, j, i)
                                  : pmb->coords.Volume<TE::F1>(k, j, i + 1))
                    : ((axis == 1)
                           ? ((side < 0) ? pmb->coords.Volume<TE::F2>(k, j, i)
                                         : pmb->coords.Volume<TE::F2>(k, j + 1, i))
                           : ((side < 0) ? pmb->coords.Volume<TE::F3>(k, j, i)
                                         : pmb->coords.Volume<TE::F3>(k + 1, j, i)));
            const Real d = (axis == 0) ? pmb->coords.Dxc<X1DIR>(k, j, i)
                                       : ((axis == 1) ? pmb->coords.Dxc<X2DIR>(k, j, i)
                                                      : pmb->coords.Dxc<X3DIR>(k, j, i));
            const Real subface_area = face_area / static_cast<Real>(nsub);

            std::array<int, 3> fine_face_anchor = {2 * gi, 2 * gj, 2 * gk};
            fine_face_anchor[axis] += (side < 0) ? -1 : 2;
            const int tan_axis0 = (axis + 1) % solver->ndim;
            const int tan_axis1 = (axis + 2) % solver->ndim;

            Real sum_k_sub = 0.0;
            if (solver->ndim == 2) {
              for (int s = 0; s < 2; ++s) {
                std::array<int, 3> to = fine_face_anchor;
                to[tan_axis0] = 2 * ((tan_axis0 == 0) ? gi : gj) + s;
                const int comp =
                    DfcComponentFromGlobal(axis, to[0], to[1], to[2], solver->ndim);
                const Real Dsub =
                    (axis == 0)
                        ? ((side < 0)
                               ? pack(pb, TE::F1, diffusion_package::Dfc(comp), k, j, i)
                               : pack(pb, TE::F1, diffusion_package::Dfc(comp), k, j,
                                      i + 1))
                        : ((side < 0)
                               ? pack(pb, TE::F2, diffusion_package::Dfc(comp), k, j, i)
                               : pack(pb, TE::F2, diffusion_package::Dfc(comp), k, j + 1,
                                      i));
                const Real ksub = Dsub * subface_area / d;
                sum_k_sub += ksub;
                set_graph_value(-(2.0 / 3.0) * ksub);
              }
            } else {
              for (int s0 = 0; s0 < 2; ++s0) {
                for (int s1 = 0; s1 < 2; ++s1) {
                  std::array<int, 3> to = fine_face_anchor;
                  to[tan_axis0] =
                      2 * ((tan_axis0 == 0) ? gi : ((tan_axis0 == 1) ? gj : gk)) + s0;
                  to[tan_axis1] =
                      2 * ((tan_axis1 == 0) ? gi : ((tan_axis1 == 1) ? gj : gk)) + s1;
                  const int comp =
                      DfcComponentFromGlobal(axis, to[0], to[1], to[2], solver->ndim);
                  const Real Dsub =
                      (axis == 0)
                          ? ((side < 0)
                                 ? pack(pb, TE::F1, diffusion_package::Dfc(comp), k, j, i)
                                 : pack(pb, TE::F1, diffusion_package::Dfc(comp), k, j,
                                        i + 1))
                          : ((axis == 1)
                                 ? ((side < 0)
                                        ? pack(pb, TE::F2, diffusion_package::Dfc(comp),
                                               k, j, i)
                                        : pack(pb, TE::F2, diffusion_package::Dfc(comp),
                                               k, j + 1, i))
                                 : ((side < 0)
                                        ? pack(pb, TE::F3, diffusion_package::Dfc(comp),
                                               k, j, i)
                                        : pack(pb, TE::F3, diffusion_package::Dfc(comp),
                                               k + 1, j, i)));
                  const Real ksub = Dsub * subface_area / d;
                  sum_k_sub += ksub;
                  set_graph_value(-(2.0 / 3.0) * ksub);
                }
              }
            }

            A(lin, 0) -= (2.0 / 3.0) * sum_k_sub;
          }
        }
      }
    }
  }

  HYPRE_SStructMatrixSetBoxValues(solver->A, part, const_cast<int *>(il.data()),
                                  const_cast<int *>(iu.data()), 0, solver->nstencil,
                                  stencil_entries.data(), matvals.data());

  HYPRE_SStructVectorSetBoxValues(solver->b, part, const_cast<int *>(il.data()),
                                  const_cast<int *>(iu.data()), 0, rhsvals.data());
  HYPRE_SStructVectorSetBoxValues(solver->x, part, const_cast<int *>(il.data()),
                                  const_cast<int *>(iu.data()), 0, xvals.data());

  return parthenon::TaskStatus::complete;
}

parthenon::TaskStatus HypreSolver::Solve(HypreSolver *solver) {
  HYPRE_SStructMatrixAssemble(solver->A);
  HYPRE_SStructVectorAssemble(solver->b);
  HYPRE_SStructVectorAssemble(solver->x);

  HYPRE_ParCSRMatrix parA;
  HYPRE_ParVector parb, parx;
  HYPRE_SStructMatrixGetObject(solver->A, reinterpret_cast<void **>(&parA));
  HYPRE_SStructVectorGetObject(solver->b, reinterpret_cast<void **>(&parb));
  HYPRE_SStructVectorGetObject(solver->x, reinterpret_cast<void **>(&parx));

  if (!solver->solver_is_setup) {
    solver->SetupSolver();
  }

  HYPRE_Int niter = 0;
  HYPRE_Real rnorm = 0.0;

  if (solver->solver_type == "pcg") {
    HYPRE_ParCSRPCGSetup(solver->solver_handle, parA, parb, parx);
    HYPRE_ParCSRPCGSolve(solver->solver_handle, parA, parb, parx);
    HYPRE_ParCSRPCGGetNumIterations(solver->solver_handle, &niter);
    HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(solver->solver_handle, &rnorm);
  } else {
    HYPRE_ParCSRBiCGSTABSetup(solver->solver_handle, parA, parb, parx);
    HYPRE_ParCSRBiCGSTABSolve(solver->solver_handle, parA, parb, parx);
    HYPRE_ParCSRBiCGSTABGetNumIterations(solver->solver_handle, &niter);
    HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver->solver_handle, &rnorm);
  }

  if (parthenon::Globals::my_rank == 0 && solver->print_level > 0) {
    std::cout << "[hypre] iterations=" << niter << " rel_resid=" << rnorm << "\n";
  }

  HYPRE_SStructVectorGather(solver->x);

  return parthenon::TaskStatus::complete;
}

parthenon::TaskStatus HypreSolver::UpdateSolution(HypreSolver *solver, int b,
                                                  parthenon::MeshBlock *pmb) {
  using namespace parthenon;

  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const int ni = ib.e - ib.s + 1;
  const int nj = jb.e - jb.s + 1;
  const int nk = kb.e - kb.s + 1;
  const int ncell = ni * nj * nk;

  const int part = solver->block_part[b];
  const auto &il = solver->block_ilower[b];
  const auto &iu = solver->block_iupper[b];

  std::vector<HYPRE_Complex> soln(static_cast<std::size_t>(ncell), 0.0);
  HYPRE_SStructVectorGetBoxValues(solver->x, part, const_cast<int *>(il.data()),
                                  const_cast<int *>(iu.data()), 0, soln.data());

  auto &uvar = pmb->meshblock_data.Get()->Get(diffusion_package::u::name()).data;
  auto lin_idx = [&](const int k, const int j, const int i) {
    return (k - kb.s) * nj * ni + (j - jb.s) * ni + (i - ib.s);
  };

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        uvar(k, j, i) = soln[lin_idx(k, j, i)];
      }
    }
  }

  return parthenon::TaskStatus::complete;
}

void HypreSolver::SetupSolver() {
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

  const bool use_amg_preconditioner = (preconditioner == "amg");
  if (use_amg_preconditioner) {
    HYPRE_BoomerAMGCreate(&precond_handle);
    HYPRE_BoomerAMGSetTol(precond_handle, 0.0);
    HYPRE_BoomerAMGSetMaxIter(precond_handle, 1);
    HYPRE_BoomerAMGSetCoarsenType(precond_handle, amg_coarsen_type);
    HYPRE_BoomerAMGSetInterpType(precond_handle, amg_interp_type);
    HYPRE_BoomerAMGSetRelaxType(precond_handle, amg_relax_type);
    HYPRE_BoomerAMGSetStrongThreshold(precond_handle, amg_strong_threshold);
    HYPRE_BoomerAMGSetNumSweeps(precond_handle, amg_num_sweeps);
    HYPRE_BoomerAMGSetPrintLevel(precond_handle, 0);
  }

  if (solver_type == "pcg") {
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_ParCSRPCGSetTol(solver_handle, tol);
    HYPRE_ParCSRPCGSetMaxIter(solver_handle, max_iter);
    HYPRE_ParCSRPCGSetPrintLevel(solver_handle, print_level);
    HYPRE_ParCSRPCGSetLogging(solver_handle, (print_level > 0) ? 1 : 0);
    if (use_amg_preconditioner) {
      HYPRE_ParCSRPCGSetPrecond(solver_handle, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup,
                                precond_handle);
    }
  } else {
    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver_handle);
    HYPRE_ParCSRBiCGSTABSetTol(solver_handle, tol);
    HYPRE_ParCSRBiCGSTABSetMaxIter(solver_handle, max_iter);
    HYPRE_ParCSRBiCGSTABSetPrintLevel(solver_handle, print_level);
    HYPRE_ParCSRBiCGSTABSetLogging(solver_handle, (print_level > 0) ? 1 : 0);
    if (use_amg_preconditioner) {
      HYPRE_ParCSRBiCGSTABSetPrecond(solver_handle, HYPRE_BoomerAMGSolve,
                                     HYPRE_BoomerAMGSetup, precond_handle);
    }
  }

  solver_is_setup = true;
}

void HypreSolver::SetupGrid(parthenon::Mesh *pmesh) {
  if (grid_is_setup) return;

  auto &blocks = pmesh->block_list;
  const int nblocks = static_cast<int>(blocks.size());
  const int legacy_root_level = pmesh->GetLegacyTreeRootLevel();

  if (nblocks == 0) {
    PARTHENON_FAIL("SetupGrid called with empty block list.");
  }

  block_part.resize(nblocks, -1);
  block_ilower.resize(nblocks);
  block_iupper.resize(nblocks);
  block_neighbor_level.resize(nblocks);
  block_is_domain_boundary.resize(nblocks);
  std::vector<std::vector<std::pair<std::array<int, 3>, std::array<int, 3>>>> part_boxes;
  std::vector<std::vector<std::pair<std::array<int, 3>, std::array<int, 3>>>>
      global_part_boxes;

  // Determine globally active refinement levels.
  int local_max_level = -1;
  int local_min_level = std::numeric_limits<int>::max();
  for (const auto &pmb : blocks) {
    const auto legacy_loc = pmesh->Forest().GetLegacyTreeLocation(pmb->loc);
    const int lev = static_cast<int>(legacy_loc.level()) - legacy_root_level;
    local_max_level = std::max(local_max_level, lev);
    local_min_level = std::min(local_min_level, lev);
  }

  int max_level = -1;
  int min_level = std::numeric_limits<int>::max();
  MPI_Allreduce(&local_max_level, &max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_min_level, &min_level, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  std::vector<int> local_level_present(std::max(max_level + 1, 0), 0);
  for (const auto &pmb : blocks) {
    const auto legacy_loc = pmesh->Forest().GetLegacyTreeLocation(pmb->loc);
    local_level_present[static_cast<int>(legacy_loc.level()) - legacy_root_level] = 1;
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
  part_boxes.resize(nparts);
  global_part_boxes.resize(nparts);

  // Add block extents and cache per-block metadata.
  for (int b = 0; b < nblocks; ++b) {
    auto *pmb = blocks[b].get();
    const auto legacy_loc = pmesh->Forest().GetLegacyTreeLocation(pmb->loc);
    const int lev = static_cast<int>(legacy_loc.level()) - legacy_root_level;
    const int part = level_to_part[lev];
    block_part[b] = part;

    const int nx1 = pmb->block_size.nx(parthenon::X1DIR);
    const int nx2 = pmb->block_size.nx(parthenon::X2DIR);
    const int nx3 = (ndim == 3) ? pmb->block_size.nx(parthenon::X3DIR) : 1;

    const int i0 = static_cast<int>(legacy_loc.lx1()) * nx1;
    const int j0 = static_cast<int>(legacy_loc.lx2()) * nx2;
    const int k0 = (ndim == 3) ? static_cast<int>(legacy_loc.lx3()) * nx3 : 0;

    block_ilower[b] = {i0, j0, k0};
    block_iupper[b] = {i0 + nx1 - 1, j0 + nx2 - 1, (ndim == 3) ? (k0 + nx3 - 1) : 0};

    HYPRE_SStructGridSetExtents(grid, part, block_ilower[b].data(),
                                block_iupper[b].data());
    part_boxes[part].push_back({block_ilower[b], block_iupper[b]});

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

      const auto nlegacy_loc = pmesh->Forest().GetLegacyTreeLocation(nb.origin_loc);
      const int nlev = static_cast<int>(nlegacy_loc.level()) - legacy_root_level;
      if (nlev > lev) {
        block_neighbor_level[b][face] = parthenon::CellLevel::fine;
      } else if (nlev < lev) {
        block_neighbor_level[b][face] = parthenon::CellLevel::coarse;
      } else {
        block_neighbor_level[b][face] = parthenon::CellLevel::same;
      }
    }
  }

  {
    const int local_nboxes = nblocks;
    std::vector<int> counts(parthenon::Globals::nranks, 0);
    MPI_Allgather(&local_nboxes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(parthenon::Globals::nranks, 0);
    int total_nboxes = 0;
    for (int r = 0; r < parthenon::Globals::nranks; ++r) {
      displs[r] = total_nboxes;
      total_nboxes += counts[r];
    }

    std::vector<int> sendbuf(static_cast<std::size_t>(local_nboxes * 7), 0);
    for (int b = 0; b < nblocks; ++b) {
      sendbuf[7 * b + 0] = block_part[b];
      sendbuf[7 * b + 1] = block_ilower[b][0];
      sendbuf[7 * b + 2] = block_ilower[b][1];
      sendbuf[7 * b + 3] = block_ilower[b][2];
      sendbuf[7 * b + 4] = block_iupper[b][0];
      sendbuf[7 * b + 5] = block_iupper[b][1];
      sendbuf[7 * b + 6] = block_iupper[b][2];
    }

    std::vector<int> recv_counts(parthenon::Globals::nranks, 0);
    std::vector<int> recv_displs(parthenon::Globals::nranks, 0);
    for (int r = 0; r < parthenon::Globals::nranks; ++r) {
      recv_counts[r] = counts[r] * 7;
      recv_displs[r] = displs[r] * 7;
    }

    std::vector<int> recvbuf(static_cast<std::size_t>(total_nboxes * 7), 0);
    MPI_Allgatherv(sendbuf.data(), local_nboxes * 7, MPI_INT, recvbuf.data(),
                   recv_counts.data(), recv_displs.data(), MPI_INT, MPI_COMM_WORLD);

    for (int n = 0; n < total_nboxes; ++n) {
      const int part = recvbuf[7 * n + 0];
      std::array<int, 3> lo_g{recvbuf[7 * n + 1], recvbuf[7 * n + 2], recvbuf[7 * n + 3]};
      std::array<int, 3> hi_g{recvbuf[7 * n + 4], recvbuf[7 * n + 5], recvbuf[7 * n + 6]};
      PARTHENON_REQUIRE(part >= 0 && part < nparts,
                        "Invalid part while constructing global_part_boxes.");
      global_part_boxes[part].push_back({lo_g, hi_g});
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
    const auto legacy_loc = pmesh->Forest().GetLegacyTreeLocation(pmb->loc);
    const int lev = static_cast<int>(legacy_loc.level()) - legacy_root_level;
    const int part = block_part[b];
    const auto &lo = block_ilower[b];
    const auto &hi = block_iupper[b];

    auto add_entry = [&](const std::array<int, 3> &from, const std::array<int, 3> &to,
                         int to_part) {
      bool valid_to = false;
      for (const auto &bx : global_part_boxes[to_part]) {
        const auto &lbx = bx.first;
        const auto &ubx = bx.second;
        if (to[0] >= lbx[0] && to[0] <= ubx[0] && to[1] >= lbx[1] && to[1] <= ubx[1] &&
            to[2] >= lbx[2] && to[2] <= ubx[2]) {
          valid_to = true;
          break;
        }
      }
      if (!valid_to) {
        const int lev_to = part_to_level[to_part];
        std::stringstream msg;
        msg << "SetupGrid graph target index not found in any registered box: from=("
            << from[0] << "," << from[1] << "," << from[2] << ") to=(" << to[0] << ","
            << to[1] << "," << to[2] << ") to_part=" << to_part << " to_level=" << lev_to;
        PARTHENON_FAIL(msg);
      }
      HYPRE_SStructGraphAddEntries(graph, part, const_cast<int *>(from.data()), 0,
                                   to_part, const_cast<int *>(to.data()), 0);
    };

    for (const auto &nb : pmb->GetNeighbors()) {
      const int ax = std::abs(nb.offsets(parthenon::X1DIR));
      const int ay = std::abs(nb.offsets(parthenon::X2DIR));
      const int az = std::abs(nb.offsets(parthenon::X3DIR));
      if (ax + ay + az != 1) continue;

      const int face = FaceFromOffsets(nb.offsets);
      if (face == parthenon::BoundaryFace::undef) continue;

      const auto nlegacy_loc = pmesh->Forest().GetLegacyTreeLocation(nb.origin_loc);
      const int nlev = static_cast<int>(nlegacy_loc.level()) - legacy_root_level;
      const int relative_nbr_level = (nlev > lev) ? 1 : ((nlev < lev) ? -1 : 0);
      if (relative_nbr_level == 0) continue;

      const int to_level = lev + relative_nbr_level;
      PARTHENON_REQUIRE(to_level >= 0 &&
                            to_level < static_cast<int>(level_to_part.size()),
                        "Invalid neighbor level mapping in SetupGrid.");
      const int to_part = level_to_part[to_level];
      PARTHENON_REQUIRE(to_part >= 0, "Invalid neighbor part mapping in SetupGrid.");

      int is, ie, js, je, ks, ke, axis, side;
      NeighborFaceBounds(lo, hi, ndim, nb, relative_nbr_level > 0, is, ie, js, je, ks, ke,
                         axis, side);
      PARTHENON_REQUIRE(axis >= 0 && axis < ndim,
                        "Invalid face axis in SetupGrid graph construction.");

      const int ni = ie - is + 1;
      const int nj = je - js + 1;
      const int nk = ke - ks + 1;
      const int nface_cells = ni * nj * nk;
      std::vector<std::array<int, 3>> from_cells(static_cast<std::size_t>(nface_cells));

      parthenon::seq_for(ks, ke, js, je, is, ie,
                         [&](const int k, const int j, const int i) {
                           const int lin = (k - ks) * nj * ni + (j - js) * ni + (i - is);
                           from_cells[lin] = {i, j, k};
                         });

      for (const auto &from : from_cells) {
        if (relative_nbr_level < 0) {
          std::array<int, 3> to{from[0] / 2, from[1] / 2, from[2] / 2};
          to[axis] = (from[axis] + ((side < 0) ? -1 : 1)) / 2;
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
  needs_grid_setup = false;
  solver_is_setup = false;
}

} // namespace diffusion_package
