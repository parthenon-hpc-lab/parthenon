//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_POISSON_GMG_POISSON_EQUATION_STAGES_HPP_
#define EXAMPLE_POISSON_GMG_POISSON_EQUATION_STAGES_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "poisson_package.hpp"

namespace poisson_package {

// This class implement methods for calculating A.x = y and returning the diagonal of A,
// where A is the the matrix representing the discretized Poisson equation on the grid.
// Here we implement the Laplace operator in terms of a flux divergence to (potentially)
// consistently deal with coarse fine boundaries on the grid. Only the routines Ax and
// SetDiagonal need to be defined for interfacing this with solvers. The other methods
// are internal, but can't be marked private or protected because they launch kernels
// on device.
template <class var_t, class D_t>
class PoissonEquationStages {
 public:
  bool do_flux_cor = false;
  bool set_flux_boundary = false;
  bool include_flux_dx = false;
  enum class ProlongationType { Constant, Linear, Kwak };
  ProlongationType prolongation_type = ProlongationType::Constant;

  using IndependentVars = parthenon::TypeList<var_t>;

  PoissonEquationStages(parthenon::ParameterInput *pin, const std::string &label) {
    do_flux_cor = pin->GetOrAddBoolean(label, "flux_correct", false);
    set_flux_boundary = pin->GetOrAddBoolean(label, "set_flux_boundary", false);
    include_flux_dx =
        (pin->GetOrAddString(label, "boundary_prolongation", "Linear") == "Constant");
    auto pro_int = pin->GetOrAddString(label, "interior_prolongation", "Linear");
    if (pro_int == "Constant") {
      prolongation_type = ProlongationType::Constant;
    } else if (pro_int == "Linear") {
      prolongation_type = ProlongationType::Linear;
    } else if (pro_int == "Kwak") {
      prolongation_type = ProlongationType::Kwak;
    } else {
      PARTHENON_FAIL("Invalid user prolongation type.");
    }
  }

  // Add tasks to calculate the result of the matrix A (which is implicitly defined by
  // this class) being applied to x_t and store it in field out_t
  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    auto flux_res = tl.AddTask(depends_on, CalculateFluxes, md_mat, md_in);
    // if (set_flux_boundary) {
    //   flux_res = tl.AddTask(flux_res, SetFluxBoundaries<x_t>, md, include_flux_dx);
    // }
    if (do_flux_cor && !(md_mat->grid.type == parthenon::GridType::two_level_composite)) {
      auto start_flxcor =
          tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
      auto send_flxcor =
          tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
      auto recv_flxcor =
          tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md_in);
      flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
    }
    return tl.AddTask(flux_res, FluxMultiplyMatrix, md_in, md_out);
  }

  // Calculate an approximation to the diagonal of the matrix A and store it in diag_t.
  // For a uniform grid or when flux correction is ignored, this diagonal calculation
  // is exact. Exactness is (probably) not required since it is just used in Jacobi
  // iterations.
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    const int ndim = md_mat->GetMeshPointer()->ndim;
    IndexRange ib = md_mat->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_mat->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_mat->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md_mat->GetMeshPointer()->packages.Get("poisson_package");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    int nblocks = md_mat->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc_mat = parthenon::MakePackDescriptor<D_t>(md_mat.get());
    auto desc_diag = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack_mat = desc_mat.GetPack(md_mat.get(), include_block);
    auto pack_diag = desc_diag.GetPack(md_diag.get(), include_block);
    using TE = parthenon::TopologicalElement;
    parthenon::par_for(
        "StoreDiagonal", 0, pack_mat.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack_mat.GetCoordinates(b);
          // Build the unigrid diagonal of the matrix
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          Real diag_elem = -(pack_mat(b, TE::F1, D_t(), k, j, i) +
                             pack_mat(b, TE::F1, D_t(), k, j, i + 1)) /
                               (dx1 * dx1) -
                           alpha;
          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            diag_elem -= (pack_mat(b, TE::F2, D_t(), k, j, i) +
                          pack_mat(b, TE::F2, D_t(), k, j + 1, i)) /
                         (dx2 * dx2);
          }
          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            diag_elem -= (pack_mat(b, TE::F3, D_t(), k, j, i) +
                          pack_mat(b, TE::F3, D_t(), k + 1, j, i)) /
                         (dx3 * dx3);
          }
          pack_diag(b, te, var_t(), k, j, i) = diag_elem;
        });
    return TaskStatus::complete;
  }

  static parthenon::TaskStatus
  CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                  std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    using TE = parthenon::TopologicalElement;

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get(), include_block);
    auto desc_mat = parthenon::MakePackDescriptor<D_t>(md_mat.get(), {});
    auto pack_mat = desc_mat.GetPack(md_mat.get(), include_block);
    parthenon::par_for(
        "CaclulateFluxes", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          pack.flux(b, X1DIR, var_t(), k, j, i) =
              pack_mat(b, TE::F1, D_t(), k, j, i) / dx1 *
              (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i));
          if (i == ib.e)
            pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
                pack_mat(b, TE::F1, D_t(), k, j, i + 1) / dx1 *
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j, i + 1));

          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            pack.flux(b, X2DIR, var_t(), k, j, i) =
                pack_mat(b, TE::F2, D_t(), k, j, i) *
                (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) / dx2;
            if (j == jb.e)
              pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                  pack_mat(b, TE::F2, D_t(), k, j + 1, i) *
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j + 1, i)) /
                  dx2;
          }

          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            pack.flux(b, X3DIR, var_t(), k, j, i) =
                pack_mat(b, TE::F3, D_t(), k, j, i) *
                (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) / dx3;
            if (k == kb.e)
              pack.flux(b, X2DIR, var_t(), k + 1, j, i) =
                  pack_mat(b, TE::F3, D_t(), k + 1, j, i) *
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k + 1, j, i)) /
                  dx3;
          }
        });
    return TaskStatus::complete;
  }

  template <class... var_ts>
  parthenon::TaskID Prolongate(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                               std::shared_ptr<parthenon::MeshData<Real>> &md) {
    if (prolongation_type == ProlongationType::Constant) {
      return tl.AddTask(depends_on, ProlongateImpl<ProlongationType::Constant, var_ts...>,
                        md);
    } else if (prolongation_type == ProlongationType::Linear) {
      return tl.AddTask(depends_on, ProlongateImpl<ProlongationType::Linear, var_ts...>,
                        md);
    } else if (prolongation_type == ProlongationType::Kwak) {
      return tl.AddTask(depends_on, ProlongateImpl<ProlongationType::Kwak, var_ts...>,
                        md);
    }
    return depends_on;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static Real LinearFactor(int d, bool lo_bound, bool up_bound) {
    if (d == 0) return 1.0; // Indicates this dimension is not included
    if (d == 1) return (2.0 + !up_bound) / 4.0;
    if (d == -1) return (2.0 + !lo_bound) / 4.0;
    if (d == 3) return !up_bound / 4.0;
    if (d == -3) return !lo_bound / 4.0;
    return 0.0;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static Real QuadraticFactor(int d) {
    if (d == 0) return 1.0; // Indicates this dimension is not included
    if (d == 1 || d == -1) return 30.0 / 32.0;
    if (d == 3 || d == -3) return 5.0 / 32.0;
    if (d == 5 || d == -5) return -3.0 / 32.0;
    return 0.0;
  }

  template <ProlongationType prolongation_type, class... var_ts>
  static parthenon::TaskStatus
  ProlongateImpl(std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    IndexRange cib = md->GetBoundsI(CellLevel::coarse, IndexDomain::interior);
    IndexRange cjb = md->GetBoundsJ(CellLevel::coarse, IndexDomain::interior);
    IndexRange ckb = md->GetBoundsK(CellLevel::coarse, IndexDomain::interior);

    using TE = parthenon::TopologicalElement;

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);
    for (int b = 0; b < nblocks; ++b) {
      include_block[b] =
          md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level();
    }
    const auto desc = parthenon::MakePackDescriptor<var_ts...>(md.get());
    const auto desc_coarse =
        parthenon::MakePackDescriptor<var_ts...>(md.get(), {}, {PDOpt::Coarse});
    auto pack = desc.GetPack(md.get(), include_block);
    auto pack_coarse = desc_coarse.GetPack(md.get(), include_block);

    parthenon::par_for(
        "Prolongate", 0, pack.GetNBlocks() - 1, pack.GetLowerBoundHost(0),
        pack.GetUpperBoundHost(0), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int n, const int fk, const int fj,
                      const int fi) {
          const int ck = (ndim > 2) ? (fk - kb.s) / 2 + ckb.s : ckb.s;
          const int cj = (ndim > 1) ? (fj - jb.s) / 2 + cjb.s : cjb.s;
          const int ci = (ndim > 0) ? (fi - ib.s) / 2 + cib.s : cib.s;
          const int fok = (fk - kb.s) % 2;
          const int foj = (fj - jb.s) % 2;
          const int foi = (fi - ib.s) % 2;
          const bool bound[6]{pack.IsPhysicalBoundary(b, 0, 0, -1) && (ib.s == fi),
                              pack.IsPhysicalBoundary(b, 0, 0, 1) && (ib.e == fi),
                              pack.IsPhysicalBoundary(b, 0, -1, 0) && (jb.s == fj),
                              pack.IsPhysicalBoundary(b, 0, 1, 0) && (jb.e == fj),
                              pack.IsPhysicalBoundary(b, -1, 0, 0) && (kb.s == fk),
                              pack.IsPhysicalBoundary(b, 1, 0, 0) && (kb.e == fk)};
          // Use both pack and pack_coarse outside of the constexpr if
          // statements to prevent compilation errors in some CUDA compilers
          pack(b, n, fk, fj, fi) = pack_coarse(b, n, ck, cj, ci);
          if constexpr (ProlongationType::Constant == prolongation_type) {
            pack(b, n, fk, fj, fi) = pack_coarse(b, n, ck, cj, ci);
          } else if constexpr (ProlongationType::Linear == prolongation_type) {
            pack(b, n, fk, fj, fi) = 0.0;
            for (int ok = -(ndim > 2); ok < 1 + (ndim > 2); ++ok) {
              for (int oj = -(ndim > 1); oj < 1 + (ndim > 1); ++oj) {
                for (int oi = -(ndim > 0); oi < 1 + (ndim > 0); ++oi) {
                  const int dx3 = (ndim > 2) ? 4 * ok - (2 * fok - 1) : 0;
                  const int dx2 = (ndim > 1) ? 4 * oj - (2 * foj - 1) : 0;
                  const int dx1 = 4 * oi - (2 * foi - 1);
                  pack(b, n, fk, fj, fi) += LinearFactor(dx1, bound[0], bound[1]) *
                                            LinearFactor(dx2, bound[2], bound[3]) *
                                            LinearFactor(dx3, bound[4], bound[5]) *
                                            pack_coarse(b, n, ck + ok, cj + oj, ci + oi);
                }
              }
            }
          } else if constexpr (ProlongationType::Kwak == prolongation_type) {
            pack(b, n, fk, fj, fi) = 0.0;
            if (ndim > 2 && !bound[4 + fok]) {
              for (int ok = fok - 1; ok <= fok; ++ok) {
                pack(b, n, fk, fj, fi) += pack_coarse(b, n, ck + ok, cj, ci);
              }
            }
            if (ndim > 1 && !bound[2 + foj]) {
              for (int oj = foj - 1; oj <= foj; ++oj) {
                pack(b, n, fk, fj, fi) += pack_coarse(b, n, ck, cj + oj, ci);
              }
            }
            if (ndim > 0 && !bound[foi]) {
              for (int oi = foi - 1; oi <= foi; ++oi) {
                pack(b, n, fk, fj, fi) += pack_coarse(b, n, ck, cj, ci + oi);
              }
            }
            pack(b, n, fk, fj, fi) /= 2.0 * ndim;
          }
        });
    return TaskStatus::complete;
  }

  static parthenon::TaskStatus
  SetFluxBoundaries(std::shared_ptr<parthenon::MeshData<Real>> &md, bool do_flux_dx) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);

    using TE = parthenon::TopologicalElement;

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc =
        parthenon::MakePackDescriptor<var_t, D_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get(), include_block);
    const std::size_t scratch_size_in_bytes = 0;
    const std::size_t scratch_level = 1;

    const parthenon::Indexer3D idxers[6]{
        parthenon::Indexer3D(kb, jb, {ib.s, ib.s}),
        parthenon::Indexer3D(kb, jb, {ib.e + 1, ib.e + 1}),
        parthenon::Indexer3D(kb, {jb.s, jb.s}, ib),
        parthenon::Indexer3D(kb, {jb.e + 1, jb.e + 1}, ib),
        parthenon::Indexer3D({kb.s, kb.s}, jb, ib),
        parthenon::Indexer3D({kb.e + 1, kb.e + 1}, jb, ib)};
    constexpr int x1off[6]{-1, 1, 0, 0, 0, 0};
    constexpr int x2off[6]{0, 0, -1, 1, 0, 0};
    constexpr int x3off[6]{0, 0, 0, 0, -1, 1};
    constexpr TE tes[6]{TE::F1, TE::F1, TE::F2, TE::F2, TE::F3, TE::F3};
    constexpr int dirs[6]{X1DIR, X1DIR, X2DIR, X2DIR, X3DIR, X3DIR};
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "SetFluxBoundaries", DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, pack.GetNBlocks() - 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
          const auto &coords = pack.GetCoordinates(b);
          const int gid = pack.GetGID(b);
          const int level = pack.GetLevel(b, 0, 0, 0);
          const Real dxs[3]{coords.template Dxc<X1DIR>(), coords.template Dxc<X2DIR>(),
                            coords.template Dxc<X3DIR>()};
          for (int face = 0; face < ndim * 2; ++face) {
            const Real dx = dxs[dirs[face] - 1];
            const auto &idxer = idxers[face];
            const auto dir = dirs[face];
            const auto te = tes[face];
            // Impose the zero Dirichlet boundary condition at the actual boundary
            if (pack.IsPhysicalBoundary(b, x3off[face], x2off[face], x1off[face])) {
              const int koff = x3off[face] > 0 ? -1 : 0;
              const int joff = x2off[face] > 0 ? -1 : 0;
              const int ioff = x1off[face] > 0 ? -1 : 0;
              const int sign = x1off[face] + x2off[face] + x3off[face];
              parthenon::par_for_inner(
                  DEFAULT_INNER_LOOP_PATTERN, member, 0, idxer.size() - 1,
                  [&](const int idx) {
                    const auto [k, j, i] = idxer(idx);
                    pack.flux(b, dir, var_t(), k, j, i) =
                        sign * pack(b, te, D_t(), k, j, i) *
                        pack(b, var_t(), k + koff, j + joff, i + ioff) / (0.5 * dx);
                  });
            }
            // Correct for size of neighboring zone at fine-coarse boundary when using
            // constant prolongation
            if (do_flux_dx &&
                pack.GetLevel(b, x3off[face], x2off[face], x1off[face]) == level - 1) {
              parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                                       idxer.size() - 1, [&](const int idx) {
                                         const auto [k, j, i] = idxer(idx);
                                         pack.flux(b, dir, var_t(), k, j, i) /= 1.5;
                                       });
            }
          }
        });
    return TaskStatus::complete;
  }

  // Calculate A in_t = out_t (in the region covered by md) for a given set of fluxes
  // calculated with in_t (which have possibly been corrected at coarse fine boundaries)
  static parthenon::TaskStatus
  FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    auto pkg = md->GetMeshPointer()->packages.Get("poisson_package");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    static auto desc =
        parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    static auto desc_out = parthenon::MakePackDescriptor<var_t>(md_out.get());
    auto pack = desc.GetPack(md.get(), include_block);
    auto pack_out = desc_out.GetPack(md_out.get(), include_block);
    parthenon::par_for(
        "FluxMultiplyMatrix", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          pack_out(b, te, var_t(), k, j, i) = -alpha * pack(b, te, var_t(), k, j, i);
          pack_out(b, te, var_t(), k, j, i) +=
              (pack.flux(b, X1DIR, var_t(), k, j, i) -
               pack.flux(b, X1DIR, var_t(), k, j, i + 1)) /
              dx1;

          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            pack_out(b, te, var_t(), k, j, i) +=
                (pack.flux(b, X2DIR, var_t(), k, j, i) -
                 pack.flux(b, X2DIR, var_t(), k, j + 1, i)) /
                dx2;
          }

          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            pack_out(b, te, var_t(), k, j, i) +=
                (pack.flux(b, X3DIR, var_t(), k, j, i) -
                 pack.flux(b, X3DIR, var_t(), k + 1, j, i)) /
                dx3;
          }
        });
    return TaskStatus::complete;
  }
};

} // namespace poisson_package

#endif // EXAMPLE_POISSON_GMG_POISSON_EQUATION_STAGES_HPP_
