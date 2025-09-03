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
#ifndef SOLVERS_INTERNAL_PROLONGATION_HPP_
#define SOLVERS_INTERNAL_PROLONGATION_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "kokkos_abstraction.hpp"

namespace parthenon {

namespace solvers {
// This uses the prolongation operator set in the fields metadata when doing prolongation
// in the interior of a block during multigrid
class ProlongationBlockInteriorDefault {
 public:
  ProlongationBlockInteriorDefault() = default;
  ProlongationBlockInteriorDefault(parthenon::ParameterInput *pin,
                                   const std::string &label) {
    auto pro_int =
        pin->GetOrAddString(label, "block_interior_prolongation", "MetadataDefault");
    PARTHENON_REQUIRE(
        pro_int == "MetadataDefault",
        "Parameter input specifies an unsupported block interior prolongation type.");
  }

  template <class>
  parthenon::TaskID Prolongate(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                               std::shared_ptr<parthenon::MeshData<Real>> &md) {
    return tl.AddTask(depends_on, TF(ProlongateBounds<BoundaryType::gmg_prolongate_recv>),
                      md);
  }
};

// Using this class overrides the prolongation operator set in a fields metadata when
// doing prolongation over the interior of a block during multigrid
class ProlongationBlockInteriorZeroDirichlet {
 public:
  enum class ProlongationType { MetadataDefault, Constant, Linear, Kwak };
  ProlongationType prolongation_type = ProlongationType::Linear;

  ProlongationBlockInteriorZeroDirichlet() = default;
  ProlongationBlockInteriorZeroDirichlet(parthenon::ParameterInput *pin,
                                         const std::string &label) {
    auto pro_int = pin->GetOrAddString(label, "block_interior_prolongation", "Linear");
    if (pro_int == "Constant") {
      prolongation_type = ProlongationType::Constant;
    } else if (pro_int == "Linear") {
      prolongation_type = ProlongationType::Linear;
    } else if (pro_int == "Kwak") {
      prolongation_type = ProlongationType::Kwak;
    } else if (pro_int == "MetadataDefault") {
      prolongation_type = ProlongationType::MetadataDefault;
    } else {
      PARTHENON_FAIL("Invalid zero Dirichlet prolongation type.");
    }
  }

  template <class VarTL>
  parthenon::TaskID Prolongate(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                               std::shared_ptr<parthenon::MeshData<Real>> &md) {
    if (prolongation_type == ProlongationType::Constant) {
      return tl.AddTask(depends_on, TF(ProlongateImpl<ProlongationType::Constant, VarTL>),
                        md);
    } else if (prolongation_type == ProlongationType::Linear) {
      return tl.AddTask(depends_on, TF(ProlongateImpl<ProlongationType::Linear, VarTL>),
                        md);
    } else if (prolongation_type == ProlongationType::Kwak) {
      return tl.AddTask(depends_on, TF(ProlongateImpl<ProlongationType::Kwak, VarTL>),
                        md);
    } else if (prolongation_type == ProlongationType::MetadataDefault) {
      return tl.AddTask(depends_on,
                        TF(ProlongateBounds<BoundaryType::gmg_prolongate_recv>), md);
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

  template <ProlongationType prolongation_type, class VarTL>
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
    const auto desc = parthenon::MakePackDescriptorFromTypeList<VarTL>(md.get());
    const auto desc_coarse = parthenon::MakePackDescriptorFromTypeList<VarTL>(
        md.get(), std::vector<MetadataFlag>{}, std::set<PDOpt>{PDOpt::Coarse});
    auto pack = desc.GetPack(md.get(), include_block);
    auto pack_coarse = desc_coarse.GetPack(md.get(), include_block);
    if (pack.GetNBlocks() == 0) return TaskStatus::complete;

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
};
} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_INTERNAL_PROLONGATION_HPP_
