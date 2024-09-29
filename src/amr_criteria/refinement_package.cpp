//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include "amr_criteria/refinement_package.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "amr_criteria/amr_criteria.hpp"
#include "interface/make_pack_descriptor.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/instrument.hpp"

namespace parthenon {
namespace Refinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ref = std::make_shared<StateDescriptor>("Refinement");
  bool check_refine_mesh =
      pin->GetOrAddBoolean("parthenon/mesh", "CheckRefineMesh", false);
  ref->AddParam("check_refine_mesh", check_refine_mesh);

  int numcrit = 0;
  while (true) {
    std::string block_name = "parthenon/refinement" + std::to_string(numcrit);
    if (!pin->DoesBlockExist(block_name)) {
      break;
    }
    std::string method =
        pin->GetOrAddString(block_name, "method", "PLEASE SPECIFY method");
    ref->amr_criteria.push_back(AMRCriteria::MakeAMRCriteria(method, pin, block_name));
    numcrit++;
  }
  return ref;
}

ParArray1D<AmrTag> CheckAllRefinement(MeshData<Real> *mc) {
  const int nblocks = mc->NumBlocks();
  // maybe not great to allocate this all the time
  auto delta_levels = ParArray1D<AmrTag>(Kokkos::View<AmrTag *>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "delta_levels"), nblocks));
  Kokkos::deep_copy(delta_levels.KokkosView(), AmrTag::derefine);

  Mesh *pm = mc->GetMeshPointer();
  static const bool check_refine_mesh =
      pm->packages.Get("Refinement")->Param<bool>("check_refine_mesh");

  for (auto &pkg : pm->packages.AllPackages()) {
    auto &desc = pkg.second;
    desc->CheckRefinement(mc, delta_levels);

    if (check_refine_mesh) {
      for (auto &amr : desc->amr_criteria) {
        (*amr)(mc, delta_levels);
      }
    }
  }

  return delta_levels;
}

AmrTag CheckAllRefinement(MeshBlockData<Real> *rc, const AmrTag &level) {
  // Check all refinement criteria and return the maximum recommended change in
  // refinement level:
  //   delta_level = -1 => recommend derefinement
  //   delta_level = 0  => leave me alone
  //   delta_level = 1  => recommend refinement
  // NOTE: recommendations from this routine are NOT always followed because
  //    1) the code will not refine more than the global maximum level defined in
  //       <parthenon/mesh>/numlevel in the input
  //    2) the code must maintain proper nesting, which sometimes means a block that is
  //       tagged as "derefine" must be left alone (or possibly refined?) because of
  //       neighboring blocks.  Similarly for "do nothing"
  PARTHENON_INSTRUMENT
  MeshBlock *pmb = rc->GetBlockPointer();
  static const bool check_refine_mesh =
      pmb->packages.Get("Refinement")->Param<bool>("check_refine_mesh");
  // delta_level holds the max over all criteria.  default to derefining, or level from
  // MeshData check.
  AmrTag delta_level = level;
  for (auto &pkg : pmb->packages.AllPackages()) {
    auto &desc = pkg.second;
    delta_level = std::max(delta_level, desc->CheckRefinement(rc));
    if (delta_level == AmrTag::refine) {
      // since 1 is the max, we can return without having to look at anything else
      return AmrTag::refine;
    }
    if (check_refine_mesh) continue;
    // call parthenon criteria that were registered
    for (auto &amr : desc->amr_criteria) {
      // get the recommended change in refinement level from this criteria
      AmrTag temp_delta = (*amr)(rc);
      if ((temp_delta == AmrTag::refine) && pmb->loc.level() >= amr->max_level) {
        // don't refine if we're at the max level
        temp_delta = AmrTag::same;
      }
      // maintain the max across all criteria
      delta_level = std::max(delta_level, temp_delta);
      if (delta_level == AmrTag::refine) {
        // 1 is the max, so just return
        return AmrTag::refine;
      }
    }
  }
  return delta_level;
}

AmrTag FirstDerivative(const AMRBounds &bnds, const ParArray3D<Real> &q,
                       const Real refine_criteria, const Real derefine_criteria) {
  PARTHENON_INSTRUMENT
  const int ndim = 1 + (bnds.je > bnds.js) + (bnds.ke > bnds.ks);
  Real maxd = 0.0;
  par_reduce(
      loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), bnds.ks, bnds.ke,
      bnds.js, bnds.je, bnds.is, bnds.ie,
      KOKKOS_LAMBDA(int k, int j, int i, Real &maxd) {
        Real scale = std::abs(q(k, j, i));
        Real d =
            0.5 * std::abs((q(k, j, i + 1) - q(k, j, i - 1))) / (scale + TINY_NUMBER);
        maxd = (d > maxd ? d : maxd);
        if (ndim > 1) {
          d = 0.5 * std::abs((q(k, j + 1, i) - q(k, j - 1, i))) / (scale + TINY_NUMBER);
          maxd = (d > maxd ? d : maxd);
        }
        if (ndim > 2) {
          d = 0.5 * std::abs((q(k + 1, j, i) - q(k - 1, j, i))) / (scale + TINY_NUMBER);
          maxd = (d > maxd ? d : maxd);
        }
      },
      Kokkos::Max<Real>(maxd));

  if (maxd > refine_criteria) return AmrTag::refine;
  if (maxd < derefine_criteria) return AmrTag::derefine;
  return AmrTag::same;
}

AmrTag SecondDerivative(const AMRBounds &bnds, const ParArray3D<Real> &q,
                        const Real refine_criteria, const Real derefine_criteria) {
  PARTHENON_INSTRUMENT
  const int ndim = 1 + (bnds.je > bnds.js) + (bnds.ke > bnds.ks);
  Real maxd = 0.0;
  par_reduce(
      loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), bnds.ks, bnds.ke,
      bnds.js, bnds.je, bnds.is, bnds.ie,
      KOKKOS_LAMBDA(int k, int j, int i, Real &maxd) {
        Real aqt = std::abs(q(k, j, i)) + TINY_NUMBER;
        Real qavg = 0.5 * (q(k, j, i + 1) + q(k, j, i - 1));
        Real d = std::abs(qavg - q(k, j, i)) / (std::abs(qavg) + aqt);
        maxd = (d > maxd ? d : maxd);
        if (ndim > 1) {
          qavg = 0.5 * (q(k, j + 1, i) + q(k, j - 1, i));
          d = std::abs(qavg - q(k, j, i)) / (std::abs(qavg) + aqt);
          maxd = (d > maxd ? d : maxd);
        }
        if (ndim > 2) {
          qavg = 0.5 * (q(k + 1, j, i) + q(k - 1, j, i));
          d = std::abs(qavg - q(k, j, i)) / (std::abs(qavg) + aqt);
          maxd = (d > maxd ? d : maxd);
        }
      },
      Kokkos::Max<Real>(maxd));

  if (maxd > refine_criteria) return AmrTag::refine;
  if (maxd < derefine_criteria) return AmrTag::derefine;
  return AmrTag::same;
}

void FirstDerivative(const AMRBounds &bnds, MeshData<Real> *mc, const std::string &field,
                     const int &idx, ParArray1D<AmrTag> &delta_levels,
                     const Real refine_criteria_, const Real derefine_criteria_) {
  const auto desc =
      MakePackDescriptor(mc->GetMeshPointer()->resolved_packages.get(), {field});
  auto pack = desc.GetPack(mc);
  const int ndim = mc->GetMeshPointer()->ndim;
  const int nvars = pack.GetMaxNumberOfVars();

  const Real refine_criteria = refine_criteria_;
  const Real derefine_criteria = derefine_criteria_;
  const int var = idx;
  auto scatter_levels = delta_levels.ToScatterView<Kokkos::Experimental::ScatterMax>();
  par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, pack.GetNBlocks() - 1, bnds.ks, bnds.ke, bnds.js,
      bnds.je,
      KOKKOS_LAMBDA(team_mbr_t team_member, const int b, const int k, const int j) {
        Real maxd = 0.;
        par_reduce_inner(
            inner_loop_pattern_ttr_tag, team_member, bnds.is, bnds.ie,
            [&](const int i, Real &maxder) {
              Real scale = std::abs(pack(b, var, k, j, i));
              Real d = 0.5 *
                       std::abs((pack(b, var, k, j, i + 1) - pack(b, var, k, j, i - 1))) /
                       (scale + TINY_NUMBER);
              maxder = (d > maxder ? d : maxder);
              if (ndim > 1) {
                d = 0.5 *
                    std::abs((pack(b, var, k, j + 1, i) - pack(b, var, k, j - 1, i))) /
                    (scale + TINY_NUMBER);
                maxder = (d > maxder ? d : maxder);
              }
              if (ndim > 2) {
                d = 0.5 *
                    std::abs((pack(b, var, k + 1, j, i) - pack(b, var, k - 1, j, i))) /
                    (scale + TINY_NUMBER);
                maxder = (d > maxder ? d : maxder);
              }
            },
            Kokkos::Max<Real>(maxd));
        auto levels_access = scatter_levels.access();
        auto flag = AmrTag::same;
        if (maxd > refine_criteria) flag = AmrTag::refine;
        if (maxd < derefine_criteria) flag = AmrTag::derefine;
        levels_access(b).update(flag);
      });
  delta_levels.ContributeScatter(scatter_levels);
}

void SecondDerivative(const AMRBounds &bnds, MeshData<Real> *mc, const std::string &field,
                      const int &idx, ParArray1D<AmrTag> &delta_levels,
                      const Real refine_criteria_, const Real derefine_criteria_) {
  const auto desc =
      MakePackDescriptor(mc->GetMeshPointer()->resolved_packages.get(), {field});
  auto pack = desc.GetPack(mc);
  const int ndim = mc->GetMeshPointer()->ndim;
  const int nvars = pack.GetMaxNumberOfVars();

  const Real refine_criteria = refine_criteria_;
  const Real derefine_criteria = derefine_criteria_;
  const int var = idx;
  auto scatter_levels = delta_levels.ToScatterView<Kokkos::Experimental::ScatterMax>();
  par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, pack.GetNBlocks() - 1, bnds.ks, bnds.ke, bnds.js,
      bnds.je,
      KOKKOS_LAMBDA(team_mbr_t team_member, const int b, const int k, const int j) {
        Real maxd = 0.;
        par_reduce_inner(
            inner_loop_pattern_ttr_tag, team_member, bnds.is, bnds.ie,
            [&](const int i, Real &maxder) {
              Real aqt = std::abs(pack(b, var, k, j, i)) + TINY_NUMBER;
              Real qavg = 0.5 * (pack(b, var, k, j, i + 1) + pack(b, var, k, j, i - 1));
              Real d = std::abs(qavg - pack(b, var, k, j, i)) / (std::abs(qavg) + aqt);
              maxder = (d > maxder ? d : maxder);
              if (ndim > 1) {
                qavg = 0.5 * (pack(b, var, k, j + 1, i) + pack(b, var, k, j - 1, i));
                d = std::abs(qavg - pack(b, var, k, j, i)) / (std::abs(qavg) + aqt);
                maxder = (d > maxder ? d : maxder);
              }
              if (ndim > 2) {
                qavg = 0.5 * (pack(b, var, k + 1, j, i) + pack(b, var, k - 1, j, i));
                d = std::abs(qavg - pack(b, var, k, j, i)) / (std::abs(qavg) + aqt);
                maxder = (d > maxder ? d : maxder);
              }
            },
            Kokkos::Max<Real>(maxd));
        auto levels_access = scatter_levels.access();
        auto flag = AmrTag::same;
        if (maxd > refine_criteria) flag = AmrTag::refine;
        if (maxd < derefine_criteria) flag = AmrTag::derefine;
        levels_access(b).update(flag);
      });
  delta_levels.ContributeScatter(scatter_levels);
}

void SetRefinement_(MeshBlockData<Real> *rc,
                    const AmrTag &delta_level = AmrTag::derefine) {
  auto pmb = rc->GetBlockPointer();
  pmb->pmr->SetRefinement(CheckAllRefinement(rc, delta_level));
}

template <>
TaskStatus Tag(MeshBlockData<Real> *rc) {
  PARTHENON_INSTRUMENT
  SetRefinement_(rc);
  return TaskStatus::complete;
}

template <>
TaskStatus Tag(MeshData<Real> *rc) {
  PARTHENON_INSTRUMENT
  ParArray1D<AmrTag> delta_levels = CheckAllRefinement(rc);
  auto delta_levels_h = delta_levels.GetHostMirrorAndCopy();

  for (int i = 0; i < rc->NumBlocks(); i++) {
    SetRefinement_(rc->GetBlockData(i).get(), delta_levels_h(i));
  }
  return TaskStatus::complete;
}

} // namespace Refinement
} // namespace parthenon
