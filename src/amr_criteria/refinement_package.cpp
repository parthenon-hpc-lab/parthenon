//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
#include <vector>

#include "amr_criteria/amr_criteria.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "pack/make_pack_descriptor.hpp"
#include "parameter_input.hpp"
#include "utils/instrument.hpp"

namespace parthenon {
namespace Refinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ref = std::make_shared<StateDescriptor>("Refinement");

  for (InputBlock *pib = pin->pfirst_block; pib != nullptr; pib = pib->pnext) {
    if (pib->block_name.compare(0, 20, "parthenon/refinement") != 0) {
      continue;
    }
    std::string method = pin->GetString(
        pib->block_name, "method",
        std::vector<std::string>{"derivative_order_1", "derivative_order_2", "magnitude"},
        "method to use to check for refinement");
    ref->amr_criteria.push_back(
        AMRCriteria::MakeAMRCriteria(method, pin, pib->block_name));
  }
  return ref;
}

ParArray1D<AmrTag> CheckAllRefinement(MeshData<Real> *md) {
  const int nblocks = md->NumBlocks();
  Mesh *pm = md->GetMeshPointer();
  auto amr_tags = pm->GetAmrTags();
  Kokkos::deep_copy(amr_tags.KokkosView(), AmrTag::derefine);

  for (auto &pkg : pm->packages.AllPackages()) {
    auto &desc = pkg.second;
    desc->CheckRefinement(md, amr_tags);

    for (auto &amr : desc->amr_criteria) {
      (*amr)(md, amr_tags);
    }
  }

  return amr_tags;
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
  }
  return delta_level;
}

void FirstDerivative(const AMRBounds &bnds, MeshData<Real> *md, const std::string &field,
                     const int &idx, ParArray1D<AmrTag> &amr_tags,
                     const Real refine_criteria_, const Real derefine_criteria_,
                     const int max_level_) {
  const auto desc = MakePackDescriptor(md, {field});
  auto pack = desc.GetPack(md);
  const int ndim = md->GetMeshPointer()->ndim;
  const int nvars = pack.GetMaxNumberOfVars();

  const Real refine_criteria = refine_criteria_;
  const Real derefine_criteria = derefine_criteria_;
  const int max_level = max_level_;
  const int var = idx;
  // get a scatterview for the tags that will use Kokkos::Max as the reduction operation
  auto scatter_tags = amr_tags.ToScatterView<Kokkos::Experimental::ScatterMax>();
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
        auto tags_access = scatter_tags.access();
        auto flag = AmrTag::same;
        if (maxd > refine_criteria && pack.GetLevel(b, 0, 0, 0) < max_level)
          flag = AmrTag::refine;
        if (maxd < derefine_criteria) flag = AmrTag::derefine;
        tags_access(b).update(flag);
      });
  amr_tags.ContributeScatter(scatter_tags);
}

void SecondDerivative(const AMRBounds &bnds, MeshData<Real> *md, const std::string &field,
                      const int &idx, ParArray1D<AmrTag> &amr_tags,
                      const Real refine_criteria_, const Real derefine_criteria_,
                      const int max_level_) {
  const auto desc = MakePackDescriptor(md, {field});
  auto pack = desc.GetPack(md);
  const int ndim = md->GetMeshPointer()->ndim;
  const int nvars = pack.GetMaxNumberOfVars();

  const Real refine_criteria = refine_criteria_;
  const Real derefine_criteria = derefine_criteria_;
  const int max_level = max_level_;
  const int var = idx;
  // get a scatterview for the tags that will use Kokkos::Max as the reduction operation
  auto scatter_tags = amr_tags.ToScatterView<Kokkos::Experimental::ScatterMax>();
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
        auto tags_access = scatter_tags.access();
        auto flag = AmrTag::same;
        if (maxd > refine_criteria && pack.GetLevel(b, 0, 0, 0) < max_level)
          flag = AmrTag::refine;
        if (maxd < derefine_criteria) flag = AmrTag::derefine;
        tags_access(b).update(flag);
      });
  amr_tags.ContributeScatter(scatter_tags);
}

void Magnitude(const AMRBounds &bnds, MeshData<Real> *md, const std::string &field,
               const int &idx, ParArray1D<AmrTag> &amr_tags, const Real sign,
               const Real refine_criteria_, const Real derefine_criteria_,
               const int max_level_) {
  const auto desc = MakePackDescriptor(md, {field});
  auto pack = desc.GetPack(md);
  const int ndim = md->GetMeshPointer()->ndim;
  const int nvars = pack.GetMaxNumberOfVars();

  const Real refine_criteria = refine_criteria_;
  const Real derefine_criteria = derefine_criteria_;
  const int max_level = max_level_;
  const int var = idx;
  // get a scatterview for the tags that will use Kokkos::Max as the reduction operation
  auto scatter_tags = amr_tags.ToScatterView<Kokkos::Experimental::ScatterMax>();
  par_for_outer(
      PARTHENON_AUTO_LABEL, 0, 0, 0, pack.GetNBlocks() - 1, bnds.ks, bnds.ke, bnds.js,
      bnds.je,
      KOKKOS_LAMBDA(team_mbr_t team_member, const int b, const int k, const int j) {
        // JMM: sign = 1  if you want to refine on mag > threshold
        //      sign = -1 if you want to regine on mag < threshold
        Real maxval;
        par_reduce_inner(
            inner_loop_pattern_ttr_tag, team_member, bnds.is, bnds.ie,
            [&](const int i, Real &r) {
              Real val = sign * pack(b, var, k, j, i);
              r = std::max(r, val);
            },
            Kokkos::Max<Real>(maxval));
        auto tags_access = scatter_tags.access();
        auto flag = AmrTag::same;
        if (maxval > sign * refine_criteria && pack.GetLevel(b, 0, 0, 0) < max_level)
          flag = AmrTag::refine;
        if (maxval < sign * derefine_criteria) flag = AmrTag::derefine;
        tags_access(b).update(flag);
      });
  amr_tags.ContributeScatter(scatter_tags);
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
TaskStatus Tag(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  ParArray1D<AmrTag> amr_tags = CheckAllRefinement(md);
  auto amr_tags_h = amr_tags.GetHostMirrorAndCopy();

  for (int i = 0; i < md->NumBlocks(); i++) {
    SetRefinement_(md->GetBlockData(i).get(), amr_tags_h(i));
  }
  return TaskStatus::complete;
}

} // namespace Refinement
} // namespace parthenon
