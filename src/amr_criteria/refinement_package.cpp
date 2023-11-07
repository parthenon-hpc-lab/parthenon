//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#include <utility>

#include "amr_criteria/amr_criteria.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

namespace parthenon {
namespace Refinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ref = std::make_shared<StateDescriptor>("Refinement");

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

AmrTag CheckAllRefinement(MeshBlockData<Real> *rc) {
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
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  // delta_level holds the max over all criteria.  default to derefining.
  AmrTag delta_level = AmrTag::derefine;
  for (auto &pkg : pmb->packages.AllPackages()) {
    auto &desc = pkg.second;
    delta_level = std::max(delta_level, desc->CheckRefinement(rc));
    if (delta_level == AmrTag::refine) {
      // since 1 is the max, we can return without having to look at anything else
      return AmrTag::refine;
    }
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

void SetRefinement_(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  pmb->pmr->SetRefinement(CheckAllRefinement(rc));
}

template <>
TaskStatus Tag(MeshBlockData<Real> *rc) {
  PARTHENON_TRACE
  PARTHENON_INSTRUMENT
  SetRefinement_(rc);
  return TaskStatus::complete;
}

template <>
TaskStatus Tag(MeshData<Real> *rc) {
  PARTHENON_TRACE
  PARTHENON_INSTRUMENT
  for (int i = 0; i < rc->NumBlocks(); i++) {
    SetRefinement_(rc->GetBlockData(i).get());
  }
  return TaskStatus::complete;
}

} // namespace Refinement
} // namespace parthenon
