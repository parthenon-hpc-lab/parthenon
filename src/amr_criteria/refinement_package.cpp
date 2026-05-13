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
  MeshBlock *pmb = rc->GetBlockPointer();
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
  PARTHENON_INSTRUMENT
  SetRefinement_(rc);
  return TaskStatus::complete;
}

template <>
TaskStatus Tag(MeshData<Real> *rc) {
  PARTHENON_INSTRUMENT
  std::vector<std::string> vars = {"U"};
  auto &v = rc->PackVariables(vars);
  IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
  IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
  const int nblocks = rc->NumBlocks();
  Kokkos::View<double*> d_maxd("d_maxd", nblocks);
  AMRBounds bnds(ib,jb,kb);
  const int ndim = 1 + (bnds.je > bnds.js) + (bnds.ke > bnds.ks);  
  parthenon::par_for_outer
    (DEFAULT_OUTER_LOOP_PATTERN, "FusedFirstDerivative", DevExecSpace(), 0, 0,
     0, nblocks - 1, 
     KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
      Real t_maxd = 0;
      const int ksize = bnds.ke - bnds.ks + 1;
      const int jsize = bnds.je - bnds.js + 1;
      const int isize = bnds.ie - bnds.is + 1;
      const int nsize = isize*jsize*ksize;
      Kokkos::parallel_reduce
        (Kokkos::TeamThreadRange(member, nsize),
          [=](const int ii, Real &maxd) {
          int k = ii / (isize*jsize);
          int j = (ii - k*isize*jsize)/isize;
          int i = ii - j*isize - k*isize*jsize;
          k += bnds.ks;
          j += bnds.js;
          i += bnds.is;
          Real scale = std::abs(v(b, 3, k, j, i));
          Real d =
            0.5 * std::abs((v(b, 3, k, j, i + 1) - v(b, 3, k, j, i - 1))) / (scale + TINY_NUMBER);
          maxd = (d > maxd ? d : maxd);
          if (ndim > 1) {
            d = 0.5 * std::abs((v(b, 3, k, j + 1, i) - v(b, 3, k, j - 1, i))) / (scale + TINY_NUMBER);
            maxd = (d > maxd ? d : maxd);
          }
          if (ndim > 2) {
            d = 0.5 * std::abs((v(b, 3, k + 1, j, i) - v(b, 3, k - 1, j, i))) / (scale + TINY_NUMBER);
            maxd = (d > maxd ? d : maxd);
          }
        }, Kokkos::Max<Real>(t_maxd));
      if (member.team_rank() == 0) d_maxd(b) = t_maxd;
    });
  Kokkos::View<double*,Kokkos::CudaSpace>::HostMirror h_maxd = Kokkos::create_mirror_view(d_maxd);
  Kokkos::deep_copy(h_maxd, d_maxd);

  std::vector<AmrTag> flags(nblocks);
  for (int i = 0; i < nblocks; i++) {
    AmrTag t;    
    for (auto &pkg : rc->GetBlockData(i).get()->GetBlockPointer()->packages.AllPackages()) {
      for (auto &amr: pkg.second->amr_criteria) {
        if (h_maxd(i) > amr->refine_criteria) {
          t = AmrTag::refine;
        } else if (h_maxd(i) < amr->derefine_criteria) {
          t = AmrTag::derefine;
        } else {
          t = AmrTag::same;
        }      
      }
    }
    flags[i] = t;
  }
  for (int i = 0; i < nblocks; i++) {
    auto pmb = rc->GetBlockData(i).get()->GetBlockPointer();
    pmb->pmr->SetRefinement(flags[i]);
  }
  NVTX_POP();
  Globals::et_Tag += timer.GetET();  
  return TaskStatus::complete;
  }


} // namespace Refinement
} // namespace parthenon
