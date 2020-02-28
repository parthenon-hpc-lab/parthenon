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

#include "better_refinement.hpp"

#include "mesh/mesh.hpp"
#include "interface/StateDescriptor.hpp"

namespace BetterRefinement {

int CheckRefinement(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  int delta_level = -1;
  for (auto &phys : pmb->physics) {
    auto& desc = phys.second;
    if (desc->CheckRefinement != nullptr) {
        delta_level = desc->CheckRefinement(rc);
        if (delta_level == 1) break;
    }
  }

  if (delta_level != 1) {
    for (auto & phys : pmb->physics) {
      for (auto & amr : phys.second->amr_criteria) {
        Variable<Real> q = pmb->real_container.Get(amr._field);
        delta_level = amr._refine_func(q, amr._refine_criteria, amr._derefine_criteria);
        if (delta_level == 1) break;
      }
    }
  }

  return delta_level;
}

int FirstDerivative(Variable<Real>& q, const Real refine_criteria, const Real derefine_criteria) {
    return -1;
}

}