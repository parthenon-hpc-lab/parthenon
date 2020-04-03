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
#include <algorithm>
#include <exception>
#include <memory>
#include <utility>

#include "amr_criteria.hpp"
#include "refinement.hpp"
#include "defs.hpp"
#include "interface/StateDescriptor.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace parthenon {
namespace Refinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ref = std::make_shared<StateDescriptor>("Refinement");
  Params& params = ref->AllParams();

  int numcrit = 0;
  while(true) {
    std::string block_name = "Refinement" + std::to_string(numcrit);
    if (!pin->DoesBlockExist(block_name)) {
      break;
    }
    std::string method = pin->GetOrAddString(block_name, "method", "PLEASE SPECIFY method");
    ref->amr_criteria.push_back(
      AMRCriteria::MakeAMRCriteria(method, pin, block_name)
    );
    numcrit++;
  }
  return std::move(ref);
}


int CheckAllRefinement(Container<Real>& rc) {
  // Check all refinement criteria and return the maximum recommended change in
  // refinement level:
  //   delta_level = -1 => recommend derefinement
  //   delta_level = 0  => leave me alone
  //   delta_level = 1  => recommend refinement
  // NOTE: recommendations from this routine are NOT always followed because
  //    1) the code will not refine more than the global maximum level defined in
  //       <mesh>/numlevel in the input
  //    2) the code must maintain proper nesting, which sometimes means a block that is
  //       tagged as "derefine" must be left alone (or possibly refined?) because of
  //       neighboring blocks.  Similarly for "do nothing"
  MeshBlock *pmb = rc.pmy_block;
  // delta_level holds the max over all criteria.  default to derefining.
  int delta_level = -1;
  for (auto &pkg : pmb->packages) {
    auto& desc = pkg.second;
    // call package specific function, if set
    if (desc->CheckRefinement != nullptr) {
        // keep the max over all criteria up to date
        delta_level = std::max(delta_level, desc->CheckRefinement(rc));
        if (delta_level == 1) {
          // since 1 is the max, we can return without having to look at anything else
          return 1;
        }
    }
    // call parthenon criteria that were registered
    for (auto & amr : desc->amr_criteria) {
      // get the recommended change in refinement level from this criteria
      int temp_delta = (*amr)(rc);
      if ( (temp_delta == 1) && rc.pmy_block->loc.level >= amr->max_level) {
        // don't refine if we're at the max level
        temp_delta = 0;
      }
      // maintain the max across all criteria
      delta_level = std::max(delta_level, temp_delta);
      if (delta_level == 1) {
        // 1 is the max, so just return
        return 1;
      }
    }
  }
  return delta_level;
}

int FirstDerivative(Variable<Real>& q,
                    const Real refine_criteria, const Real derefine_criteria) {
  Real maxd = 0.0;
  const int dim1 = q.GetDim1();
  const int dim2 = q.GetDim2();
  const int dim3 = q.GetDim3();
  int kl=0, ku=0, jl=0, ju=0, il=0, iu=0;
  if (dim3 > 1) {
    kl = 1;
    ku = dim3-2;
  }
  if (dim2 > 1) {
    jl = 1;
    ju = dim2-2;
  }
  if (dim1 > 1) {
    il = 1;
    iu = dim1-2;
  }
  for (int k=kl; k<=ku; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real scale = std::abs(q(k,j,i));
        Real d = 0.5*std::abs((q(k,j,i+1)-q(k,j,i-1)))/(scale+TINY_NUMBER);
        maxd = (d > maxd ? d : maxd);
        if (dim2 > 1) {
          d = 0.5*std::abs((q(k,j+1,i)-q(k,j-1,i)))/(scale+TINY_NUMBER);
          maxd = (d > maxd ? d : maxd);
        }
        if (dim3 > 1) {
          d = 0.5*std::abs((q(k+1,j,i) - q(k-1,j,i)))/(scale+TINY_NUMBER);
          maxd = (d > maxd ? d : maxd);
        }
      }
    }
  }
  if (maxd > refine_criteria) return 1;
  if (maxd < derefine_criteria) return -1;
  return 0;;
}

} // namespace Refinement
} // namespace parthenon
