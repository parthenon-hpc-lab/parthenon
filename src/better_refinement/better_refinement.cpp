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

#include "better_refinement.hpp"
#include "interface/StateDescriptor.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace parthenon {
namespace BetterRefinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ref = std::make_shared<StateDescriptor>("Refinement");
  Params& params = ref->AllParams();

  std::string base("Refinement");
  int numcrit = 0;
  for(;;) {
    std::string block_name = base + std::to_string(numcrit);
    if (!pin->DoesBlockExist(block_name)) {
      break;
    }
    int method = pin->GetOrAddInteger(block_name, "method", 0);
    switch(method) {
      case 0:
        ref->amr_criteria.push_back(
          std::make_unique<AMRFirstDerivative>(pin, block_name)
        );
        break;
      default:
        throw std::invalid_argument(
          "Invalid selection for refinment method in " + block_name
        );
    }
    numcrit++;
  }
  return std::move(ref);
}


int CheckRefinement(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  int delta_level = -1;
  for (auto &phys : pmb->physics) {
    auto& desc = phys.second;
    if (desc->CheckRefinement != nullptr) {
        int package_delta_level = desc->CheckRefinement(rc);
        delta_level = std::max(delta_level, package_delta_level);
        if (delta_level == 1) break;
    }
  }

  if (delta_level != 1) {
    for (auto & phys : pmb->physics) {
      for (auto & amr : phys.second->amr_criteria) {
        int package_delta_level = (*amr)(rc);
        if (package_delta_level == 0) {
          delta_level = std::max(delta_level, package_delta_level);
        } else if (package_delta_level == 1 && rc.pmy_block->loc.level < amr->max_level) {
          delta_level = 1;
          break;
        }
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
  int klo=0, khi=1, jlo=0, jhi=1, ilo=0, ihi=1;
  if (dim3 > 1) {
    klo = 1;
    khi = dim3-1;
  }
  if (dim2 > 1) {
    jlo = 1;
    jhi = dim2-1;
  }
  if (dim1 > 1) {
    ilo = 1;
    ihi = dim1-1;
  }
  for (int k=klo; k<khi; k++) {
    for (int j=jlo; j<jhi; j++) {
      for (int i=ilo; i<ihi; i++) {
        Real scale = std::abs(q(k,j,i));
        Real d = 0.5*std::abs((q(k,j,i+1)-q(k,j,i-1)))/(scale+1.e-16);
        maxd = (d > maxd ? d : maxd);
        if (dim2 > 1) {
          d = 0.5*std::abs((q(k,j+1,i)-q(k,j-1,i)))/(scale+1.e-16);
          maxd = (d > maxd ? d : maxd);
        }
        if (dim3 > 1) {
          d = 0.5*std::abs((q(k+1,j,i) - q(k-1,j,i)))/(scale+1.e-16);
          maxd = (d > maxd ? d : maxd);
        }
      }
    }
  }
  if (maxd > refine_criteria) return 1;
  if (maxd < derefine_criteria) return -1;
  return 0;;
}

} // namespace BetterRefinement
} // namespace parthenon
