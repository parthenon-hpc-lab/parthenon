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

#ifndef BETTER_REFINEMENT_HPP_PK
#define BETTER_REFINEMENT_HPP_PK

#include <string>
#include "athena.hpp"
#include "interface/Variable.hpp"
#include "interface/Container.hpp"

namespace parthenon {
using RefineFunction = int (Variable<Real>& , const Real , const Real);

struct AMRCriteria {
  public:
    AMRCriteria(std::string field, RefineFunction* rfunc, const Real refine_criteria, const Real derefine_criteria) :
                _field(field), _refine_func(rfunc), _refine_criteria(refine_criteria), _derefine_criteria(derefine_criteria) {}
    std::string _field;
    RefineFunction* _refine_func;
    const Real _refine_criteria, _derefine_criteria;
};

namespace BetterRefinement {
  int CheckRefinement(Container<Real>& rc);
  int FirstDerivative(Variable<Real>& q, const Real refine_criteria, const Real derefine_criteria);
};

}
#endif
