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

#ifndef PARTHENON_REFINEMENT_REFINEMENT_HPP_
#define PARTHENON_REFINEMENT_REFINEMENT_HPP_

#include <memory>
#include <string>
#include "athena.hpp"
#include "interface/Container.hpp"
#include "interface/StateDescriptor.hpp"
#include "interface/Variable.hpp"

namespace parthenon {

class ParameterInput;

namespace Refinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

int CheckAllRefinement(Container<Real>& rc);

int FirstDerivative(Variable<Real>& q,
                    const Real refine_criteria, const Real derefine_criteria);

} // namespace Refinement

} // namespace parthenon

#endif // PARTHENON_REFINEMENT_REFINEMENT_HPP_
