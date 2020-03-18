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
#include "amr_criteria.hpp"
#include "better_refinement.hpp"
#include "interface/Container.hpp"
#include "interface/Variable.hpp"
#include "parameter_input.hpp"

namespace parthenon {

AMRFirstDerivative::AMRFirstDerivative(ParameterInput *pin, std::string block_name) {
    _field = pin->GetOrAddString(block_name, "field", "none");
    if (!_field.compare("none")) {
      std::cerr << "Error in " << block_name << ": no field set" << std::endl;
      exit(1);
    }
    refine_criteria = pin->GetOrAddReal(block_name, "refine_tol", 0.5);
    derefine_criteria = pin->GetOrAddReal(block_name, "derefine_tol", 0.05);
    int global_max_level = pin->GetOrAddInteger("mesh", "numlevel", 1);
    max_level = pin->GetOrAddInteger(block_name, "max_level", global_max_level);
}

int AMRFirstDerivative::operator()(Container<Real>& rc) {
  Variable<Real>& q = rc.Get(_field);
  return BetterRefinement::FirstDerivative(q, refine_criteria, derefine_criteria);
}

} // namespace parthenon
