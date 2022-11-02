//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef REFINEMENT_AMR_CRITERIA_HPP_
#define REFINEMENT_AMR_CRITERIA_HPP_

#include <memory>
#include <string>

#include "defs.hpp"
#include "interface/meshblock_data.hpp"

namespace parthenon {

class ParameterInput;

struct AMRCriteria {
  AMRCriteria() = default;
  virtual ~AMRCriteria() {}
  virtual AmrTag operator()(const MeshBlockData<Real> *rc) const = 0;
  std::string field;
  Real refine_criteria, derefine_criteria;
  int max_level;
  int comp6, comp5, comp4;
  static std::shared_ptr<AMRCriteria>
  MakeAMRCriteria(std::string &criteria, ParameterInput *pin, std::string &block_name);
};

struct AMRFirstDerivative : public AMRCriteria {
  AMRFirstDerivative(ParameterInput *pin, std::string &block_name);
  AmrTag operator()(const MeshBlockData<Real> *rc) const override;
};

} // namespace parthenon

#endif // REFINEMENT_AMR_CRITERIA_HPP_
