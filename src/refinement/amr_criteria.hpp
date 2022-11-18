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

struct AMRBounds {
  AMRBounds(const IndexRange &ib, const IndexRange &jb, const IndexRange &kb)
      : is(ib.s - (ib.e != ib.s)), ie(ib.e + (ib.e != ib.s)), js(jb.s - (jb.e != jb.s)),
        je(jb.e + (jb.e != jb.s)), ks(kb.s - (kb.e != kb.s)), ke(kb.e + (kb.e != kb.s)) {}
  const int is, ie, js, je, ks, ke;
};

struct AMRCriteria {
  AMRCriteria(ParameterInput *pin, std::string &block_name);
  virtual ~AMRCriteria() {}
  virtual AmrTag operator()(const MeshBlockData<Real> *rc) const = 0;
  std::string field;
  Real refine_criteria, derefine_criteria;
  int max_level;
  int comp6, comp5, comp4;
  static std::shared_ptr<AMRCriteria>
  MakeAMRCriteria(std::string &criteria, ParameterInput *pin, std::string &block_name);
  AMRBounds GetBounds(const MeshBlockData<Real> *rc) const;
};

struct AMRFirstDerivative : public AMRCriteria {
  AMRFirstDerivative(ParameterInput *pin, std::string &block_name)
      : AMRCriteria(pin, block_name) {}
  AmrTag operator()(const MeshBlockData<Real> *rc) const override;
};

struct AMRSecondDerivative : public AMRCriteria {
  AMRSecondDerivative(ParameterInput *pin, std::string &block_name)
      : AMRCriteria(pin, block_name) {}
  AmrTag operator()(const MeshBlockData<Real> *rc) const override;
};

} // namespace parthenon

#endif // REFINEMENT_AMR_CRITERIA_HPP_
