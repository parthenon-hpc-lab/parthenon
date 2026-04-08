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
#ifndef AMR_CRITERIA_AMR_CRITERIA_HPP_
#define AMR_CRITERIA_AMR_CRITERIA_HPP_

#include <memory>
#include <string>
#include <vector>

#include "defs.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

class ParameterInput;
template <class>
class MeshBlockData;

struct AMRBounds {
  AMRBounds(const IndexRange &ib, const IndexRange &jb, const IndexRange &kb)
      : is(ib.s - (ib.e != ib.s)), ie(ib.e + (ib.e != ib.s)), js(jb.s - (jb.e != jb.s)),
        je(jb.e + (jb.e != jb.s)), ks(kb.s - (kb.e != kb.s)), ke(kb.e + (kb.e != kb.s)) {}
  const int is, ie, js, je, ks, ke;
};

struct AMRCriteria {
  AMRCriteria(ParameterInput *pin, std::string &block_name);
  virtual ~AMRCriteria() {}
  virtual void operator()(MeshData<Real> *md, ParArray1D<AmrTag> &delta_level) const = 0;
  std::string field;
  Real refine_criteria, derefine_criteria;
  int max_level;
  int comp6, comp5, comp4;
  static std::shared_ptr<AMRCriteria>
  MakeAMRCriteria(std::string &criteria, ParameterInput *pin, std::string &block_name);
};

struct AMRFirstDerivative : public AMRCriteria {
  AMRFirstDerivative(ParameterInput *pin, std::string &block_name)
      : AMRCriteria(pin, block_name) {}
  void operator()(MeshData<Real> *md, ParArray1D<AmrTag> &delta_level) const override;
};

struct AMRSecondDerivative : public AMRCriteria {
  AMRSecondDerivative(ParameterInput *pin, std::string &block_name)
      : AMRCriteria(pin, block_name) {}
  void operator()(MeshData<Real> *md, ParArray1D<AmrTag> &delta_level) const override;
};

struct AMRMagnitude : public AMRCriteria {
  AMRMagnitude(ParameterInput *pin, std::string &block_name)
      : AMRCriteria(pin, block_name) {
    std::string comparator =
        pin->GetOrAddString(block_name, "comparator", "greater_than",
                            std::vector<std::string>{"greater_than", "less_than"},
                            "greater_than implies large magnitudes trigger refinement. "
                            "less_than implies small magnitudes trigger refinement.");
    if (comparator == "greater_than") {
      sign = 1.0;
    } else { // if (comarator == "less_than") {
      sign = -1.0;
    }
  }
  void operator()(MeshData<Real> *md, ParArray1D<AmrTag> &delta_level) const override;
  Real sign;
};

} // namespace parthenon

#endif // AMR_CRITERIA_AMR_CRITERIA_HPP_
