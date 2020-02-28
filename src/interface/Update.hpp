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
#ifndef UPDATE_HPP_PK
#define UPDATE_HPP_PK

#include "athena.hpp"
#include "interface/Container.hpp"

using PreFillDerivedFunc = std::function<void(Container<Real> &)>;

namespace Update {

void FluxDivergence(Container<Real> &in, Container<Real> &dudt_cont);
void UpdateContainer(Container<Real> &in, Container<Real> &dudt_cont,
                     const Real dt, Container<Real> &out);
void AverageContainers(Container<Real> &c1, Container<Real> &c2,
                       const Real wgt1);

void FillDerived(PreFillDerivedFunc pre_fill_derived, Container<Real> &rc);

Real EstimateTimestep(Container<Real> &rc);

} // namespace Update

#endif
