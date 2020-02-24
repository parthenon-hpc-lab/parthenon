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
#include "mesh/mesh.hpp"

class MeshBlock;

namespace Update {

void FluxDivergence(Container<Real> &in, Container<Real> &dudt_cont);
void UpdateContainer(Container<Real> &in, Container<Real> &dudt_cont,
                     const Real dt, Container<Real> &out);
void AverageContainers(Container<Real> &c1, Container<Real> &c2,
                       const Real wgt1);

void FillDerived(Container<Real> &rc);

Real EstimateTimestep(Container<Real> &rc);

} // namespace Update

using FillDerivedFunc = void (Container<Real>&);
class FillDerivedVariables {
  public:
    static void SetFillDerivedFunctions(FillDerivedFunc *pre, FillDerivedFunc *post) {pre_package_fill = pre; post_package_fill = post;}
    static void FillDerived(Container<Real> &rc); 
    static FillDerivedFunc *pre_package_fill;
    static FillDerivedFunc *post_package_fill;
  private:
    FillDerivedVariables() {}; // private constructor means we'll never instantiate a FillDerivedVariables object
};

#endif
