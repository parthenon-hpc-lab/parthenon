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

// Self Include
#include "tasks.hpp"

// Imports commonly used names in Parthenon packages
using namespace parthenon::package::prelude;

namespace calculate_pi {
TaskStatus ComputeArea(MeshBlock *pmb) {
  // compute 1/r0^2 \int d^2x in_or_out(x,y) over the block's domain
  Container<Real> &rc = pmb->real_containers.Get();
  int is = pmb->is;
  int js = pmb->js;
  int ks = pmb->ks;
  int ie = pmb->ie;
  int je = pmb->je;
  int ke = pmb->ke;
  Coordinates *pcoord = pmb->pcoord.get();
  CellVariable<Real> &v = rc.Get("in_or_out");
  const auto &radius = pmb->packages["calculate_pi"]->Param<Real>("radius");
  Real area = 0.0;
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      for (int i = is; i <= ie; i++) {
        area += v(k, j, i) * pcoord->dx1f(i) * pcoord->dx2f(j);
      }
    }
  }
  // std::cout << "area = " << area << std::endl;
  area /= (radius * radius);
  // just stash the area somewhere for later
  v(0, 0, 0) = area;
  return TaskStatus::complete;
}
} // namespace calculate_pi
