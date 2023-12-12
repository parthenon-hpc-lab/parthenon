//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef BENCHMARKS_BURGERS_BURGERS_PACKAGE_HPP_
#define BENCHMARKS_BURGERS_BURGERS_PACKAGE_HPP_

#include <algorithm>
#include <memory>

#include <parthenon/package.hpp>

namespace burgers_package {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
void CalculateDerived(MeshData<Real> *md);
Real EstimateTimestepMesh(MeshData<Real> *md);
TaskStatus CalculateFluxes(MeshData<Real> *md);
Real MassHistory(MeshData<Real> *md, const Real x1min, const Real x1max, const Real x2min,
                 const Real x2max, const Real x3min, const Real x3max);
Real MeshCountHistory(MeshData<Real> *md);

// compute the hll flux for Burgers' equation
KOKKOS_INLINE_FUNCTION
void lr_to_flux(const Real uxl, const Real uxr, const Real uyl, const Real uyr,
                const Real uzl, const Real uzr, const Real upl, const Real upr, Real &sl,
                Real &sr, Real &fux, Real &fuy, Real &fuz) {
  sl = std::min(std::min(upl, upr), 0.0);
  sr = std::max(std::max(upl, upr), 0.0);
  const Real islsr = 1.0 / (sr - sl + (sl * sr == 0.0));

  fux = 0.5 * (sr * uxl * upl - sl * uxr * upr + sl * sr * (uxr - uxl)) * islsr;
  fuy = 0.5 * (sr * uyl * upl - sl * uyr * upr + sl * sr * (uyr - uyl)) * islsr;
  fuz = 0.5 * (sr * uzl * upl - sl * uzr * upr + sl * sr * (uzr - uzl)) * islsr;
}

struct Region {
  std::array<Real, 3> xmin, xmax;
};

} // namespace burgers_package

#endif // BENCHMARKS_BURGERS_BURGERS_PACKAGE_HPP_
