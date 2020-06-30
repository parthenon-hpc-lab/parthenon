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
#ifndef EXAMPLE_POISSON_POISSON_PACKAGE_HPP_
#define EXAMPLE_POISSON_POISSON_PACKAGE_HPP_

#include <memory>

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/container.hpp"
#include "interface/state_descriptor.hpp"
#include "task_list/tasks.hpp"

using parthenon::Container;
using parthenon::Coordinates_t;
using parthenon::ParameterInput;
using parthenon::ParArrayND;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::TaskStatus;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

namespace poisson {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus Smooth(Container<Real> &rc_in, Container<Real> &rc_out);
Real GetResidual(Container<Real> &rc);

KOKKOS_INLINE_FUNCTION
Real residual(ParArrayND<Real> &phi, ParArrayND<Real> &rho, Coordinates_t &coords, Real K,
              int ndim, int k, int j, int i) {
  Real dx = coords.Dx(X1DIR, k, j, i);
  Real dy = coords.Dx(X2DIR, k, j, i);
  Real dz = coords.Dx(X3DIR, k, j, i);
  Real dx2 = dx * dx;
  Real dy2 = dy * dy;
  Real dz2 = dz * dz;
  Real residual = 0;
  residual += (phi(k, j, i + 1) + phi(k, j, i - 1) - 2 * phi(k, j, i)) / dx2;
  if (ndim >= 2) {
    residual += (phi(k, j + 1, i) + phi(k, j - 1, i) - 2 * phi(k, j, i)) / dy2;
  }
  if (ndim >= 3) {
    residual += (phi(k + 1, j, i) + phi(k - 1, j, i) - 2 * phi(k, j, i)) / dz2;
  }
  residual -= K * rho(k, j, i);
  return residual;
}

} // namespace poisson

#endif // EXAMPLE_POISSON_POISSON_PACKAGE_HPP_
