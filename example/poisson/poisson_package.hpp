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

#include <parthenon/package.hpp>

namespace poisson {

using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus Smooth(std::shared_ptr<Container<Real>> &rc_in,
                  std::shared_ptr<Container<Real>> &rc_out);
Real GetL1Residual(std::shared_ptr<Container<Real>> &rc);
// Residual and diagonal coalesced into a single kernel for performance
TaskStatus ComputeResidualAndDiagonal(std::shared_ptr<Container<Real>> &div,
                                      std::shared_ptr<Container<Real>> &update);
TaskStatus CalculateFluxes(std::shared_ptr<Container<Real>> &rc);

} // namespace poisson

#endif // EXAMPLE_POISSON_POISSON_PACKAGE_HPP_
