//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_POISSON_CG_POISSON_CG_PACKAGE_HPP_
#define EXAMPLE_POISSON_CG_POISSON_CG_PACKAGE_HPP_

#include <memory>

#include <parthenon/package.hpp>

namespace poisson_package {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
template <typename T>
TaskStatus Jacobian(T *u);
TaskStatus PrintComplete();

template <typename T>
TaskStatus Residual(T *u, Real *res);

} // namespace poisson_package

#endif // EXAMPLE_POISSON_CG_POISSON_CG_PACKAGE_HPP_
