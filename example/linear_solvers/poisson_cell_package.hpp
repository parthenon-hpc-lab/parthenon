//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_LINEAR_SOLVERS_POISSON_CELL_PACKAGE_HPP_
#define EXAMPLE_LINEAR_SOLVERS_POISSON_CELL_PACKAGE_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "linear_solver_driver.hpp"
#include "variable_type.hpp"

namespace poisson_cell_package {
using namespace parthenon::package::prelude;

VARIABLE(poisson, D);
VARIABLE(poisson, u);
VARIABLE(poisson, rhs);
VARIABLE(poisson, exact);

// Meshdata container labels
inline const std::string u_label = "cell_u";
inline const std::string rhs_label = "cell_rhs";
inline const std::string exact_label = "cell_exact";

// This just provides a convenient short hand for TE::CC and will make it
// easier for testing solves with different topological elements in the
// future (although other types of fields require significantly different
// condition boundary implementations)
constexpr parthenon::TopologicalElement te = parthenon::TopologicalElement::CC;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
parthenon::TaskStatus SetVector(parthenon::ParameterInput *pin, bool use_exponential,
                                std::shared_ptr<parthenon::MeshData<parthenon::Real>> md);
parthenon::TaskStatus SetD(parthenon::ParameterInput *pin,
                           std::shared_ptr<parthenon::MeshData<parthenon::Real>> md);
void AddTaskRegion(parthenon::TaskCollection &tc,
                   linear_solver_example::LinearSolverDriver *driver);
} // namespace poisson_cell_package

#endif // EXAMPLE_LINEAR_SOLVERS_POISSON_CELL_PACKAGE_HPP_
