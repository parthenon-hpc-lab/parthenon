//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_LINEAR_SOLVERS_HELMHOLTZ_EQUATION_HPP_
#define EXAMPLE_LINEAR_SOLVERS_HELMHOLTZ_EQUATION_HPP_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "helmholtz_package.hpp"

namespace helmholtz_package {
using namespace parthenon::package::prelude;

// This class implement methods for calculating A.x = y and returning the diagonal of A,
// where A is the the matrix representing the discretized Poisson equation on the grid.
// Here we implement the Laplace operator in terms of a flux divergence to (potentially)
// consistently deal with coarse fine boundaries on the grid. Only the routines Ax and
// SetDiagonal need to be defined for interfacing this with solvers. The other methods
// are internal, but can't be marked private or protected because they launch kernels
// on device.
class HelmholtzEquation {
 public:
  using vcc_t = u;
  using vfc_t = F;
  using IndependentVars = parthenon::TypeList<vcc_t, vfc_t>;

  HelmholtzEquation(parthenon::ParameterInput *pin, const std::string &label) {}

  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> & /*md_mat*/,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    return tl.AddTask(depends_on, AxImpl, md_in, md_out);
  }

  static parthenon::TaskStatus AxImpl(std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                                      std::shared_ptr<parthenon::MeshData<Real>> &md_out);

  static parthenon::TaskStatus SetBoundary(std::shared_ptr<parthenon::MeshData<Real>> &md,
                                           bool coarse);

  parthenon::TaskStatus
  SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> & /*md_mat*/,
              std::shared_ptr<parthenon::MeshData<Real>> &md_diag);
};

} // namespace helmholtz_package

#endif // EXAMPLE_LINEAR_SOLVERS_HELMHOLTZ_EQUATION_HPP_
