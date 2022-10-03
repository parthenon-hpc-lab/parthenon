//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_COUNT_CELLS_COUNT_CELLS_HPP_
#define EXAMPLE_COUNT_CELLS_COUNT_CELLS_HPP_

// C++ Includes
#include <memory>

// Parthenon Includes
#include <coordinates/coordinates.hpp>
#include <interface/state_descriptor.hpp>
#include <parthenon/package.hpp>

namespace count_cells {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using parthenon::Mesh;
using parthenon::MeshBlock;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
AmrTag CheckRefinement(MeshBlockData<Real> *rc);
bool BlockInRegion(const StateDescriptor *pkg, MeshBlock *pmb);
bool SufficientlyRefined(const StateDescriptor *pkg, const Coordinates_t &coords);
void CountCells(Mesh *pmesh);

} // namespace count_cells

#endif // EXAMPLE_COUNT_CELLS_COUNT_CELLS_HPP_
