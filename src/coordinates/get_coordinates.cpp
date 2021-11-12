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

#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {

template <typename Pack>
ParArray1D<Coordinates_t> GetCoordinates(MeshBlockData<Real> *rc, Pack &p) {
  ParArray1D<Coordinates_t> coords("Parthenon::Coordinates_t array", 1);
  coords(0) = rc->GetParentPointer()->coords;
  return coords;
}

template <typename Pack>
ParArray1D<Coordinates_t> GetCoordinates(MeshData<Real> *rc, Pack &p) {
  return p.coords;
}

} // namespace parthenon
