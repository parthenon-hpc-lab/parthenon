//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_

#include "config.hpp"
#include "kokkos_abstraction.hpp"
#include "uniform_cartesian.hpp"

namespace parthenon {

using Coordinates_t = COORDINATE_TYPE;

template <typename Data, typename Pack>
ParArray1D<Coordinates_t> GetCoordinates(Data *rc, Pack &p);

} // namespace parthenon

#endif // COORDINATES_COORDINATES_HPP_
