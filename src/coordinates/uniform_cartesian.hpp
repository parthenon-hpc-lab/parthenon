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
#ifndef COORDINATES_UNIFORM_CARTESIAN_HPP_
#define COORDINATES_UNIFORM_CARTESIAN_HPP_

#include "uniform_coordinates.hpp"

namespace parthenon {

class UniformCartesian : public UniformCoordinates<UniformCartesian> {
 public:
  UniformCartesian() = default;
  UniformCartesian(const RegionSize &rs, ParameterInput *pin) 
    : UniformCoordinates<UniformCartesian>(rs, pin) {}
  UniformCartesian(const UniformCartesian &src, int coarsen)
      : UniformCoordinates<UniformCartesian>(src, coarsen) {}

  constexpr static const char *name_ = "UniformCartesian";
 private:
};

} // namespace parthenon

#endif // COORDINATES_UNIFORM_CARTESIAN_HPP_
