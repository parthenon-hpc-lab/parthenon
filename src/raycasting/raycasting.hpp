//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

#ifndef RAYCASTING_RAYCASTING_HPP_
#define RAYCASTING_RAYCASTING_HPP_

#include <array>
#include <memory>
#include <string>

#include "defs.hpp"
#include "kokkos_abstraction.hpp"

namespace parthenon {
struct StateDescriptor;
struct ParameterInput;

namespace Raycasting {

struct Camera {
 public:
  Camera(ParameterInput *pin, const std::string &blockname);
  KOKKOS_FORCEINLINE_FUNCTION
  Real AspectRatio() const { return aspect_ratio_; }
  KOKKOS_FORCEINLINE_FUNCTION
  const auto &Resolution() const { return resolution_; }
  KOKKOS_FORCEINLINE_FUNCTION
  const auto &Position() const { return position_; }
  KOKKOS_FORCEINLINE_FUNCTION
  const auto &Target() const { return target_; }
  KOKKOS_FORCEINLINE_FUNCTION
  const auto &FieldOfView() const { return field_of_view_; }

 private:
  // aspect ratio
  Real aspect_ratio_;
  // number of pixels in x,y directions
  std::array<std::size_t, 2> resolution_;
  // location of camera
  std::array<Real, 3> position_;
  // where the camera is looking at. Defaults to center of physical
  // domain.
  std::array<Real, 3> target_;
  // physical size of camera plane
  std::array<Real, 2> field_of_view_;
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

} // namespace Raycasting
} // namespace parthenon

#endif // RAYCASTING_RAYCASTING_HPP_
