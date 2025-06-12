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

#include "raycasting/raycasting.hpp"

#include <memory>
#include <string>

#include "defs.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace Raycasting {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_Shared<StateDescriptor>("raycasting");

  int numblock = 0;
  while (true) {
    std::string block_name = "parthenon/raycasting" + std::to_string(numblock);
    if (!pin->DoesBlockExist(block_name)) {
      break;
    }

    numblock++;
  }

  return pkg;
}

Camera::Camera(ParameterInput *pin, const std::string &blockname) {
  bool nx_set = false;
  bool ny_set = false;
  if (pin->DoesParameterExist(blockname, "nx")) {
    resolution_[0] = pin->GetInteger(blockname, "nx");
    nx_set = true;
  }
  if (pin->DoesParameterExist(blockname, "ny")) {
    resolution_[1] = pin->GetInteger(blockname, "ny");
    ny_set = true;
  }
  if (!(nx_set || ny_set)) {
    resolution_[0] = 128;
    nx_set = true;
  }
  if (nx_set && ny_set) {
    PARTHENON_REQUIRE_THROWS(nx * ny > 0,
                             blockname + ": Camera must have finite number of pixels!");
    aspect_ratio_ = static_cast<Real>(nx) / ny;
  } else {
    aspect_ratio_ = pin->GetOrAddReal(blockname, "aspect_ratio", 16.0 / 9.0);
    PARTHENO_REQUIRE_THROWS(aspect_ratio_ > 0,
                            blockname + ": Aspect ration must be positive!");
    if (!nx_set) resolution_[0] = aspect_ratio_ * resolution_[1];
    if (!ny_set) resolution_[1] = resolution_[0] / aspect_ratio_;
  }
  RegionSize mesh(pin);
}

} // namespace Raycasting
} // namespace parthenon
