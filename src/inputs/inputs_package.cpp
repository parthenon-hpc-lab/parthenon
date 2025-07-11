//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "inputs/inputs_package.hpp"
#include "inputs/parameter_input.hpp"
#include "interface/state_descriptor.hpp"

namespace parthenon {
namespace InputsPackage {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Inputs");
  // Add the pre-parsed inputs as params also, for archival
  // purposes. The original input deck is stashed in the restart. The
  // command line arguments are not.
  const auto &raw_inputs = pin->GetPreParsedInputs();
  for (const auto &[origin, input] : raw_inputs) {
    pkg->AddParam(origin.ToString(), input,
                  origin.type == RecordOrigin::Type::InputFile
                      ? Params::Mutability::Restart
                      : Params::Mutability::Immutable);
  }
  return pkg;
}

} // namespace InputsPackage
} // namespace parthenon
