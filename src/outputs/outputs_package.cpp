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

#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "outputs/outputs_package.hpp"
#include "parameter_input.hpp"

namespace parthenon {

namespace OutputsPackage {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Outputs");

  // loop over input block names.  Find those that start with "parthenon/output" and
  // add/initialize `Params` for further processing (so that they're available to be read
  // from restart files or are cleanly initialized).
  for (InputBlock *pib = pin->pfirst_block; pib != nullptr; pib = pib->pnext) {
    if (pib->block_name.compare(0, 16, "parthenon/output") == 0) {
      std::string outn = pib->block_name.substr(16); // 16 because counting starts at 0!
      std::string block_name = pib->block_name;

      // These will be updated later or restarted from
      int file_number = 0;

      // JMM: Limits to indicate these haven't been set yet. The reason
      // to set these to a "signal" number, rather than to start_time
      // is that we want to ensure a first output is performed.
      auto last_time = std::numeric_limits<Real>::lowest();
      auto last_n = std::numeric_limits<int>::lowest();

      bool next_time_exists = pin->DoesParameterExist(block_name, "next_time");
      bool next_n_exists = pin->DoesParameterExist(block_name, "next_n");
      if (next_time_exists || next_n_exists) {
        std::stringstream msg;
        msg << "You have used the next_time or next_n parameter in the " << block_name
            << " output block. This parameter is deprecated. Instead change"
            << " the output cadence with dt or dn." << std::endl;
        if (parthenon::Globals::is_restart) {
          if (Globals::my_rank == 0) {
            msg << "The parameters will automatically be updated internally and the "
                   "warning should not be shown for subsequent "
                   "restarts.\n";
            PARTHENON_WARN(msg);
          }
        } else {
          PARTHENON_THROW(msg);
        }
      }
      // It should be safe here to just use outn as output blocks are unique
      pkg->AddParam(outn + "/file_number", file_number, Params::Mutability::Restart);
      pkg->AddParam(outn + "/last_time", last_time, Params::Mutability::Restart);
      pkg->AddParam(outn + "/last_n", last_n, Params::Mutability::Restart);
    }
  }

  return pkg;
}

} // namespace OutputsPackage
} // namespace parthenon
