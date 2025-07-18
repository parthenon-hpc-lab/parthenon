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

#include "inputs/parameter_input.hpp"
#include "interface/state_descriptor.hpp"
#include "outputs/outputs_package.hpp"

namespace parthenon {

namespace OutputsPackage {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Outputs");

  std::string basename = pin->GetOrAddString("parthenon/job", "problem_id", "parthenon",
                                             "prefix for output files");
  std::vector<std::string> block_names;
  std::vector<int> block_numbers;

  std::vector<bool> active;
  std::vector<int> file_numbers;
  std::vector<Real> last_times;
  std::vector<int> last_ns;

  // loop over input block names.  Find those in the parthenon.output blocks, read
  // parameters, and construct singly linked list of OutputTypes.
  for (auto pib : pin->Blocks("parthenon")) {
    std::string block_name = std::string(pib.first);
    if (block_name.compare(0, 6, "output") == 0) {
      std::string outn = block_name.substr(6); // 6 because counting starts at 0!

      if (pin->DoesParameterExist(block_name, "next_time")) {
        std::stringstream msg;
        msg << "You have used the next_time parameter in the " << block_name
            << " output block. This parameter is deprecated. Instead change"
            << " the output cadence with dt." << std::endl;
        PARTHENON_THROW(msg);
      }
      if (pin->DoesParameterExist(block_name, "next_n")) {
        std::stringstream msg;
        msg << "You have used the next_n parameter in the " << block_name
            << " output block. This parameter is deprecated. Instead change"
            << " the output cadence with dn." << std::endl;
        PARTHENON_THROW(msg);
      }

      // these are used for book-keeping
      block_names.push_back("parthenon." + block_name);
      block_numbers.push_back(atoi(outn.c_str()));

      // These will be updated later or restarted from
      active.push_back(false);
      file_numbers.push_back(0);

      // JMM: Limits to indicate these haven't been set yet. The reason
      // to set these to a "signal" number, rather than to start_time
      // is that we want to ensure a first output is performed.
      last_times.push_back(std::numeric_limits<Real>::lowest());
      last_ns.push_back(std::numeric_limits<int>::lowest());
    }
  }
  pkg->AddParam("block_names", block_names);
  pkg->AddParam("block_numbers", block_numbers);
  pkg->AddParam("active", active, Params::Mutability::Restart);
  pkg->AddParam("file_numbers", file_numbers, Params::Mutability::Restart);
  pkg->AddParam("last_times", last_times, Params::Mutability::Restart);
  pkg->AddParam("last_ns", last_ns, Params::Mutability::Restart);

  return pkg;
}

} // namespace OutputsPackage
} // namespace parthenon
