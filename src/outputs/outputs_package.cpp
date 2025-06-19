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
#include <string>
#include <vector>

#include "interface/state_descriptor.hpp"
#include "outputs/outputs_package.hpp"
#include "parameter_input.hpp"

namespace parthenon {

namespace OutputsPackage {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Outputs");

  std::string basename = pin->GetOrAddString("parthenon/job", "problem_id", "parthenon");
  std::vector<std::string> block_names;
  std::vector<int> block_numbers;

  std::vector<bool> active;
  std::vector<int> file_numbers;
  std::vector<Real> last_times;
  std::vector<int> last_ns;

  // loop over input block names.  Find those that start with "parthenon/output", read
  // parameters, and construct singly linked list of OutputTypes.
  for (InputBlock *pib = pin->pfirst_block; pib != nullptr; pib = pib->pnext) {
    if (pib->block_name.compare(0, 16, "parthenon/output") == 0) {
      std::string outn = pib->block_name.substr(16); // 6 because counting starts at 0!
      std::string block_name = pib->block_name;
      // these are used for book-keeping
      block_names.push_back(block_name);
      block_numbers.push_back(atoi(outn.c_str()));

      // These will be updated later or restarted from
      active.push_back(false);
      file_numbers.push_back(0);
      // JMM: Limits to indicate these haven't been set yet
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
