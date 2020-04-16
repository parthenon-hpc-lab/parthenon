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

#ifndef PARTHENON_DRIVER_HPP_
#define PARTHENON_DRIVER_HPP_

// Internal Includes
#include <basic_types.hpp>
#include <driver/driver.hpp>
#include <interface/metadata.hpp>
#include <interface/params.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/mesh.hpp>
#include <outputs/outputs.hpp>
#include <parameter_input.hpp>
#include <task_list/tasks.hpp>

// Local Includes
#include "prelude.hpp"

namespace parthenon {
namespace driver {
namespace prelude {
using namespace ::parthenon::prelude;

using ::parthenon::AmrTag;
using ::parthenon::BlockTask;
using ::parthenon::DerivedOwnership;
using ::parthenon::Driver;
using ::parthenon::DriverStatus;
using ::parthenon::Mesh;
using ::parthenon::MeshBlock;
using ::parthenon::Metadata;
using ::parthenon::Outputs;
using ::parthenon::ParameterInput;
using ::parthenon::Params;
using ::parthenon::TaskID;
using ::parthenon::TaskList;
using ::parthenon::DriverUtils::ConstructAndExecuteBlockTasks;
} // namespace prelude
} // namespace driver
} // namespace parthenon

#endif // PARTHENON_DRIVER_HPP_