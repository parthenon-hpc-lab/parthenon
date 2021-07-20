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

#ifndef PARTHENON_DRIVER_HPP_
#define PARTHENON_DRIVER_HPP_

// Internal Includes
#include <application_input.hpp>
#include <basic_types.hpp>
#include <bvals/boundary_conditions.hpp>
#include <driver/driver.hpp>
#include <driver/multistage.hpp>
#include <interface/update.hpp>
#include <mesh/mesh.hpp>
#include <mesh/mesh_refinement.hpp>
#include <mesh/meshblock_pack.hpp>
#include <outputs/outputs.hpp>
#include <parameter_input.hpp>
#include <tasks/task_id.hpp>
#include <tasks/task_list.hpp>
#include <tasks/task_types.hpp>
#include <utils/partition_stl_containers.hpp>

// Local Includes
#include "prelude.hpp"

namespace parthenon {
namespace driver {
namespace prelude {
using namespace ::parthenon::prelude;

using ::parthenon::ApplicationInput;
using ::parthenon::ApplyBoundaryConditions;
using ::parthenon::BlockList_t;
using ::parthenon::Driver;
using ::parthenon::DriverStatus;
using ::parthenon::EvolutionDriver;
using ::parthenon::Mesh;
using ::parthenon::MeshBlock;
using ::parthenon::MeshBlockFluxMetaPack;
using ::parthenon::MeshBlockPack;
using ::parthenon::MeshBlockVarMetaPack;
using ::parthenon::MultiStageBlockTaskDriver;
using ::parthenon::MultiStageDriver;
using ::parthenon::Outputs;
using ::parthenon::Packages_t;
using ::parthenon::ParameterInput;
using ::parthenon::ParthenonManager;
using ::parthenon::StagedIntegrator;
using ::parthenon::Task;
using ::parthenon::TaskCollection;
using ::parthenon::TaskID;
using ::parthenon::TaskList;
using ::parthenon::TaskListStatus;
using ::parthenon::TaskRegion;
using ::parthenon::TaskStatus;
using ::parthenon::DriverUtils::ConstructAndExecuteBlockTasks;
using ::parthenon::DriverUtils::ConstructAndExecuteTaskLists;

namespace partition {
using ::parthenon::partition::Partition_t;
using ::parthenon::partition::ToNPartitions;
using ::parthenon::partition::ToSizeN;
} // namespace partition
} // namespace prelude
} // namespace driver
} // namespace parthenon

#endif // PARTHENON_DRIVER_HPP_
