//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef PARTHENON_PACKAGE_HPP_
#define PARTHENON_PACKAGE_HPP_

// Internal Includes
#include <application_input.hpp>
#include <basic_types.hpp>
#include <coordinates/coordinates.hpp>
#include <interface/make_pack_descriptor.hpp>
#include <interface/metadata.hpp>
#include <interface/params.hpp>
#include <interface/sparse_pack.hpp>
#include <interface/sparse_pool.hpp>
#include <interface/state_descriptor.hpp>
#include <interface/variable_pack.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/mesh.hpp>
#include <mesh/meshblock.hpp>
#include <mesh/meshblock_pack.hpp>
#include <parameter_input.hpp>
#include <parthenon_manager.hpp>
#include <utils/index_split.hpp>
#include <utils/partition_stl_containers.hpp>

// Local Includes
#include "prelude.hpp"

namespace parthenon {
namespace package {
namespace prelude {
using namespace ::parthenon::prelude;

using ::parthenon::AmrTag;
using ::parthenon::ApplicationInput;
using ::parthenon::BlockList_t;
using ::parthenon::DevExecSpace;
using ::parthenon::HostExecSpace;
using ::parthenon::IndexSplit;
using ::parthenon::Mesh;
using ::parthenon::MeshBlock;
using ::parthenon::MeshBlockPack;
using ::parthenon::MeshBlockVarFluxPack;
using ::parthenon::MeshBlockVarPack;
using ::parthenon::Metadata;
using ::parthenon::PackIndexMap;
using ::parthenon::par_for;
using ::parthenon::ParameterInput;
using ::parthenon::Params;
using ::parthenon::SparsePack;
using ::parthenon::SparsePool;
using ::parthenon::StateDescriptor;
using ::parthenon::TaskStatus;
using ::parthenon::VariableFluxPack;
using ::parthenon::VariablePack;
using ::parthenon::X1DIR;
using ::parthenon::X2DIR;
using ::parthenon::X3DIR;

namespace partition {
using ::parthenon::partition::Partition_t;
using ::parthenon::partition::ToNPartitions;
using ::parthenon::partition::ToSizeN;
} // namespace partition
} // namespace prelude
} // namespace package
} // namespace parthenon

#endif // PARTHENON_PACKAGE_HPP_
