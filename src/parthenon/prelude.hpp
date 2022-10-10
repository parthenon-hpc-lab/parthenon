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

#ifndef PARTHENON_PRELUDE_HPP_
#define PARTHENON_PRELUDE_HPP_

// Internal Includes
#include <basic_types.hpp>
#include <defs.hpp>
#include <globals.hpp>
#include <interface/mesh_data.hpp>
#include <interface/meshblock_data.hpp>
#include <interface/swarm_container.hpp>
#include <interface/variable.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parthenon_arrays.hpp>
#include <parthenon_manager.hpp>
#include <parthenon_mpi.hpp>

namespace parthenon {
namespace prelude {
using ::parthenon::BoundaryCommSubset;
using ::parthenon::IndexDomain;
using ::parthenon::IndexRange;
using ::parthenon::MeshBlock;
using ::parthenon::MeshBlockData;
using ::parthenon::MeshData;
using ::parthenon::par_for_inner;
using ::parthenon::ParArrayHost;
using ::parthenon::ParArrayND;
using ::parthenon::ParthenonStatus;
using ::parthenon::Real;
using ::parthenon::Swarm;
using ::parthenon::SwarmContainer;
using ::parthenon::Variable;
using ::parthenon::Globals::my_rank;
using ::parthenon::Globals::nranks;
} // namespace prelude
} // namespace parthenon

#endif // PARTHENON_PRELUDE_HPP_
