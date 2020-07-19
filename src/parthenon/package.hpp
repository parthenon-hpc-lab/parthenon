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

#ifndef PARTHENON_PACKAGE_HPP_
#define PARTHENON_PACKAGE_HPP_

// Internal Includes
#include <application_input.hpp>
#include <basic_types.hpp>
#include <coordinates/coordinates.hpp>
#include <interface/container_iterator.hpp>
#include <interface/metadata.hpp>
#include <interface/params.hpp>
#include <interface/state_descriptor.hpp>
#include <interface/variable_pack.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon_manager.hpp>

// Local Includes
#include "prelude.hpp"

namespace parthenon {
namespace package {
namespace prelude {
using namespace ::parthenon::prelude;

using ::parthenon::AmrTag;
using ::parthenon::Coordinates;
using ::parthenon::DerivedOwnership;
using ::parthenon::DevExecSpace;
using ::parthenon::ApplicationInput;
using ::parthenon::MeshBlock;
using ::parthenon::Metadata;
using ::parthenon::PackIndexMap;
using ::parthenon::par_for;
using ::parthenon::ParameterInput;
using ::parthenon::Params;
using ::parthenon::ParthenonManager;
using ::parthenon::StateDescriptor;
using ::parthenon::TaskStatus;
using ::parthenon::X1DIR;
using ::parthenon::X2DIR;
using ::parthenon::X3DIR;
} // namespace prelude
} // namespace package
} // namespace parthenon

#endif // PARTHENON_PACKAGE_HPP_
