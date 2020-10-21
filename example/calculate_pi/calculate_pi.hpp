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
#ifndef EXAMPLE_CALCULATE_PI_CALCULATE_PI_HPP_
#define EXAMPLE_CALCULATE_PI_CALCULATE_PI_HPP_

// Standard Includes
#include <memory>
#include <vector>

// Parthenon Includes
#include <interface/state_descriptor.hpp>
#include <parthenon/package.hpp>

namespace calculate_pi {
using namespace parthenon::package::prelude;
using parthenon::Packages_t;
using parthenon::ParArrayHost;
using Pack_t = parthenon::MeshBlockVarPack<Real>;

// Package Callbacks
void SetInOrOut(std::shared_ptr<MeshBlockData<Real>> &rc);
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// Task Implementations
// Note pass by value here. Required for capture.
// All objects here have reference semantics, so capture by value is ok.
// TODO(JMM) A std::shared_ptr might be better.
// Computes area on a given meshpack
parthenon::TaskStatus ComputeArea(std::shared_ptr<MeshData<Real>> &md,
                                  ParArrayHost<Real> areas, int i);
// Sums up areas accross packs.
parthenon::TaskStatus AccumulateAreas(ParArrayHost<Real> areas, Packages_t &packages);
} // namespace calculate_pi

#endif // EXAMPLE_CALCULATE_PI_CALCULATE_PI_HPP_
