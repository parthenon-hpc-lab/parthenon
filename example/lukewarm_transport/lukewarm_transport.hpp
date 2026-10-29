//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_LUKEWARM_TRANSPORT_LUKEWARM_TRANSPORT_HPP_
#define EXAMPLE_LUKEWARM_TRANSPORT_LUKEWARM_TRANSPORT_HPP_

// This file was made in part with generative AI.

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "pack/default_names.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;

// Tracers that each move in their own direction and are pushed iteratively: every tracer
// is pushed until it reaches the end of the step or enters a block of another MeshData,
// tracers that dropped out are exchanged and pushed again by their new MeshData, and so
// on until every tracer has reached the end of the step
namespace lukewarm_transport {

static const char lukewarm_name[] = "tracers";

namespace tracer {
// The time a tracer has been pushed to
PAR_VAR(tracer, time);
PAR_VAR(tracer, velocity);
} // namespace tracer

namespace field {
PAR_VAR(field, deposition);
} // namespace field

class TracerDriver : public EvolutionDriver {
 public:
  TracerDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm) {}
  TaskListStatus Step();
  TaskCollection StepTasks();

 private:
  // Iterations of the push so far this step, per partition
  std::vector<int> push_iterations_;
};

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

namespace tracers_package {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
Real EstimateTimestepMesh(MeshData<Real> *md);
void SourceTracers(Mesh *pm, ParameterInput *pin, MeshData<Real> *md);

} // namespace tracers_package

// Push the tracers of md from t_start to t_start + dt, all of them on the first
// iteration and only the ones the last exchange delivered on later ones, stopping each
// tracer that enters a block of another MeshData
TaskStatus PushTracers(MeshData<Real> *md, const Real t_start, const Real dt,
                       int *iteration);
// Iterate while any tracer on md has not reached t_end
TaskStatus CheckCompletion(MeshData<Real> *md, const Real t_end);
// Fail unless every tracer on md has reached t_end
TaskStatus CheckTracersFinished(MeshData<Real> *md, const Real t_end);
TaskStatus DepositTracers(MeshData<Real> *md);

} // namespace lukewarm_transport

#endif // EXAMPLE_LUKEWARM_TRANSPORT_LUKEWARM_TRANSPORT_HPP_
