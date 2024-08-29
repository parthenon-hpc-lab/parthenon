//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
using namespace parthenon::package::prelude;

#include "mccirc/mccirc.hpp"

void GenerateCircle(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto &data = pmb->meshblock_data.Get();

  // pull out information/global params from package
  auto pkg = pmb->packages.Get("MCCirc");
  auto rng_pool = pkg->Param<MCCirc::RNGPool>("rng_pool");
  const int N = pkg->Param<int>("num_particles");

  // Pull out swarm object
  auto swarm = data->GetSwarmData()->Get("samples");

  // Create an accessor to particles, allocate particles
  auto newParticlesContext = swarm->AddEmptyParticles(N);

  // Meshblock geometry
  const IndexRange &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int &nx_i = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int &nx_j = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int &nx_k = pmb->cellbounds.ncellsk(IndexDomain::interior);
  const Real &dx_i = pmb->coords.Dxf<1>(pmb->cellbounds.is(IndexDomain::interior));
  const Real &dx_j = pmb->coords.Dxf<2>(pmb->cellbounds.js(IndexDomain::interior));
  const Real &dx_k = pmb->coords.Dxf<3>(pmb->cellbounds.ks(IndexDomain::interior));
  const Real &minx_i = pmb->coords.Xf<1>(ib.s);
  const Real &minx_j = pmb->coords.Xf<2>(jb.s);
  const Real &minx_k = pmb->coords.Xf<3>(kb.s);

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &weight = swarm->Get<Real>(MCCirc::weight::name()).Get();

  // Make a SwarmPack via types to get positions
  static auto desc_swarm =
      parthenon::MakeSwarmPackDescriptor<swarm_position::x, swarm_position::y,
                                         swarm_position::z, MCCirc::weight>("samples");
  auto pack_swarm = desc_swarm.GetPack(data.get());
  auto swarm_d = swarm->GetDeviceContext();

  // loop over new particles created
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      newParticlesContext.GetNewParticlesMaxIndex(),
      // new_n ranges from 0 to N_new_particles
      KOKKOS_LAMBDA(const int new_n) {
        // this is the particle index inside the swarm
        const int n = newParticlesContext.GetNewParticleIndex(new_n);
        auto rng_gen = rng_pool.get_state();

        // Normally b would be free-floating and set by pack.GetBlockparticleIndices
        // but since we're on a single meshblock for this loop, it's just 0
        // because block index = 0
        const int b = 0;
        // auto [b, n] = pack_swarm.GetBlockparticleIndices(idx);

        // randomly sample particle positions
        pack_swarm(b, swarm_position::x(), n) = minx_i + nx_i * dx_i * rng_gen.drand();
        pack_swarm(b, swarm_position::y(), n) = minx_j + nx_j * dx_j * rng_gen.drand();
        pack_swarm(b, swarm_position::z(), n) = minx_k + nx_k * dx_k * rng_gen.drand();

        // set weights to 1
        pack_swarm(b, MCCirc::weight(), n) = 1.0;

        // release random number generator
        rng_pool.free_state(rng_gen);
      });
}
