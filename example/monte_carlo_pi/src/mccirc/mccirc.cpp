#include <memory>
#include <parthenon/package.hpp>
using namespace parthenon::package::prelude;

#include "mccirc.hpp"

namespace MCCirc {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("MCCirc");

  const Real radius = pin->GetOrAddReal("circle", "radius", 1.0);
  pkg->AddParam("radius", radius);

  const int npart = pin->GetOrAddInteger("MonteCarlo", "num_particles_per_block", 1000);
  pkg->AddParam("num_particles", npart);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("MonteCarlo", "rng_seed", 1234);
  pkg->AddParam("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  pkg->AddParam("rng_pool", rng_pool);

  // TO access later
  // pkg->Param<Type>("name");
  // e.g., pkg->Param<RNGPool>("rng_pool");

  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  pkg->AddSwarm("samples", swarm_metadata);

  Metadata real_swarmvalue_metadata({Metadata::Real});
  pkg->AddSwarmValue(weight::name(), "samples", real_swarmvalue_metadata);

  Metadata m({Metadata::Cell});
  pkg->AddField<NumParticles>(m);

  return pkg;
}

// in task list should return task status but I'm being lazy.
void ComputeParticleCounts(Mesh *pm) {
  // get mesh data
  auto md = pm->mesh_data.Get();

  // Make a SwarmPack via types to get positions
  static auto desc_swarm =
    parthenon::MakeSwarmPackDescriptor<swarm_position::x,
                                       swarm_position::y,
                                       swarm_position::z,
                                       MCCirc::weight>("samples");
  auto pack_swarm = desc_swarm.GetPack(md.get());

  // pull out circle radius from params
  auto pkg = pm->packages.Get("MCCirc");
  auto r = pkg->Param<Real>("radius");

  // build a type-based variable pack
  static auto desc = parthenon::MakePackDescriptor<MCCirc::NumParticles>(md.get());
  auto pack = desc.GetPack(md.get());

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  parthenon::par_for(
                     PARTHENON_AUTO_LABEL, 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                     KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                       pack(b, MCCirc::NumParticles(), k, j, i) = 0;
                     });

  parthenon::par_for(DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL,
                     DevExecSpace(), 0,
                     pack_swarm.GetMaxFlatIndex(),
                     // loop over all particles
                     KOKKOS_LAMBDA(const int idx) {
                       // block and particle indices
                       auto [b, n] = pack_swarm.GetBlockParticleIndices(idx);
                       const auto swarm_d = pack_swarm.GetContext(b);
                       if (swarm_d.IsActive(n)) {
                         // computes block-local cell indices of this particle
                         int i, j, k;
                         Real x = pack_swarm(b, swarm_position::x(), n);
                         Real y = pack_swarm(b, swarm_position::y(), n);
                         Real z = pack_swarm(b, swarm_position::z(), n);
                         swarm_d.Xtoijk(x, y, z, i, j, k);

                         Kokkos::atomic_add(&pack(b, MCCirc::NumParticles(), k, j, i),
                                            pack_swarm(b, MCCirc::weight(), n));
                       }
                     });

  // local reductions over all meshblocks in meshdata object
  Real total_particles;
  parthenon::par_reduce(parthenon::LoopPatternMDRange(),
                        PARTHENON_AUTO_LABEL, DevExecSpace(),
                        0, pack.GetNBlocks() - 1,
                        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &tot) {
                       tot += pack(b, MCCirc::NumParticles(), k, j, i);
                     }, total_particles);
  Real total_particles_in_circle;
  parthenon::par_reduce(parthenon::LoopPatternMDRange(),
                        PARTHENON_AUTO_LABEL, DevExecSpace(),
                        0, pack.GetNBlocks() - 1,
                        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &tot) {
                          auto &coords = pack.GetCoordinates(b);
                          Real x = coords.Xc<X1DIR>(k, j, i);
                          Real y = coords.Xc<X2DIR>(k, j, i);
                          bool in_circle = x*x + y*y < r*r;
                          tot += in_circle*pack(b, MCCirc::NumParticles(), k, j, i);
                     }, total_particles_in_circle);

  // just print for simplicity but if we were doing this right, we would call parthenon's reductions
  // which take the above data and reduce accross MPI ranks and task lists
  printf("particles in circle, particles total, pi = %.14e %.14e %.14e\n",
         total_particles_in_circle, total_particles, 4.*total_particles_in_circle/total_particles);
  
}

} // namespace MCCirc
