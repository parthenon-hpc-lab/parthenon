# Particles

Parthenon provides a data framework for particle methods that allows for Kokkos
accelerated parallel dispatch of compute operations. Particle memory is allocated as a
separate pool for each variable in the particle.

## Swarms

A `Swarm` contains all the particle data for all particles of a given species. It owns a
set of `ParticleVariable`s, one for each value of each particle. For example, the spatial
positions `x`, `y`, and `z` of the particles in a swarm are three separate
`ParticleVariable`s. `ParticleVariable`s can be either `Real`- or `int`-valued, which is
specified by the metadata values `Metadata::Real` and `Metadata::Integer`. All `Swarm`s by
default contain `x`, `y`, and `z` `ParticleVariable`s; additional fields can be added as:
```c++
Swarm.Add(name, metadata)
```
Each `Swarm` belongs to a `MeshBlock`, which is pointed to by `Swarm::pmy_block`.

To add particles to a `Swarm`, one calls
```c++
ParArrayND<bool> new_particles_mask = swarm->AddEmptyParticles(num_to_add)
```
This call automatically resizes the memory pools as necessary and returns a
`ParArrayND<bool>` mask indicating which indices in the `ParticleVariable`s are newly
available.

To remove particles from a `Swarm`, one first calls
```c++
swarm.MarkParticleForRemoval(index_to_remove)
```
inside device code. This only indicates that this particle should be removed from the pool,
it does not actually update any data. To remove all particles so marked, one then calls
```c++
swarm.RemoveMarkedParticles()
```
in host code. This updates the swarm such that the marked particles are seen as free slots
in the memory pool.

## Parallel Dispatch

Parallel computations on particle data can be performed with the usual `MeshBlock`
`par_for` calls. Typically one loops over the entire range of active indices and uses a
mask variable to only perform computations on currently active particles:
```c++
auto &mask = swarm.GetMask().Get();
auto &x = swarm.Get("x").Get();
swarm.pmy_block->par_for("Simple loop", 0, swarm.get_max_active_index(),
  KOKKOS_LAMBDA(const int n) {
    if (mask(n)) {
      x(n) += 1.0;
    }
  });
```

## SwarmContainer

A `SwarmContainer` contains a set of related `Swarm`s, such as the different stages used
by a higher order time integrator. This feature is currently not exercised in detail.

## `particles` Example

An example showing how to create a Parthenon application that defines a `Swarm` and
creates, destroys, and transports particles is available in
`parthenon/examples/particles`.
