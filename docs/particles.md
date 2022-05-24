# Particles

Parthenon provides a data framework for particle methods that allows for Kokkos
accelerated parallel dispatch of compute operations. Particle memory is allocated as a
separate pool for each variable in the particle.

## Swarms

A `Swarm` contains all the particle data for all particles of a given species. It owns a
set of `ParticleVariable`s, one for each value of each particle. For example, the spatial
positions `x`, `y`, and `z` of the particles in a swarm are three separate
`ParticleVariable`s. `ParticleVariable`s can be either `Real`- or `int`-valued, which is
specified by the metadata values `Metadata::Real` and `Metadata::Integer`.
`ParticleVariable`s should also contain the `Metadata::Particle` flag. By default,
`ParticleVariable`s provide one scalar quantity per particle, but up to 2D data per particle is
currently supported, by passing `std::vector<int>{N1, N2}` as the second argument to the
`ParticleVariable` `Metadata`. All `Swarm`s by default contain `x`, `y`, and `z`
`ParticleVariable`s; additional fields can be added as:
```c++
Swarm.Add(name, metadata)
```
For a given species, each `MeshBlock` contains its own `Swarm` that holds the particles of
that species that are spatially contained by that `MeshBlock`. The `MeshBlock` is pointed
to by `Swarm::pmy_block`.

The `Swarm` is a host-side object, but some of its data members are required for device-
side compution. To access this data, a `SwarmDeviceContext` object is created via
`Swarm::GetDeviceContext()`. This object can then be passed by copy into Kokkos lambdas.
Hereafter we refer to it as `swarm_d`.

To add particles to a `Swarm`, one calls
```c++
ParArray1D<bool> new_particles_mask = swarm->AddEmptyParticles(num_to_add, new_indices)
```
This call automatically resizes the memory pools as necessary and returns a
`ParArray1D<bool>` mask indicating which indices in the `ParticleVariable`s are newly
available. `new_indices` is a reference to a `ParArrayND<int>` of size `num_to_add` which
contains the indices of each newly added particle.

To remove particles from a `Swarm`, one first calls
```c++
swarm_d.MarkParticleForRemoval(index_to_remove)
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
auto &x = swarm.Get("x").Get();
swarm.pmy_block->par_for("Simple loop", 0, swarm.GetMaxActiveIndex(),
  KOKKOS_LAMBDA(const int n) {
    if (swarm_d.IsActive(n)) {
      x(n) += 1.0;
    }
  });
```

## Sorting

By default, particles are stored in per-meshblock pools of memory. However, one frequently wants
convenient access to all the particles in each computational cell separately. To facilitate this,
the Swarm provides the method `SortParticlesByCell` (and the `SwarmContainer` provides the matching
task `SortParticlesByCell`). Calling this function populates internal data structures that map from
per-cell indices to the per-meshblock data array. These are accessed by the `SwarmDeviceContext`
member functions `GetParticleCountPerCell` and `GetFullIndex`. See `examples/particles` for example
usage.

## Defragmenting

Because one typically loops over particles from 0 to `max_active_index`, if only a small
fraction of particles in that range are active, significant effort will be wasted. To
clean up these situations, `Swarm` provides a `Defrag` method which, when called, will
copy all active particles to be contiguous starting from the 0 index. `Defrag` is not
fully parallelized so should be called only sparingly.

## SwarmContainer

A `SwarmContainer` contains a set of related `Swarm`s, such as the different stages used
by a higher order time integrator. This feature is currently not exercised in detail.

## `particles` Example

An example showing how to create a Parthenon application that defines a `Swarm` and
creates, destroys, and transports particles is available in
`parthenon/examples/particles`.

## Communication

Communication of particles across `MeshBlock`s, including across MPI
processors, is supported. Particle communication is currently handled via
paired asynchronous/synchronous tasking regions on each MPI processor. The
asynchronous tasks include transporting particles and `SwarmContainer::Send`
and `SwarmContainer::Receive` calls. The synchronous task checks every
`MeshBlock` on that MPI processor for whether the `Swarm`s are finished
transporting. This set of tasks must be repeated in the driver's evolution
function until all particles are completed. See the `particles` example for
further details. Note that this pattern is blocking, and may be replaced in the
future.

AMR is currently not supported, but support will be added in the future.

## Variable Packing

Similarly to grid variables, particle swarms support `ParticleVariable` packing, by the
function `Swarm::PackVariables`. This also supports `FlatIdx` for indexing; see the
`particle_leapfrog` example for usage.

## Boundary conditions

Particle boundary conditions are not applied in separate kernel calls; instead, inherited
classes containing boundary condition functions for updating particles or removing them
when they are in boundary regions are allocated depending on the boundary flags specified
in the input file. Currently, outflow and periodic boundaries are supported natively.
User-specified boundary conditions must be set by specifying the "user" flag in the input
parameter file and then updating the appropriate Swarm::bounds array entries to factory
functions that allocate device-side boundary condition objects. An example is given in the
`particles` example when ix1 and ox1 are set to `user` in the input parameter file.
