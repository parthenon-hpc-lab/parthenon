Particles
=========

Parthenon provides a data framework for particle methods that allows for
Kokkos accelerated parallel dispatch of compute operations. Particle
memory is allocated as a separate pool for each variable in the
particle.

Swarms
------

A ``Swarm`` contains all the particle data for all particles of a given
species. It owns a set of ``ParticleVariable``\ s, one for each value of
each particle. For example, the spatial positions ``x``, ``y``, and
``z`` of the particles in a swarm are three separate
``ParticleVariable``\ s. ``ParticleVariable``\ s can be either ``Real``-,
``uint64_t``-, or ``int``-valued, which is specified by the metadata values
``Metadata::Real``, ``MetaData::UInt64``, and ``Metadata::Integer``. ``ParticleVariable``\ s
should also contain the ``Metadata::Particle`` flag. By default,
``ParticleVariable``\ s provide one scalar quantity per particle, but up
to 2D data per particle is currently supported, by passing
``std::vector<int>{N1, N2}`` as the second argument to the
``ParticleVariable`` ``Metadata``. All ``Swarm``\ s by default contain
positional ``x``, ``y``, and ``z`` ``ParticleVariable``\ s and a ``uint64`` particle
``id`` that is persistent and unique for a given simulation.
The latter field can be disabled (e.g., because it is not required for a method
that constant creates and destroys anonymous particles) by providing the
``Metadata::NoPersistentParticleIds`` flag;
additional fields can be added as:

.. code:: cpp

   Swarm.Add(name, metadata)

Instances of ``MeshBlockData`` own the ``Swarm``s that hold particles spatially contained
by the associated ``MeshBlock``. The ``MeshBlock`` is pointed to by ``Swarm::pmy_block``.
``Swarm``s can be retrieved via both ``MeshBlockData::GetSwarmData()`` or
``MeshData::GetSwarmData(b)`` where the latter returns the ``Swarm``s associated with the
``MeshBlockData`` pointed to by the ``b`` index within a ``MeshData``.  We currently only
permit ``Swarm``s to be retrieved from ``"base"`` ``MeshBlockData`` and ``MeshData``.


The ``Swarm`` is a host-side object, but some of its data members are
required for device- side compution. To access this data, a
``SwarmDeviceContext`` object is created via
``Swarm::GetDeviceContext()``. This object can then be passed by copy
into Kokkos lambdas. Hereafter we refer to it as ``swarm_d``.

To add particles to a ``Swarm``, one calls

.. code:: cpp

   NewParticlesContext context = swarm->AddEmptyParticles(num_to_add);

This call automatically resizes the memory pools as necessary and
returns a ``NewParticlesContext`` object that provides the methods
``int GetNewParticlesMaxIndex()`` to get the max index of the contiguous block
of indices into the swarm, and ``int GetNewParticleIndex(const int n)`` to
convert a new particle index into the swarm index.

To remove particles from a ``Swarm``, one first calls

.. code:: cpp

   swarm_d.MarkParticleForRemoval(index_to_remove)

inside device code. This only indicates that this particle should be
removed from the pool, it does not actually update any data. To remove
all particles so marked, one then calls

.. code:: cpp

   swarm.RemoveMarkedParticles()

in host code. This updates the swarm such that the marked particles are
seen as free slots in the memory pool.

Parallel Dispatch
-----------------

Parallel computations on particle data can be performed with the usual
``MeshBlock`` ``par_for`` calls. Typically one loops over the entire
range of active indices and uses a mask variable to only perform
computations on currently active particles:

.. code:: cpp

   auto &x = swarm.Get("x").Get();
   swarm.pmy_block->par_for("Simple loop", 0, swarm.GetMaxActiveIndex(),
     KOKKOS_LAMBDA(const int n) {
       if (swarm_d.IsActive(n)) {
         x(n) += 1.0;
       }
     });

Sorting
-------

By default, particles are stored in per-meshblock pools of memory.
However, one frequently wants convenient access to all the particles in
each computational cell separately. To facilitate this, the Swarm
provides the method ``SortParticlesByCell`` (and the ``SwarmContainer``
provides the matching task ``SortParticlesByCell``). Calling this
function populates internal data structures that map from per-cell
indices to the per-meshblock data array. These are accessed by the
``SwarmDeviceContext`` member functions ``GetParticleCountPerCell`` and
``GetFullIndex``. See ``examples/particles`` for example usage.

Defragmenting
-------------

Because one typically loops over particles from 0 to
``max_active_index``, if only a small fraction of particles in that
range are active, significant effort will be wasted. To clean up these
situations, ``Swarm`` provides a ``Defrag`` method which, when called,
will copy all active particles to be contiguous starting from the 0
index. ``Defrag`` is not fully parallelized so should be called only
sparingly.

SwarmContainer
--------------

A ``SwarmContainer`` contains a set of related ``Swarm``\ s, such as the
different stages used by a higher order time integrator. This feature is
currently not exercised in detail.

``particles`` Example
---------------------

An example showing how to create a Parthenon application that defines a
``Swarm`` and creates, destroys, and transports particles is available
in ``parthenon/examples/particles``.

Communication
-------------

Communication of particles across ``MeshBlock``\ s, including across MPI
processors, is supported. Particle communication is currently handled
via paired asynchronous/synchronous tasking regions on each MPI
processor. The asynchronous tasks include transporting particles and
``SwarmContainer::Send`` and ``SwarmContainer::Receive`` calls. The
synchronous task checks every ``MeshBlock`` on that MPI processor for
whether the ``Swarm``\ s are finished transporting. This set of tasks
must be repeated in the driver’s evolution function until all particles
are completed. See the ``particles`` example for further details. Note
that this pattern is blocking, and may be replaced in the future.

AMR is currently not supported, but support will be added in the future.

Variable Packing
----------------

Similarly to grid variables, particle swarms support
``ParticleVariable`` packing, by the function ``Swarm::PackVariables``.
This also supports ``FlatIdx`` for indexing; see the
``particle_leapfrog`` example for usage.

.. _swarm_packs:

``SwarmPack``\ s
----------------

Similar to grid variables, swarms can be packed over ``MeshBlock``\ s via ``SwarmPack``\ s.
``SwarmPack``\ s are the particle analog to ``SparsePack``\ s for field variables.  A single
``SwarmPack`` can contain either ``int`` or ``Real`` entries, but not both.  One can pack
a ``SwarmPack`` via a ``std::vector<std::string>`` or the type-based variable prescription
previously used by ``SparsePack``\ s (see :ref:`sparse_packs`).

For packing via string (wherein below, ``swarm_position::x::name()`` returns a string),
one must specify the data type by template argument:

.. code:: cpp

   std::vector<std::string> vars{swarm_position::x::name(),
                                 swarm_position::y::name(),
                                 swarm_position::z::name()};
   static auto desc = MakeSwarmPackDescriptor<Real>(swarm_name, vars);
   auto pack = desc.GetPack(md);


For packing via type-based variables (see pack/swarm_default_names.hpp for an
example), the type can be inferred automatically:

.. code:: cpp

   static auto desc = MakeSwarmPackDescriptor<swarm_position::x,
                                              swarm_position::y,
                                              swarm_position::z>(swarm_name);
   auto pack = desc.GetPack(md);


For example ``SwarmPack`` usage, see the ``particle_leapfrog`` example.



Boundary conditions
-------------------

Particle boundary conditions are applied in per-block per-boundary kernel
launches analogous to grid-based variables. Outflow and periodic boundaries
are supported natively, but other boundary conditions (including reflecting)
must be provided by the downstream application. Particle boundary conditions are
enrolled by setting entries in ``ApplicationInput::swarm_boundary_conditions``
to per-boundary (inner ``x1``, outer ``x2``, etc.) custom boundary functions
with signature

.. code:: cpp

   void SwarmUserInnerX1(std::shared_ptr<Swarm> &swarm);

The ``particles`` example demonstrates how to create and enroll custom particle
boundary conditions.

Note that periodic boundary conditions cannot be enrolled by the user; the
default ``periodic`` option for Parthenon must be requested in the input file.

Outputs
--------

Outputs for swarms can be set in an output block, just like any other
variable. The user must specify a comma separated list denoting which
swarms are marked for output:

::

   swarms = swarm1, swarm2, ...

By default every swarm is initialized with ``x``, ``y``, and ``z``
position variables. These are automatically output.

To specify additional outputs, one may add an additional comma
separated list:

::

   swarmname_variables = var1, var2, ...

Here ``swarmname`` is the name of the swarm in question, and ``var1``,
``var2``, etc., are the variables to output for that particular
swarm. You may still specify ``x``, ``y``, and ``z``, but specifying
them is superfluous as they are automatically output for any swarm
that is output.

Alternatively, you may provide the

::

   swarm_variables = var1, var2, ...

input as a comma separated list. This will output each variable in the
``swarm_variables`` list for **every** swarm. This is most useful if
all the swarms contain similar variable structure, or if you only have
one swarm to output. The per-swarm lists can be composed with the
``swarm_variables`` field. Every swarm will output the vars in
``swarm_variables`` but then **additionally** the variables in a
per-swarm list will be output for that swarm.

.. note::

   Some visualization tools, like Visit and Paraview, prefer to have
   access to an ``id`` field for each particle, however it's not clear
   that a unique ID is required for each particle in
   general. Therefore, if swarms do not automatically contain a unique ID swarm
   variable (because the ``Metadata::NoPersistentParticleIds`` has been passed when
   creating the swarm), Parthenon will generate an ``id`` variable just for output
   (i.e., it is still not available within the simulation itself) and write it to file.

.. warning::

   The automatically generated ``id`` is unique for a snapshot in time,
   but not guaranteed to be time invariant. Indeed it is likely
   **not** the same between dumps.

Putting it all together, you might have an output block that looks like this:

::

   <parthenon/output1>
   file_type = hdf5
   dt = 1.0
   swarms = swarm1, swarm2
   swarm_variables = shared_var
   swarm1_variables = per_swarm_var

The result would be that both ``swarm1`` and ``swarm2`` output the
variables ``id``, ``x``, ``y``, ``z``, and ``shared_var``. But only ``swarm1``
outputs ``per_swarm_var``.
