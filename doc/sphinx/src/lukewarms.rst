.. _lukewarms:

Lukewarms
=========

A lukewarm is a collection of particles owned by a ``MeshData`` rather than by any one
``MeshBlock``. Lukewarms exist alongside :doc:`swarms <particles>`, which are unchanged.
The two differ in where particles live and how they move:

* A ``Swarm`` belongs to a ``MeshBlockData``, so every particle that crosses a block
  boundary is communicated, even between blocks of the same ``MeshData``. A lukewarm
  belongs to the ``MeshData``, so a particle moving between its blocks only updates an
  index. Communication happens only when a particle reaches a block of another
  ``MeshData``, on this rank or another.
* Lukewarm storage is always dense. The live particles are exactly indices
  ``[0, NumParticles())``, so kernels need no mask and no free-list.
* Particles can find their block from inside a kernel (see `Locating particles`_). This
  makes it possible to move a particle across several blocks in one kernel, and to stop
  only when it leaves its ``MeshData``.
* Communication buffers stay on the device. Only message sizes pass through the host.

Lukewarm API names follow the names of the field API: ``AddLukewarmField`` matches
``AddField``, ``MakeLukewarmPackDescriptor`` matches ``MakePackDescriptor``, and fields are
named by ``PAR_VAR`` types.

Including ``parthenon/package.hpp`` makes ``lukewarm/lukewarm_pack.hpp`` and
``lukewarm/lukewarm_relocate.hpp`` available.

Declaring a lukewarm
--------------------

A package declares a lukewarm and its fields in its ``StateDescriptor``:

.. code:: cpp

   namespace tracer {
   PAR_VAR(tracer, time);
   PAR_VAR(tracer, velocity);
   } // namespace tracer

   pkg->AddLukewarm("tracers", Metadata({Metadata::Provides}));
   pkg->AddLukewarmField<tracer::time>("tracers", Metadata({Metadata::Real}));
   pkg->AddLukewarmField<tracer::velocity>("tracers",
                                           Metadata({Metadata::Real}, std::vector<int>{3}));

A field may be ``Metadata::Real``, ``Metadata::Integer``, or ``Metadata::UInt64``, and may
have a shape, just as a field does. ``Metadata::Particle`` is set automatically. A field
can be named by a string as well as by a ``PAR_VAR`` type, as in
``AddLukewarmField(lukewarm_name, field_name, metadata)``.

Every lukewarm also gets these default fields, declared in ``pack/default_names.hpp``:

=====================  ==================  ==========================================
Field                  Type                Meaning
=====================  ==================  ==========================================
``lukewarm::x``        ``Real``            Position in the mesh's coordinates
``lukewarm::y``        ``Real``
``lukewarm::z``        ``Real``
``lukewarm::id``       ``std::uint64_t``   Unique id, persistent for the simulation
=====================  ==================  ==========================================

Each particle also remembers the block it was last located in. That is internal: it is
not a field, and no pack can read or write it. It is read through ``Locate(n)`` and
``GetBlockIndex(n)`` (see `Locating particles`_).

Ids are handed out without communication. The ``k``-th id given out on rank ``r`` is
``k * nranks + r``.

Ownership
---------

Lukewarms live only on the ``"base"`` ``MeshData`` of each partition of the default
partitioning, i.e., ``parthenon/mesh/pack_size``:

.. code:: cpp

   for (int i = 0; i < pmesh->DefaultNumPartitions(); ++i) {
     auto &base = pmesh->mesh_data.GetOrAdd("base", i);
     Lukewarm &tracers = base->GetLukewarm("tracers");
   }

Asking any other ``MeshData`` for a lukewarm throws, and ``md->HasLukewarms()`` tells
the two apart. This matters, e.g., for history outputs, which receive a ``MeshData`` of
all blocks on a rank. Unless that is also the only partition, a history function has to
sum over the default partitions, as the examples do.

Each owning ``MeshData`` has an id, ``md->GetLukewarmContainer().GetId()``, which is the
smallest gid among its blocks. It is unique across ranks, which makes it useful, e.g., as
a random number seed.

Creating and removing particles
-------------------------------

``Lukewarm`` is a host-side object. Its storage methods are called from the host:

* ``IndexRange Create(int n)`` appends ``n`` particles and gives each an id. They start
  unlocated, and the caller sets every other field, usually in a kernel over the
  returned range.
* ``ParArray1DRaw<int> Create(const ParArray1DRaw<int> &counts)`` creates a different
  number of particles for each of several places at once, and reports where each
  place's particles went. ``counts`` is a device array, and ``counts(i)`` is how many
  particles you want for place ``i``. What ``i`` stands for is up to you, e.g., a block
  index, or a flattened cell index. The returned ``offsets`` has one more entry than
  ``counts``, and place ``i``'s new particles are the indices ``offsets(i)`` through
  ``offsets(i + 1) - 1``. For example, if the lukewarm already holds 10 particles and
  ``counts`` is ``{2, 0, 3}``, then ``offsets`` is ``{10, 12, 12, 15}``: place 0 gets
  particles 10 and 11, place 1 gets none, and place 2 gets 12, 13, and 14.
* ``RemoveMarked()`` removes every particle marked by ``pack.MarkForRemoval(n)``, or by
  setting ``GetRemovalMarks()(n) = 1``. Particles from the end of storage fill the holes,
  so survivors may change index. ``pack.IsMarkedForRemoval(n)`` tells a kernel to skip a
  particle that is marked but not yet removed.
* ``Clear()`` removes every particle and keeps the storage. ``Reserve(n)`` makes room
  ahead of time.
* ``NumParticles()`` and ``Capacity()`` report the current sizes.

Particles that exist from the start are best created in the package's
``PostInitializationMesh``, which is called for every owning ``MeshData``. For example,
to give every block the same number of particles at random positions:

.. code:: cpp

   void SourceTracers(Mesh *pm, ParameterInput *pin, MeshData<Real> *md) {
     auto &tracers = md->GetLukewarm("tracers");
     const int nblocks = md->NumBlocks();

     ParArray1DRaw<int> counts("counts", nblocks);
     Kokkos::deep_copy(counts, num_per_block);
     const auto offsets = tracers.Create(counts);

     // A pack is a snapshot, so get it after creating the particles
     static auto desc =
         MakeLukewarmPackDescriptor<lukewarm::x, lukewarm::y, lukewarm::z>(md, "tracers");
     auto pack = desc.GetPack(md);
     const auto geometry = md->GetLukewarmGeometry();
     auto rng_pool = RNGPool(md->GetLukewarmContainer().GetId());

     par_for_outer(
         PARTHENON_AUTO_LABEL, 0, 0, 0, nblocks - 1,
         KOKKOS_LAMBDA(team_mbr_t member, const int b) {
           const Real x_min = geometry.Xmin(b, X1DIR);
           const Real x_max = geometry.Xmax(b, X1DIR);
           // ... and likewise for X2DIR and X3DIR
           par_for_inner(member, offsets(b), offsets(b + 1) - 1, [&](const int n) {
             auto rng_gen = rng_pool.get_state();
             pack(lukewarm::x(), n) = x_min + rng_gen.drand() * (x_max - x_min);
             // ...
             rng_pool.free_state(rng_gen);
           });
         });
   }

   pkg->PostInitializationMesh = SourceTracers;

Lukewarm packs
--------------

Lukewarm fields are reached on the device through a ``LukewarmPack``, which works like a
field pack without the block index:

.. code:: cpp

   static auto desc =
       MakeLukewarmPackDescriptor<lukewarm::x, tracer::velocity>(md, "tracers");
   auto pack = desc.GetPack(md);
   par_for(
       PARTHENON_AUTO_LABEL, 0, pack.GetNParticles() - 1, KOKKOS_LAMBDA(const int n) {
         pack(lukewarm::x(), n) += dt * pack(tracer::velocity(0), n);
       });

The descriptor checks, when it is made, that every field exists and that its
``PAR_VAR`` type has the right data type. The descriptor holds only the lukewarm's name,
so it may be ``static`` and stays valid across remeshing.

.. warning::

   A pack is a snapshot of the storage and particle count when it was made. Packs are
   not cached, so getting one is cheap, but a new pack must be made after particles are
   created, removed, or communicated.

Locating particles
------------------

Every particle remembers the block it was last located in. After moving a particle, a
kernel calls ``pack.Locate(n)``, which updates that block, wraps the position across
periodic boundaries, and returns a ``LukewarmLocation``:

* ``InMeshData()``: the particle is in block ``Block()`` of this ``MeshData``.
* ``InOtherMeshData()``: the particle is in block ``Gid()`` of another ``MeshData``,
  where the next exchange delivers it.
* ``LeftMesh()``: the particle left through a non-periodic boundary, and the next
  exchange removes it.
* ``Lost()``: the particle never settled in a block, e.g., because its position is NaN.
  The next exchange throws an error if it finds one.

The local block index ``b`` is the same as the block index of a field pack of the same
``MeshData``, so the two can be used together. ``pack.GetCellIndices(b, n)`` gives the
interior cell holding the particle, in the order a field pack takes them:

.. code:: cpp

   pack(lukewarm::x(), n) += dt * pack(tracer::velocity(0), n);
   const auto loc = pack.Locate(n);
   if (loc.InMeshData()) {
     const int b = loc.Block();
     const auto [k, j, i] = pack.GetCellIndices(b, n);
     Kokkos::atomic_add(&field_pack(b, field::deposition(), k, j, i), 1.0);
   }

If a particle has not moved since it was last located, ``pack.GetBlockIndex(n)`` returns
its block without searching. In debug builds it checks that the particle is still there.

``Locate`` does not search a tree. Each block is split in half along every active
direction, and space just outside it is half a block wide. This gives a
:math:`4 \times 4 \times 4` grid of half-blocks. Each half-block outside the block lies in
exactly one neighbor, whether that neighbor is coarser, finer, or at the same level. A
table built at setup gives that neighbor's local index, gid, and periodic shift.
``Locate`` steps from block to neighbor until it reaches the block holding the particle:

* A particle that stayed in its block costs a few comparisons.
* One that crossed into a neighbor takes one hop.
* An unlocated particle, e.g., one just created, starts from the block of the
  ``MeshData`` nearest to it. A particle within half a block of the ``MeshData`` is
  therefore found in one hop.

Periodic shifts are applied once, at the end, so a particle that crosses a periodic
boundary and comes back is restored exactly.

``md->GetLukewarmGeometry()`` and ``pack.GetGeometry()`` expose the same table on the
device:

* ``Xmin(b, dir)`` and ``Xmax(b, dir)`` give block faces.
* ``Gid(b)`` gives the global id of a block.
* ``Contains(b, x1, x2, x3)`` checks whether a block holds a position.
* ``FindBlock(x1, x2, x3)`` searches every block linearly.

Communication
-------------

Particles that reach a block of another ``MeshData`` are sent to it. Each ``MeshData``
exchanges only with the ``MeshData`` next to it, whether on this rank or another, much as
fields exchange their boundaries. ``lukewarm/lukewarm_relocate.hpp`` provides three ways
to do this.

**One round in a task list.** ``AddLukewarmExchangeTasks`` adds one round of locating,
sending, and receiving to a task list, like ``AddBoundaryExchangeTasks`` for fields.
Every ``MeshData`` must take part, e.g., from the task lists of one ``TaskRegion``:

.. code:: cpp

   TaskRegion &region = tc.AddRegion(num_partitions);
   for (int i = 0; i < num_partitions; i++) {
     auto &tl = region[i];
     auto &base = pmesh->mesh_data.GetOrAdd("base", i);
     auto push = tl.AddTask(none, PushTracers, base.get(), dt);
     auto exchange = AddLukewarmExchangeTasks(push, tl, base.get(), "tracers");
     auto deposit = tl.AddTask(exchange, DepositTracers, base.get());
   }

A particle goes to the first block of another ``MeshData`` it reaches, so one round is
enough as long as no particle moves past a whole block. This holds, e.g., when moves are
limited by a CFL condition.

**Rounds until settled.** ``RelocateLukewarm(pmesh, name)`` repeats rounds until every
particle on every rank is in its block. It is collective over all ranks and works for
moves of any length.

**Iterative pushes.** A kernel can push each particle until it is done or leaves its
``MeshData``, exchange, and then push again only the particles that just arrived.
``GetLastReceived(md, name)`` gives the range the last exchange appended:

.. code:: cpp

   auto &tracers = md->GetLukewarm("tracers");
   const IndexRange range = first ? IndexRange{0, tracers.NumParticles() - 1}
                                  : GetLastReceived(md, "tracers");
   par_for(
       PARTHENON_AUTO_LABEL, range.s, range.e, KOKKOS_LAMBDA(const int n) {
         while (/* particle n has work left */) {
           // ... move particle n a little
           if (!pack.Locate(n).InMeshData()) break; // finish after the exchange
         }
       });

Since a particle may arrive on any ``MeshData``, every ``MeshData`` has to keep pushing
and exchanging until no particle on any ``MeshData`` of any rank has work left.
``AddLukewarmCompletionTasks`` ends an ``AddSublist`` loop this way. It takes a function
giving how many particles a ``MeshData`` has left, and sums these over the ``TaskRegion``
and over ranks in an ``AllReduce<int>``. All task lists of the region share that
``AllReduce``, which must outlive the task collection, e.g., as a member of the driver:

.. code:: cpp

   auto [itl, push_id] = tl.AddSublist(start, {1, max_iterations});
   auto push = itl.AddTask(none, PushTracers, base.get(), dt);
   auto exchange = AddLukewarmExchangeTasks(push, itl, base.get(), "tracers");
   auto complete =
       AddLukewarmCompletionTasks(exchange, itl, base.get(), &num_left_, CountUnfinished);
   auto deposit = tl.AddTask(push_id, DepositTracers, base.get());

This works with any ``pack_size``. ``lukewarm_transport`` shows the whole loop.

The individual steps of a round are also available as task functions:

1. ``LocateLukewarm`` (returns how many particles are waiting to be sent)
2. ``StartReceiveLukewarm``
3. ``SendLukewarm``
4. ``ReceiveLukewarm``

Between ``LocateLukewarm`` and ``SendLukewarm``, no particles may be created or removed on
that ``MeshData``.

Every exchange also calls ``RemoveMarked()``. A particle marked for removal is therefore
gone after the next exchange, whether or not anything was sent.

Boundaries
----------

* Periodic boundaries wrap particle positions.
* A particle that leaves through any other boundary is removed at the next exchange.

Lukewarms have no user-defined particle boundary conditions yet.

Mesh refinement and load balancing
----------------------------------

Lukewarms survive remeshing and load balancing with no work by the user. Both happen
during ``Mesh::RedistributeAndRefineMeshBlocks``:

1. Before the old ``MeshData`` are destroyed, every particle is settled in its block and
   detached.
2. After the new ``MeshData`` are built, each particle goes to the new owner of its
   block. On refinement, that is the daughter holding it; on derefinement, the parent.
   Particles that move between ranks are sent device to device.

Particle values are carried over unchanged. Particle indices and blocks change, so
packs and ranges from before the remesh must not be reused.

Output
------

HDF5 and openPMD outputs write lukewarms named in the output block, much as they write
swarms:

::

   <parthenon/output1>
   file_type = hdf5
   dt = 0.1
   lukewarms = tracers, markers            # nonexistent lukewarms are ignored
   lukewarm_variables = tracer.time        # fields written for every lukewarm
   tracers_variables = tracer.velocity     # fields written for one lukewarm
   write_lukewarm_xdmf = true              # also describe lukewarms in XDMF

``lukewarm::x``, ``lukewarm::y``, ``lukewarm::z``, and ``lukewarm::id`` are always
written, and a requested field that a lukewarm lacks is skipped. A lukewarm may not
share a name with a swarm in the same output. An HDF5 output without ``lukewarms``
writes none, while an openPMD output without it writes every field of every lukewarm,
as each does for swarms.

In HDF5 files the layout is the one swarms use, but with fields in ``LukewarmVars``
where a swarm has ``SwarmVars``. ``phdf(filename).GetLukewarm("tracers")`` reads one, and
gives positions through ``.x``, ``.y``, and ``.z``. Each lukewarm has a group ``/<name>``
with:

* ``counts`` and ``offsets``, with one entry per block in gid order, give each block's
  number of particles and where they start.
* ``LukewarmVars/<field>`` holds a field, with its shape first and the particle index
  last.
  Particles are ordered by block.
* ``LukewarmVars/lukewarm_positions`` holds positions as (particle, direction) when
  ``write_lukewarm_xdmf`` is set, for the XDMF file ``<file>.lukewarm.xdmf``.
* The attribute ``id_bound`` is above every id handed out so far, on any rank.

In openPMD files each lukewarm is a particle species ``<name>``, with particles ordered
by block, and with:

* the attributes ``counts`` and ``offsets``, as in HDF5 files, and ``id_bound``.
* the records ``position`` (components ``x``, ``y``, and ``z``), ``positionOffset``
  (zero), and ``id``, as the openPMD standard names them.
* a record for every other field, named after the field. A scalar field has the
  component ``openPMD::RecordComponent::SCALAR``, and any other field has components
  ``0``, ``1``, and so on, in the order HDF5 files store them. A field may not be named
  ``position``, ``positionOffset``, or ``id``.

A lukewarm with no particles still has these records, with zero length, since openPMD
readers skip a species without records.

To place every particle in its block, writing a lukewarm first calls
``RelocateLukewarm``. This also removes particles that are marked or have left the mesh.

Restarts
--------

HDF5 and openPMD restart files hold every field of every lukewarm, with no input
needed. On restart, each block's particles go to the MeshData that now owns the block, so
a restart may use a different number of ranks or a different ``pack_size``. Particles
keep their ids, and particles created after the restart never get the id of any
particle before it, even one since removed. Package hooks that create particles at
initialization, such as ``PostInitializationMesh``, are not called on restart, as for
every other initialization hook.

A field missing from the restart file, e.g., one added since the file was written, is
set to zero. A lukewarm missing from the file starts empty.

An analysis run (``-a``, see :ref:`analysis`) keeps a lukewarm only if it was added with
``Metadata::Analysis``, and reads its particles as a restart does, from either a restart
file or a data output that lists it in ``lukewarms``. A ``Metadata::Private`` lukewarm
keeps the ``package::name`` it is written under.

.. note::

   openPMD record names may only hold letters, digits, and ``_``, so, as for mesh fields
   and swarm variables, a lukewarm field whose name has a ``.``, such as any declared
   with ``PAR_VAR``, cannot be written to openPMD files. An openPMD restart, which holds
   every field of every lukewarm, fails if any of them is named with a ``.``.

Examples
--------

* ``example/lukewarm_tracers`` advects a field and passive tracers, with optional AMR.
  It covers:

  * creating particles in ``PostInitializationMesh`` with ``Create(counts)``;
  * a single ``AddLukewarmExchangeTasks`` round per step;
  * depositing onto a field through ``Locate`` and ``GetCellIndices``.

* ``example/lukewarm_transport`` pushes tracers that may cross many blocks in a step.
  It uses the iterative push with ``GetLastReceived`` and
  ``AddLukewarmCompletionTasks``.

.. note::
    This file was made in part with generative AI.
