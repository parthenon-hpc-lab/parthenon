.. _loop abstraction:

Loop Abstraction
================

The loop abstraction in ``src/loop_abstraction`` (header
``loop_abstraction/loop_abstraction.hpp``, namespace
``parthenon::loop_abstraction``) is a higher-level way to write block-structured
kernels than raw :ref:`nested par for` calls. It lets a kernel describe *what*
index space it iterates over and *how* variables are laid out, and then chooses an
efficient loop structure for the target backend at compile time.

It is a newer, more experimental interface than ``par_for``/``par_for_outer`` and is
primarily used by downstream applications with demanding reconstruction/flux kernels.
The precise semantic contracts each path must satisfy are recorded in
``src/loop_abstraction/LOOP_ABSTRACTION_CONTRACTS.md``; that document is the
(somewhat) authoritative reference for the invariants summarized here.

.. warning::

   The loop-abstraction headers encode subtle index-space and scratch contracts.
   Read ``LOOP_ABSTRACTION_CONTRACTS.md`` before changing them.

Overview
--------

Two objects and two free functions form the core of the API:

- ``IndexSpace<loop_tag, inner_tag, backend>`` describes the logical
  ``(block, k, j, i)`` iteration space and the memory layout of a block. The three
  template parameters fix the loop shape, the inner traversal, and the backend at
  compile time. The backend has a default that is raw for loops with simd markings 
  on host and kokkos based loops on device.
- ``InnerIndexRange`` is one slice of an ``IndexSpace`` (a block plus the current
  chunk of ``kji`` space). It is the object handed to inner loop bodies and knows how
  to translate between flat, memory, and logical ``(k, j, i)`` indices.
- ``outer(idx_space, f)`` launches the outer loop. Its body ``f(idx_range, b)``
  receives an ``InnerIndexRange`` and the block index ``b``.
- ``inner(idx_range, g)`` runs the inner loop over one slice. Its body ``g`` may take
  either a single index (``g(auto idx)``) or explicit coordinates
  (``g(int k, int j, int i)``).

A minimal kernel looks like:

.. code:: cpp

   namespace la = parthenon::loop_abstraction;
   using IST = la::IndexSpace<la::loop_tag::bovi, la::inner_tag::logical_flat>;

   IST idx_space(ninner, IndexDomain::interior,
                 0, nblocks, md, TopologicalElement::CC); 

   la::outer(idx_space, KOKKOS_LAMBDA(const IST::idx_range_t &idx_range, int b) {
     la::inner(idx_range, [&](auto idx) {
       const auto [k, j, i] = idx_range.GetKJI(idx);
       // ... work at (b, k, j, i) ...
     });
   });

Lambda markings follow the usual Kokkos hierarchical-parallelism rule: mark the
``outer(...)`` body with ``KOKKOS_LAMBDA`` (it is stored and invoked on the device),
and leave the ``inner(...)`` bodies as plain ``[&]`` lambdas (they are defined inside
the outer device lambda, so they are already device code and capture by reference).

The outer body must name its parameter type rather than use ``auto``: nvcc rejects
generic extended (``__host__ __device__``) lambdas. ``IST::idx_range_t`` is the
(base, no-halo) range ``outer`` hands the body; in code templated on the index space,
spell ``la::InnerIndexRange<IST>`` directly to avoid a dependent-name ``typename``.

Loop tags
---------

The ``loop_tag`` selects where the variable (``v``) loop sits in the
``block -> outer -> inner`` hierarchy. This controls the generated loop structure.

The "variable" (``v``) level is an intermediate level in the hierarchy at which a user
can write their own loops -- typically a loop over the fields/components a kernel
updates, but more generally any per-block work that sits between the block level and
the innermost ``kji`` traversal. The name was chosen early in prototyping (the loop is
usually over *variables*) and stuck; read ``v`` as "the level where you write your own
block-local loop." Its placement relative to the ``outer`` and ``inner`` loops is what
each tag names, and it determines what state is in scope for that loop:

- In ``bvoi`` the ``v`` loop sits *above* the ``outer``/``inner`` traversal, so a
  ``v`` iteration spans the whole ``kji`` space of a block.
- In ``bovi`` the ``v`` loop sits *inside* the chunk ``outer`` loop but *around* the
  ``inner`` traversal, so each ``v`` iteration covers one ``kji`` chunk.
- In ``boiv`` the ``v`` loop is innermost, at a single logical cell.

Where ``v`` sits also governs how it composes with scratch and halos (which are
chunk-relative in ``bovi``/``boiv`` but block-relative in ``bvoi``); the tradeoffs
between these placements are discussed further in
``LOOP_ABSTRACTION_CONTRACTS.md``.

``bvoi`` (block, var, outer, inner)
   ``outer(...)`` runs over blocks only; ``inner(...)`` walks the whole ``kji``
   space (internally possibly split into an outer/inner pair). A mixed
   logical/memory path.

``bovi`` (block, outer, var, inner)
   ``outer(...)`` runs over blocks *and* chunks of ``kji`` space; ``inner(...)``
   runs over one chunk. This is the main contiguous-span path.

``boiv`` (block, outer, inner, var)
   The inner "loop" is a single logical cell -- the hot path for point-wise,
   coordinate-based access. Logically the ``inner_chunk = 1`` limit of ``bovi``, but
   with its own code path for performance.

Inner tags
----------

The ``inner_tag`` selects how one inner chunk is traversed:

``logical_flat``
   Visits each logical cell exactly once, passing a flat integer suitable for
   ``var[idx]``-style access. Requires that all fields touched in the kernel share a
   memory layout so they can share the flat index. Most likely to vectorize.

``logical_coords``
   Same logical-cell coverage, but the body receives an ``Index3`` with ``(k, j, i)``.
   Use this when fields in the kernel have *different* layouts (e.g. cell-centered and
   face-centered).

``memory``
   Iterates the contiguous memory span for the chunk, which may include ghost cells.
   The logical region is still touched exactly once; ghost cells inside the span may
   also be touched (their post-loop values are not meaningful). Can be faster by
   consuming long uniform runs of memory.

.. note::

   ``boiv`` combined with ``memory`` is rejected at compile time: a single-cell
   inner range has no meaningful contiguous-span traversal.

Backend selection
-----------------

The third ``IndexSpace`` template parameter is the ``loop_backend``:

- ``loop_backend::raw`` -- a plain host loop nest (with ``#pragma omp simd``).
- ``loop_backend::kokkos`` -- dispatch through Kokkos parallel policies.

It defaults to ``default_loop_backend_v``, which is ``raw`` when the device execution
space is the host space and ``kokkos`` otherwise. ``outer``/``inner`` dispatch on this
tag with ``if constexpr``, so the selection is zero-cost. Pinning the tag explicitly
is mostly useful in tests that want to exercise a specific backend regardless of the
build.

Body signatures
---------------

An inner body may be written as ``f(auto idx)`` or ``f(int k, int j, int i)``. When
both are viable the three-argument coordinate form is selected. The coordinate form
may cost some performance (the internal index is converted back to ``(k, j, i)``
before the call) but is often clearer.

Scratch
-------

Per-point scratch is registered on the ``IndexSpace`` at setup and requested inside
the outer body:

.. code:: cpp
   using recon_halo = la::halo::minus_i_t;
   idx_space.AddPerPointScratch<Real>();          // one Real per point
   idx_space.AddPerPointScratch<Real, 2, 3>();    // a 2x3 block per point
   idx_space.AddPerPointScratch<Real, recon_halo>(); // One Real per point, but with
                                                     // enough storage to cover the
                                                     // extended inner halo range

   la::outer(idx_space, KOKKOS_LAMBDA(const IST::idx_range_t &idx_range, int b) {
     auto scratch = la::GetPerPointScratch<Real>(idx_range);
     // scratch(idx), scratch(Index3{k,j,i}), scratch(c0, c1, idx), ...
   });

The scratch object specializes per loop pattern and backend (compact per-cell storage
for the point-wise ``boiv`` paths, and a host scratch for the raw backend or Kokkos team
scratch for the Kokkos backend for other paths), but the user-facing indexing is uniform.
As with raw nested parallelism, call ``idx_range.TeamBarrier()`` between a producer inner
loop and a consumer that reads values written by other threads.

Reductions
----------

``outer_reduce``/``inner_reduce`` mirror ``outer``/``inner`` but fold a single Kokkos
reducer over the index space. They are **Kokkos-only**: they always dispatch to the
Kokkos backend regardless of the ``IndexSpace`` backend tag (on a host-only build the
device execution space *is* the host, so the Kokkos reduce still runs there), and
there is no raw reduction path.

The reducer is baked into the index-space type. Build a reduction space with
``ReductionIndexSpace<lt, it, R>`` (which hides the backend template parameter) or by
rebinding an existing space with ``idx_space.WithReducer<R>()``. Its ``idx_range_t``
then carries the reduction, so the outer body is just ``(idx_range, int b)`` -- no
handle to thread through. The preferred ``outer_reduce`` overload constructs the reducer
over a fresh result and **returns** it (the result is a host scalar, so the reduce is
synchronous and the value is valid on return, no fence needed):

.. code:: cpp

   using rist = la::ReductionIndexSpace<lt, it, Kokkos::Sum<Real>>;
   rist idx_space(/* ... */);

   auto result = la::outer_reduce(idx_space,
     // Outer body param types must be named, not auto (nvcc rejects generic extended
     // lambdas). The inner_reduce body is an ordinary lambda, so auto is fine there.
     KOKKOS_LAMBDA(const rist::idx_range_t &idx_range, int b) {
       la::inner_reduce(idx_range, [&](auto idx, auto &v) {
         v += /* something at idx */;
       });
     });

An overload instead takes a caller-constructed reducer instance *last*, matching ``Kokkos::parallel_reduce(policy, functor, reducer)``. This returns void and its reducer type must match the space's ``reduction_t``. This overload is necessary for, e.g., reducing into a
``View``, ``ScatterView``, or device memory.

Because the reducer type lives on the index space, ``inner_reduce`` uses the 
reducer's merge operation to combine each inner team reduction into the enclosing
reduction without the caller restating it, and a single ``outer_reduce`` region may
contain several ``inner_reduce`` calls (interleaved with plain ``inner`` calls that
only fill scratch) that all join into one accumulator. There is one reducer per region.
The ``inner_reduce`` body takes the usual index form plus a trailing reduction-value
reference.

Two rules keep reductions off ghost cells:

- **No reductions over halo ranges.** ``inner_reduce`` ``static_assert``\ s that the
  range's halo is ``none_t``. Extend a range only for producer (scratch) ``inner``
  loops and reduce over the base, halo-free range.
- **The** ``memory`` **inner tag degenerates to** ``logical_flat`` **-- but only for**
  ``inner_reduce``. Under a reduction the ``memory`` tag iterates logical cells rather
  than a contiguous memory span, so no swept ghost cell is folded in (the body still
  gets a memory-relative flat index, so call sites are unchanged). This is scoped
  strictly to ``inner_reduce``: a plain ``inner()`` inside an ``outer_reduce`` region
  behaves exactly as under ``outer()`` and does **not** degenerate -- with the
  ``memory`` tag it still sweeps whole contiguous spans, ghosts included. A
  ``memory``-tag producer feeding an ``inner_reduce`` consumer is therefore fine; just
  don't assume the producer stayed inside the logical set.

Halos
-----

The common reconstruction-to-flux pattern is a producer inner loop that writes reconstructed
states into scratch over an extended range, followed by a consumer flux loop over
the base range that accessess the scratch memory with offsets. The parthenon loop abstraction
implements patterns like this through the concept of inner range halos. As an example, a simple
flux calculation kernel in the loop abstraction might look like (see below for pack views used 
in the example):

.. code:: cpp
   const auto desc = MakePackDescriptor<var>(md, {}, {parthenon::PDOpt::WithFluxes});
   auto pack = desc.GetPack(md);
   // Declare an index space over F1 faces 
   IndexSpace<loop_tag, inner_tag> idx_space(ninner, IndexDomain::interior,
                                             0, pack.GetNBlocks(), md, TopologicalElement::F1); 
   
   using recon_halo = la::halo::minus_i_t;
   idx_space.AddPerPointScratch<Real, recon_halo>(2);  // at setup

   // Get a value that offsets the index of an inner loop by +1 in the X1 direction
   const auto dx1 = idx_space.GetDelta(X1DIR);

   la::outer(idx_space, KOKKOS_LAMBDA(const IST::idx_range_t &idx_range, int b) {
     const auto halo_range = idx_range.AddHalo<recon_halo>();
     auto scratch_plus = la::GetPerPointScratch<Real>(halo_range);
     auto scratch_minus = la::GetPerPointScratch<Real>(halo_range);

     auto pv = la::make_pack_view(halo_range, pack); 
     
     // Produce reconstructed left and right states across the halo range
     la::inner(halo_range, [&](auto kji) {
       scratch_plus(kji) = reconstruct_plus(pv(var(), kji - dx1), pv(var(), kji), pv(var(), kji + dx1)); 
       scratch_minus(kji) = reconstruct_minus(pv(var(), kji - dx1), pv(var(), kji), pv(var(), kji + dx1)); 
     });
     idx_range.TeamBarrier();  // producer must finish before the consumer reads

     // Consume (use) reconstructed states in Riemann solver to calculate fluxes
     auto fv = la::make_flux_pack_view(idx_range, pack, X1DIR); 
     la::inner(idx_range, [&](auto kji) {
       fv(var(), kji) = riemann(scratch_plus(kji - dx1), scratch_minus(kji));
     });
   });

More explicitly, a halo is a compile-time annotation naming the neighboring produced
values a consumer loop needs. If a consumer inner loop runs over a logical point set
``S``, then a producer that fills scratch for the consumer must cover ``S`` plus the
shifted copies named by the halo:

.. code:: cpp

   AddHalo<halo_t<h1, h2, ...>>(S) == S ∪ shift(S, h1) ∪ shift(S, h2) ∪ ...

A halo is *not* the same as a reconstruction stencil width: the stencil is internal to
computing one value, while the halo describes which produced neighbors must exist.

Pack and variable views
------------------------

Views adapt a ``SparsePack`` to the loop abstraction so variable access follows
the same index conventions as the loop body:

- ``make_pack_view(idx_range, pack)`` -- a view over all non-sparse variables contained 
  in ``pack``.
- ``make_sparse_pack_view(idx_range, pack, sparse_idx)`` -- a view over all sparse variables
  contained in ``pack`` at sparse index ``sparse_index``.
- ``make_var_view(idx_range, pack, var)`` -- a single-variable view.
- ``make_flux_pack_view(idx_range, pack, dir)`` / ``make_flux_view(...)`` -- the
  flux-array counterparts, for one sweep direction. Note that this is different from how sparse 
  packs and variable packs work in Parthenon, where you can request fluxes from the pack and do 
  it for any direction. Here the flux view only contains fluxes and only for the direction 
  requested on construction.

Each view accepts the same index forms the body produces (flat ``int``, ``Index3``,
or explicit ``k, j, i``), so a kernel can be written once and reused across inner
tags. In `inner_tag::logical_coords` loops, these are just light wrappers that call
through to the sparse packs. For all other `inner_tag`s, pack view construction directly
pulls out pointers to the variables. This can promote vectorization and be a significant
performance benefit. 

.. code:: cpp
  auto desc = parthenon::MakePackDescriptor<v1, vf>(md);
  auto pack = desc.GetPack(md);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const idx_space_t::idx_range_t &idx_range, int b) {
        auto pv = loop_abstraction::make_pack_view(idx_range, pack);
        loop_abstraction::inner(idx_range, [&](auto idx) {
          pv(v1(), idx) = ...;
          pv(TE::F1, vf(), idx) = ...;
          pv(TE::F2, vf(), idx) = ...;
          pv(TE::F3, vf(), idx) = ...;
        });
      });
