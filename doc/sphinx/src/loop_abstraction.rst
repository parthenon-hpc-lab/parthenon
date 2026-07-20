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
authoritative reference for the invariants summarized here.

.. warning::

   The loop-abstraction headers encode subtle index-space and scratch contracts.
   Read ``LOOP_ABSTRACTION_CONTRACTS.md`` before changing them.

Overview
--------

Two objects and two free functions form the core of the API:

- ``IndexSpace<loop_tag, inner_tag, backend>`` describes the logical
  ``(block, k, j, i)`` iteration space and the memory layout of a block. The three
  template parameters fix the loop shape, the inner traversal, and the backend at
  compile time.
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

   IST idx_space(nblocks, nx, ny, nz, nghost);

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

Body signatures
---------------

An inner body may be written as ``f(auto idx)`` or ``f(int k, int j, int i)``. When
both are viable the three-argument coordinate form is selected. The coordinate form
may cost some performance (the internal index is converted back to ``(k, j, i)``
before the call) but is often clearer.

Halos
-----

A halo is a compile-time annotation naming the neighboring produced values a consumer
loop needs. If a consumer inner loop runs over a logical point set ``S``, then a
producer that fills scratch for the consumer must cover ``S`` plus the shifted copies
named by the halo:

.. code:: cpp

   AddHalo<halo_t<h1, h2, ...>>(S) == S ∪ shift(S, h1) ∪ shift(S, h2) ∪ ...

The common reconstruction-to-flux pattern is a producer that writes reconstructed
states into scratch over a halo-extended range, followed by a consumer flux loop over
the base range:

.. code:: cpp

   using recon_halo = la::halo::minus_i_t;
   idx_space.AddPerPointScratch<Real, recon_halo>();  // at setup
   const auto dx1 = idx_space.GetDelta(X1DIR);

   la::outer(idx_space, KOKKOS_LAMBDA(const IST::idx_range_t &idx_range, int b) {
     const auto halo_range = idx_range.AddHalo<recon_halo>();
     auto scratch = la::GetPerPointScratch<Real>(halo_range);

     la::inner(halo_range, [&](auto kji) { scratch(kji) = reconstruct(kji); });
     idx_range.TeamBarrier();  // producer must finish before the consumer reads

     la::inner(idx_range, [&](auto kji) {
       flux(kji) = riemann(scratch(kji - dx1), scratch(kji));
     });
   });

A halo is *not* the same as a reconstruction stencil width: the stencil is internal to
computing one value, while the halo describes which produced neighbors must exist.

Scratch
-------

Per-point scratch is registered on the ``IndexSpace`` at setup and requested inside
the outer body:

.. code:: cpp

   idx_space.AddPerPointScratch<Real>();          // one Real per point
   idx_space.AddPerPointScratch<Real, 2, 3>();    // a 2x3 block per point

   la::outer(idx_space, KOKKOS_LAMBDA(const IST::idx_range_t &idx_range, int b) {
     auto scratch = la::GetPerPointScratch<Real>(idx_range);
     // scratch(idx), scratch(Index3{k,j,i}), scratch(c0, c1, idx), ...
   });

The scratch object specializes per backend (compact per-cell storage for the
point-wise ``boiv`` path, a bump arena for the raw backend, Kokkos team scratch for
the Kokkos backend), but the user-facing indexing is uniform. As with raw nested
parallelism, call ``idx_range.TeamBarrier()`` between a producer inner loop and a
consumer that reads values written by other threads.

Pack and variable views
------------------------

Views adapt a ``SparsePack`` to the current loop contract so variable access follows
the same index conventions as the loop body:

- ``make_pack_view(idx_range, pack)`` -- a view over a set of variable types.
- ``make_var_view(idx_range, pack, var)`` -- a single-variable view.
- ``make_flux_pack_view(idx_range, pack, dir)`` / ``make_flux_view(...)`` -- the
  flux-array counterparts, for one sweep direction.

Each view accepts the same index forms the body produces (flat ``int``, ``Index3``,
or explicit ``k, j, i``), so a kernel can be written once and reused across inner
tags.

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
