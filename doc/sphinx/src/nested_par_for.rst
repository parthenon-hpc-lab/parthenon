.. _nested par for:

Nested Parallelism
==================

The loop wrappers documented here abstracts a hierarchical parallelism
model, which allows more fine grain control on core level vs. thread and
vector level parallelism and allows explicitly defined caches for
tightly nested loops. These wrappers provide a simplified interface of
the `hierarchical parallelism in
Kokkos <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html>`__

For an example of the nested parallel wrappers in use, see `the unit
test <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/unit/kokkos_abstraction.cpp>`__

``par_for_outer``
-----------------

``par_for_outer`` abstracts the team or multicore parallelism for outer
loops. Inside the loop body, in the lambda function provided to
``par_for_outer``, synchronization and memory sharing between the
threads in the single team is possible through the ``member_type`` team
member type from Kokkos.

The bytes of scratch memory for cache needed by a single team is
specified via the ``scratch_size_in_bytes``, which needs be computed
using ``ScratchPadXD::shmem_size``.

The argument ``scratch_level`` defines where the scratch memory should
be allocated. For CUDA GPUs, ``scratch_level=0`` allocates the cache in
the faster by smaller ``shared`` memory and ``scratch_level=1``
allocates the cache in the slower but larger ``global`` or on device RAM
memory. For CPUs, currently ``scratch_level`` makes no difference.

Note that every thread within a team will execute code inside a
``par_for_outer`` but outside of the ``par_for_inner``.

``par_for_inner``
-----------------

``par_for_inner`` abstracts the thread and vector level parallelism of
compute units within a single team or core. Work defined through a
``par_for_inner`` will be distributed between individual threads and
vector lanes within the team.

``ScratchPadXD``
----------------

Data type for memory in scratch pad/cache memory. Use
``ScratchPadXD::shmem_size``, which is documented in `the Kokkos
documentation <https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html?highlight=hierarchical>`__
for determining scratch pad memory needs before kernel launch.

On Barriers
---------------------

In order to ensure that individual threads of a team are synchronized
always call ``team_member.team_barrier();`` after an ``par_for_inner``
if the following execution depends on the results of the
``par_for_inner``. This pertains, for example, to filling a
``ScratchPadXD`` array in one ``par_inner_for`` and using the scratch
array in the next one, see `the unit
test <https://github.com/parthenon-hpc-lab/parthenon/blob/develop/tst/unit/kokkos_abstraction.cpp>`__ for sample usage.

In addition, the entry to a ``par_for_inner`` does **not** imply a
barrier and not all threads of a team may even enter an inner parallel
region (e.g., if there is not enough work – read indices – for all team
members). This can lead to unintended side-effects when all team member
write to common variables, see this
`code <https://github.com/parthenon-hpc-lab/parthenon/issues/659#issuecomment-1346871509>`__
for an example.


Cmake Options
-------------

``PAR_LOOP_INNER_LAYOUT`` controls how the inner loop is implemented.

``PAR_LOOP_INNER_LAYOUT=TVR_INNER_LOOP`` uses the Kokkos
``TeamVectorRange``, which merges ``TeamThreadRange`` and
``ThreadVectorRange`` into one loop, to distribute work between threads.
``PAR_LOOP_INNER_LAYOUT=TVR_INNER_LOOP`` is the only option supported
for CUDA since the Kokkos loops are required for parallelization on
GPUs.

``PAR_LOOP_INNER_LAYOUT=SIMDFOR_INNER_LOOP`` uses a ``for`` loop with a
``#pragma omp simd`` to vectorize the loop, which typically gives better
vectorization loops than ``PAR_LOOP_INNER_LAYOUT=TVR_INNER_LOOP`` on
CPUs and so is the default on CPUs.


Performance Considerations
---------------------------

Hierarchical parallelism can produce very performant code, but a
deeper awareness of how hardware is mapped to threads is required to
get optimal performance. Here we list a few strategies/considerations.

* On CPU, with `SIMDFOR_INNER_LOOP` you may have trouble vectorizing
  unless you help the compiler along. One way to do so is to work with
  raw pointers to contiguous memory, rather than working with views
  and strides. Even for stencil ops, if you can pull out pointers that
  represent the different points on the stencil, this can help with
  vectorization.
* Similarly on CPUs, due to the cost of starting up a vector op,
  vectorization will only be a performance win if there's enough work
  in the inner loop. A minimum of 16 points is required for the op to
  vectorize at all. Experience shows, however, that at least 64 is
  really required to see big wins. One strategy for providing enough
  vector cells in the inner loop is to do a 1D ``SIMDFOR`` inner loop
  but combine the ``j`` and ``i`` indices by simply looping over the
  contiguous memory in a rasterized plane on a block.
* On GPUs, the outer loop typically maps to blocks, while the inner
  maps to threads. To see good performance, you must both provide
  enough work in the inner loop to create enough threads to fill in
  CUDA terms a streaming multiprocessor (SM, equivalent to a Compute
  Unit or CU on AMD GPUs) with multiple warps (or wavefronts for AMD)
  to take advantage of pipelining and enough work in the outer loop to
  create enough blocks to fill all SMs on the GPU divided by the
  number of simultaneous streams. The number of warps in flight on the
  inner loop per SM (which is related to "occupancy") will depend
  positively on length of the inner loop and negatively on higher
  shared memory usage (scratch pad memory in Kokkos parlance and Local
  Data Store or LDS on AMD GPUs) and higher register usage. Note that
  the number of SMs and the available shared memory and registers per
  SM will vary between GPU architectures and especially between GPU
  vendors.

IndexSplit
-------------

To balance the CPU vs GPU hardware considerations of hierarchical
parallelism, ``Parthenon`` provides a utility, the ``IndexSplit``
class, defined in the ``utils/index_split.hpp`` header file and
available in ``<parthenon/package.hpp>`` in the
``parthenon::package::prelude`` namespace.

In our experience ``IndexSplit`` is most beneficial when working with
small meshblocks, especially in two dimensions. For small blocks, we
want vectorized operations over contiguous memory for our innermost
loop, but we want that loop to contain enough work for, e.g., vector
ops to function. We have often found that the optimal split is to fuse
j, and i into the inner loop and use k and blocks in the outer loop.

The ``IndexSplit`` class can be constructed as

.. code:: cpp

  IndexSplit(MeshData<Real> md, IndexDomain domain, const int nkp, const int njp);

where here ``md`` is a ``MeshData`` object on which you want to
operate. ``domain`` specifies where in the ``MeshBlock`` you wish to
operate, for example ``IndexDomain::Interior``. ``nkp`` and ``njp``
are the number of points in ``X3`` and ``X2`` respectively that are in
the outer loop. All remaining points are in the inner loop; each team will iterate over multiple `k` and/or `j` indices to cover the specified `k/j` range. Typically
``MeshBlock`` index in the pack is also assumed to be in the outer
loop. ``nkp`` and ``njp`` also accept special flags
``IndexSplit::all_outer`` and ``IndexSplit::no_outer``, which specify
that all and none of the indices in that direction should be in the
outer loop.

A second constructor alternatively sets the range for ``X3``, ``X2``,
and ``X1`` explicitly:

.. code:: cpp

  IndexSplit(MeshData<Real> *md, const IndexRange &kb, const IndexRange &jb,
             const IndexRange &ib, const int nkp, const int njp);

where here ``kb``, ``jb``, and ``ib`` specify the starting and ending
indices for ``X3``, ``X2``, and ``X1`` respecively.

An ``IndexSplit`` object is typically used as:

.. code:: cpp

  using namespace parthenon::package::prelude;
  using parthenon::ScratchPad1D;
  using parthenon::IndexSplit;
  using parthenon::par_for_outer;
  using parthenon::par_for_inner;
  using parthenon::team_mbr_t;
  // Initialize index split object
  IndexSplit idx_sp(md, IndexDomain::interior, nkp, njp);
  
  // Request maximum size in i and j in the inner loop, for scratch
  const int Ni = idx_sp.get_max_ni();
  const int Nj = idx_sp = get_max_nj();
  const in tNmax = Ni * Nj;
  
  // single scratch array for i,j
  auto scratch_size = ScratchPad1D<Real>::shmem_size(Nmax);
  constexpr int scratch_level = 0;
  
  // Par for
  par_for_outer(
	  DEFAULT_OUTER_LOOP_PATTERN, "KernalOuter", DevExecSpace(), scratch_size,
	  scratch_level, 0, nblocks - 1, 0, idx_sp.outer_size() - 1,
	  KOKKOS_LAMBDA(team_mbr_t member, const int b, const int outer_idx) {
	    ScratchPad1D<Real> scratch(member.team_scratch(scratch_level), Nmax);
	    // Get index ranges. Note they depend on where we are in the outer index!
	    // These give us a sense for where we are in k,j space
	    const auto krange = idx_sp.GetBoundsK(outer_idx);
	    const auto jrange = idx_sp.GetBoundsJ(outer_idx);
	    // This is the loop of contiguous inner memory. May contain i and j!
	    const auto irange = idx_sp.GetInnerBounds(jrange);

	    // Whatever part of k is not in the outer loop can be looped over
	    // with a normal for loop here
	    for (int k = krange.s; k <= krange.e; ++k) {

	      // pull out a pointer some variable in some pack. Note
	      // we pick the 0th index of i at k and jrange.s
	      Real *var = &pack(b, ivar, k, jrange.s, 0);

	      // Do something with the pointer in the inner loop.
	      par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, irange.s, irange.e,
	        [&](const int i) {
		  foo(var[i]);
		});
	    }
	  });
