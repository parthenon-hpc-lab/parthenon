
# Nested Parallelism

The loop wrappers documented here abstracts a hierarchical parallelism model,
which allows more fine grain control on core level vs. thread and vector level
parallelism and allows explicitly defined caches for tightly nested loops.
These wrappers provide a simplified interface of the [hierarchical parallelism
in Kokkos](https://github.com/kokkos/kokkos/wiki/HierarchicalParallelism)


For an example of the nested parallel wrappers in use, see [the unit
test](../tst/unit/kokkos_abstraction.cpp)

## `par_for_outer`

`par_for_outer` abstracts the team or multicore parallelism for outer loops.
Inside the loop body, in the lambda function provided to `par_for_outer`,
synchronization and memory sharing between the threads in the single team is
possible through the `member_type` team member type from Kokkos.

The bytes of scratch memory for cache needed by a single team is specified via
the `scratch_size_in_bytes`, which needs be computed using `ScratchPadXD::shmem_size`.

The argument `scratch_level` defines where the scratch memory should be
allocated. For CUDA GPUs, `scratch_level=0` allocates the cache in the faster
by smaller `shared` memory and `scratch_level=1` allocates the cache in the
slower but larger `global` or on device RAM memory. For CPUs, currently
`scratch_level` makes no difference.

Note that every thread within a team will execute code inside a `par_for_outer`
but outside of the `par_for_inner`.

## `par_for_inner`

`par_for_inner` abstracts the thread and vector level parallelism of compute
units within a single team or core. Work defined through a `par_for_inner` will
be distributed between individual threads and vector lanes within the team.

## `ScratchPadXD`

Data type for memory in scratch pad/cache memory. Use
`ScratchPadXD::shmem_size`, which is documented in [the Kokkos
documentation](https://github.com/kokkos/kokkos/wiki/HierarchicalParallelism)
for determining scratch pad memory needs before kernel launch.


## Cmake Options

`PAR_LOOP_INNER_LAYOUT` controls how the inner loop is implemented.

`PAR_LOOP_INNER_LAYOUT=TVR_INNER_LOOP` uses the Kokkos `TeamVectorRange`, which
merges `TeamThreadRange` and `ThreadVectorRange` into one loop, to distribute
work between threads. `PAR_LOOP_INNER_LAYOUT=TVR_INNER_LOOP` is the only option
supported for CUDA since the Kokkos loops are required for parallelization on
GPUs.

`PAR_LOOP_INNER_LAYOUT=SIMDFOR_INNER_LOOP` uses a `for` loop with a `#pragma
omp simd` to vectorize the loop, which typically gives better vectorization
loops than `PAR_LOOP_INNER_LAYOUT=TVR_INNER_LOOP` on CPUs and so is the default
on CPUs.
