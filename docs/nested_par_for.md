
# Nested Parallelism

The loop wrappers documented here abstracts a hierarchical parallelism model,
which allows more fine grain control on core level vs. vector level parallelism
and allows explicitly defined caches for tightly nested loops. These wrappers
provide a simplified interface of the [hierarchical parallelism in
Kokkos](https://github.com/kokkos/kokkos/wiki/HierarchicalParallelism)

## `par_outer_for`

`par_outer_for` abstracts the team or multicore parallelism for outer loops.
Inside the loop body, in the lambda function provided to `par_outer_for`,
synchronization and memory sharing between the threads in the single team is
possible through the `member_type` team member type from Kokkos.

The bytes of scratch memory for cache needed by a single team is specified via
the `scratch_size_in_bytes`, which needs be computed using `ScratchPadXD::shmem_size`.

The argument `scratch_level` defines where the scratch memory should be
allocated. For CUDA GPUs, `scratch_level=0` allocates the cache in the faster
by smaller `shared` memory and `scratch_level=1` allocates the cache in the
slower but larger `global` or on device RAM memory. For CPUs, currently
`scratch_level` makes no difference.


## `par_inner_for`

`par_inner_for` abstracts the vector level parallelism of compute units within a team.

## `ScratchPadXD`

Data type for memory in scratch pad/cache memory. Use `ScratchPadXD::shmem_size`, which is documented in [the 
Kokkos documentation](https://github.com/kokkos/kokkos/wiki/HierarchicalParallelism) for determining scratch pad memory needs before kernel launch.


