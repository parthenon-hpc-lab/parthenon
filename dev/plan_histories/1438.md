<!-- This file was made in part with generative AI. -->

# Loop Abstraction OpenMP and Variable-View Test Plan

## Goal

Add focused regression coverage for vector-component offsets in `make_var_view` and
for the raw backend's moderate OpenMP support. Keep the tests useful in non-OpenMP
builds and avoid duplicating the existing full loop-contract matrix unnecessarily.

## Phase 1: Variable-view offsets

### Dense vector components

- Define a fixed-size, three-component cell variable and construct one `var_view`
  rooted at its first component.
- Write distinct coordinate- and component-dependent values through
  `view(component_offset, point)`.
- Cover every supported `(loop_tag, inner_tag)` pair so the test exercises:
  - flat integer indices;
  - `MemoryOffset` in `boiv/logical_flat`;
  - `Index3` and explicit `(component, k, j, i)` access in `logical_coords`;
  - cached pack-entry strides in flat/memory paths;
  - direct pack forwarding in coordinate paths;
  - block- and chunk-relative origins.
- Put scalar sentinel variables on both sides of the vector in the pack and verify
  that they remain unchanged, catching an incorrect component stride.

## Phase 2: OpenMP correctness

### `boiv` range isolation

- Use one block with enough `(k, j)` work to distribute iterations across multiple
  OpenMP workers.
- Atomically count visits at coordinates recovered from each `InnerIndexRange` and
  require every logical cell to be visited exactly once.
- Cover `boiv/logical_flat` and `boiv/logical_coords` so a regression to a shared,
  mutable current-point range is detected.

### Raw scratch isolation

- Add a focused raw `bovi` case with one block, many outer chunks, and several scratch
  allocations per outer invocation.
- Encode the block/chunk coordinates into scratch, then validate the values before the
  invocation returns. This exercises concurrent reset/allocation on the thread-local
  bump arenas.
- Add a smaller multi-block `bvoi` companion to cover block-parallel arena use.
- Do not add an arena test for `boiv`, which uses stack scratch.

### Guaranteed multithreaded execution

- Tag the focused concurrency tests with `[loop_abstraction][openmp]`.
- When `PARTHENON_ENABLE_RAW_OPENMP` is enabled, register an additional Catch2 invocation
  that marks OpenMP execution as required. In that invocation, configure four
  non-dynamic workers through the test-only OpenMP runtime before entering the raw
  loop.
- Mark that CTest invocation as consuming four processors.
- Keep production code independent of `<omp.h>` and OpenMP runtime APIs. Guard the
  test-only runtime calls with `_OPENMP` so non-OpenMP builds continue to compile and
  run the ordinary serial coverage.

## Validation

- Run `git diff --check`.
- Build `unit_tests`.
- Run the focused variable-view tests after Phase 1.
- After Phase 2, run the focused OpenMP invocation with four threads and the existing
  loop-abstraction contract/scratch tests.

## Scope boundary

Phase 1 changes only variable-view tests. Phase 2 changes only loop-abstraction tests
and conditional CTest registration; it does not add an OpenMP runtime dependency to
production code.
