# Loop Benchmark Goals

This document defines what the loop benchmark is trying to measure, what
constraints matter for interpreting the results, and what we should verify
before trusting conclusions about a given loop pattern.

## Problem We Are Modeling

Parthenon operations work over packs that combine:

- blocks
- variables on each block
- spatial values on each block

In the benchmark code, `LoopData` stores those values as `View5D`s with the
index order `(block, variable, k, j, i)`.

The real Parthenon layout is more complicated:

- a pack is conceptually a 2D view over block and variable
- each element points at a 3D spatial view
- the number of variables can be ragged across blocks

This benchmark keeps the data model simpler than the production implementation,
but it should still preserve the important performance questions.

## Abstraction We Want To Test

We want a loop abstraction that looks close to the Kokkos style while still
allowing a CPU backend to lower to raw `for` loops when that is faster.

The intended shape is roughly:

```cpp
Indexer idxer(0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e);
loop_abstraction_outer(idxer, KOKKOS_LAMBDA(IndexCounter &id, const int b) {
  for (int v = 0; v < pack.nvars(b); ++v) {
    loop_abstraction_inner(idxer, id, [&](const int b, const int k, const int j,
                                          const int i) {
      // kernel
    });
  }
});
```

The key idea is that the same source-level structure should be able to express:

- flat loops over the full memory range
- active-cell-only loops
- chunked logical loops
- portable Kokkos execution
- CPU execution that lowers to raw `for` loops internally

In the simplest flat case, `loop_abstraction_inner` should be able to act like a
single call rather than a real inner loop.

## What We Want To Learn

The benchmark should answer these questions on a new machine:

- Which loop pattern is fastest for a given kernel?
- How does the answer change as block size changes?
- How does the answer change as arithmetic intensity changes?
- How does the answer change as the kernel access pattern changes?
- How sensitive is performance to chunk size in hierarchical patterns?
- Which variants actually vectorize on CPU?
- Can the abstraction preserve vectorization when the CPU backend lowers to raw
  loops?
- How much performance do we lose if the abstraction does not explicitly pull
  pointers before the inner loop?
- What is the best achievable full-memory throughput if ghost-zone avoidance is
  not part of the problem?

The goal is not just a raw timing table. The goal is a result set that explains
why one pattern wins on one architecture and not another.

The benchmark metric is cell-updates per second, not raw elapsed time. For any
pattern that traverses the full memory footprint, the update count should be
normalized by the total number of cells that loop actually touches. For
active-cell-only patterns, the normalization should use the number of active
cells.

## Core Loop Patterns To Compare

The benchmark should compare a small set of patterns that represent different
ways Parthenon could traverse pack data.

### 1. Flat dense loops

This is the simplest baseline:

- iterate directly over `(b, v, k, j, i)`
- assume the same variable count on every block
- walk memory in the natural contiguous order

This is the pattern that is most likely to vectorize when there are no ragged
counts and no complicated index translation.

This baseline is intentionally somewhat outside the main test matrix. It
answers the question: "what is the best throughput we could plausibly get if we
did not need to skip ghost zones?"

On GPU, this should be expressed as a one-dimensional Kokkos loop over the
flattened index space, not an MDRange kernel, unless a tuned MDRange version is
added later for comparison.

### 2. SIMD-friendly CPU loops

These loops use raw `for` loops and `#pragma omp simd` on the innermost
dimension.

They are important because they test whether we can get reliable CPU
vectorization from a loop written in a style close to production code.

The abstraction should be compared against this baseline, because this is the
most direct answer to "can the generic form still vectorize?"

### 3. Hierarchical raw-span loops

These loops flatten some chunk of the logical index space, then use raw pointers
inside the inner loop.

The current hand-written version in `loop_patterns_luke.cpp` is an example of
this style:

- inner loop walks contiguous memory
- outer loops handle block, variable, and chunk selection
- pointers are pulled out before the inner loop

This pattern is useful because it can preserve portability of the indexing
logic while still giving the compiler a simple inner loop.

This is also the main candidate for "generic abstraction plus CPU lowering."
The benchmark should tell us whether the indirection cost is negligible or
material.

### 4. Logical-index chunked loops

These loops iterate over chunks of the logical active-cell domain rather than
the full memory domain.

They matter because production Parthenon code often wants to visit active cells
only, not every ghost cell.

### 5. Kokkos-based portable loops

These loops are the portable comparison points.

They are important because the project is not trying to replace Kokkos. It is
trying to understand when a portable loop abstraction is good enough, and when
more explicit indexing or layout-aware code is needed.

For GPU-oriented runs, a flattened one-dimensional Kokkos loop over the logical
or dense index space should be included as the portable baseline. MDRange can
remain a secondary comparison if we want to study its tuning sensitivity, but
it should not be the only GPU reference point.

## Unified Kernel Family

The benchmark should use one unified kernel family instead of several
human-named kernels that are easy to misread.

The unified kernel should be parameterized by:

- arithmetic intensity, using a template or compile-time `NITER`
- stencil width in each spatial direction
- the number and arrangement of neighbor pointers passed to the kernel

The goal is to make the kernel behavior explicit in terms of:

- how much computation it does per cell
- how many neighbor values it reads per cell
- whether those neighbor values are contiguous or direction-specific

This should replace the old light/flux/stencil/heavy naming in the benchmark
spec.

Representative parameter points should be chosen so the benchmark spans:

- low arithmetic intensity
- moderate arithmetic intensity
- high arithmetic intensity
- narrow stencils
- wider stencils
- cases that are likely to vectorize well and cases that are intentionally more
  challenging

The exact parameter sets can evolve, but the kernel family should stay unified.

The benchmark should support two kernel-call styles:

- a pointer-hoisted form, where the loop structure assembles pointers before
  calling the cell update
- an abstraction-native form, where the loop abstraction passes structured
  indices and the kernel resolves accesses through the abstraction

## Constraints That Matter

These constraints are required if we want the results to be interpretable.

### CPU vectorization must be verified, not assumed

For any raw CPU loop we care about, we should prove that the compiler emitted
vector code.

That means checking compiler reports or generated assembly, not just timing the
binary.

If a loop is marked SIMD but does not vectorize, the benchmark is lying about
what it is measuring.

This is especially important for the abstraction, because a clean source-level
API may hide an implementation that prevents the compiler from seeing a simple
inner loop.

### The inner loop must be contiguous and simple

For vectorization to be plausible, the innermost loop should:

- have unit-stride access
- avoid unnecessary address arithmetic
- avoid hidden control flow
- avoid function calls that the compiler cannot inline

The hand-written raw-span style is attractive because it lets us pull pointer
setup outside the SIMD loop.

For the abstraction, we should test two CPU implementations if possible:

- one that passes explicit pointers into the inner kernel
- one that relies on the abstraction alone

That isolates whether pointer pulling is a real requirement or just an
optimization detail.

The unified kernel should also be written so that stencil inputs can be
assembled from arrays of pointers instead of hard-coded neighbor fetches. That
lets us vary stencil width without changing the kernel shape every time.

### Ghost cells must be handled explicitly

Most production kernels only need active cells, but it is usually safe to
operate on ghost cells as well.

That distinction matters because:

- full-memory loops are simpler and often more vector-friendly
- active-cell-only loops are closer to real Parthenon use cases
- the benchmark should tell us the cost of each choice

### Ragged variable counts must be represented, but not overcomplicated

The benchmark should capture the fact that blocks may have different numbers of
active variables.

It does not need to model every detail of the production ragged pack
implementation, but it should preserve the key performance effect: loop bounds
may differ by block.

### The benchmark should separate layout effects from kernel effects

We should avoid conflating:

- data layout
- index order
- chunk size
- kernel arithmetic intensity
- stencil width

If one experiment changes all of them, the result is hard to explain.

## What The Benchmark Output Should Support

The benchmark should produce output that makes it easy to compare:

- different block sizes
- different kernels
- different loop patterns
- different chunk sizes
- CPU and portable Kokkos variants
- full-memory baseline throughput versus active-cell test-matrix throughput

The current analysis script structure is useful because it already produces a
PDF-style narrative from command-line runs. The benchmark suite should preserve
that workflow.

The preferred workflow is:

- choose loop type, block size, kernel parameters, and backend from the command
  line
- run the C++ executable once per configuration
- let the Python script orchestrate repeated runs and collect results

That keeps the benchmark binary focused on execution and the Python layer
focused on experiment management.

## What We Need To Trust Before Using Results

Before we treat a result as meaningful, we should confirm:

- the selected benchmark variant is actually exercised
- the CPU raw loops vectorize when they are supposed to
- the output is numerically consistent across the compared variants
- the chosen block sizes and chunk sizes are representative of the target
  machine

If any of those are not true, the performance numbers are not reliable enough
to guide a design decision.

## Practical Experiment Matrix

A useful first matrix would be:

- one dense full-memory baseline outside the main matrix
- one SIMD-oriented CPU active-cell loop
- one hierarchical raw-span loop with pointer extraction
- one generic abstraction-based CPU loop without explicit pointer pulling
- one generic abstraction-based CPU loop with explicit pointer pulling if we can
  express it cleanly
- one logical-index chunked loop
- one portable Kokkos counterpart for each major family
- one unified kernel family swept over a small set of arithmetic-intensity and
  stencil-width parameter combinations

Then sweep:

- `ni` for vector-length and cleanup behavior
- chunk size for hierarchical patterns
- `NITER` for arithmetic intensity
- stencil width per direction for neighborhood access patterns

The dense full-memory baseline should be reported alongside the matrix, but it
should not be treated as one of the active-cell loop patterns under evaluation.

## Non-Goals

This benchmark is not trying to:

- reproduce the full Parthenon pack implementation exactly
- measure MPI effects
- measure threading scalability in the general case
- replace the real application kernels
- hide compiler behavior behind opaque abstractions

## Open Questions

We should still decide:

- which loop pattern should be the reference CPU baseline
- whether the hand-written raw-span pattern becomes a new named variant
- what compiler-vectorization checks we want to automate
- how many variants are enough to keep the matrix interpretable
- whether the benchmark should bias toward active-cell-only loops or full-memory
  loops when the two conflict
- whether the abstraction can cover all target patterns without obscuring the
  CPU fast path
- whether pointer pulling needs to be a first-class abstraction feature or just
  an optional optimization hook
- whether the abstraction can preserve vectorization without specialized CPU
  code outside the common interface
- how to encode the unified kernel parameters so they remain readable in the
  benchmark output
- whether stencil width should be compile-time, runtime, or both
- whether the Python layer should run each configuration serially or allow
  limited parallelism without making the results noisy
- whether we want a tuned GPU MDRange variant as a separate reference point or
  just the flattened Kokkos loop

## Relationship To The Next Abstraction Work

This benchmark is meant to inform the next Parthenon loop abstraction, not to
freeze it.

The abstraction work should probably be driven by the same questions this
benchmark answers:

- how do we express block/variable/spatial traversal cleanly
- how do we preserve vectorization on CPU
- how do we stay portable across Kokkos and hand-written CPU paths
- how do we keep the active-cell and ghost-cell cases understandable

If the abstraction cannot express the fast patterns identified here, that is a
design problem worth catching early.
