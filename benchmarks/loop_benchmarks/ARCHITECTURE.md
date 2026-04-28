# Loop Benchmark Architecture

This document describes the intended v2 structure for the loop benchmark.
The goal is to keep the loop and kernel specifications clear, concise, local,
and easy to inspect, while allowing the surrounding harness to be more complex
if that improves usability and measurement quality.

## Design Principles

- Keep the tested loop body easy to read in isolation.
- Keep the tested kernel easy to read in isolation.
- Separate loop structure from kernel math.
- Make the benchmark cases explicit and reproducible.
- Prefer clarity over clever generalization inside the core test code.
- Allow the orchestration layer to be more complex if it reduces duplication.

## What Should Stay Local

The following pieces should remain small and directly readable:

- the loop pattern implementation
- the unified kernel implementation
- the pointer-hoisted kernel variant
- the abstraction-native kernel variant

Someone looking at the code should be able to answer:

- what indices are traversed
- what data is read
- what data is written
- what arithmetic happens per cell
- what changes between one loop pattern and another

## What Can Be More Flexible

The surrounding benchmark harness may be more elaborate if needed:

- case matrix parsing
- fixed-footprint problem construction
- repeated-run orchestration
- reporting and CSV output
- compiler-vectorization checks
- experiment labeling

The harness should help users run clean comparisons, but it should not obscure
the actual loop/kernel code under test.

## Core Objects

### Case Spec

One case spec should describe one benchmark run.

It should include:

- problem size:
  - `nblocks`
  - `nvars`
  - `nz_interior`
  - `ny_interior`
  - `nx_interior`
  - `nghost`
  - an array of per-block variable counts, each strictly less than or equal to
    `nvars`, to simulate sparsity
- loop configuration:
  - loop type
  - any loop-specific parameters, mainly `ninner`
- kernel configuration:
  - kernel template parameters only
- backend

The problem-size specification should support two modes:

- explicit `nblocks`
- derived `nblocks` from a target total cell count plus the rest of the
  problem-size specification

That lets us hold the footprint fixed while sweeping loop or kernel choices.

### Dataset Setup

The dataset setup layer should turn a case spec into:

- allocated views
- block counts
- variable counts
- active-cell counts
- index ranges

If `nblocks` is not explicitly provided, this layer or a nearby helper should
derive it from the requested total cell count and the remaining problem-size
parameters.

This layer may be more complicated than the loop itself because it is allowed
to encode the mechanics of the test setup.

### Loop Backend

The loop backend should implement the tested traversal strategy.

Examples:

- `flat`: full-memory memory-order traversal
- `bovi`: block, outer, variable, inner
- `boiv`: block, outer, inner, variable
- `bvoi`: block, variable, outer, inner
- `logical`: active-cell traversal over logical coordinates
- `kokkos1d`: Kokkos 1D traversal over a flattened index space
- `kokkos_team`: Kokkos team-based traversal

## Loop Naming

Loop names should be short and encode the traversal order directly.

Suggested convention:

- `b` = block
- `o` = outer chunk or outer logical span
- `v` = variable
- `i` = inner contiguous span

Examples:

- `bovi` means `(block, outer, var, inner)`
- `boiv` means `(block, outer, inner, var)`
- `bvoi` means `(block, var, outer, inner)`

For the hierarchical families, we should also encode whether the inner span is
contiguous in memory or only walks active logical cells.

Suggested suffixes:

- `_contiguous` for raw contiguous inner spans
- `_logical` for active-cell-only inner spans

Examples:

- `cpu_bovi_contiguous`
- `cpu_bovi_logical`
- `cpu_bvoi_contiguous`
- `cpu_bvoi_logical`

For Kokkos, use a shorter family name that reflects the team decomposition:

- `kokkos_boiv_flat`
- `kokkos_bovi_team_contiguous`
- `kokkos_bovi_team_logical`

For hierarchical Kokkos forms, distinguish whether the implementation uses:

- pointer-hoisted raw spans
- direct view access

That distinction matters only for true hierarchical implementations. It should
be reflected in the case name or a suffix if we decide to benchmark both.

The point of the short names is to keep the benchmark output readable while
still making the traversal order explicit.

The loop backend should not hide the test structure behind unnecessary helper
logic.

### Kernel Backend

The kernel backend should implement the math performed per cell.

We want two forms:

- pointer-hoisted
- abstraction-native

Both forms should expose the same cell-update behavior.

## Kernel Contract

The unified kernel should be parameterized so the tested behavior is obvious.

Expected axes:

- arithmetic intensity, via `NITER`
- stencil width per dimension
- neighbor access pattern

The kernel should remain local and readable. It should not require reading the
whole benchmark harness to understand what it computes.

## Loop Contract

Each loop pattern should answer one specific question.

Examples:

- what is the best full-memory throughput?
- what is the best active-cell throughput?
- does the abstraction preserve vectorization?
- does pointer hoisting matter?
- does chunking help or hurt?

The loop code should make that intent obvious.

## Execution Model

The executable should support running a matrix of cases in one process.

Recommended flow:

- Python generates the matrix or experiment description.
- The executable reads the matrix and runs all cases.
- The executable emits one result row per case.
- Python postprocesses the result rows into plots and reports.

This keeps the measurement process stable and avoids repeated startup cost.

## Fixed Footprint

The benchmark should support a fixed memory footprint as part of case setup.

That means the problem size may be derived from:

- a target cell count
- a target byte count
- or an explicit `(blocks, ni, nj, nk)` tuple

Whatever convention we choose, it should be reported in the output so the
result is reproducible.

## Reporting

Each row should report at least:

- loop pattern
- kernel parameters
- problem size
- total cells touched
- active cells, when applicable
- elapsed time
- cell-updates/s
- compiler vectorization evidence when available

The main metric remains cell-updates/s.

## Recommended File Boundaries

This is a suggested organization for the v2 implementation:

- `kernel.hpp` or similar: unified kernel definitions
- `loops.hpp` or similar: loop pattern implementations
- `case_matrix.hpp/cpp`: case parsing and expansion
- `dataset.hpp/cpp`: memory setup and footprint handling
- `runner.hpp/cpp`: execution and timing
- `reporting.hpp/cpp`: CSV/summary output

The exact names can change, but the split between kernel, loop, setup, and
reporting should stay clear.

## Migration Strategy

The existing prototype can remain as a reference while v2 is built.

The safest migration path is:

- define the case schema first
- implement one unified kernel
- implement one or two loop patterns end-to-end
- add the matrix runner
- port the remaining patterns only after the structure is stable

That avoids trying to repair a large prototype before the design is settled.
