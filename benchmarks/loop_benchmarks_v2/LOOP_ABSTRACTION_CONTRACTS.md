# Loop Abstraction Contracts

This document describes the current loop-abstraction paths in `benchmarks/loop_benchmarks_v2` and the contracts they are intended to satisfy.

The goal is to keep the semantic choices explicit before changing the implementation again.

## Core Types

`IndexSpace<loop_tag, inner_tag>`

- Describes the logical iteration space and the memory space for a block.
- Carries `nblocks`, `ninner`, and the logical/memory indexers.
- Selects the outer-loop shape at compile time.

`InnerIndexRange<IndexSpaceType>`

- Describes one outer slice of work.
- Carries the block index and the current slice state.
- Is the object passed into `inner(...)`.
- Exposes `GetKJI(int idx)` so tests and pack-view helpers can recover `(k, j, i)` from the current inner index contract.

## Loop Tags

### `bvoi`

Shape: block -> var -> outer -> inner

- The outer loop is chunked over the logical index space.
- The inner loop contract depends on `inner_tag`.
- This is a mixed logical/memory path and needs the current slice state to translate indices correctly.

### `bovi`

Shape: block -> outer -> var -> inner

- The outer loop is also chunked, but the variable sits outside the inner loop.
- This is the main contiguous-span path.
- The inner loop contract depends on `inner_tag`.

### `boiv`

Shape: block -> outer -> inner -> var

- The inner loop walks one logical cell at a time.
- This is the hot-path shape for coordinate-based access.
- The range object carries the current `(k, j, i)` point directly.

## Inner Tags

### `logical_flat`

- The body receives a flat integer index.
- The logical region must be touched exactly once.
- Non-logical cells must not be touched.
- The intent is that the integer is usable as a flat logical-space contract, not a coordinate tuple.

### `logical_coords`

- The body receives coordinates, either as `Index3` or as `(k, j, i)`.
- The logical region must be touched exactly once.
- Non-logical cells must not be touched.
- This is the preferred contract when the caller wants coordinate access.

### `memory`

- The body receives a flat integer index over a contiguous memory span.
- The logical region must still be touched exactly once.
- Halo cells may also be touched if they lie inside the contiguous span.
- The exact span is an implementation detail, but raw and Kokkos must agree.

## Body Signatures

The inner body may be written in two common forms:

- `f(auto idx)` or `f(int idx)`
- `f(int k, int j, int i)`

When both forms are viable, the coordinate form must be selected explicitly and the dispatch order must be stable.

## Current Backend Requirement

The raw and Kokkos implementations are expected to satisfy the same contract for a given `IndexSpace`.

For tests, the safest reference is:

- a plain host nested loop over `(b, v, k, j, i)` for logical-cell correctness
- direct raw-vs-Kokkos comparison for backend parity

The tests should not reimplement the abstraction logic as a second source of truth.

## Pack View Contract

`make_pack_view(inner_range, pack, b)` is the helper that adapts a `SparsePack` to the current loop contract.

The current intent is:

- `logical_coords` should forward coordinate access directly to the pack.
- `logical_flat` and `memory` should support flat-index access.
- `boiv` should remain a very thin adapter around the pack and current range.

The pack-view implementation is temporary scaffolding, so the important thing is the loop contract it preserves, not the exact storage shape.

## Contract Summary

The invariant that matters most is:

1. Every loop pattern must touch every logical cell exactly once.
2. `logical_flat` and `logical_coords` must not touch halo cells.
3. `memory` may touch halo cells, but must still satisfy the logical-cell contract.
4. Raw and Kokkos must agree on the contract for the same loop pattern and body signature.

## Open Questions

The current code is still being shaped around these contracts, so the following should be treated as intentional design questions until the implementation is stabilized:

- Whether a flat integer in `logical_flat` should be interpreted as logical-span-relative or memory-span-relative in every loop tag.
- Whether `boiv` should always carry coordinate state only, or whether some flat-index forms should remain range-aware.
- Whether a pack-view specialization should store raw pointers, a pack pointer, or both depending on the loop contract.

