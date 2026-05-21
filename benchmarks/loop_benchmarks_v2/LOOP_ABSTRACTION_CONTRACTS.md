# Loop Abstraction Contracts

This document describes the current loop-abstraction paths in `benchmarks/loop_benchmarks_v2` and the contracts they are intended to satisfy.

The goal is to keep the semantic choices explicit before changing the implementation again.

One need to be careful about making changes to the loop abstraction headers without thinking carefully. When in doubt, ask someone else.

## Core Types

`IndexSpace<loop_tag, inner_tag>`
- Defines an space of (block, k, j, i) points to iterate over.
  - What we call the v index is an intermediate level that a user of the loop hierarchy can write for loops in, do block level work, etc. 
  - The level of the v loops in the hierarchy of the (b, k, j, i) space is determined by the loop tags.
- Describes the logical iteration space and the memory space for a block.
- Carries `nblocks`, `ninner`, and the logical/memory indexers.
- Selects the outer-loop shape at compile time.
- Is the object that is passed into `outer(...)`. 

`InnerIndexRange<IndexSpaceType>`

- Describes one slice of an index space.
- Carries the block index and the current slice state.
- Is the object passed into `inner(...)`.
- Exposes `GetKJI(int idx)` so tests and pack-view helpers can recover `(k, j, i)` from the current inner index contract.

## Loop Tags

### `bvoi`

Shape: block -> var -> outer -> inner

- The loop that occurs in `outer(...)` only goes over blocks
- The call to `inner(...)` goes over the whole kji space defined in the IndexSpace, but internally this may be split into two levels of loops.
  - The split into an outer and inner is really only relevant when the inner range goes over memory space.
- The outer work (that actually goes on inside the function `inner(...)` is chunked over the logical index space.
- The inner loop contract depends on `inner_tag`.
- This is a mixed logical/memory path and needs the current slice state to translate indices correctly.
- When we directly index into memory from this loop type, it is relative to the starting index of the index space.

### `bovi`

Shape: block -> outer -> var -> inner
 
- The `outer(...)` loop now runs over blocks and chunks of the kji space. 
- The `inner(...)` loop now runs over a single chunk of kji space.
- The variable loop(s) now sit between the {block, kji chunk} and {chunk members} spaces
- This is the main contiguous-span path.
- The inner loop contract depends on `inner_tag`.
- When we directly index into memory from this loop type, it is relative to the starting index of the current inner chunk. 

### `boiv`

Shape: block -> outer -> inner -> var

- The inner loop walks one logical cell at a time. So it is really not a loop at all. Logically, it is the limit of bovi for inner chunk 
  size one, but it requires its own code path for performance reasons.
- This is the hot-path shape for coordinate-based access.
- The range object carries the current `(k, j, i)` point directly.
- Direct memory access is relative to the current `(k, j, i)` index. 

## Inner Tags

These define how to traverse the one inner chunk of the index range. 

### `logical_flat`

- A logical variant just iterates over the cell indices contained in the inner chunk.
- The logical region must be touched exactly once.
- Non-logical cells (i.e. ghost cells) must not be touched.
- The logical variant `logical_flat` calls the passed auto functor with an integer index for directly indexing a pointer. The indexing must
  agree with the Loop tag memory indexing conventions above.   
  - This requires all fields being accessed within a given kernel to have the same memory layout so that they can share the same flat index
  - This logical form will be most likely to vectorize since calls should inline to look like `var[idx]` within the innermost loop.

### `logical_coords`

- A logical variant just iterates over the cell indices contained in the inner chunk.
- The logical region must be touched exactly once.
- Non-logical cells (i.e. ghost cells) must not be touched.
- The auto functor receives an `Index3` object that contains the k, j, i indices of the current iteration point. 
  - This contract is required when fields accessed within a kernel have a different memory layout (say a face centered field and a cell centered field). 
  - Different memory layouts are probably the only time when this layout is preferred. 

### `memory`

- The memory variant iterates over all points in memory between the the start and end of the inner logical iteration space.
- This will touch inactive zones, but for most use cases their values are safe to mutate. It is just unecessary work.
- Nevertheless, this pattern can be more performant since it can consume long runs of memory uniformly.
- The auto functor receives an integer index for directly indexing a pointer. The indexing must
  agree with the Loop tag memory indexing conventions above. 
- The logical region must still be touched exactly once.
- Ghost (i.e. non-logical) cells may also be touched if they lie inside the contiguous span.
- The exact span is an implementation detail, but raw and Kokkos must agree.

## Body Signatures

The inner body may be written in two common forms:

- `f(auto idx)`
- `f(int k, int j, int i)`

When both forms are viable, the `f(int k, int j, int i)` form must be selected explicitly and the dispatch order must be stable. When the `f(int k, int j, int i)`
form is selected for a given [loop_tag, inner_tag] pair, the loop structure is as described above but before calling the functor the internal index 
is transformed back to (k,j,i) space and then passed to the functor. This form may hurt performance, but is likely clearer to many users.  

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

The pack-view implementation is still under development, but this will be the first class way to access variabless in kernels written using the loop abstraction.

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

