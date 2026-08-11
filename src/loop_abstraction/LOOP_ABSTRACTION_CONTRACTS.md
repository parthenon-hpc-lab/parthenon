# Loop Abstraction Contracts
[THIS FILE IS INTENDED AS RULES FOR LLMs]

This document describes the current loop-abstraction paths in `src/loop_abstraction` and the contracts they are intended to satisfy.

The goal is to keep the semantic choices explicit before changing the implementation again.

One needs to be careful about making changes to the loop abstraction headers without thinking carefully. When in doubt, ask someone else.

## Core Types

`IndexSpace<loop_tag, inner_tag, backend>`
- Defines a space of (block, k, j, i) points to iterate over.
  - What we call the `v` index is an intermediate level that a user of the loop hierarchy can write loops in, do block-level work, etc.
  - The level of the v loops in the hierarchy of the (b, k, j, i) space is determined by the loop tags.
- Describes the logical iteration space and the memory space for a block.
- Carries `nblocks`, `ninner`, and the logical/memory indexers.
- Selects the outer-loop shape at compile time.
- Selects the loop backend at compile time via `backend_v`.
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
- The outer work that actually goes on inside the function `inner(...)` is chunked over the logical index space.
- The inner loop contract depends on `inner_tag`.
- This is a mixed logical/memory path and needs the current slice state to translate indices correctly.
- When we directly index into memory from this loop type, it is relative to the starting index of the index space.

### `bovi`

Shape: block -> outer -> var -> inner
 
- The `outer(...)` loop now runs over blocks and chunks of the kji space. 
- The `inner(...)` loop now runs over a single chunk of kji space.
- The variable loop(s) now sit between the {block, kji chunk} and {chunk members} spaces.
- This is the main contiguous-span path.
- The inner loop contract depends on `inner_tag`.
- When we directly index into memory from this loop type, it is relative to the starting index of the current inner chunk.

### `boiv`

Shape: block -> outer -> inner -> var

- The inner loop walks one logical cell at a time. So it is really not a loop at all. Logically, it is the limit of bovi for inner chunk 
  size one, but it requires its own code path for performance reasons.
- The range object carries the current `(k, j, i)` point directly.so
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

The intended `logical_flat` integer contract is:

| Loop tag | `logical_flat` integer passed to the functor |
| --- | --- |
| `bvoi` | Flat index into the current memory-space view of the visited points. The origin is the memory start implied by the current `IndexSpace` and outer slice. |
| `bovi` | Flat index into the current memory-space view of the visited points. The origin is the memory start implied by the current `InnerIndexRange`. |
| `boiv` | Flat index into the current logical point. This is the one-cell contract used by the hot coordinate path, so the integer form is only meaningful if the caller wants a flat logical index. |

### `logical_coords`

- A logical variant just iterates over the cell indices contained in the inner chunk.
- The logical region must be touched exactly once.
- Non-logical cells (i.e. ghost cells) must not be touched (aside from in halo inner ranges).
- The auto functor receives an `Index3` object that contains the k, j, i indices of the current iteration point. 
  - This contract is required when fields accessed within a kernel have a different memory layout (say a face centered field and a cell centered field).
  - Different memory layouts are probably the only time when this layout is preferred.

The intended `logical_coords` contract is the same logical-cell coverage contract as `logical_flat`, but the body receives coordinates instead of a flat integer.

### `memory`

- The memory variant iterates over the contiguous memory span for the current inner range, which may include non-logical cells.
- This will touch inactive zones, but for most use cases their values are safe to mutate. It is just unnecessary work.
- Memory cells outside the original logical range are not required to retain any meaningful value after the loop.
- Nevertheless, this pattern can be more performant since it can consume long runs of memory uniformly.
- The auto functor receives an integer index for directly indexing a pointer. The indexing must
  agree with the Loop tag memory indexing conventions above.
- The logical region must still be touched exactly once.
- Ghost (i.e. non-logical) cells may also be touched if they lie inside the contiguous span.
- The exact span is an implementation detail, but raw and Kokkos must agree.

The intended `memory` integer contract is:

| Loop tag | `memory` integer passed to the functor |
| --- | --- |
| `bvoi` | Flat index into the current memory-space view of the visited points. The origin is the memory start implied by the current `IndexSpace` and outer slice. |
| `bovi` | Flat index into the current memory-space view of the visited points. The origin is the memory start implied by the current `InnerIndexRange`. |
| `boiv` | Not a `memory` contract. `boiv` uses logical coordinates or flat logical indexing, not a memory-span inner contract. |

## Body Signatures

The inner body may be written in two common forms:

- `f(auto idx)`
- `f(int k, int j, int i)`

If both `f(auto idx)` and `f(int, int, int)` are viable, the three-argument form wins. When the `f(int k, int j, int i)`
form is selected for a given `[loop_tag, inner_tag]` pair, the loop structure is as described above, but before calling the functor the internal index
is transformed back to `(k, j, i)` space and then passed to the functor. This form may hurt performance, but is likely clearer to many users.

## Current Backend Requirement

The raw and Kokkos implementations are expected to satisfy the same contract for a given `IndexSpace`.

For tests, the safest reference is:

- a plain host nested loop over `(b, v, k, j, i)` for logical-cell correctness
- direct raw-vs-Kokkos comparison for backend parity

The tests should not reimplement the abstraction logic as a second source of truth.

## Contract Summary

The invariant that matters most is:

1. Every loop pattern must touch every logical cell exactly once.
2. `logical_flat` and `logical_coords` must not touch halo cells.
3. `memory` may touch halo cells, but must still satisfy the logical-cell contract.
4. Raw and Kokkos must agree on the contract for the same loop pattern and body signature.

## Pack View Information

`make_pack_view(inner_range, pack, b)` is the helper that adapts a `SparsePack` to the current loop contract.

The current intent is:

- `logical_coords` should forward coordinate access directly to the pack.
- `logical_flat` and `memory` should support flat-index access.
- `boiv` should remain a very thin adapter around the pack and current range.

This is the first-class way to access variables in kernels written using the loop abstraction.

## Planned Extensions

These are known, deliberately-deferred extensions rather than open design questions. They are not implemented yet and are out of scope for the initial version. If you (an LLM assistant) are asked to change `NInner`/chunk shaping or per-point scratch, surface the relevant item below in conversation before proposing an implementation, since a naive change may conflict with the intended direction.

- **Expressive `NInner` arithmetic.** Allow chunk-shape expressions such as `NInner(2 * i_pencil)` (a chunk of two extended i-rows), rather than only a bare cell count or a single `chunk_shape`. This would extend the `chunk_shape`/`NInner` vocabulary so callers can describe chunk sizes as multiples of a shape resolved against the (possibly halo-extended) indexer.
- **Partially runtime-sized scratch.** Today per-point scratch is sized entirely by the template `Dims...`. For every loop tag except `boiv`, the size could instead be chosen at run time: keep the template argument as an upper bound (a capacity) but accept a runtime actual size, ignored for the `boiv` stack-scratch path. This is blocked on understanding the GPU tradeoffs of the fixed stack scratch vs. a more flexible runtime scratch -- register pressure is expected to be the deciding factor -- so it should not be implemented before that study.


# Halo ranges for inner loops Implementation Ideas

## Concept

A halo extends the **logical point set** visited by an `inner` loop.

If the original inner range visits a set of logical points `S`, then a halo is a set of additional logical offsets applied to every point in `S`.

The base range `S` is always included implicitly.

For a halo containing offsets `{h1, h2, ...}`, the extended set is

```text
S_halo = S
       ∪ {p + h1 : p ∈ S}
       ∪ {p + h2 : p ∈ S}
       ∪ ...
````

For example, a `-i` halo means

```text
S_halo = S ∪ {p - i_hat : p ∈ S}
```

This is a statement about the **logical index space**, not the memory layout.

## Why this exists

The intended use case is a producer/consumer pattern inside an `outer` loop.

For example, one inner loop computes reconstructed states into scratch, and a later inner loop computes fluxes from those reconstructed states:

```cpp
constexpr auto recon_halo = halo::minus_i;

auto scratch_p = idx_range.GetScratch<Real, recon_halo>();
auto scratch_m = idx_range.GetScratch<Real, recon_halo>();

inner(idx_range.AddHalo<recon_halo>(), KOKKOS_LAMBDA(auto kji) {
  scratch_p(kji) = reconstruct_plus(kji);
  scratch_m(kji) = reconstruct_minus(kji);
});

inner(idx_range, KOKKOS_LAMBDA(auto kji) {
  auto dx1 = idx_range.GetOffset<X1DIR>();

  flux(kji) = riemann(scratch_p(kji - dx1),
                      scratch_m(kji));
});
```

The flux loop runs over `idx_range`, but it consumes a reconstructed value at `kji - dx1`. Therefore, the reconstruction loop must produce values over `idx_range` plus that neighboring logical point set. The halo expresses this dependency.

## Halo is not reconstruction stencil width

A halo is **not** the same thing as the stencil width used by the reconstruction operator.

A reconstruction routine may read a wide input stencil:

```cpp
q(kji - 2*dx1)
q(kji - dx1)
q(kji)
q(kji + dx1)
q(kji + 2*dx1)
```

That is internal to computing one reconstructed value.

The halo instead describes which neighboring **produced scratch values** must exist for a later consumer loop.

In common reconstruction-to-flux patterns, the reconstruction stencil may be wide, but the producer halo is often only one cell.

## Offset-set representation

A halo can be represented as a small compile-time set of `Index3` offsets.

Conceptually:

```cpp
template <Index3... Offsets>
struct halo_t;
```

or equivalently:

```cpp
halo = set of logical offsets
```

The base point is implicit. Users specify only the additional shifted copies of `S`.

For example:

```cpp
using minus_i_halo = halo_t<Index3{0, 0, -1}>;
```

means

```text
S_halo = S ∪ {p + (0,0,-1) : p ∈ S}
```

not just the shifted set.

Common aliases can make this readable:

```cpp
constexpr auto recon_halo = halo::minus_i;
constexpr auto transverse_halo = halo::plus_j;
```

For the expected use cases, the number of offsets is small: usually one, sometimes two or six, and perhaps up to around twelve in more general cases.

## Important implementation rule

Do not define halos by manipulating the original flat index space directly.

A geometric offset such as `+i` means

```text
(k, j, i) -> (k, j, i + 1)
```

It should not be treated as `flat + 1` in the original interior indexer, because that can accidentally wrap across row boundaries.

Instead, the implementation should:

```text
1. Start with the original logical domain D.
2. Given halo offsets, construct an extended logical domain D_h
   that contains both S and all shifted copies of S.
3. Flatten S in D_h.
4. Flatten each shifted copy of S in D_h.
5. Merge the resulting flat spans if they overlap or touch.
```

This keeps the halo operation geometric and avoids conflating logical coordinates, memory coordinates, and flat indexing.

## Range construction

For a one-offset halo, the halo range is the union of at most two flat spans in the halo-aware indexer:

```text
span 0: S flattened in D_h
span 1: shift(S, h) flattened in D_h
```

If the spans overlap or touch, they can be merged into one span. If they are disjoint, the range is represented as two spans.

For a multi-offset halo, the same idea generalizes:

```text
span 0: S
span 1: shift(S, h1)
span 2: shift(S, h2)
...
```

After flattening these spans in the halo-aware logical domain, sort and merge them into a compact span union.

Since the number of halo offsets is expected to be small, this can be represented with a small fixed-capacity span list.

```cpp
struct flat_span {
  int start;
  int stop; // inclusive
};

template <int MaxSpans>
struct span_union {
  int nspans;
  flat_span spans[MaxSpans];
};
```

## Scratch indexing

Scratch should use the same halo-aware flat index space as the halo range.

Hierarchical scratch currently allocates the whole memory-flat span covered by
the halo-extended range. This uses more storage than the compact union of touched
points, but lets flat-index scratch access use a simple base subtraction:

```text
[span_start, span_stop] -> [0, span_stop - span_start]
```

For flat-index bodies, the index passed to the body is already relative to the
current inner range's memory origin, so scratch maps it as:

```cpp
KOKKOS_INLINE_FUNCTION
int scratch_index(int idx) const {
  return idx - scratch_index_start;
}
```

For coordinate bodies, scratch first maps `(k,j,i)` into the memory-flat indexer
and then subtracts the cached memory-flat span start:

```cpp
KOKKOS_INLINE_FUNCTION
int scratch_index(int k, int j, int i) const {
  return memory_kji.GetFlatIdx(k, j, i) - scratch_flat_start;
}
```

The logical coverage size of an `InnerIndexRange` is cached when the range is
constructed. `size()` returns that cached value rather than recomputing the
merged span lengths on every call.

## Backend interpretation

The user-facing semantics stay the same:

```cpp
constexpr auto recon_halo = halo::minus_i;

auto scratch = idx_range.GetScratch<Real, recon_halo>();

inner(idx_range.AddHalo<recon_halo>(), KOKKOS_LAMBDA(auto kji) {
  scratch(kji) = reconstruct(kji);
});

inner(idx_range, KOKKOS_LAMBDA(auto kji) {
  auto dx1 = idx_range.GetOffset<X1DIR>();
  flux(kji) = riemann(scratch(kji - dx1), scratch(kji));
});
```

But different backends can implement the same logical operation differently.

### Point-wise / GPU-style backend

For a `boiv`-style point-wise loop, the base range `S` is a single logical point.

A one-offset halo gives a tiny local point set:

```text
S_halo = {p, p + h}
```

The scratch object can specialize to compact per-cell local storage, for example:

```cpp
Real scratch[2];
```

This backend may recompute neighboring reconstructions in adjacent point-wise iterations. That is intentional: it exposes more parallelism and avoids shared-memory coordination.

### Hierarchical / CPU-style backend

For a hierarchical loop, the base range `S` may contain many logical points.

The halo range covers

```text
S ∪ shift(S, h1) ∪ shift(S, h2) ∪ ...
```

and scratch is allocated over the enclosing memory-flat span for those shifted
sets. This allows reconstructed values to be reused across multiple flux
calculations while reducing per-access indexing arithmetic.

## Summary

A halo is a compile-time logical dependency annotation for an inner loop.

It answers the question:

> If the consumer loop runs over `S`, which neighboring produced values must also exist?

The core semantic rule is:

```text
AddHalo<halo_t<h1, h2, ...>>(S)
=
S ∪ shift(S, h1) ∪ shift(S, h2) ∪ ...
```

The implementation should first extend the logical domain, then flatten the base and shifted sets in that extended domain, then merge the resulting spans.

This keeps the API simple while giving the backend enough information to choose either compact per-cell scratch or reusable team scratch.
