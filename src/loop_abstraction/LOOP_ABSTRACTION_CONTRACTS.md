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

## Reductions

`outer_reduce` / `inner_reduce` fold a single Kokkos reducer over an `IndexSpace`,
mirroring `outer` / `inner`. They are **Kokkos-only**: they always dispatch to the
Kokkos backend regardless of `IndexSpace::backend_v` (on a host-only build
`DevExecSpace` is a host space, so the Kokkos reduce still runs correctly). There is
no raw reduction path.

The reducer is baked into the index-space type. Build a reduction space with
`ReductionIndexSpace<lt, it, R>` (which hides the backend parameter) or by rebinding an
existing space with `idx_space.WithReducer<R>()`. Its `idx_range_t` then carries the
reduction, so the body needs no reduction handle -- it is `(idx_range, int b)`, exactly
like `outer`. The preferred `outer_reduce` overload constructs the reducer over a fresh
result and returns it (the result is a host scalar, so the Kokkos reduce is synchronous
and the value is valid on return -- no fence):

```cpp
using rist = ReductionIndexSpace<loop_tag::bovi, inner_tag::logical_flat, Kokkos::Sum<Real>>;
rist idx_space(/* ... */);

auto result = outer_reduce(idx_space,
  // Body param types must be named, not `auto`: KOKKOS_LAMBDA is an extended
  // __host__ __device__ lambda and nvcc forbids `auto` params. Name the range with
  // rist::idx_range_t (the inner_reduce body lambdas are ordinary lambdas, so `auto` is
  // fine there).
  KOKKOS_LAMBDA(const rist::idx_range_t &range, int b) {
    // Plain inner() calls still work here -- e.g. fill scratch.
    inner(range, [&](auto idx) { scratch(idx) = compute(idx); });
    range.TeamBarrier();
    // inner_reduce contributes to the reduction. Body takes a trailing value ref.
    inner_reduce(range, [&](auto idx, auto &v) { v += scratch(idx); });
  });
```

An escape-hatch overload takes a caller-constructed reducer instance last (matching
`Kokkos::parallel_reduce(policy, functor, reducer)`) for reducing into a `View`,
`ScatterView`, or device memory; it returns void. Its reducer type must match the
space's `reduction_t`.

Rules:

1. **Single reducer per region.** One reducer op per `outer_reduce` (the space's
   `reduction_t`). Multiple `inner_reduce` calls in the same region all join into the
   same accumulator (the reducer type comes from the index space, so the join op is not
   restated), so a region may freely interleave plain `inner` (no reduction) and
   `inner_reduce` calls.
2. **Body signature.** `inner_reduce`'s body takes the usual index form plus a trailing
   reduction-value reference: `[](auto idx, auto &v)`, `[](Index3 idx, Real &v)`, or
   `[](int k, int j, int i, Real &v)`.
3. **No reductions over halo ranges.** Reductions must never touch ghost/halo cells.
   `inner_reduce` `static_assert`s that the range's halo is `none_t`; extend a range
   only for producer (scratch) `inner` loops and reduce over the base range.
4. **`memory` degenerates to `logical_flat` — but only for `inner_reduce`.** For a
   reduction the `memory` inner tag iterates logical cells (not a contiguous memory
   span), so no swept ghost cell is ever folded in. The body still receives a
   memory-relative flat index, so call sites are identical to `logical_flat`. This
   degeneration is scoped strictly to `inner_reduce`: a plain `inner()` call inside an
   `outer_reduce` region behaves exactly as it does under `outer()` and does **not**
   degenerate — with the `memory` tag it still sweeps whole contiguous memory spans
   (ghost cells included). So mixing a `memory`-tag `inner()` producer (which may write
   ghosts) with an `inner_reduce()` consumer (which will not read them) is fine and
   intended; just don't assume the producer stayed inside the logical set.
5. **Custom reducers must not read their bound target in `join`/`init`.** The reducer
   instance is copied by value into the device kernel, carrying the reference it was
   bound to. The implementation only ever calls `join(a, b)` and `init(a)` on it,
   neither of which dereferences the bound target, so all built-in Kokkos reducers
   (`Sum`, `Min`, `Max`, `MinLoc`, ...) are safe. A custom reducer whose `join`/`init`
   read the bound target would dereference a host pointer on the device — don't write
   one. (`value_type` must also be device-copyable, as for any Kokkos reducer.) The
   returning `outer_reduce` overload requires `reduction_t` to be constructible from a
   `value_t&` and `value_t` to be default-constructible; use the instance-bound overload
   for a reducer that needs anything else.

Note on the result: the returning overload discards its fresh result's initial value --
Kokkos initializes the accumulator to the reducer's identity, not a seed. `value_t` is
`IndexSpace::value_t` (the reducer's `value_type`).

## Current Backend Requirement

The raw and Kokkos implementations are expected to satisfy the same indexing and
coverage contract for a given `IndexSpace`. Backend choice may change execution
order and which invocations run concurrently; user kernels must not depend on either.

### Raw backend and OpenMP

The raw backend is still selected as `loop_backend::raw`; OpenMP is not a separate
loop backend. Its loop nests contain OpenMP directives that become active when the
build sets `PARTHENON_ENABLE_RAW_OPENMP=ON` and otherwise leave an ordinary serial
host loop (apart from any compiler handling of the SIMD directives).

The current raw outer-loop decomposition is:

| Loop tag | OpenMP work-sharing in `outer(...)` |
| --- | --- |
| `bvoi` | Parallelize the block loop. |
| `bovi` | Parallelize the collapsed `(block, outer chunk)` loops. This exposes parallel work even for a one-block run when it has multiple chunks. |
| `boiv` | Keep the block loop outside the parallel region; for each block, parallelize the collapsed `(k, j)` loops and mark the `i` loop SIMD. |

Raw inner loops use `omp simd` on their contiguous innermost loops where applicable.
This is deliberately moderate OpenMP support rather than a general nested-parallel
execution model. In particular:

- An `outer(...)` body may execute concurrently for different ranges, and no ordering
  between those invocations is guaranteed.
- Each invocation must observe its own correct `InnerIndexRange` state. Backend
  implementation must not introduce races by sharing mutable current-point/range state.
- The user's writes must be disjoint across outer ranges or use appropriate
  synchronization. Captured host state is not made thread-safe by the abstraction.
- Do not assume that raw `outer(...)` composes safely or efficiently inside another
  OpenMP parallel region, a separate host-threaded region, or a Kokkos parallel region.
  Nested use requires deliberate coordination of the threading models.
- Completion of `outer(...)` remains a synchronization point for the work it launches.

For tests, the safest reference is:

- a plain host nested loop over `(b, v, k, j, i)` for logical-cell correctness
- direct raw-vs-Kokkos comparison for backend parity
- raw-backend coverage and race-sensitive checks with more than one OpenMP thread

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

### Single-variable views and component offsets

`make_var_view(inner_range, pack, var)` resolves `var` to one absolute variable
index in the pack for the current block. `var` may be a typed index or a raw integral
pack index. The returned view supports the ordinary single-variable forms:

```cpp
pv(kji);
pv(k, j, i);
```

It also supports a relative packed-variable offset as the first argument:

```cpp
auto pv = make_var_view(idx_range, pack, my_vec_var());

inner(idx_range, [&](auto kji) {
  pv(component, kji) = value;
});
```

The meaning is:

```text
pv(component, point) == pack(block, base_variable_index + component, point)
```

This makes a view rooted at the first component of a vector/tensor-type variable
usable for all of that variable's components without constructing a separate view
for each component. The point argument follows the selected inner-tag contract:

- flat and memory paths accept the loop's flat `int`/`MemoryOffset` index;
- `logical_coords` accepts `Index3` or explicit `(k, j, i)` coordinates, with the
  component offset prepended.

The first argument is a pack-variable offset, not a logical-space
offset. It may select a vector/tensor component, which are represented
as consecutive variable entries in the pack.

The caller must keep the offset within the consecutive pack entries represented by the
variable family; there is no bounds check. Those entries must have the same topology
and memory layout. Flat/memory views cache the stride between consecutive pack entries,
while `logical_coords` views forward `base_variable_index + offset` to the pack, but
both forms must have the semantics above.

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
