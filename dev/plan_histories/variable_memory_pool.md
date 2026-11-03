<!-- This file was made in part with generative AI. -->

# Mesh-owned variable memory pool plan

## Goal

Extend the mesh-owned chunked allocation mechanism used for communication buffers to
the storage of `Variable<Real>` objects. The mechanism must serve fine and coarse
variable storage, dense and sparse variables, and block/stage copies through one
allocation path.

The optimization target is allocation churn from sparse-state changes and AMR/block
creation. It must not substantially inflate retained device-memory high-water marks,
and must preserve the existing semantics of copied Kokkos views and cached variable
packs.

## Existing relevant behavior

- `ObjectPoolMap<BufArray1D<Real>>` in `Mesh` pools same-sized, 1-D comm-buffer
  subviews. On exhaustion it allocates a contiguous chunk and adds the remaining
  subviews to the free list.
- `ObjectPoolMap` is deliberately limited to a scalar `size_t` key and 1-D views.
  It is not directly suitable for variables, which have full seven-dimensional
  shapes.
- `Variable::AllocateData()` and `Variable::AllocateCoarse()` are the central fine
  and coarse storage allocation paths. Sparse activation, ordinary dense-field
  creation, `AllocateCopy`, and the non-shallow stages of sparse variables all
  converge on them.
- `Variable::Deallocate()` only calls `Reset()` today. A Kokkos view copied into a
  pack or kernel retains the backing allocation through ordinary Kokkos reference
  counting.

## Non-goals

- Do not pool particle/swarm arrays, scratch arrays, communication buffers, or
  arbitrary application-owned Kokkos views in this change.
- Do not change the public type of `Variable::data` or `Variable::coarse_s`.
- Do not make pool operations device-side or thread-safe; allocation remains host
  infrastructure, as it is for the communication-buffer pool.

## Design

### 1. A variable-specific pool map

Factor only reusable mechanics from `ObjectPool`; introduce a dedicated
`VariablePoolMap<Real>` (or a generalized internal chunked pool used by it) rather
than expanding the public comm-buffer `ObjectPoolMap` API around variable-specific
rank and lifetime rules.

The pool stores state-free rank-7 views:

```cpp
using raw_variable_view_t = ParArrayND<Real>;
using raw_variable_owner_t = ObjectPool<raw_variable_view_t>::owner_t;
```

Its key is a `VariablePoolKey` containing all seven extents. The memory space and
layout are fixed by the `Real` instantiation, but should be part of the key if this
is later made polymorphic over element type or allocation space. `VariableState` is
not part of the pooled object: it is variable-specific and is applied when the view
is checked out.

When a pool for shape `(d1, ..., d7)` needs a chunk of `n` slots, allocate a rank-8
view with a leading slot dimension and the normal rank-7 dimensions. Put subviews
`chunk(slot, ALL(), ..., ALL())` for slots `1..n-1` on the free list and return slot
`0`. Each subview is a managed Kokkos view and therefore retains the backing chunk.

### 2. One generic storage-acquisition path

Give `Variable` a private storage-handle member for fine data and one for owned
coarse data. A handle contains a state-free rank-7 view and, when pooled, its
`raw_variable_owner_t` lease. `data` and `coarse_s` retain their current public
`ParArrayND<Real, VariableState>` type.

Allocation follows this shape:

```cpp
auto handle = AcquireVariableStorage(pmb, dims_, label());
data = ParArrayND<Real, VariableState>(handle.view.KokkosView(), MakeVariableState());
fine_storage_ = std::move(handle);
```

`AcquireVariableStorage` is a host-side acquisition-provider interface:

- a mesh-backed provider obtains a raw view/lease from the mesh variable pool;
- a direct provider constructs a normal Kokkos view when no live `MeshBlock`/`Mesh`
  allocation context exists.

This makes the body of `Variable::AllocateData()` generic. It is analogous to the
comm-buffer acquisition functor: the pool's acquisition function creates a chunk
only when its free list is empty. The direct provider is intentionally *not* modeled
as a temporary `ObjectPool`, because an `owner_t` referring to a short-lived pool
would be invalid.

The direct path keeps standalone variable uses and expired-`MeshBlock` test paths
working. It is only a small host-side branch at allocation/release time and has no
device-kernel cost.

Apply this one mechanism to all `Variable<Real>` allocations rather than creating a
sparse-only branch. Dense variables retain their leases for their block lifetime;
sparse variables release them more often; both have identical storage and aliasing
rules. `AllocateCopy()` also uses the normal fine-data acquisition path. Continue
to shallow-copy coarse storage where `CopyCoarseBuffer()` does so today.

### 3. Deferred recycling and quiescent points

Do not immediately return a pooled slot in `Variable::Deallocate()`. The
`ObjectPool` owner count cannot observe arbitrary copies of the bare `ParArrayND` or
underlying Kokkos view that occur in variable packs and submitted kernels.

Unsafe sequence:

```text
A.data is backed by pool slot S
a pack copies A.data
A is deallocated and S is put on the free list
B receives S
the old pack/kernel reads or writes A.data, now aliasing B
```

Instead, deallocation must:

1. Mark the variable unallocated and reset its public data/coarse views as today.
2. Move pooled storage handles to `Mesh::retired_variable_allocations_`.
3. Update logical per-block memory accounting as today; physical retained pool bytes
   are reported separately at mesh scope.

Add `Mesh::RecycleVariablePool()` (or an equivalently named internal operation).
It may run only at a documented quiescent point:

1. Complete the relevant task region and `Kokkos::fence()`.
2. Clear/invalidate all framework caches which can retain copied variable views:
   `MeshData` variable, sparse-pack, flux-pack, and boundary caches, plus the
   corresponding `MeshBlockData` caches.
3. Release retired leases to the pool free lists.

The contract must state that application code may not retain a raw variable/Kokkos
view across variable deallocation and a recycle point. A fence completes already
submitted kernels, but cannot discover arbitrary host-side copies of a view.

Initially call recycling only after conservative structural boundaries such as AMR
block-list replacement and known sparse-management boundaries. Do not expose an
unconditional public per-variable recycle call.

### 4. Pool lifetime and trimming

Declare the variable pool so it outlives every `MeshBlock`, `MeshBlockData`, and
retired lease. The ordering requirement is the same as the documented ordering of
comm-buffer pools and boundary comm buffers. In particular, a member declared after
`Mesh::block_list` would be destroyed before the variables it serves. Either declare
the pool before the block-list member or explicitly clear all block/GMG lists and
retired leases while the pool remains alive in `Mesh::~Mesh()`.

Only trim/reset pools after the same quiescent protocol and when no checked-out or
retired leases exist for the affected pool. The first implementation may reset all
idle pools at a block-list rebuild. Reclaiming individual partially idle chunks is a
later optimization: it requires chunk/slot accounting, because freeing a subview does
not free its parent Kokkos allocation while another slot uses that chunk.

### 5. Preallocation and bounded chunks

Before initial block construction and before new AMR block construction, make a
lightweight allocation plan keyed by shape. It counts the known active allocations:

- dense fields;
- force-allocated sparse fields;
- non-shallow stages/copies;
- owned coarse buffers when multilevel storage is required.

Seed each pool with its exact currently planned demand, not a speculative maximum
number of sparse fields. Dynamic sparse activation and unforeseen AMR demand grow
the pool in chunks.

Do **not** use the communication-buffer default of roughly one chunk per local block
for variables. Variable sizes and the number of independent shapes vary too widely.
Use a byte-bounded default:

```text
objects_per_chunk(shape) = max(1,
  min(configured_object_cap, target_chunk_bytes / bytes_per_variable(shape)))
```

For a first allocation, use `min(objects_per_chunk, known_initial_demand)` so a
small initial mesh does not allocate a full speculative chunk. This bounds the
retained slack in each shape to approximately one configured chunk (except where one
variable itself exceeds the target).

Proposed controls:

- `parthenon/mesh/variable_pool_chunk_bytes` — normal default policy;
- `parthenon/mesh/variable_pool_chunk_size` — expert object-count override;
- `parthenon/mesh/variable_pool_reset_fraction`;
- `parthenon/time/variable_pool_reset_cadence`.

Add mesh-level diagnostics for total physical pool bytes, active slots, free slots,
retired slots, allocation misses/chunks, and peak physical bytes. Keep
`MeshBlock::ReportMemUsage()` as logical active variable storage unless its existing
contract is deliberately revised; shared retained pool memory cannot be meaningfully
charged to an individual block.

## File-level implementation outline

- `src/utils/object_pool.hpp`
  - Reuse `ObjectPool` ownership mechanics where appropriate.
  - Do not weaken existing comm-buffer behavior.
  - Optionally factor a private generic chunk-allocation helper; retain a separate
    variable map/key API rather than making the existing 1-D API misleading.
- New `src/utils/variable_pool.hpp` (preferred) or an equivalent focused location
  - `VariablePoolKey`, shape hashing, rank-8 chunk acquisition, pool statistics,
    storage-handle types, and trim/reset checks.
- `src/mesh/mesh.hpp`, `src/mesh/mesh.cpp`
  - Mesh ownership, parameter registration, preallocation plan, retirement queue,
    recycle/reset, diagnostics, and destruction ordering.
- `src/interface/variable.hpp`, `src/interface/variable.cpp`
  - Private fine/coarse storage handles and generic acquisition in `AllocateData()` /
    `AllocateCoarse()`; retire rather than immediately free in `Deallocate()`.
- `src/interface/meshblock_data.hpp/.cpp`, `src/interface/mesh_data.hpp`
  - Add narrowly scoped variable-cache invalidation methods required by the recycle
    protocol. Do not indiscriminately clear unrelated user data.
- Sparse-management and AMR lifecycle call sites
  - Establish the specific safe recycle boundaries only after all task dependencies
    and caches can be invalidated correctly.
- Documentation and generated parameter tables
  - Describe pool lifetime, byte-bounded chunk behavior, physical-vs-logical memory
    accounting, and the view-lifetime contract.

## Delivery sequence

1. Implement the rank-7 pool and direct/pooled storage handle, but route all
   `Variable<Real>` fine and coarse allocation through it immediately.
2. Seed deterministic dense/force-allocated initial capacity from the allocation
   plan and add metrics. Keep recycle disabled except at destruction/block-list
   replacement.
3. Add cache invalidation plus fenced retired-lease recycling at conservative sparse
   and AMR quiescent points.
4. Add optional idle-pool reset cadence. Consider per-chunk reclamation only if
   profiling shows retained slack matters after byte-bounded chunking.

This is an operational rollout, not a sparse-vs-dense implementation split: one
generic allocation path is less complex and safer than maintaining separate storage
semantics for what are otherwise the same `Variable` objects.

## Validation

- Unit-test rank-8 chunk subdivision: exact dimensions, independent
  `VariableState`, reuse only after a recycle, byte/count statistics, and direct
  fallback behavior.
- Test fine and coarse storage for dense variables, sparse variables, all relevant
  topologies, and multi-stage shallow/non-shallow copies.
- Test a retained pack/view across `Deallocate()` and verify its slot is not reused
  until the explicit fenced cache-clear recycle point.
- Run sparse activation/deactivation and controller-group tests across multiple
  stages.
- Run AMR/load-balancing regression coverage through repeated refine/derefine cycles;
  verify memory does not grow beyond planned active demand plus bounded chunk slack.
- Run sanitizer/debug builds where available, normal CPU tests, and representative
  device-backend tests. Validate numerical results and pool telemetry, not only lack
  of allocation failures.
