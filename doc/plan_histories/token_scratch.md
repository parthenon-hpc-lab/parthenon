# Token-Based Scratch Memory System

## Overview

The token scratch system provides efficient, thread-safe scratch memory management for Kokkos parallel kernels using `Kokkos::Experimental::UniqueToken`. Each parallel thread acquires a unique token ID and gets access to a pre-allocated scratch buffer that can be carved into multiple typed views.

## Key Features

- **Thread-safe**: Uses Kokkos UniqueToken to avoid race conditions
- **Type-safe**: Automatic type deduction and alignment handling
- **Zero runtime overhead**: All inline functions, compile-time resolution
- **Flexible allocation**: Supports 1D, 2D, and 3D views of arbitrary types
- **RAII-based**: Automatic token acquisition and release
- **Debug validation**: Bounds checking in debug builds

## Components

### 1. `TokenScratchPool<MemorySpace, ExecutionSpace, TokenScope>`

The main pool manager that allocates scratch memory for all tokens.

**Template Parameters:**
- `MemorySpace` - Kokkos memory space (e.g., `HostSpace`, device memory space)
- `ExecutionSpace` - Kokkos execution space
- `TokenScope` (optional) - `UniqueTokenScope::Global` (default) or `UniqueTokenScope::Instance`

**Constructor:**
```cpp
explicit TokenScratchPool(size_t bytes_per_token)
```

**Methods:**
- `acquire()` - Returns a `ScratchAllocator` for the current thread
- `size()` - Number of tokens in the pool
- `bytes_per_token()` - Bytes allocated per token
- `total_bytes()` - Total pool size

### 2. `ScratchAllocator<MemorySpace, ExecutionSpace>`

Per-token allocator that carves typed views from the token's buffer.

**Methods:**
- `allocate_view<T>(n)` - Allocate 1D view with n elements
- `allocate_view<T>(n1, n2)` - Allocate 2D view
- `allocate_view<T>(n1, n2, n3)` - Allocate 3D view
- `remaining()` - Get remaining capacity
- `reset()` - Reset allocator to beginning (for reuse)

### 3. `TypedScratchBundle<MemorySpace, ExecutionSpace, ArraySpecs...>`

(Optional) Pre-configured bundle for compile-time known layouts.

## Usage Examples

### Basic Usage

```cpp
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

// Create pool with 64KB per token
TokenScratchPool<MemSpace, ExecSpace> pool(64 * 1024);

Kokkos::parallel_for("my_kernel", N, KOKKOS_LAMBDA(int i) {
  // Acquire scratch allocator
  auto scratch = pool.acquire();
  
  // Allocate multiple typed views
  auto doubles = scratch.template allocate_view<double>(100);
  auto ints = scratch.template allocate_view<int>(50);
  auto flags = scratch.template allocate_view<bool>(200);
  
  // Use the views...
  for (int j = 0; j < 100; ++j) {
    doubles(j) = compute(i, j);
  }
  
  // Token automatically released when scratch goes out of scope
});
```

### Multi-dimensional Views

```cpp
TokenScratchPool<MemSpace, ExecSpace> pool(128 * 1024);

Kokkos::parallel_for("multidim", N, KOKKOS_LAMBDA(int i) {
  auto scratch = pool.acquire();
  
  // 2D and 3D scratch arrays
  auto work_2d = scratch.template allocate_view<double>(ni, nj);
  auto work_3d = scratch.template allocate_view<double>(ni, nj, nk);
  
  // Access as: work_2d(i, j), work_3d(i, j, k)
});
```

## Design Considerations

### Memory Layout

The pool allocates a 2D view: `View<char**>` with shape `[num_tokens, bytes_per_token]`.

- Each row is owned by one token
- Allocations within a token are sequential with proper alignment
- No fragmentation within a token's buffer

### Token Scope

Two scope options (specified as a template parameter):

1. **`UniqueTokenScope::Global`** (default)
   - Tokens shared across all kernel invocations
   - More token reuse, potentially fewer tokens needed
   - May see contention under high concurrency
   ```cpp
   // Explicit Global scope (or omit for default)
   TokenScratchPool<MemSpace, ExecSpace, 
                    Kokkos::Experimental::UniqueTokenScope::Global> 
       pool(scratch_bytes);
   ```

2. **`UniqueTokenScope::Instance`**
   - Tokens private to each kernel instance
   - Better for very high thread counts
   - May allocate more tokens
   ```cpp
   // Use Instance scope for high concurrency
   TokenScratchPool<MemSpace, ExecSpace,
                    Kokkos::Experimental::UniqueTokenScope::Instance>
       pool(scratch_bytes);
   ```

### Sizing Guidelines

Choose `bytes_per_token` based on:

1. **Per-thread scratch needs**: Sum of all allocations in your kernel
2. **Alignment overhead**: Add ~20% for alignment padding
3. **Safety margin**: Add 10-20% buffer for debug assertions

Example calculation:
```cpp
// Kernel needs:
// - 100 doubles = 800 bytes
// - 50 ints = 200 bytes  
// - 200 bools = 200 bytes
// Total: 1200 bytes
// With alignment (20%): 1440 bytes
// With safety margin (20%): 1728 bytes → round to 2KB
constexpr size_t scratch_bytes = 2 * 1024;
```

### Performance Notes

- **Zero overhead in release builds**: All functions inline, no virtual dispatch
- **Debug validation**: Bounds checking only in debug builds (`#ifndef NDEBUG`)
- **No dynamic allocation**: All memory pre-allocated at pool creation
- **Cache-friendly**: Each token's buffer is contiguous

## Integration with the rest of Parthenon

```cpp
// In package initialization or task setup
namespace Package {

void MyTask(MeshData<Real>* md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto &ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto &jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto &kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  
  // Size scratch based on block dimensions
  const size_t ni = ib.e - ib.s + 1;
  const size_t nj = jb.e - jb.s + 1;
  const size_t nk = kb.e - kb.s + 1;
  
  const size_t scratch_per_k = (ni * nj) * sizeof(Real) * n_vars;
  const size_t total_scratch = scratch_per_k * 1.3; // 30% padding
  
  TokenScratchPool<DeviceSpace, ExecSpace> pool(total_scratch);
  
  // Use in block loop...
}

} // namespace Package
```

## Error Handling

- **Overflow detection**: In debug builds, `Kokkos::abort()` if allocation exceeds capacity
- **Release builds**: No checks for performance; ensure proper sizing
- **Diagnostic methods**: Use `remaining()` and `current_offset()` to debug sizing

## Thread Safety

- **Acquire/release**: Managed by Kokkos UniqueToken (thread-safe)
- **Within token**: No synchronization needed - each thread owns its token
- **Across tokens**: No shared state between tokens

## Future Extensions

Possible enhancements:

1. **Named allocations**: Track allocation names for debugging
2. **High-water mark tracking**: Monitor maximum usage per token
3. **Hierarchical scratch**: Separate team-shared and thread-private levels
4. **Template bundle interface**: Compile-time view configuration

## References

- [Kokkos UniqueToken Documentation](https://kokkos.org/kokkos-core-wiki/API/core/UniqueToken.html)
- Parthenon scratch memory patterns
