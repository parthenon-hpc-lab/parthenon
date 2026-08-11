# Add reduction support to the loop abstraction

## Context

The loop abstraction in `src/loop_abstraction/*` currently exposes `outer()` /
`inner()` for pure `parallel_for`-style traversals but has **no reduction
support** — a gap explicitly noted in `doc/loop_abstraction/loop_abstraction.tex`
("when `par_reduce_inner` is supported need to check") and in
`kokkos_abstraction.hpp` ("par_reduce does not currently work with either
team-based patterns").

We want to compute reductions (min/max/sum/custom) over an `IndexSpace` using the
same `IndexSpace` / `InnerIndexRange` vocabulary as the existing loops. The key
usability goal: inside a single `outer_reduce` region the user can freely mix
plain `inner()` calls (e.g. fill scratch) with `inner_reduce()` calls that
contribute to the reduction — so one can compute something in scratch and then
reduce over it.

### Decisions (from the user)
1. **All three loop tags** (`bvoi`, `bovi`, `boiv`) supported.
2. **Always dispatch to the Kokkos backend** for the reduction entry points,
   regardless of `IndexSpace::backend_v`. (On CPU-only builds `DevExecSpace` is a
   host space, so the Kokkos reduce still runs. No raw reduction path is written.)
3. **Single reducer per region** — one Kokkos reducer op per `outer_reduce`;
   multiple `inner_reduce` calls join into it.
4. **New names** `outer_reduce` / `inner_reduce` alongside existing `outer`/`inner`.
5. `inner_reduce` takes one **value-type argument** in its body; `outer_reduce`
   takes a **Kokkos reducer instance** (e.g. `Kokkos::Min<double>(result)`).
6. **Reductions over halo ranges are disallowed** (`static_assert` that
   `halo_t == none_t` in `inner_reduce`).
7. **The reducer argument comes last**, consistent with Kokkos'
   `parallel_reduce(policy, functor, reducer)` ordering.
8. **The `memory` inner tag degenerates to `logical_flat` for reductions.** The
   `memory` tag sweeps whole contiguous memory spans, which include ghost cells;
   reducing over those would violate the "reductions never touch ghost/halo
   cells" rule. So in `inner_kokkos_reduce`, when `inner_tag_v == memory`, we
   traverse the *logical* cells (as `logical_flat` does) instead of the memory
   span. The body still receives a flat memory-relative index so call sites are
   unchanged.

## Design

### Public API (`src/loop_abstraction/loop_abstraction.hpp`)

```cpp
// Reducer is a bound Kokkos reducer instance, e.g. Kokkos::Min<double>(result).
// Reducer comes last, matching Kokkos::parallel_reduce(policy, functor, reducer).
template <class IndexSpaceType, class F, class Reducer>
void outer_reduce(IndexSpaceType idx_space, F &&f, Reducer reducer) {
  // Always Kokkos, per decision 2.
  impl::outer_kokkos_reduce(idx_space, std::forward<F>(f), reducer);
}

// The handle (carries the reducer + accumulator) comes before the body here,
// because the body's own trailing parameter is the reduction value.
template <class InnerIndexRangeType, class Handle, class F>
KOKKOS_FORCEINLINE_FUNCTION
void inner_reduce(const InnerIndexRangeType &idx_range, const Handle &handle, F &&f) {
  impl::inner_kokkos_reduce(idx_range, handle, std::forward<F>(f));
}
```

`outer_reduce`'s body `f` has signature `(InnerIndexRange range, int b, Handle handle)`
— same as `outer()` plus a trailing `handle`. Plain `inner(range, ...)` calls still
work unchanged inside this body (the handle is simply ignored by them).

### The reduction handle (new, in `kokkos.hpp`)

A lightweight struct threaded from `outer_reduce` into `inner_reduce`. It carries
the reducer type (so `inner_reduce` reuses the *same* join op without the user
restating it), a pointer to the per-team/per-thread accumulator `update`, and the
team member (null for `boiv`).

```cpp
template <class Reducer>
struct ReduceHandle {
  using value_type = typename Reducer::value_type;
  const device_team_member_t *member = nullptr; // null in the boiv (flat) case
  value_type *update = nullptr;                  // the enclosing reduce's accumulator
  Reducer reducer;                               // for join(); Kokkos reducers are
                                                 // trivially copyable to device
};
```

### `impl::outer_kokkos_reduce` (mirrors `outer_kokkos` in `kokkos.hpp`)

- **bvoi / bovi**: replace `Kokkos::parallel_for(policy, lambda)` with
  `Kokkos::parallel_reduce(policy, lambda, reducer)`. The team lambda gains a
  trailing `value_type &update`; build the `InnerIndexRange` exactly as today,
  construct `ReduceHandle{&member, &update, reducer}`, and call
  `f(idx_range, b, handle)`. Scratch sizing (`set_scratch_size`) is identical.
- **boiv**: replace the flat `parallel_for(RangePolicy, ...)` with
  `parallel_reduce(RangePolicy, lambda, reducer)`; lambda gains `value_type &update`,
  builds the single-point `idx_range` as today, and calls `f` with
  `ReduceHandle{nullptr, &update, reducer}`.

### `impl::inner_kokkos_reduce` (mirrors `inner_kokkos`)

Body `f` signature: `(idx, value_type &partial)` (matching the existing
index-form dispatch: `auto idx`, `Index3`, `MemoryOffset`, or `(k,j,i)` are not
supported here since the second arg is the accumulator — support the `auto idx` /
coord forms as in `inner_kokkos`, threading `partial` as the last body param).

- `static_assert(std::is_same_v<typename InnerIndexRangeType::halo_t, halo::none_t>,
   "Reductions over halo ranges are not allowed.");`
- **`memory` tag degenerates to `logical_flat`** (decision 8): since `halo_t ==
  none_t`, the region spans equal the base logical range. When `inner_tag_v ==
  memory`, iterate the logical cells (`logical_kji(idx + start)`) and hand the
  body a memory-relative flat index (`memory.GetFlatIdx(k,j,i) - mem_start`) —
  i.e. take the `logical_flat` branch of the existing `inner_kokkos` dispatch
  rather than the memory-span/chunk branch. This guarantees ghost cells are
  never visited by a reduction.
- **bvoi / bovi** (team case, `handle.member != nullptr`): for each region span,
  run a **nested** `Kokkos::parallel_reduce(TeamThreadRange(member, 0, n), body,
  Reducer(team_result))` into a team-local `value_type team_result`
  (init from `reducer` identity). This is the exact idiom already in
  `par_reduce_inner` (`kokkos_abstraction.hpp:675-729`). Then join once into the
  enclosing accumulator, guarded so only one thread writes:
  ```cpp
  Kokkos::single(Kokkos::PerTeam(*handle.member), [&]() {
    handle.reducer.join(*handle.update, team_result);
  });
  ```
  Loop over `idx_range.nregions` / the memory-chunk logic exactly as
  `inner_kokkos` does, accumulating each region's `team_result` before the join
  (or join per region — join per region is simplest and correct).
- **boiv** (flat case, `handle.member == nullptr`): a single logical point, no
  halo, no team. Just call `f(idx, *handle.update)` once (map the index the same
  way `inner_kokkos`'s boiv branch does for the identity offset `n = hrange.begin`,
  which is `{0,0,0}` given the no-halo assertion).

### Interleaving guarantee

Because the handle only *contributes* to `update` on `inner_reduce`, a user can do:
```cpp
outer_reduce(space,
  KOKKOS_LAMBDA(auto range, int b, auto handle) {
    inner(range, [&](auto idx){ scratch(idx) = compute(idx); });  // no reduction
    range.TeamBarrier();
    inner_reduce(range, handle, [&](auto idx, Real &v){            // reduces
      v = fmax(v, scratch(idx));
    });
  },
  Kokkos::Max<Real>(result));                                     // reducer last
```

## Files to modify

- **`src/loop_abstraction/kokkos.hpp`** — add `ReduceHandle`,
  `outer_kokkos_reduce`, `inner_kokkos_reduce` (bulk of the work; parallels the
  three-tag structure already there).
- **`src/loop_abstraction/loop_abstraction.hpp`** — add public `outer_reduce` /
  `inner_reduce` that always call the Kokkos impl.
- **`src/loop_abstraction/LOOP_ABSTRACTION_CONTRACTS.md`** — document the reduction
  entry points, the single-reducer / handle model, and the no-halo-reduction rule.
- **`doc/loop_abstraction/loop_abstraction.tex`** — brief section on reductions
  (and update the "when par_reduce_inner is supported" note).
- **`tst/unit/test_loop_abstraction.cpp`** — new tests (see below).

## Reuse of existing code

- Nested team reduction idiom: `par_reduce_inner` in
  `src/kokkos_abstraction.hpp:675-729` (`TeamThreadRange` + trailing reducer).
- Custom multi-value reducer example if needed for tests:
  `summable_array_t` in `src/utils/summable_array.hpp`.
- All index mapping (`GetKJI`, region spans, memory-chunk logic) reused verbatim
  from `inner_kokkos` — the reduce variants differ only in wrapping
  `parallel_for` → `parallel_reduce` + the single-writer join.

## Verification

- **Unit tests** in `tst/unit/test_loop_abstraction.cpp`:
  - For each loop tag × inner tag, run `outer_reduce` with `Kokkos::Sum<Real>`
    over a known field; assert the result equals the analytic sum (reuse the
    `EncodeValue` pattern already in the test file).
  - Repeat with `Kokkos::Min` and `Kokkos::Max`.
  - **Interleave test**: one `outer_reduce` region that first fills scratch via a
    plain `inner()`, barriers, then `inner_reduce()` reduces over the scratch —
    assert the combined result.
  - **Two `inner_reduce` in one region** joining into the same handle (e.g. sum of
    two sub-ranges) equals the full sum.
  - Confirm a `Sum` over a full space matches the corresponding `par_reduce`
    reference already used elsewhere in the test file.
  - Confirm the `memory` inner tag gives the *same* reduction result as
    `logical_flat` (i.e. ghost cells are excluded — the degeneration works).
  - (Compile-only) confirm `inner_reduce` on a halo-extended range fails the
    `static_assert` — document rather than assert in CI.
- **Build & run**: build the unit test target and run
  `ctest -R loop_abstraction` (or the existing test binary) on both a CPU (raw →
  diverted-to-Kokkos-host) and, if available, a GPU build to confirm the team
  reduction path.

---

## Deviations from the plan and final result

This section records how the implementation actually unfolded during the
LLM-assisted session, including where it diverged from the plan above. It is
meant to help a reader reconstruct *how* we arrived at the PR, not just *what*
the PR contains. (Experimental for this repo; may not persist as a convention.)

### What matched the plan

- `outer_reduce` / `inner_reduce` were added in `kokkos.hpp` with a `ReduceHandle`
  threading the reducer type + accumulator, and public entry points in
  `loop_abstraction.hpp`. Reductions always dispatch to the Kokkos backend.
- All three loop tags are supported; halo ranges are rejected with a
  `static_assert`; the `memory` inner tag degenerates to `logical_flat` under a
  reduction so ghost cells are never folded in.
- The reducer is passed **last**, matching `Kokkos::parallel_reduce`.
- Docs were added to the contracts `.md`, the `.tex`, and (added later at the
  user's request) `doc/sphinx/src/loop_abstraction.rst`.

### Deviations, decided mid-implementation

1. **`memory` → `logical_flat` degeneration + reducer-last ordering.** Not in the
   first draft plan. The user flagged during review that (a) the reducer should
   come last for Kokkos consistency, and (b) reducing over a `memory`-tag span is
   unsafe because those spans sweep ghost cells. Both were folded into the plan
   before coding. This required a small addition to `InnerIndexRange`
   (`chunk_logical_start/end`) so the reduce path can iterate logical cells for
   every inner tag.

2. **Merged bvoi/bovi inner-reduce path.** The plan implied following
   `inner_kokkos`'s separate per-tag structure. In practice the bvoi and bovi
   team cases collapse into one branch *because reductions forbid halos*
   (`nregions == 1`, no swept ghosts). The user questioned this; the merge is
   correct but now carries an explicit comment warning that adding halo support
   would break it. bvoi reduces over the whole-block logical span, bovi over one
   chunk; Kokkos combines league entries via the reducer.

3. **`ReduceHandle` moved to `types.hpp`.** Originally defined in `kokkos.hpp`.
   The user asked for it to live with the other loop vocabulary; it is
   backend-agnostic (templated on the reducer) so this was clean.

4. **Device-safety: named handle type instead of `auto`.** The initial tests used
   `KOKKOS_LAMBDA(..., auto handle)`, which compiles on host but is rejected by
   nvcc for extended `__host__ __device__` lambdas. The user caught this. First
   fix was a bare `reduce_handle_t<R>` alias; the user found it too noisy, so it
   was replaced by a `Reduction<R>` bundle exposing `reducer_t` / `value_t` /
   `handle_t`. Convention in docs/tests is `using reduce_t = Reduction<...>`.
   `reduce_handle_t` was removed to keep a single spelling.

5. **Test coverage widened to the full pattern matrix.** The first version of the
   scratch-interleave and two-`inner_reduce` tests ran only a couple of tag
   combinations, with an incorrect comment claiming scratch was team-tag-only.
   The user corrected this: scratch works for all tags (stack-allocated for
   `boiv`), and any loop-abstraction feature must hold across every allowed tag
   combination. Both tests now enumerate the full matrix. The interleave test was
   also fixed to put the variable loop *outside* the inner call, matching the
   documented per-cell-accumulator pattern.

6. **Doc emphasis on degeneration scope.** At the user's request, the docs were
   updated to stress that the `memory` → `logical_flat` degeneration applies
   **only** to `inner_reduce`; a plain `inner()` inside an `outer_reduce` region
   still sweeps memory spans (ghosts included), which is intended for producers.

### Final result

- Files changed: `src/loop_abstraction/{kokkos,loop_abstraction,types,inner_range}.hpp`,
  `src/loop_abstraction/LOOP_ABSTRACTION_CONTRACTS.md`,
  `doc/loop_abstraction/loop_abstraction.tex`, `doc/sphinx/src/loop_abstraction.rst`,
  `tst/unit/test_loop_abstraction.cpp` (8 files, ~667 insertions).
- Tests: full `[loop_abstraction]` suite passes (19 cases); `[reduction]` covers
  Sum/Min/Max over the full tag matrix, scratch interleave, and two-region joins.