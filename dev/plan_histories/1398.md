<!-- This file was made in part with generative AI. -->

## Goal

Generalize sparse identifiers so Parthenon can represent more than a single integer
index per sparse field, while keeping the existing scalar behavior and restart behavior
unchanged for current codes. At the same time, introduce a cleaner control-group model
for sparse fields so allocation and deallocation can operate on groups of controlling
fields rather than only a single controller label.

## Functional Plan

- Introduce a dedicated `SparseID` value type and thread it through the sparse field
  model instead of storing raw integers directly in the core interfaces.
- Keep the sparse label format backward compatible for current scalar IDs so existing
  sparse names continue to look the same.
- Add support for multi-component sparse IDs in a way that remains device-visible and
  inexpensive enough to be used in hot code paths.
- Mark the sparse-ID comparison operators as device-available so pack selection and
  other device code can reason about sparse ordering without special handling.
- Keep the host-facing sparse APIs compatible with existing scalar call sites by
  allowing integer inputs to be converted to `SparseID` at the API boundary.
- Update sparse pools so they store and operate on typed sparse IDs internally rather
  than treating sparse IDs as anonymous integers.
- Add a sparse-ID projection rule for sparse pools so a pool can group fields by a
  controller key derived from the sparse index space.
- Extend `StateDescriptor` so it resolves sparse-pool control groups into the same
  controller-reverse-map machinery used by ordinary fields.
- Treat control groups as the source of truth for grouped sparse control, while keeping
  the legacy single-controller accessor available as a compatibility shim.
- Keep per-field sparse deallocation counters intact, but evaluate deallocation at the
  group level by requiring every control member to be ready before the group is removed.
- Make sparse deallocation ignore dense-only controller groups so dense fields do not
  enter the sparse dealloc path.
- Preserve restart exactness by keeping the sparse deallocation metadata and sparse
  allocation behavior compatible with existing restart files.
- Keep the existing sparse-advection example and restart flow working as the primary
  exactness check while the sparse-control machinery changes underneath it.
- Add focused unit coverage for the new sparse-ID type and for grouped sparse control.
- Update sparse documentation so the split between sparse IDs, sparse pools, and
  controlling groups is described clearly.

## Scope Boundaries

- Do not change the public downstream package API unless it is required for backward
  compatibility with existing scalar sparse call sites.
- Keep dense fields and sparse fields conceptually separate; only unify their internal
  registration and control bookkeeping where that reduces duplication.
- Do not change the numerical update kernels in the example problems unless a restart or
  sparse-control bug forces a targeted fix.
- Preserve the current sparse-advection restart gold output as the exactness oracle for
  this PR.

