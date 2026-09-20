# Plan: load RIOT analysis state from a `.phdf` file

## Goal and command-line contract

Support

```sh
riot -a dump.phdf -i analysis.in
```

as an analysis-only start in which RIOT:

1. builds the mesh described by `dump.phdf`;
2. constructs packages and registered fields from `analysis.in`;
3. loads every compatible registered field present in the dump, independent of its `Independent`, `Restart`, or `FillGhost` metadata; and
4. fills ghost zones for every loaded field before `UserWorkBeforeLoop` and analysis outputs run.

The driver must not enter the evolution loop. Existing invocations retain their current meaning: `-i` starts a new problem, `-r` performs a restart, and `-a` on a restart file uses the existing analysis-restart behavior. A `.phdf` analysis load requires `-i`, because ordinary dumps need not contain a complete archived input deck.

## Implementation

### 1. Introduce an explicit analysis-load policy in Parthenon

In argument parsing/`ParthenonManager`, distinguish the existing restart policy from a new `analysis_data` policy when `-a` names an HDF5 data dump. Do not infer this merely from the filename suffix: inspect the HDF5 metadata that identifies the output type, with the suffix used only for a clear early diagnostic if necessary.

Keep the existing initialization order—parse `analysis.in`, initialize RIOT packages, and construct the mesh from the file—but pass the load policy into the HDF5 reader and package-loading path. Require an input file for `analysis_data`; leave restart input merging unchanged.

### 2. Generalize field loading without changing restart selection

Refactor the current restart-package loader into shared mechanics plus a selection policy:

- `restart`: retain the current `Independent || Restart` selection and all existing validation.
- `analysis_data`: enumerate field datasets in the file and load the intersection with fields registered by the packages created from `analysis.in`, regardless of field metadata.

For each selected field, validate centering, component dimensions, scalar type/precision conversion, block dimensions, and sparse identity before reading. Recreate sparse allocation from `SparseInfo` before copying data. Read interior data into the base `MeshData`; even when the file contains output ghost zones, regenerate ghosts to make the result independent of the writer's rank layout and stale neighboring values.

Fail with an actionable error for a same-named but incompatible registered field. Warn once on rank 0 for datasets that have no registered RIOT field, and for registered fields absent from the dump; do not dynamically invent fields because the input/package metadata supplies their semantics. Initially support every field topology already supported symmetrically by the HDF5 writer and reader; explicitly reject unsupported topologies rather than silently skipping them.

### 3. Add a one-time, explicit-field ghost exchange

Extend Parthenon's boundary-exchange API to accept an explicit list of loaded field UIDs (or an equivalent selector) that overrides only the usual `FillGhost` filtering for that call. The override must:

- allocate/use boundary buffers for the selected dense and allocated sparse fields;
- perform same-level exchange and AMR prolongation/restriction as applicable;
- apply the configured physical boundary conditions; and
- leave each field's persistent metadata unchanged.

After loading all interiors, run this exchange to completion across the base mesh before any RIOT analysis callback. Do not call RIOT's derived-fill hooks during this step: fields read from the dump, including derived fields, are authoritative and must not be recomputed or overwritten.

Document the boundary rule: internal and coarse/fine ghosts are always regenerated; physical ghosts use the boundary conditions in `analysis.in`. A user boundary callback must therefore handle any analysis-loaded field for which it requests physical ghost values.

### 4. Wire RIOT to the generalized path

RIOT's main program should continue to use Parthenon's normal `-a` control flow. Add only the RIOT-side callback or small driver hook needed to request the post-load explicit-field exchange, if this cannot live entirely in `ParthenonManager`. Record the loaded UID set during the read so RIOT does not guess from metadata afterward.

Ensure the sequence is:

```text
parse input -> initialize packages -> build file mesh -> allocate sparse fields
-> load field interiors -> fill ghosts for loaded UIDs
-> UserWorkBeforeLoop -> analysis_output blocks -> exit
```

### 5. Compatibility and diagnostics

- Make the new behavior conditional on `-a` plus a data-output file; add no new required input parameters.
- Do not change restart output contents, restart field selection, normal `FillGhost` selection, problem generation, or time evolution.
- Take time/cycle and mesh topology from the dump for analysis output metadata; take package configuration, boundary conditions, and requested analysis outputs from `analysis.in`.
- Validate coordinate system, dimensionality, block shape, and ghost width early. Report file and input values in mismatch errors.
- Print a concise rank-0 summary listing loaded, missing, ignored, and ghost-filled fields.

## Tests and acceptance criteria

Add a regression fixture that writes a multiblock `.phdf` containing: an independent `FillGhost` field, a derived field without `FillGhost`, a `Restart`-only field, and an allocated/deallocated sparse field. Analyze it with a separate input file and verify:

- all compatible file fields have exact interior values;
- all loaded fields have correct same-level, periodic/physical, and AMR coarse/fine ghost values;
- sparse allocation matches the file;
- `UserWorkBeforeLoop` observes the completed ghost fill;
- analysis outputs preserve the dump's time/cycle; and
- loading works with a different MPI rank count from the writer.

Retain regression coverage for ordinary `-i`, `-r`, and `-a` on restart files. Add negative tests for missing `-i`, incompatible field shapes, incompatible mesh geometry, and unsupported field topology. The feature is complete when these tests pass without any metadata changes to the fields used by a normal RIOT evolution run.
