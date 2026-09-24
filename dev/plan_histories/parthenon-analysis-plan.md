# Plan: native dump analysis in Parthenon

## Objective and starting point

Implement an analysis-only startup from **Parthenon `develop`**, with none of PR #1454 assumed present:

```sh
application -a snapshot.phdf -i analysis.in
```

Reconstruct the dumped mesh, import selected registered mesh variables and swarms, prepare valid boundary data, run analysis callbacks and outputs, and exit without evolution. Use an analysis-specific state descriptor so ordinary allocation, packing, and communication retain their normal meaning.

This plan incorporates the goals discussed in PR #1454 but replaces its runtime metadata overrides. The supplied diff was reviewed; the current `develop` source was not available when this plan was written. All API names introduced below are conceptual. Begin by inspecting the actual checkout and adapt names and integration points accordingly.

## 1. Required behavior

### Command line and input precedence

- Select the new pathway when `-a` names an ordinary dump and `-i` supplies an input file. Require `-i` for this pathway.
- Preserve existing new-problem `-i`, restart `-r`, and restart-file `-a` behavior. Identify dump kind from file metadata where available, not solely from the suffix. Define a tested legacy-file fallback using available structure and format information; diagnose ambiguous files rather than guessing silently. Reject slices or incomplete geometry that cannot reconstruct a mesh.
- Merge inputs in this order: archived dump input, if present; `analysis.in`; explicit command-line parameter overrides. Later values replace earlier values for the same block/key. An archived input deck must not be required for ordinary dump analysis.
- Apply this merge before package construction and before reading initialization-sensitive settings. Do not restore serialized package parameters afterward in a way that overwrites this merged configuration.
- The file supplies mesh topology, geometry, time, and cycle. Input overrides configure packages, boundary conditions, analysis, and outputs. Reject conflicting geometry requests with file/input values in the error; changing a mesh parameter is not implicit regridding. Distinguish explicitly overridden geometry from inherited archived values.
- A different destination ghost width is allowed when supported by the operators and block dimensions. File ghost width controls source slicing only; it must not dictate the newly allocated ghost width.
- Suppress evolution and automatic writes from inherited ordinary output schedules. Execute only outputs selected for analysis by the established analysis-output contract. Document how archived and overridden output blocks are treated; key overlay does not delete old blocks.

### State selection

Resolve application packages from the merged input into an immutable source catalog. For mesh fields, select:

```text
((registered fields intersect file fields) union analysis-requested fields)
    minus explicitly excluded fields
```

Introduce `Metadata::Analysis` as an optional allocation request if no suitable existing mechanism exists. It has no effect on evolution, restart selection, or ordinary output selection. Add `<parthenon/analysis>/exclude_fields`, default empty; exclusions take precedence. Use full variable labels, including sparse IDs. Diagnose unknown exclusion names deterministically.

Apply an analogous, explicitly documented selection policy to registered swarms and their registered particle variables. Use qualified identifiers and separate swarm/particle exclusion parameters to avoid collisions with mesh field names. Selecting a particle variable selects its swarm. Required position and ownership information are dependencies: reject an exclusion that makes the selected swarm uninterpretable rather than inventing data or silently undoing the exclusion.

Do not allocate unrelated simulation state or automatically expand flux/controller groups. Retain unselected registrations in the source catalog, not in the analysis mesh's operational descriptor. Report unregistered file datasets and omitted registrations. Do not invent physical semantics for unregistered datasets; generic schema-only import is a separate feature.

Imported values are authoritative. Analysis-requested variables absent from the file require an explicit analysis initialization callback. Missing mandatory inputs to that callback are errors. No selected variable may reach analysis consumers with undefined values.

### Meaning of complete support

Support every variable category and scalar type supported by the framework, every mesh topological centering, and every **legal** metadata combination. Invalid metadata remains invalid. No category may be silently skipped to claim completion.

| Category | Required behavior |
| --- | --- |
| Cell-centered fields | Load interiors; fill same-level, periodic, physical, and AMR ghosts |
| Face-, edge-, and node-centered fields | Load all orientation components and native extents; fill topology-appropriate ghosts and shared entities |
| Generated flux fields and `WithFluxes` parents | Import independently when selected; provide spatial ghost filling for the sampled flux field without invoking evolution flux correction |
| Dense and sparse fields | Preserve imported interior values and sparse provenance; supply valid boundary data wherever the analysis contract exposes field values |
| Scalars, vectors, tensors, multi-component arrays | Preserve shape, labels, orientation, and component behavior at boundaries |
| Non-spatial (`None`) variables | Restore the framework-defined block-local or other applicable payload; mesh ghosts are not applicable |
| Alternate-resolution fields such as `Fine`, and GMG-related fields | Preserve their actual storage layout; support boundary operations on the grid where they live, with explicit handling of hierarchy dependencies |
| Swarms and particle variables | Restore active particles and all selected attributes, redistribute ownership, and complete swarm boundary handling |

Mesh ghost validity applies to all represented ghost indices, including edges/corners and topology-specific extents, not allocation padding. Swarm attributes and non-spatial arrays do not acquire mesh ghost cells. If a swarm supports ghost particles, complete its native ghost-particle protocol and all associated attributes. Otherwise establish valid particle ownership and boundaries without fabricating ghost copies.

## 2. Audit `develop` before coding

Read repository instructions and record the starting commit. Trace argument parsing, input merge order, package resolution, mesh reconstruction, mesh/block initialization, reader/writer symmetry, sparse control groups, boundary communication, refinement registration, swarm restart, and driver analysis callbacks.

Create a support matrix from the actual built-in metadata list, variable types, topological elements, and storage grids. For each category identify writer schema, reader support, allocation rules, boundary operator, and regression fixture. Enumerate legal flag interactions rather than assuming that adding `FillGhost` makes every existing variable communicable.

Inspect especially:

- `src/parthenon_manager.*`, argument parsing, drivers, and application callbacks;
- `src/interface/` metadata, descriptors, variables, swarms, and data containers;
- `src/mesh/`, `src/bvals/`, refinement operators, and communication setup;
- `src/outputs/` readers, writers, field layouts, and swarm serialization;
- sparse packs, boundary pack caches, typed particle packs, and MPI partitioning.

If `develop` lacks writer or communication support for a required category, implementing that support is part of this work. Runtime rejection is appropriate for an ambiguous or incomplete file; rejecting an entire valid category is not completion of this plan.

## 3. Build an analysis descriptor before allocation

Introduce a small analysis module containing an immutable selection/import plan and a descriptor builder. Keep file-specific selection lists and diagnostics out of `Mesh` where possible.

Maintain two distinct objects:

1. **Source catalog:** original resolved registrations, physical semantics, custom operators, package provenance, parameters, and callback information.
2. **Analysis descriptor:** exactly the state and operational relationships required by this analysis mesh.

Construct fresh validated metadata for analysis variables before registering them or allocating storage. Preserve field identity, shape, topology, scalar type, sparse ID, user flags, and physical properties relevant to interpolation or boundaries. Record the source metadata separately for callbacks that explicitly need it. Do not mutate the source descriptor or alias mutable metadata between descriptors.

Define and test the metadata transformation explicitly:

- Imported spatial fields request ordinary `FillGhost` communication in the analysis descriptor. They need not become `Independent` or `Restart`.
- Supply valid restriction/prolongation operators at descriptor construction. Preserve custom operators. If metadata currently discards operators on fields not selected for communication, preserve that capability in the registration/catalog representation; do not force all fields into refinement operation maps.
- Remove evolution-only parent/flux allocation relationships. A selected parent does not implicitly instantiate a flux; selected flux datasets are independently represented spatial samples with their centering and physical semantics preserved. Their analysis instances must not enter refluxing/flux-correction communication merely because the source variables were fluxes.
- Preserve identities and sparse nature, but construct independent analysis allocation control where source controller groups would allocate unrelated variables. Rebuild controller maps consistently; do not leave dangling links.
- Audit `Fine`, GMG, `OneCopy`, allocation flags, `ForceRemeshComm`, topology flags, and all other built-ins. Preserve layout semantics. Reinterpret or omit evolution scheduling roles only through this documented transformation. Do not indiscriminately strip unfamiliar flags.
- Build pack descriptors and cached selectors only after the analysis descriptor exists. Preserve labels and compatible UID identity where required by application access, and test typed and runtime packs. Callbacks requiring source-flag queries must use the source catalog explicitly.

The mesh's `resolved_packages` must describe its actual operational state. Expose the source catalog through a distinct read-only accessor if needed. Preserve application package parameter access and analysis callback registration without allowing a second inconsistent field registry to drive allocation or packing.

Avoid runtime `BoundaryCommunicationOverride`, `force_refinement`, forced coarse allocation, and analysis booleans propagated through ordinary boundary loops. Missing topology operations should be implemented as general capabilities usable by normal fields with matching metadata.

## 4. Share I/O mechanics; separate restart and analysis policy

Extract reusable file inspection, mesh reconstruction, typed field reads, sparse-mask reads, and swarm reads from restart code as needed. Keep restart validation and selection unchanged. Prefer separate analysis orchestration over `RestartPackages(..., bool analysis)`.

Build a file manifest describing topology, coordinates, block locations, output mode, time/cycle, field types/shapes/orientations, sparse allocation, and swarm counts/offsets/attributes. Add versioned writer metadata where existing files cannot describe required distinctions. New metadata must be backward compatible for ordinary readers; legacy imports may use unambiguous application registrations, but must reject ambiguous layouts.

For each selected variable:

1. Validate topology, scalar type, dimensions, sparse identity, block count, and applicable grid location against the source registration and analysis representation.
2. Read the source interior using source layout, ghost width, orientation extents, and padding rules. Map it into the destination interior using destination layout. Do not calculate source offsets from destination ghost bounds.
3. Preserve integer and other supported non-real payload types using typed buffers; never route identifiers through floating point. Define supported precision conversions, overflow checks, and diagnostics.
4. Restore source sparse allocation and payloads; initialize any additional analysis support storage according to the sparse policy below.
5. Record successful imports and initialized regions. Complete device transfers before dependent tasks run.

For face/edge/node arrays, distinguish meaningful staggered entities from file padding. Do not assume dataset shape alone uniquely proves centering. Preserve per-block non-spatial arrays without applying spatial hyperslab logic.

## 5. Separate structural initialization from data initialization

Factor reusable mesh infrastructure setup from problem generation, derived filling, remeshing, and evolution initialization. Preserve the existing normal-start and restart sequence through their existing orchestration.

Analysis must execute this dependency order:

```text
inspect file and merge input
-> construct source catalog and validate import plan
-> construct analysis descriptor
-> reconstruct mesh and initialize required infrastructure
-> restore field interiors and swarm payloads
-> initialize explicitly requested missing analysis state
-> prepare sparse boundary support and swarm ownership
-> run complete mesh and swarm boundary preparation
-> establish analysis-ready state
-> UserWorkBeforeLoop and selected analysis outputs
-> finalize and exit
```

An analysis initialization hook may initialize only explicitly designated missing state. Declare dependencies and order initialization/boundary tasks accordingly; dependencies requiring imported ghosts may need an earlier boundary pass. Detect dependency cycles. Final readiness requires all selected initialized spatial fields to have valid ghosts.

Do not call problem generators, evolution derived-fill hooks, or adaptive regridding to repair analysis state. Preserve imported interiors through initialization and ghost filling. Audit before-output and user hooks too: they must not silently recompute authoritative imported fields. Provide explicit analysis callbacks and document which existing callbacks remain active.

## 6. Complete mesh ghost preparation

Use ordinary boundary setup and tasks on the analysis descriptor. Audit communicator creation, tag maps, coarse buffers, refinement subsets, and coalesced communication for every selected topology and type.

Implement or extend general topology-aware operations where needed:

- Same-level exchange and periodic mapping for all orientation components.
- Coarse-to-fine prolongation and fine-to-coarse restriction with the correct centering, staggering, geometry, and intensive/extensive interpretation.
- Shared-entity synchronization for faces, edges, and nodes, including coarse/fine interface ownership. Define the authoritative owner. Check redundant file copies against the documented tolerance before synchronization; fail on inconsistency rather than silently overwriting conflicting imported interior values.
- Edge/corner completion and physical BC ordering sufficient for refinement stencils. A face-only exchange is not sufficient evidence of full ghost validity.
- `Fine` and other alternate-grid layouts using their own bounds and neighbor mappings. Reconstruct hierarchy-only state where explicitly defined by registered operators; do not treat GMG flags as ordinary AMR flags or fabricate absent hierarchy data.

Physical BCs come from the merged input and registered operators. Generic BCs must implement the appropriate component parity and geometry for the field; custom physics may require application operators. Never substitute zero-fill, scalar reflection, or generic interpolation where semantics are unknown. An applicable missing operator is a field-specific preflight error; supply operator registration paths and fixtures for all supported categories.

For non-real values, interpolation/averaging is not automatically meaningful. Require or define type-appropriate transfer policies and test them. Numerical correctness is defined by the declared boundary and refinement operators; arbitrary physical semantics cannot be inferred from array shape.

If existing boundary algorithms modify owned staggered entities as part of restriction/synchronization, explicitly separate required ghost/shared-entity operations from evolution updates. Preserve file-authoritative owned degrees of freedom. Verify divergence/circulation or other constraints when promised by custom operators.

## 7. Sparse allocation and absent data

Keep **source allocation provenance** separate from **analysis backing allocation**. Exact preservation of source allocation and valid neighbor data can conflict: a block without a field may still need that field's halo.

Determine the closure of required sparse support across same-level neighbors, refinement stencils, and physical boundaries before communication. Allocate only required selected fields, using independent analysis controllers and ordinary allocation APIs. Do not expand to excluded siblings or unrelated source controllers.

Use the framework's documented unallocated-field value where it defines a value for absent interiors. If absence has no numerical meaning, require an application policy; do not assume zero. Initialize added backing storage accordingly, retain the original file allocation mask as provenance, and exclude artificial support allocation from claims about source occupancy. Suppress evolution allocation/deallocation heuristics during the import and initial exchange.

If the framework cannot represent required halo support independently, a small general storage improvement may be necessary. Document that choice and its invariants rather than adding an analysis bypass in sparse communication. Test that a second exchange is stable and does not keep expanding allocation unexpectedly.

## 8. Swarms and particle variables

Reuse existing swarm storage, typed attribute packs, migration, and boundary machinery. Project registered swarm descriptors just as mesh fields are projected, preserving variable types/shapes and required built-in attributes. Avoid constructing unrelated evolution swarms.

- Read particle counts and file offsets per source block. Restore active particles and attributes as complete records, independent of memory pool capacity, inactive slots, writer rank count, or current block partitions.
- Preserve particle IDs and integer attributes exactly. Handle empty swarms, empty blocks, variable particle counts, and all framework-supported particle attribute types/shapes.
- Validate that required positions and selected attributes are present and consistently sized. Missing optional analysis attributes require initialization before they are communicated or consumed.
- Redistribute to the reconstructed mesh owners using the normal ownership rules, including particles exactly on boundaries. Complete all sends/receives and maintain attribute association through compaction and migration. Do not assume `pack_size = -1`; exercise multiple partitions and rank counts.
- Apply merged-input swarm BCs according to an explicit policy. Periodic wrapping, reflection, absorption, or user BCs may transform or remove particles; report resulting changes. Never silently discard particles because the importer selected the wrong owner or lacked an attribute.
- Distinguish ownership migration from ghost-particle replication. If supported/required by the native swarm model, rebuild ghost particles and their attributes with duplicate/ownership tracking; otherwise mesh ghost terminology is not applicable to particle records.

Do not invent a general particle halo system if Parthenon's swarm model has none. The analysis-ready guarantee is complete particle payloads, correct ownership, and completed configured boundary processing.

## 9. MPI, failure handling, and readiness

All ranks must use the same manifest and deterministic selections. Empty selections and ranks with zero local blocks/particles are valid. Do not dereference a first local block unconditionally. Collective communicator creation and error handling must include all participating ranks, including ranks with no local work.

Respect task dependencies, device synchronization, and communication completion. A successful read is not sufficient to mark a field initialized or ghost-valid. Track readiness at a useful granularity and expose analysis callbacks only after the final boundary stage completes.

On error, identify dataset/swarm, component or topology, expected and actual layouts, and missing operator or dependency. Close readers and release communication resources cleanly. Rank-zero summaries should list imported, initialized, excluded, ignored, omitted, boundary-ready, and inapplicable categories, plus sparse support and particle boundary changes.

## 10. Implementation sequence and validation gates

Implement from `develop` in reviewable steps; do not apply PR #1454 first.

1. **Audit and contracts:** Record actual capabilities, metadata transformations, input precedence, and operator requirements. Add focused contract tests.
2. **Descriptor projection:** Implement catalog/analysis separation, exact state selection, identity/pack access, independent sparse control, and flux representation. Verify source metadata remains unchanged.
3. **Reader and schema:** Share typed reading mechanics, inspect files, import interiors/non-spatial payloads, and restore swarms. Add any required writer schema support with legacy compatibility tests.
4. **Initialization:** Split structural and data setup; integrate the analysis orchestration and callback lifecycle. Establish serial uniform-grid loading first as an intermediate milestone.
5. **Full boundaries:** Complete all topologies, sparse support, custom BCs/operators, alternate grids, and swarm boundary processing. Partial support is not a final deliverable.
6. **Distributed validation and compatibility:** Complete the matrix below, documentation, and normal-start/restart gates.

Use deterministic analytic fields and independently computed expectations. Poison ghost storage before exchange and inspect every meaningful ghost index afterward; finite values alone are not proof of correctness.

| Test family | Required cases |
| --- | --- |
| Input/dispatch | Archive absent/present; input overrides archive; CLI overrides input; package parameters reflect merge; dump kind/legacy detection; missing `-i`; existing `-i`, `-r`, restart `-a` |
| Field layout | 1D/2D/3D; cell, all face/edge orientations, node, non-spatial; scalar/vector/tensor shapes; all supported scalar types; source with/without ghosts; different destination ghost width |
| Metadata | No communication flags; `Derived`; `Independent`; `Restart`; `OneCopy`; `WithFluxes`; selected/excluded standalone flux; sparse; `Fine`; GMG-related flags; user flags; relevant legal combinations |
| Mesh boundaries | Multiple blocks; periodic and physical faces/edges/corners; multiple AMR levels; both transfer directions; staggered shared ownership; custom conservative/constraint-preserving operators; supported coordinate systems |
| Sparse | Independent allocation masks; absent neighbors and coarse/fine donors; controller groups with excluded members; support allocation/provenance; analysis-only sparse data; repeat exchange |
| Swarms | Multiple swarms and typed/shaped attributes; empty blocks/ranks; migration and boundary particles; periodic/custom BCs; supported ghost particles; ID/attribute association; no unexplained loss/duplication |
| Parallelism | Serial and MPI; writer/reader rank counts differ; multiple `pack_size` values including `-1` and `1`; coalesced communication modes; CPU and an available device backend |
| Lifecycle | No evolution; no accidental derived overwrite; callback observes final valid state; analysis-only initialization dependencies; output retains snapshot time/cycle |
| Negative cases | Incompatible types/shapes/geometry; ambiguous legacy topology; malformed sparse/swarm records; excluded mandatory dependencies; missing physical/refinement policies; conflicting shared values |

Keep tests tractable with pairwise combinations plus targeted interaction tests; do not require a blind Cartesian product. Preserve exact imported owned values when types are unchanged, or expected converted values under a documented conversion. Use analytic AMR expectations appropriate to the registered operators rather than assuming arbitrary fields interpolate exactly.

Run the repository's required formatting and test gates and relevant existing restart, boundary, sparse, swarm, and output regressions. Add regression coverage that confirms ordinary allocation/controller/flux behavior and normal metadata remain unchanged.

## 11. Completion criteria

- The requested command works from a fresh `develop`-based implementation, including input overrides and analysis-only exit.
- All registered/importable variable categories in the audited support matrix are implemented and tested; unsupported valid categories are not silently deferred.
- Every initialized selected spatial field has correct complete ghosts before analysis consumers run, regardless of source communication/evolution metadata. Swarms and non-spatial state satisfy their applicable validity contracts.
- Missing information or operators produce explicit errors; the implementation never reports successful readiness for undefined data.
- Normal simulation and restart semantics remain unchanged.
- Allocation and communication follow validated analysis descriptors through ordinary infrastructure; there is no pervasive analysis override state in boundary kernels or data containers.
- Documentation includes metadata projection, source-catalog access, field/swarm selection, input precedence, sparse provenance, required BC/refinement policies, initialization hooks, and supported legacy files.

Provide a final implementation report listing changes, tests run and results, format additions, application callback adjustments, and any unresolved requirements. Do not declare the feature complete with unresolved support-matrix entries.
