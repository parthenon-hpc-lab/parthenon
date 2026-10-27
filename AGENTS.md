<!-- This file was made in part with generative AI. -->

# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Copilot, Cursor, ...) working in the
Parthenon repository. Human contributors should read
[CONTRIBUTING.md](CONTRIBUTING.md); this file summarizes the parts of it that agents
must follow and adds agent-specific expectations. If the two disagree, CONTRIBUTING.md
wins.

## Project at a glance

Parthenon is a performance-portable, block-structured adaptive mesh refinement (AMR)
framework written in C++20 on top of Kokkos (5.x) and MPI. It is a library consumed by
downstream physics codes, so **API and behavior changes have downstream impact**.

| Path | Contents |
| --- | --- |
| `src/` | The library (`mesh/`, `bvals/`, `interface/`, `pack/`, `tasks/`, `solvers/`, `outputs/`, `loop_abstraction/`, `utils/`, ...) |
| `example/` | Example drivers (advection, diffusion, poisson, particles, ...); also used by regression tests |
| `tst/unit/` | Catch2 unit tests (`test_*.cpp`) |
| `tst/regression/` | Python-driven regression tests (`run_test.py`, `test_suites/<name>/`) |
| `tst/style/` | Vendored `cpplint.py` |
| `doc/sphinx/src/` | User/developer documentation (reStructuredText) |
| `scripts/python/packages/` | `parthenon_tools` (e.g. `phdf`) and the performance app |
| `cmake/` | CMake modules and machine configs (`cmake/machinecfg/`) |
| `external/` | Git submodules (Kokkos, Catch2). **Do not modify.** |

## Rules for agents (mandatory)

These come from the project's
[agentic coding policy](CONTRIBUTING.md#use-of-agentic-coding).

1. **Disclose AI involvement in every file you create or meaningfully modify.** Add this
   line near the top of the file, after the license/copyright header if there is one:
   - C++: `// This file was made in part with generative AI.`
   - Python/CMake: `# This file was made in part with generative AI.`
   - Markdown: `<!-- This file was made in part with generative AI. -->`
   - reStructuredText: a comment (`.. This file was made in part with generative AI.`)
     or a visible note, matching existing docs such as `doc/sphinx/src/solvers.rst`.

   Don't add a second copy if the line is already there.
2. **The human submitter owns the change.** Never open, merge, or force-push PRs, and
   never post `@par-hermes` commands, unless the user explicitly asks you to. When
   drafting a PR description, state clearly what was AI-generated and how much, and
   tick the generative-AI checkbox in `.github/PULL_REQUEST_TEMPLATE.md` only if that
   is true.
3. **Write a plan for large, feature-style changes.** Put it in
   `doc/plan_histories/<PR number>.md` (create the folder if it doesn't exist). The plan
   covers motivation, design, affected APIs, and testing. If there's no PR number yet,
   use a placeholder name and tell the user to rename the file.
4. **Prefer small, logically separate changes.** If a task naturally splits into
   independent pieces (e.g. a refactor plus a feature), suggest separate PRs.
5. **Condense generated code.** Match how terse and idiomatic the surrounding code is.
   Reuse existing utilities instead of re-implementing them. Don't add boilerplate,
   defensive checks the codebase doesn't use, or comments that just restate the code.
   Expect reviewers to be more skeptical of AI-assisted PRs, so make the diff easy to
   review.
6. **Don't touch unrelated code.** Don't reformat or "clean up" files outside the scope
   of the task. Never edit `external/`, build directories (`build*/`), or generated
   files by hand.

## Building

Out-of-source CMake builds; submodules must be checked out
(`git submodule update --init --recursive`).

```bash
# Host (CPU) build, with tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Build only what you need, e.g. an example target or the unit tests
cmake --build build -t advection-example
cmake --build build -t unit_tests
```

Useful options (see the top-level `CMakeLists.txt` for the full list):
`PARTHENON_DISABLE_MPI`, `PARTHENON_DISABLE_HDF5`, `PARTHENON_SINGLE_PRECISION`,
`PARTHENON_ENABLE_RAW_OPENMP`, `PARTHENON_DISABLE_SPARSE`, `PARTHENON_ENABLE_FFT`,
`PARTHENON_DISABLE_EXAMPLES`, `PARTHENON_ENABLE_{UNIT,INTEGRATION,REGRESSION,PERFORMANCE}_TESTS`,
`MACHINE_CFG` / `MACHINE_VARIANT` (e.g. `-DMACHINE_CFG=cmake/machinecfg/CI.cmake -DMACHINE_VARIANT=cuda-mpi`).

Full builds (especially with CUDA/HIP) are slow. Build only the targets you need, and
reuse an existing build directory if the user has one.

## Testing

Run from the build directory:

```bash
ctest -LE 'performance|regression'     # unit + integration (what CI "unit" runs)
ctest -R regression_test:calculate_pi  # a single serial regression test
ctest -R regression_mpi_test:poisson   # a single MPI regression test
ctest -L regression                    # all regression tests (slow; needs parthenon_tools)
```

- Regression tests need `pip install scripts/python/packages/parthenon_tools` and the
  Python modules in `requirements.txt`.
- Bug fixes should come with a test that would have caught the bug. New features need
  tests. Put unit tests in `tst/unit/test_<topic>.cpp` (Catch2, tagged `[unit]` etc.)
  and register them in `tst/unit/CMakeLists.txt`.
- For regression tests, follow the `TestCase(utils.test_case.TestCaseAbs)` pattern with
  `Prepare`/`Analyse` in `tst/regression/test_suites/<name>/<name>.py`, and register
  the test in `tst/regression/CMakeLists.txt` (see CONTRIBUTING.md for details). Use
  `NUM_MPI_PROC_TESTING` rather than hard-coding the number of ranks.
- Never change gold-standard data or loosen test tolerances to make a test pass
  without asking the user first.
- Say what you actually ran. If you couldn't build or test a configuration (e.g. GPU,
  MPI), tell the user instead of implying it works.

## Style, lint, and formatting

CI fails on any of these, so check them before calling a change done.

- **clang-format** (>= 11; CI's formatting bot uses a recent version), config in
  `.clang-format`: 90-column limit, 2-space indent. Run
  `cmake --build build -t format`. This also runs **black** on Python files.
- **cpplint**, config in `CPPLINT.cfg`: `cmake --build build -t lint`, or
  `python3 tst/style/cpplint.py --counting=detailed --recursive src example tst`.
  Header guards are path-based relative to `src/`, e.g. `MESH_MESH_HPP_` for
  `src/mesh/mesh.hpp`. Includes must be in alphabetical order within each group.
- **Copyright header**: new C++/CMake/Python files need the standard Triad/LANL header
  (copy it from a neighboring file and set the current year). LANL contributors also
  update the year range on files they change (`check-copyright` / `update-copyright`
  targets).
- **Docs**: `python scripts/check_docs_toc.py` must pass, so every `.rst` under
  `doc/sphinx/src` has to be reachable from a toctree. Docs build with
  `-W --keep-going -n` (warnings are errors). If you change runtime input parameters
  used by documented examples, regenerate `doc/sphinx/src/generated/` with
  `cmake --build build -t param-docs`.

## C++ conventions

- Everything in the library goes in `namespace parthenon`. Use `Real` (not
  `double`/`float`) for floating point, because single-precision builds are tested in
  CI.
- Use the portable loop abstractions (`par_for`, `par_reduce`, `par_for_outer` /
  `par_for_inner`, and the newer `loop_abstraction/` machinery) with `KOKKOS_LAMBDA`
  instead of raw Kokkos or host loops. Code in device lambdas must be GPU-safe: no
  `std::` containers or strings, no virtual dispatch, no host-only calls, and no
  capturing `this` or host-only objects by reference.
- Use sparse/swarm packs (`SparsePack`, `MeshData`, `MeshBlockData`) for variable
  access, following existing examples in `example/` and `src/`.
- Use the error-checking macros from `src/utils/error_checking.hpp`
  (`PARTHENON_REQUIRE`, `PARTHENON_REQUIRE_THROWS`, `PARTHENON_DEBUG_REQUIRE`,
  `PARTHENON_FAIL`, `PARTHENON_WARN`) instead of `assert` or raw `throw`.
- Write TODOs as `// TODO(<initials or handle>): ...`. Ask the user for their handle
  rather than leaving it blank.
- Consider MPI and AMR: code must work with multiple ranks, multiple blocks per rank,
  and refined meshes. Mention it in your summary if you only tested a single block or
  a single rank.

## CHANGELOG

Every PR adds one line under `## Current develop` in `CHANGELOG.md`, in the right
category, newest first:

```
- [[PR NNNN]](https://github.com/parthenon-hpc-lab/parthenon/pull/NNNN) Short description
```

The categories are Added, Changed (behavior/API changes), Fixed, Infrastructure (no
downstream impact), Removed, and Incompatibilities (breaking changes; add the entry
here *in addition to* its regular category). If there's no PR number yet, leave a
placeholder and tell the user.

## Git workflow

- `develop` is the only long-lived branch, and all PRs target it. Releases are `vYY.0M`
  tags.
- Branch names: `<username>/<short-description>`.
- Only commit, push, or rebase when the user asks you to. Never commit build
  directories, local scratch files, or generated output (`*.phdf`, `*.rhdf`,
  `*.xdmf`, ...).
- Prefix draft PR titles with `WIP:`. Link the related issue in the PR body and fill
  in `.github/PULL_REQUEST_TEMPLATE.md`, including the breaking-change and
  generative-AI items.

## Before you say you're done

- [ ] Builds for the configuration(s) you touched, with no new warnings
- [ ] Relevant `ctest` tests pass (say which ones you ran)
- [ ] `format` and `lint` targets are clean
- [ ] AI disclosure line added to new/modified files
- [ ] Tests added or updated for the fix or feature
- [ ] Docs updated in `doc/sphinx/src/` (and in a toctree) if user-facing
- [ ] `CHANGELOG.md` entry drafted
- [ ] For large features: `doc/plan_histories/<PR>.md` drafted
- [ ] Diff is minimal, with no stray reformatting or debugging leftovers
