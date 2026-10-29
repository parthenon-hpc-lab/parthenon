<!-- This file was made in part with generative AI -->
<!-- TODO: rename to <PR number>.md once the PR is opened -->

# Streaming openPMD output for live visualization

## Goal

Live visualization of a running simulation with as little code change as possible, by
extending the existing openPMD/ADIOS2 output instead of adding a new output type.
The first target is a simple Python (`openpmd_api` + matplotlib) viewer and the
`advection` example on a uniform grid (`parthenon/mesh/refinement=none`).

## Evaluation (done before implementation)

- openPMD-api 0.17.1 supports streaming through `Access::CREATE_LINEAR` and
  `Series::snapshots()`. The `.sst` file extension selects the ADIOS2 SST engine.
  Prototypes with the Python bindings and the local ADIOS2 2.12.1 confirmed the following:
  - An SST writer configured with `RendezvousReaderCount=0, QueueLimit=1,
    QueueFullPolicy=Discard` never blocks, and readers can join late.
  - A single BP5 file with one step per output can be followed with `read_linear` while
    it is being written, and can be read with random access afterwards.
- Tools:
  - Python with `openpmd_api` (already a regression-test dependency): stream-aware and
    easiest to use, so it is the first target.
  - ParaView 6.2: its openPMD reader drops every mesh whose name contains `_lvl`, so it
    currently shows no Parthenon data at all. It is also random-access only and caches
    the Series. A possible follow-up is to drop the `_lvl0` suffix for level 0 (new
    `openpmd_format_version`).
  - VisIt: no openPMD-aware ADIOS2 reader, so not recommended.
  - Ascent (already supported): renders images in situ, but is not interactive.
- Example: `advection` (2D, dense cell-centered field, periodic, cheap). The other
  examples were ruled out: `fine_advection` (`Metadata::Fine` is unsupported in openPMD
  output), `sparse_advection` and `stochastic_subgrid` (sparse/AMR leave holes per level),
  and `calculate_pi` (no time evolution). `particle_tracers` is a possible second demo.

## Design

1. New openPMD output option `stream_engine = none | sst | bp5` (default `none` keeps
   the current behaviour).
   - `sst`: in-memory staging to `<problem_id>.<id>.sst`. By default it uses the
     non-blocking engine parameters above, so the simulation is never held up by a
     viewer, or by the absence of one. It also sets `DataTransport=WAN`, because the
     automatically selected transport (e.g., UCX) may not be available to the reader.
     A user `backend_config` replaces these defaults.
   - `bp5`: a single growing `<problem_id>.<id>.bp` file.
   - The Series is created once and kept open, with one ADIOS2 step per output and
     `file_number` as the iteration index. Streams use openPMD's variable-based iteration
     encoding, as group-based encoding is limited to 100 steps in BP5 files (and its
     metadata grows with every step). It is closed by the destructor of the
     `openPMD::Series` held by `OpenPMDOutput`. As for all other device data, the
     driver must be destroyed before `ParthenonFinalize`.
   - Streams only contain regularly triggered (or requested) outputs: `now` and `final`
     signals do not write to a stream. This keeps iteration indices unique, and the final
     step is not needed for live visualization.
   - Restart outputs cannot be streamed. Restarting a simulation overwrites a `bp5`
     stream.
2. New generic output option `on_request = true`, which makes outputs viewer-triggered.
   - The output is written only if the file `<problem_id>.<id>.request` exists. `dt`/`dn`
     are ignored (a negative `dt` still disables the block). These outputs are never
     written on `final`.
   - Rank 0 checks for the file once per `MakeOutputs` call and broadcasts the result.
     The file is removed *before* the output is written, so a new request issued after
     the viewer received the step cannot be lost.
   - With SST, the simulation continues right after the step is closed. The viewer pulls
     the data asynchronously, so at most one step is buffered.
3. Viewer `parthenon_tools/opmd_live.py`:
   1. Opens the stream with `read_linear`.
   2. Re-requests every 0.2 s from a separate process until the first frame arrives.
      This is needed because SST admits late readers only at step boundaries, and the
      blocking open holds the GIL.
   3. Then loops: load → close step → request next → plot.
   4. `--min-interval` limits the request rate, and thereby the overhead for the
      simulation (10–20% when a viewer is attached for the whole run).

## Testing

- Regression test `stream_opmd` (advection with AMR, so that records of finer levels
  appear during the run). It runs a `stream_engine=bp5`
  block next to a file-based block with the same cadence, and checks that both contain
  identical data (all level records and chunks) for every streamed iteration.
- Manual test of `sst` with `on_request=true` and `opmd_live.py`.

## Changes made

- `src/outputs/outputs.{hpp,cpp}`, `output_parameters.hpp`, `parthenon_opmd.cpp`
- `scripts/python/packages/parthenon_tools/parthenon_tools/opmd_live.py`
- `tst/regression/test_suites/stream_opmd/`
- `doc/sphinx/src/outputs.rst`, `CHANGELOG.md`
