<!-- This file was made in part with generative AI. -->

# Asynchronous OpenPMD (ADIOS2/BP5) Output Plan

Rename this file to `<PR number>.md` once the PR is opened.

## Goal

Let simulations continue while OpenPMD output files are written to disk. Keep the
current file-based iteration encoding (one file per output), allow at most one file per
output block to be written in the background, and avoid concurrent background writes of
different output blocks hitting the parallel filesystem at the same time.

## Status quo

`OpenPMDOutput::WriteOutputFileImpl` copies each variable to a reused host buffer, calls
`storeChunkRaw` and flushes per variable. In openPMD-api 0.17 the default
`preferred_flush_target` is `"disk"`, i.e., each flush is a synchronous
`PerformDataWrite`. Finally, `it.close(); series.close()` ends the BP5 step and closes
the engine (metadata aggregation and remaining writes), all on the critical path.

ADIOS2's own `AsyncWrite` does not help with file-based encoding as the engine is closed
right after the single step of each file and `Close()` waits for the async write.

## Design

1. Main thread (blocking, short):
   - Wait for the previous background write of the same output block (if any).
   - Run the existing logic (attributes, device-to-host copies, `storeChunkRaw`, MPI
     gathers on `MPI_COMM_WORLD`) but flush with `preferred_flush_target = "buffer"` so
     that BP5 copies the data into its internal buffer. Reusing host buffers stays safe.
2. Background thread: `it.close(); series.close()`, i.e., attribute flush, BP5
   `EndStep` (metadata aggregation, data write) and `Close`.
3. A single FIFO worker thread (`AsyncIOQueue`, owned by `Outputs`) executes all
   background closes. This serializes background I/O across output blocks.
4. Each async output block uses its own duplicate of `MPI_COMM_WORLD` for the `Series`
   (openPMD does not duplicate the communicator itself).
5. MPI needs `MPI_THREAD_MULTIPLE` as ADIOS2 runs collectives in the background thread.
   New CMake option `PARTHENON_ENABLE_ASYNC_IO` initializes MPI with
   `MPI_Init_thread(MPI_THREAD_MULTIPLE)`. If an async output is requested but the
   provided thread level is insufficient, a warning is printed and the output is written
   synchronously.
6. Pending writes are waited for explicitly at the end of `EvolutionDriver::Execute`
   (errors in the background thread are rethrown there or at the next output of the same
   block). The `Outputs` destructor drains the queue as a backstop.

New input parameter: `<parthenon/output*>/async = false` (OpenPMD only).

Memory: about one copy of the rank-local output per file in flight (BP5 buffer).

## Batching across output blocks (evaluation)

- Serializing background writes in one FIFO worker (implemented): outputs triggered at the
  same time (e.g., multiple slices) are written back-to-back, never concurrently.
- Reducing the number of files per (small) output via the existing per-block
  `backend_config` (e.g., `NumAggregators`/`NumSubFiles` = 1 for slices): documented, no
  code needed. For small outputs the filesystem load is dominated by file creation.
- Merging several output blocks into one file: technically possible but not pursued.
  File naming/numbering is per block and diverges for different cadences or signals,
  record names of different slices collide (requiring non-standard prefixes that break
  readers and the restart reader), and the gain over the two points above is small.
- Variable-based encoding (one file with many steps per block) would allow native ADIOS2
  async writes but changes the file layout; out of scope.

## Implementation steps

1. `src/outputs/async_io.hpp`: FIFO single-thread task queue + Catch2 unit test.
2. CMake option and `MPI_Init_thread` in `ParthenonManager::ParthenonInitEnv`.
3. `Outputs`: parse `async`, own the queue, `WaitForPendingWrites()`, call from driver.
4. `OpenPMDOutput`: duplicate communicator, wait for previous write, flush to buffer,
   submit close.
5. Regression test: `restart_opmd` writes the baseline asynchronously, restarts from an
   asynchronously written file and compares against synchronously written outputs.
6. Documentation (`doc/sphinx/src/outputs.rst`) and CHANGELOG.

## Risks

- Concurrent use of different ADIOS2 instances (main thread opens block B while the
  worker closes block A). Fallback: move all openPMD calls to the worker using
  Parthenon-owned `shared_ptr` buffers.
- BP5 file creation still happens synchronously in the main thread at iteration open.
- The worker thread shares the cores the MPI rank is bound to.
