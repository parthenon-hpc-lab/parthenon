# Loop Benchmarks V2

This directory is the clean rewrite of the loop benchmark prototype.

The intent is to keep the loop and kernel definitions local and readable while
allowing the surrounding harness to be more flexible.

See:

- `ARCHITECTURE.md` in `benchmarks/loop_benchmarks`
- `GOALS.md` in `benchmarks/loop_benchmarks`

Basic usage:

- single case: `./build-make/benchmarks/loop_benchmarks_v2/loop-benchmarks-v2 --loop cpu_bovi_contiguous --nblocks 2 --nvars 3 --nz 8 --ny 8 --nx 8 --nghost 1 --ninner 64 --niter 4`
- batch analysis: `python3 benchmarks/loop_benchmarks_v2/run_analysis.py --binary build-make/benchmarks/loop_benchmarks_v2/loop-benchmarks-v2`

The analysis script performs the standard block-size sweep with `ninner = nx * ny`
for each cubic block size and writes a CSV plus a summary PDF containing the
compiler, flags, platform, and CPU information.
