# Loop Benchmarks

This benchmark provides a standalone loop microbenchmark executable inside
Parthenon's `benchmarks/` tree. It preserves the current command-line interface
of the benchmark while building directly with Parthenon's CMake configuration.

## Build

From the Parthenon repository root:

```bash
cmake -S . -B build -DKokkos_ENABLE_SERIAL=ON -DREGRESSION_GOLD_STANDARD_SYNC=OFF
cmake --build build --target loop-benchmarks -j 4
```

## Run

```bash
./build/benchmarks/loop_benchmarks/loop-benchmarks \
  --kernel stencil \
  --variant cpu_simd \
  --blocks 8 \
  --vars 16 \
  --nk 32 \
  --nj 32 \
  --ni 32 \
  --ghosts 2 \
  --repeats 5 \
  --warmup 1
```

## Analysis

The narrative analysis driver lives next to the benchmark as
`benchmarks/loop_benchmarks/run_analysis.py`.

Two analysis modes are supported:

- `default`: CPU raw-span and Kokkos raw-span use their default chunking.
- `verify`: CPU raw-span uses `--inner-chunk-length <ni>` during the `ni` sweep
  for apples-to-apples comparison against the SIMD loop orders.

The report includes:

- `stencil` and `heavy` `ni` sweeps
- a fixed-`ni` chunk-size sweep

Example:

```bash
python3 benchmarks/loop_benchmarks/run_analysis.py \
  --binary build/benchmarks/loop_benchmarks/loop-benchmarks \
  --analysis-mode default \
  --output-dir reports/loop-benchmarks
```
