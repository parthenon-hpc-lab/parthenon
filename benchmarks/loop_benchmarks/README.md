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
  --variant cpu_logical_kji \
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

By default the analysis script runs the CPU-focused suite:

- `cpu_dense_flat_bvkji` full-memory baseline
- `cpu_logical_kji`
- `cpu_rawspan_voi`
- `cpu_rawspan_ovi`
- `cpu_logical_ovi`
- `kokkos_rawspan_ovi`
- `kokkos_rawspan_view_ovi`
- `kokkos_logical_ovi`

For GPU-capable runs, pass `--gpu` to switch to a Kokkos-only suite:

- `kokkos_dense_flat_bvkji` full-memory baseline
- `kokkos_flat_kji`
- `kokkos_mdrange_kji`
- `kokkos_rawspan_ovi` via explicit `ninner=ni*nj`, `ninner=ni`, and chunk sweeps
- `kokkos_rawspan_view_ovi` via explicit `ninner=ni*nj`, `ninner=ni`, and chunk sweeps
- `kokkos_logical_ovi` via explicit `ninner=ni*nj`, `ninner=ni`, and chunk sweeps

Available benchmark variants currently include:

- `cpu_dense_flat_bvkji`
- `cpu_logical_kji`
- `cpu_rawspan_voi`
- `cpu_rawspan_ovi`
- `cpu_logical_ovi`
- `kokkos_dense_flat_bvkji`
- `kokkos_flat_kji`
- `kokkos_mdrange_kji`
- `kokkos_rawspan_ovi`
- `kokkos_rawspan_view_ovi`
- `kokkos_logical_ovi`

The report includes:

- `stencil` and `heavy` cubic edge sweeps with `ni=nj=nk`
- a fixed-edge chunk-size sweep
- a fixed-edge heavy-intensity sweep

By default the analysis script chooses the number of blocks so
`blocks * ni * nj * nk` stays close to an architecture-dependent
`--target-total-cells` value. The CPU and `--gpu` modes use different defaults,
and you can override them directly. The option accepts forms like `1e9`.

Example:

```bash
python3 benchmarks/loop_benchmarks/run_analysis.py \
  --binary build/benchmarks/loop_benchmarks/loop-benchmarks \
  --analysis-mode default \
  --target-total-cells 2e6 \
  --output-dir reports/loop-benchmarks

python3 benchmarks/loop_benchmarks/run_analysis.py \
  --binary build/benchmarks/loop_benchmarks/loop-benchmarks \
  --gpu \
  --analysis-mode default \
  --target-total-cells 2e7 \
  --output-dir reports/loop-benchmarks-gpu
```
