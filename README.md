# Parthenon

[![testing](https://github.com/lanl/parthenon/actions/workflows/ci-short.yml/badge.svg?branch=develop)](https://github.com/lanl/parthenon/actions/workflows/ci-short.yml)
[![Extended CI](https://github.com/lanl/parthenon/actions/workflows/ci-extended.yml/badge.svg?branch=develop)](https://github.com/lanl/parthenon/actions/workflows/ci-extended.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Matrix chat](https://img.shields.io/matrix/parthenon-general:matrix.org)](https://app.element.io/#/room/#parthenon-general:matrix.org)

Parthenon -- a performance portable block-structured adaptive mesh refinement framework

# Key features

* High performance by
  * device first/device resident approach (work data only in device memory to prevent expensive transfers between host and device)
  * transparent packing of data across blocks (to reduce/hide kernel launch latency)
  * direct device-to-device communication via asynchronous, one-sided  MPI communication
* Intermediate abstraction layer to hide complexity of device kernel launches
* Flexible, plug-in package system
* Abstract variables controlled via metadata flags
* Support for particles
* Multi-stage drivers/integrators with support for task-based parallelism

# Community
* [Chat room on matrix.org](https://app.element.io/#/room/#parthenon-general:matrix.org)

# Dependencies

## Required

* CMake 3.16 or greater
* C++17 compatible compiler
* Kokkos 3.6 or greater

## Optional (enabling features)

* MPI
* OpenMP
* HDF5 (for outputs)
* Ascent (for in situ visualization and analysis)

## Other

* catch2 (for unit tests)
* python3 (for regression tests)
* numpy (for regression tests)
* matplotlib (optional, for plotting results of regression tests)

# Quick start guide

For detailed instructions for a given system, see our [build doc](docs/building.md).

## Basics

    mkdir build
    cd build
    cmake ../
    cmake --build . -j 8
    ctest

## Import Into Your Code
```c++
// Imports all of parthenon's public interface
#include <parthenon/parthenon.hpp>

// You can use one of the following headers instead if you want to limit how
// much you import. They import Parthenon's Driver and Package APIs,
// respectively
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// The following namespaces are good short-hands to import commonly used names
// for each set of Parthenon APIs.
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
```

## Parallel_for wrapper options

Following options are available to configure the default behavior of the `par_for` wrappers.

- `PAR_LOOP_LAYOUT` (sets default layout)
  - `MANUAL1D_LOOP` maps to `Kokkos::RangePolicy` (default for CUDA backend)
  - `MDRANGE` maps to `Kokkos::MDRangePolicy`
  - `SIMDFOR_LOOP` maps to standard `for` loops with `#pragma omp simd` (default for OpenMP backend)
  - `TPTTR_LOOP` maps to double nested loop with `Kokkos::TeamPolicy` and `Kokkos::ThreadVectorRange`
  - `TPTVR_LOOP` maps to double nested loop with `Kokkos::TeamPolicy` and `Kokkos::ThreadVectorRange`
  - `TPTTRTVR_LOOP` maps to triple nested loop with `Kokkos::TeamPolicy`, `Kokkos::TeamThreadRange` and `Kokkos::ThreadVectorRange`

Similarly, for explicit nested paralellism the `par_for_outer` and `par_for_inner` wrappers are available.
`par_for_outer` always maps to a `Kokkos::TeamPolicy` and the `par_for_inner` mapping is controlled by the
- `PAR_LOOP_INNER_LAYOUT` (sets default innermost loop layout for `par_for_inner`)
  - `SIMDFOR_INNER_LOOP` maps to standard `for` loops with `#pragma omp simd` (default for OpenMP backend)
  - `TVR_INNER_LOOP` maps to `Kokkos::TeamVectorRange` (default for CUDA backend)


## Kokkos options
Kokkos can be configured through `cmake` options, see https://github.com/kokkos/kokkos/wiki/Compiling

For example to build with the OpenMP backend for Intel Skylake architecture using Intel compilers

    mkdir build-omp-skx && cd build-omp-skx
    cmake -DKokkos_ENABLE_OPENMP=ON -DCMAKE_CXX_COMPILER=icpc -DKokkos_ARCH_SKX=ON ../

or to build for NVIDIA V100 GPUs (using `nvcc` compiler for GPU code, which is automatically picked up by `Kokkos`)

    mkdir build-cuda-v100 && cd build-cuda-v100
    cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=On ../

or to build for AMD MI100 GPUs (using `hipcc` compiler)

    mkdir build-hip-mi100 && cd build-hip-mi100
    cmake -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc -DKokkos_ARCH_Vega908=ON ../

# Developing/Contributing

Please see the [developer guidelines](CONTRIBUTING.md) for additional information.

# Documentation

Please see the [docs/](docs/README.md) folder for additional documentation on features and
how to use them.

We are migrating our legacy docs to sphinx, which can be found [here](https://parthenon-hpc-lab.github.io/parthenon).

# Contributors

| Name     | Handle       | Team       |
|----------|--------------|------------|
| Jonah Miller | @Yurlungur  | LANL Physics  |
| Josh Dolence | @jdolence | LANL Physics |
| Andrew Gaspar | @AndrewGaspar | LANL Computer Science |
| Philipp Grete | @pgrete | Athena Physics |
| Forrest Glines | @forrestglines | Athena Physics |
| Jim Stone | @jmstone | Athena Physics |
| Jonas Lippuner | @jlippuner | LANL Computer Science |
| Joshua Brown | @JoshuaSBrown | LANL Computer Science |
| Christoph Junghans | @junghans | LANL Computer Science |
| Sriram Swaminarayan | @nmsriram | LANL Computer Science |
| Daniel Holladay | @dholladay00 | LANL Computer Science |
| Galen Shipman | @gshipman | LANL Computer Science |
| Ben Ryan | @brryan | LANL Physics |
| Clell J. (CJ) Solomon | @clellsolomon | LANL Physics |
| Luke Roberts | @lroberts36 | LANL Physics |
| Ben Prather | @bprather | LANL Physics |
