# parthenon

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3c7f326d05b34929a847657a9674f524)](https://app.codacy.com/gh/lanl/parthenon?utm_source=github.com&utm_medium=referral&utm_content=lanl/parthenon&utm_campaign=Badge_Grade)
[![deepcode](https://www.deepcode.ai/api/gh/badge?key=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwbGF0Zm9ybTEiOiJnaCIsIm93bmVyMSI6ImxhbmwiLCJyZXBvMSI6InBhcnRoZW5vbiIsImluY2x1ZGVMaW50IjpmYWxzZSwiYXV0aG9ySWQiOjE2MzAxLCJpYXQiOjE2MjM5NjA4Njh9.7W8akiFnSjPx7tPq5Ra6NqnJUOLq0sKnwaHEpD0_YH0)](https://www.deepcode.ai/app/gh/lanl/parthenon/_/dashboard?utm_content=gh%2Flanl%2Fparthenon)
[![codecov](https://codecov.io/gh/lanl/parthenon/branch/master/graph/badge.svg)](https://codecov.io/gh/lanl/parthenon)
[![testing](https://gitlab.com/theias/hpc/jmstone/athena-parthenon/parthenon-ci-mirror/badges/develop/pipeline.svg)](https://gitlab.com/theias/hpc/jmstone/athena-parthenon/parthenon-ci-mirror/-/commits/develop)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Parthenon performance portable AMR framework

# Community
* [Chat room on matrix.org](https://app.element.io/#/room/#parthenon-general:matrix.org)

# Dependencies

## Required

* CMake 3.16 or greater
* C++14 compatible compiler
* Kokkos 3.0 or greater

## Optional (enabling features)

* MPI
* OpenMP
* HDF5 (for outputs)

## Other

* catch2 (for unit tests)
* python3 (for regression tests)
* numpy (for regression tests)
* matplotlib (for regression tests)

# Installation

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
    cmake -DKokkos_ENABLE_OPENMP=On -DCMAKE_CXX_COMPILER=icpc -DKokkos_ARCH_SKX=On ../

or to build for NVIDIA V100 GPUs (using `nvcc` compiler for GPU code)

    mkdir build-cuda-v100 && cd build-cuda-v100
    cmake -DKokkos_ENABLE_CUDA=On -DCMAKE_CXX_COMPILER=$(pwd)/../external/Kokkos/bin/nvcc_wrapper -DKokkos_ARCH_VOLTA70=On ../

# Developing/Contributing

Please see the [developer guidelines](CONTRIBUTING.md) for additional information.

# Documentation

Please see the [docs/](docs/README.md) folder for additional documentation on features and
how to use them.

# Contributors

| Name          | Handle                | Team              |
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

