# parthenon

[![codecov](https://codecov.io/gh/lanl/parthenon/branch/master/graph/badge.svg)](https://codecov.io/gh/lanl/parthenon)

Parthenon AMR infrastructure

# Community
* [Matrix](https://riot.im/app/#/room/#parthenon-general:matrix.org)

# Required Dependencies

* CMake 3.10 or greater
* gcc or intel compiler
* mpi
* openMP
* hdf5
* kokkos

# Dependencies Other

* catch2
* python3
* h5py
* numpy
* matplotlib

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
    cmake -DKokkos_ENABLE_CUDA=On -DCMAKE_CXX_COMPILER=$(pwd)/../external/kokkos/bin/nvcc_wrapper -DKokkos_ARCH_VOLTA70=On ../

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

