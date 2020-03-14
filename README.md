# parthenon

Parthenon AMR infrastructure 

# Required Dependencies

* CMake 3.10 or greater
* gcc or intel compiler
* mpi 
* openMP
* hdf5

# Dependencies Other

* catch2 

# Installation

## Basics

    mkdir build
    cd build
    cmake ../
    make
    make test

## Parallel_for wrapper options

Following options are available to configure the default behavior of the `par_for` wrappers.

- `PAR_LOOP_LAYOUT` (sets default layout)
  - `MANUAL1D_LOOP` maps to `Kokkos::RangePolicy` (default for CUDA backend)
  - `MDRANGE` maps to `Kokkos::MDRangePolicy`
  - `SIMDFOR_LOOP` maps to standard `for` loops with `#pragma omp simd` (default for OpenMP backend)
  - `TPTTRTVR_LOOP` maps to triple nested loop with `Kokkos::TeamPolicy`, `Kokkos::TeamThreadRange` and `Kokkos::ThreadVectorRange`
  - `TP_INNERX_LOOP`
    - maps double nested loop with outer `Kokkos::TeamPolicy`
    - inner loop is set by `PAR_LOOP_LAYOUT_INNER`. Options are
      - `INNER_TTR` for `Kokkos::TeamThreadRange` and
      - `INNER_TVR` for `Kokkos::ThreadVectorRange`
- `PAR_VECTOR_LENGTH` (int; set the vector length used in Kokkos::TeamPolicy based wrappers)

## Kokkos options
Kokkos can be configured through `cmake` options, see https://github.com/kokkos/kokkos/wiki/Compiling

For example to build with the OpenMP backend for Intel Skylake architecture using Intel compilers

    mkdir build-omp-skx && cd build-omp-skx
    cmake -DKokkos_ENABLE_OPENMP=On -DCMAKE_CXX_COMPILER=icpc -DKokkos_ARCH_SKX=On ../

or to build for NVIDIA V100 GPUs (using `nvcc` compiler for GPU code)

    mkdir build-cuda-v100 && cd build-cuda-v100
    cmake -DKokkos_ENABLE_CUDA=On -DCMAKE_CXX_COMPILER=$(pwd)/../external/kokkos/bin/nvcc_wrapper -DKokkos_ARCH_VOLTA70=On -DKokkos_ENABLE_CUDA_LAMBDA=True ../

# Developing/Contributing

Please see the [developer guidelines](CONTRIBUTING.md) for additional information.

# Documentation

Please see the [docs/](docs/README.md) folder for additional documentation on features and
how to use them.
