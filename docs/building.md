# Building Parthenon on Various Systems

*IMPORTANT: We try our best to keep the instructions up-to-date.
However, Parthenon itself, dependencies, and environments constantly changes so that the instruction may not work any more.
If you come across a disfunctional setup, please report it by open an issue or propose an updated description in a pull request*

## General list of cmake options:

   |           Option             | Default  | Type   | Description |
   | ---------------------------: | :------- | :----- | :---------- |
   |            PARTHENON\_NGHOST | 2        | String | Number of ghost cells | 
   | PARTHENON\_SINGLE\_PRECISION | OFF      | Option | Enable single precision mode if requested |
   |     PARTHENON\_DISABLE\_HDF5 | OFF      | Option | HDF5 is enabled by default if found, set this to True to disable HDF5 |
   |      PARTHENON\_DISABLE\_MPI | OFF      | Option | MPI is enabled by default if found, set this to True to disable MPI |
   |   PARTHENON\_DISABLE\_OPENMP | OFF      | Option | OpenMP is enabled by default if found, set this to True to disable OpenMP |
   |   ENABLE\_COMPILER\_WARNINGS | OFF      | Option | Enable compiler warnings |
   |        TEST\_ERROR\_CHECKING | OFF      | Option | Enables the error checking unit test. This test will FAIL |
   |    TEST\_INTEL\_OPTIMIZATION | OFF      | Option | Test intel optimization and vectorization |
   |    CHECK\_REGISTRY\_PRESSURE | OFF      | Option | Check the registry pressure for Kokkos CUDA kernels |
   |               BUILD\_TESTING | ON       | Option | Multi-testing enablement |
   | PARTHENON\_DISABLE\_EXAMPLES | OFF      | Option | Toggle building of examples, if regression tests are on, drivers needed by the tests will still be built |   
   |   ENABLE\_INTEGRATION\_TESTS | ${BUILD\_TESTING} | Option | Enable integration tests |
   |    ENABLE\_REGRESSION\_TESTS | ${BUILD\_TESTING} | Option | Enable regression tests |
   |  REGRESSION\_GOLD\_STANDARD\_VER | #     | Int    | Version of current gold standard file used in regression tests. Default is set to latest version matching the source. |
   | REGRESSION\_GOLD\_STANDARD\_HASH | SHA512=... | String | Hash value of gold standard file to be downloaded. Used to ensure that the download is not corrupted. |
   | REGRESSION\_GOLD\_STANDARD\_SYNC | ON    | Option | Create `gold_standard` target to download gold standard files |
   |          ENABLE\_UNIT\_TESTS | ${BUILD\_TESTING} | Option | Enable unit tests |
   |               CODE\_COVERAGE | OFF      | Option | Builds with code coverage flags |

### NB: CMake options prefixed with *PARTHENON\_* modify behavior.

## System specific instructions

Common first step: Obtain the Parthenon source including external dependencies (mostly Kokkos)

```bash
# Clone parthenon, with submodules
git clone --recursive https://github.com/lanl/parthenon.git
export PARTHENON_ROOT=$(pwd)/parthenon
```
We set the latter variable for easier reference in out-of-source builds.

### Ubuntu 20.04 LTS

The following procedure has been tested for an Ubuntu 20.04 LTS system:

```bash
# install dependencies
# openmpi is installed implicitly by the hdf5 install
sudo apt-get update
install cmake build-essentials libhdf5-openmpi-dev

# make a bin directory
mkdir bin
cd bin
# configure and build
cmake ..
cmake -j --build .
# run unit and regression tests
ctest -LE performance
# run performance tests
ctest -L performance
```

### OLCF Summit (Power9+Volta)

Last verified 7 Aug 2020.

#### Common environment

```bash
# setup environment
$ module restore system
$ module load cuda gcc cmake/3.14.2 python hdf5

# on 7 Aug 2020 that results the following version
$ module list

Currently Loaded Modules:
  1) hsi/5.0.2.p5    4) darshan-runtime/3.1.7   7) gcc/6.4.0                       10) hdf5/1.10.4
  2) xalt/1.2.0      5) DefApps                 8) cmake/3.14.2                    11) python/3.6.6-anaconda3-5.3.0
  3) lsf-tools/2.0   6) cuda/10.1.243           9) spectrum-mpi/10.3.1.2-20200121
```

#### Cuda with MPI

```bash
# configure and build. Make sure to build in an directory on the GPFS filesystem as the home directory is not writeable from the compute nodes (which will result in the regression tests failing)
$ mkdir build-cuda-mpi && cd build-cuda-mpi
$ export OMPI_CXX=${PARTHENON_ROOT}/external/Kokkos/bin/nvcc_wrapper
$ cmake -DKokkos_ARCH_POWER9=ON -DKokkos_ARCH_VOLTA70=True -DKokkos_ENABLE_CUDA=True -DKokkos_ENABLE_OPENMP=True -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpic++ ${PARTHENON_ROOT}
$ make -j10

# run all tests (assumes running within a job)
# "-gpu" is required to enable Cuda aware MPI support
$ jsrun -n 1 -g 1 --smpiargs="-gpu" ctest
```

### Cuda without MPI

```bash
# configure and build
$ mkdir build-cuda && cd build-cuda
$ cmake -DKokkos_ARCH_POWER9=ON -DKokkos_ARCH_VOLTA70=True -DKokkos_ENABLE_CUDA=True -DKokkos_ENABLE_OPENMP=True -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${PARTHENON_ROOT}/external/Kokkos/bin/nvcc_wrapper -DPARTHENON_DISABLE_MPI=On ${PARTHENON_ROOT}
$ make -j10

# run all tests (assumes running within a job)
# - jsrun is required as the test would otherwise be executed on the scheduler node rather than on a compute node
# - "off" is required as otherwise the implicit PAMI initialization would fail
$ jsrun -n 1 -g 1 --smpiargs="off" ctest
```


