# Building Parthenon on Various Systems

## Ubuntu 20.04 LTS

The following procedure has been tested for an Ubuntu 20.04 LTS system:

```bash
# install dependencies
# openmpi is installed implicitly by the hdf5 install
sudo apt-get update
install cmake build-essentials libhdf5-openmpi-dev

# Clone parthenon, with submodules
git clone --recursive https://github.com/lanl/parthenon.git
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

### List of cmake options:

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
   |          ENABLE\_UNIT\_TESTS | ${BUILD\_TESTING} | Option | Enable unit tests |
   |PARTHENON\_USE\_LIGHTWEIGHT\_HINT | OFF | Option | Decorate default parallel region wrappers to be lightweight kernels |
   |               CODE\_COVERAGE | OFF      | Option | Builds with code coverage flags |

### NB: CMake options prefixed with *PARTHENON\_* modify behavior.
