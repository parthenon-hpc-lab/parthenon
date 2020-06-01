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

   |           Option           | Default | Type   | Description |
   | -------------------------: | :------ | :----- | :---------- |
   |           PARTHENON_NGHOST | 2       | String | Number of ghost cells | 
   | PARTHENON_SINGLE_PRECISION | OFF     | Option | Enable single precision mode if requested |
   |     PARTHENON_DISABLE_HDF5 | OFF     | Option | HDF5 is enabled by default if found, set this to True to disable HDF5 |
   |      PARTHENON_DISABLE_MPI | OFF     | Option | MPI is enabled by default if found, set this to True to disable MPI |
   |   PARTHENON_DISABLE_OPENMP | OFF     | Option | OpenMP is enabled by default if found, set this to True to disable OpenMP |
   |   ENABLE_COMPILER_WARNINGS | OFF     | Option | Enable compiler warnings |
   |        TEST_ERROR_CHECKING | OFF     | Option | Enables the error checking unit test. This test will FAIL |
   |    TEST_INTEL_OPTIMIZATION | OFF     | Option | Test intel optimization and vectorization |
   |    CHECK_REGISTRY_PRESSURE | OFF     | Option | Check the registry pressure for Kokkos CUDA kernels |
   |              BUILD_TESTING | ON      | Option | Multi-testing enablement |
   |   ENABLE_INTEGRATION_TESTS | ${BUILD_TESTING} | Option | Enable integration tests |
   |    ENABLE_REGRESSION_TESTS | ${BUILD_TESTING} | Option | Enable regression tests |
   |          ENABLE_UNIT_TESTS | ${BUILD_TESTING} | Option | Enable unit tests |

   Note: Options that are prefixed with PARTHENON_ modify behavior.
