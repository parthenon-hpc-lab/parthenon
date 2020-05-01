# CMake build flags

CMake build options.

| Flag                     | Default |
| ------------------------ | ------- |
| ENABLE_UNIT_TESTS        | ON      |
| ENABLE_INTEGRATION_TESTS | ON      |
| ENABLE_REGRESSION_TESTS  | ON      |
| BUILD_EXAMPLES           | ON      | 
| DISABLE_MPI              | OFF     |
| DISABLE_OPENMP           | OFF     |
| DISABLE_HDF5             | OFF     |
| ENABLE_COMPILER_WARNINGS | OFF     |
| CHECK_REGISTRY_PRESSURE  | OFF     |
| TEST_INTEL_OPTIMIZATION  | OFF     |

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
