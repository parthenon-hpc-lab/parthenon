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

## LANL Darwin - Power 9

The following works on the Darwin Power 9 nodes. We use Spack to install `hdf5` with the `xl`
compiler. Setting up Spack is beyond the scop of this documentation.

```bash
# HDF5 is optional, but recommended
spack install hdf5%xl

# load required modules and set environment
module purge

module load cmake/3.17.0
module load ibm/xlc-16.1.1.7-xlf-16.1.1.7
module load openmpi/p9/4.0.3-xlc_16.1.1.7-xlf_16.1.1.7
module load gcc/8.3.0
module load clang/8.0.1
module load cuda/10.2

spack load hdf5%xl

export NVCC_WRAPPER_DEFAULT_COMPILER=xlc++

# clone parthenon with submodules
git clone --recursive https://github.com/lanl/parthenon.git

# make a build directory
mkdir build
cd build

# configure and build (if hdf5 is not available, add -DDISABLE_HDF5=ON)
CXX=`pwd`/../cmake/kokkos_nvcc_wrapper CC=xlc cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_POWER9=ON -DKokkos_ARCH_VOLTA70=ON ..
make -j
make test
```
