#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile

# Load system modules
module purge
module load cmake/3.17.0
module load gcc/7.3.0
module load clang/8.0.0
module load openmpi/p9/4.0.0-gcc_7.3.0
module load cuda/10.1

# Initialize spack env
. spack/share/spack/setup-env.sh

# Load spack modules
spack load hdf5

# Setup build env
export OMP_PROC_BIND=close
export CTEST_OUTPUT_ON_FAILURE=1
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

# Build
cd parthenon
mkdir build-cuda-power9
cd build-cuda-power9

cmake \
 -DKokkos_ENABLE_CUDA=ON \
 -DKokkos_ENABLE_OPENMP=ON \
 -DKokkos_ARCH_POWER9=ON \
 -DKokkos_ARCH_VOLTA70=ON \
 -DKokkos_ENABLE_CUDA_UVM=OFF \
 -DCMAKE_CXX_COMPILER=$(pwd)/../external/Kokkos/bin/nvcc_wrapper \
 -DCMAKE_BUILD_TYPE="Debug" ../ \

make -j${J}

ctest -j${J} -LE 'performance|regression'
 
 
