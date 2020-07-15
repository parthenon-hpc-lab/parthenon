#!/bin/bash

# Error functions
check_for_failure() {
  if [ $1 -ne 0 ]; then
    echo $2
    exit 1
  fi
}

# Load system env only
source /etc/bashrc
source /etc/profile

# Make sure home is pointing to current directory
export HOME=$(pwd)

# Load system modules
module purge
module load cmake/3.17.0
module load gcc/6.4.0
module load clang/8.0.0
module load openmpi/p9/4.0.0-gcc_6.4.0
module load cuda/10.1

#GPLUSPLUS_PATH=$(which g++)
#export NVCC_WRAPPER_DEFAULT_COMPILER=$GPLUSPLUS_PATH

echo "Printing build env"
env
pwd
ls

# Initialize spack env
. spack/share/spack/setup-env.sh

# Find compilers
spack compiler find

# Setup build env
export OMP_PROC_BIND=close
export CTEST_OUTPUT_ON_FAILURE=1
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

# Build
rm -rf build-power9_gcc_mpi/*
mkdir build-power9_gcc_mpi
cd build-power9_gcc_mpi

cmake \
 -DKokkos_ENABLE_CUDA=ON \
 -DKokkos_ENABLE_OPENMP=ON \
 -DKokkos_ARCH_POWER9=ON \
 -DKokkos_ARCH_VOLTA70=ON \
 -DKokkos_ENABLE_CUDA_UVM=OFF \
 -DKOKKOS_ENABLE_CXX11=On \
 -DCMAKE_BUILD_TYPE="Debug" \
 -DCMAKE_CXX_COMPILER=$(pwd)/../external/Kokkos/bin/nvcc_wrapper \
 -DPARTHENON_DISABLE_HDF5=ON ../
check_for_failure $? "CMake failed!"

make  VERBOSE=1
check_for_failure $? "Make failed!"

ctest -j 4 -LE 'performance|regression'
check_for_failure $? "Tests failed!"
 
 
