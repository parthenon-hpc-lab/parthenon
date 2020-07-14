#!/bin/bash

# Error functions
check_for_failure() {
  if [ $1 -neq 0 ]; then
    echo $2
    exit 1
  fi
}

# Load system env only
source /etc/bashrc
source /etc/profile

echo "Printing build env"
env
pwd
ls

# Load system modules
module purge
module load cmake/3.17.0
module load gcc/7.4.0
module load clang/8.0.1
module load openmpi/p9/4.0.2-gcc_7.4.0
module load cuda/10.1

# Initialize spack env
. spack/share/spack/setup-env.sh

# Find compilers
spack compiler find

# Load spack modules
spack load hdf5%gcc@7.4.0

# Setup build env
export OMP_PROC_BIND=close
export CTEST_OUTPUT_ON_FAILURE=1
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

# Build
mkdir build-cuda-power9
cd build-cuda-power9

cmake \
 -DKokkos_ENABLE_CUDA=ON \
 -DKokkos_ENABLE_OPENMP=ON \
 -DKokkos_ARCH_POWER9=ON \
 -DKokkos_ARCH_VOLTA70=ON \
 -DKokkos_ENABLE_CUDA_UVM=OFF \
 -DCMAKE_CXX_COMPILER=$(pwd)/../external/Kokkos/bin/nvcc_wrapper \
 -DCMAKE_BUILD_TYPE="Debug" ../
check_for_failure $? "CMake failed!"

make -j 4 VERBOSE=1
check_for_failure $? "Make failed!"

ctest -j 4 -LE 'performance|regression'
check_for_failure $? "Tests failed!"
 
 
