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
# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

# Load system modules
module purge
module load cmake/3.11.1
module load gcc/7.4.0
module load clang/8.0.0
module load openmpi/p9/4.0.2-gcc_7.4.0
module load cuda/10.1

# Initialize spack env
. spack/share/spack/setup-env.sh

# Find compilers
spack compiler find

spack install py-numpy
spack install py-matplotlib

# Load Spack Modules
spack load hdf5@1.10.6%gcc@7.4.0 ^openmpi@4.0.2%gcc@7.4.0
spack load py-h5py ^hdf5@1.10.6%gcc@7.4.0 ^openmpi@4.0.2%gcc@7.4.0
spack load py-matplotlib
spack load py-numpy

# Setup build env
export OMP_PROC_BIND=close
export CTEST_OUTPUT_ON_FAILURE=1

# Build
if [ -d $1 ] 
then
  echo "exists"
  #rm -rf $1/*
fi
mkdir $1 
cd $1 

cmake \
 -DCMAKE_BUILD_TYPE=$2 \
 -DCMAKE_CXX_COMPILER=$3 \
 -DKokkos_ARCH_POWER9=$4 \
 -DKokkos_ARCH_VOLTA70=$5 \
 -DKokkos_ENABLE_CUDA=$6 \
 -DKokkos_ENABLE_CUDA_UVM=$7 \
 -DKokkos_ENABLE_CXX11=$8 \
 -DKokkos_ENABLE_OPENMP=$9 \
 -DNUM_MPI_PROC_TESTING=${10} \
 -DOMP_NUM_THREADS=${11} \
 -DPARTHENON_DISABLE_HDF5=${12} ../
check_for_failure $? "CMake failed!"

make -j $J VERBOSE=1
check_for_failure $? "Make failed!"

ctest --output-on-failure -j $J -L 'performance|regression'
check_for_failure $? "Tests failed!"
 
 
