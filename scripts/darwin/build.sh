#!/bin/bash


# Load system env only
source /etc/bashrc
source /etc/profile

# Make sure home is pointing to current directory
export HOME=$(pwd)
# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

COMPILER_MODULE=${14}
MPI_MODULE=${16}

export TMPDIR=${HOME}/tmp

# Temp directory apparently needed by spack because of OSError: [Errno 18] Invalid cross-device link
if [ -d ${TMPDIR} ] 
then
  echo "Removing ${TMPDIR}"
  rm -rf ${TMPDIR}
fi
echo "Creating tmp directory ${TMPDIR}"
mkdir ${TMPDIR}

compiler_version=$(bash $HOME/scripts/darwin/get_version.sh $COMPILER_MODULE)
compiler_package=$(bash $HOME/scripts/darwin/get_package.sh $COMPILER_MODULE)
mpi_version=$(bash $HOME/scripts/darwin/get_version.sh $MPI_MODULE)
mpi_package=$(bash $HOME/scripts/darwin/get_package.sh $MPI_MODULE)

wrapper_compiler=$(bash $HOME/scripts/darwin/get_cpp.sh $compiler_package)
export NVCC_WRAPPER_DEFAULT_COMPILER=${wrapper_compiler}

# Load system modules
module purge
module load ${12} # cmake
module load ${13} # clang for formatting
module load $COMPILER_MODULE # gcc
module load $MPI_MODULE # mpi
module load ${15} # cuda

# Initialize spack env
. spack/share/spack/setup-env.sh

spack env activate ci

# Find compilers
spack -dd compiler find

spack -dd install py-numpy
spack install py-matplotlib

# Load Spack Modules

spack load hdf5@1.10.6%${compiler_package}@${compiler_version} \
  ^${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}

spack load py-h5py ^hdf5@1.10.6%${compiler_package}@${compiler_version} \
  ^${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}

spack load py-matplotlib
spack load py-numpy

# Setup build env
export OMP_PROC_BIND=close
export CTEST_OUTPUT_ON_FAILURE=1

# Build
if [ -d $1 ] 
then
  echo "Removing $1"
  rm -rf $1/*
  rmdir $1
fi
echo "Creating build folder $1"
mkdir $1 
cd $1 

# exit when any of the following commands fail
set -e

# Display build command
echo "cmake \
 -DCMAKE_BUILD_TYPE=$2 \
 -DCMAKE_CXX_COMPILER=$3 \
 -DKokkos_ARCH_POWER9=$4 \
 -DKokkos_ARCH_VOLTA70=$5 \
 -DKokkos_ENABLE_CUDA=$6 \
 -DKokkos_ENABLE_CUDA_UVM=$7 \
 -DKokkos_ENABLE_OPENMP=$8 \
 -DNUM_MPI_PROC_TESTING=${9} \
 -DOMP_NUM_THREADS=${10} \
 -DPARTHENON_DISABLE_HDF5=${11} ../"

cmake \
 -DCMAKE_BUILD_TYPE=$2 \
 -DCMAKE_CXX_COMPILER=$3 \
 -DKokkos_ARCH_POWER9=$4 \
 -DKokkos_ARCH_VOLTA70=$5 \
 -DKokkos_ENABLE_CUDA=$6 \
 -DKokkos_ENABLE_CUDA_UVM=$7 \
 -DKokkos_ENABLE_OPENMP=$8 \
 -DNUM_MPI_PROC_TESTING=${9} \
 -DOMP_NUM_THREADS=${10} \
 -DPARTHENON_DISABLE_HDF5=${11} ../

make -j $J VERBOSE=1

ctest --output-on-failure -j $J -LE 'coverage' -L 'performance|regression'
 
 
