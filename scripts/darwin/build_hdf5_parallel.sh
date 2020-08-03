#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile

# Make sure home is pointing to current directory
export HOME=$(pwd)
# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

COMPILER_MODULE=$1
MPI_MODULE=$2

compiler_version=$(bash get_version $COMPILER_MODULE)
compiler_package=$(bash get_package $COMPILER_MODULE)
mpi_version=$(bash get_version $MPI_MODULE)
mpi_package=$(bash get_package $MPI_MODULE)

# Load system modules
module purge
module load $COMPILER_MODULE # gcc
module load $MPI_MODULE # mpi

# Initialize spack env
. spack/share/spack/setup-env.sh

spack env activate ci

# Find compilers
spack compiler find

spack find hdf5

# Install hdf5
spack install -j${J} py-h5py ^hdf5@1.10.6%${compiler_package}@${compiler_version} \
  ^${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}

spack find hdf5

spack load hdf5@1.10.6%${compiler_package}@${compiler_version} \
  ^${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}

spack load py-h5py \
  ^${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}

