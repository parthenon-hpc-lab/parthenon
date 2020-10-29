#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile

# Make sure home is pointing to current directory
export PARTHENON=$(pwd)
cd ../
export HOME=$(pwd)
cd ${PARTHENON}
# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

COMPILER_MODULE=$1
MPI_MODULE=$2

compiler_version=$(bash $PARTHENON/scripts/darwin/get_version.sh $COMPILER_MODULE)
compiler_package=$(bash $PARTHENON/scripts/darwin/get_package.sh $COMPILER_MODULE)
mpi_version=$(bash $PARTHENON/scripts/darwin/get_version.sh $MPI_MODULE)
mpi_package=$(bash $PARTHENON/scripts/darwin/get_package.sh $MPI_MODULE)

# Load system modules
module purge
module load $COMPILER_MODULE # gcc
module load $MPI_MODULE # mpi

# Initialize spack env
. ../spack/share/spack/setup-env.sh

spack env activate ci

# Find compilers
spack compiler find

#spack add hdf5@1.10.6%${compiler_package}@${compiler_version} \
#  ^${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}

# Install hdf5
spack install py-h5py@2.10.0 ^hdf5@1.10.6%${compiler_package}@${compiler_version} \
  ^${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}

spack add py-numpy
spack add py-matplotlib

spack concretize -f

