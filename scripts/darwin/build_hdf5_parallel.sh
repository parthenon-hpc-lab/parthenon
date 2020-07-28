#!/bin/bash

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
module load openmpi/p9/4.0.2-gcc_7.4.0

# Initialize spack env
. spack/share/spack/setup-env.sh

# Find compilers
spack compiler find

spack find hdf5
# Install hdf5
#spack install -j${J} hdf5@1.10.6%gcc@7.4.0 ^openmpi@4.0.2%gcc@7.4.0

spack install -j${J} py-h5py ^hdf5@1.10.6%gcc@7.4.0 ^openmpi@4.0.2%gcc@7.4.0

spack find hdf5

spack load hdf5@1.10.6%gcc@7.4.0 ^openmpi@4.0.2%gcc@7.4.0
spack load py-h5py ^openmpi@4.0.2%gcc@7.4.0

