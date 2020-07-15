#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile

# Make sure home is pointing to current directory
export HOME=$(pwd)

# Download spack
if [ ! -d "spack" ]; then
  git clone https://github.com/spack/spack.git
fi

echo $HOME
pwd
ls

# Initialize spack env
. spack/share/spack/setup-env.sh

# Create .spack folder
if [ ! -d ".spack" ]; then
  mkdir -p .spack
fi

# Setup spack package yaml
echo "packages:" > .spack/packages.yaml
echo "  openmpi:" >> .spack/packages.yaml
echo "    modules:" >> .spack/packages.yaml
echo "      openmpi@4.0.2: openmpi/p9/4.0.2-gcc_7.4.0" >> .spack/packages.yaml

# Load system modules
module purge
module load cmake/3.11.1
module load gcc/7.4.0
module load clang/8.0.1
module load openmpi/p9/4.0.2-gcc_7.4.0
module load cuda/10.1

# Find compilers
spack compiler find

# Install hdf5
spack install hdf5%gcc@7.4.0 ^openmpi@4.0.2%gcc@7.4.0

