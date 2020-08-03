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

# Initialize spack env
. spack/share/spack/setup-env.sh

spack env create ci
sapck env activate ci

# Create .spack folder
if [ ! -d ".spack" ]; then
  mkdir -p .spack
fi

COMPILER_MODULE=$1
MPI_MODULE=$2

compiler_package=$(bash get_package $COMPILER_MODULE)
compiler_version=$(bash get_version $COMPILER_MODULE)
mpi_package=$(bash get_package $MPI_MODULE)
mpi_version=$(bash get_version $MPI_MODULE)

# Setup spack package yaml
echo "packages:" > .spack/packages.yaml
echo "  python:" >> .spack/packages.yaml
echo "    version: ['3:']" >> .spack/packages.yaml
echo "  openmpi:" >> .spack/packages.yaml
echo "    modules:" >> .spack/packages.yaml
echo "      ${mpi_package}@${mpi_version}: $MPI_MODULE" >> .spack/packages.yaml
