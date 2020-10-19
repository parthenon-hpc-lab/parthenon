#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile

# Make sure home is pointing to current directory
export HOME=$(pwd)

cd ../
# Download spack
if [ -d "spack" ]; then
  rm -rf spack
fi
git clone --branch v0.14.2 https://github.com/spack/spack.git

# Initialize spack env
. spack/share/spack/setup-env.sh

spack env create ci
spack env activate ci

# Create .spack folder
if [ -d ".spack" ]; then
  rm -rf .spack
fi
mkdir -p .spack

COMPILER_MODULE=$1
MPI_MODULE=$2

compiler_package=$(bash $HOME/scripts/darwin/get_package.sh $COMPILER_MODULE)
compiler_version=$(bash $HOME/scripts/darwin/get_version.sh $COMPILER_MODULE)
mpi_package=$(bash $HOME/scripts/darwin/get_package.sh $MPI_MODULE)
mpi_version=$(bash $HOME/scripts/darwin/get_version.sh $MPI_MODULE)

# Setup spack package yaml
echo "packages:" > $HOME/.spack/packages.yaml
echo "  python:" >> $HOME/.spack/packages.yaml
echo "    version: ['3:']" >> $HOME/.spack/packages.yaml
echo "  openmpi:" >> $HOME/.spack/packages.yaml
echo "    modules:" >> $HOME/.spack/packages.yaml
echo "      ${mpi_package}@${mpi_version}: $MPI_MODULE" >> $HOME/.spack/packages.yaml

cd $HOME
