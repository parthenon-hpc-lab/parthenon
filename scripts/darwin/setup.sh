#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile

# Make sure home is pointing to current directory
export PARTHENON=$(pwd)
cd ../
export HOME=$(pwd)

ls -a
# Download spack
if [ -d "${HOME}/spack" ]; then
  rm -rf ${HOME}/spack
fi
git clone --branch v0.15.4 https://github.com/spack/spack.git
ls -a
# Initialize spack env
. spack/share/spack/setup-env.sh

spack env create ci
spack env activate ci

# Create .spack folder
if [ -d "${HOME}/.spack" ]; then
  rm -rf ${HOME}/.spack
fi
mkdir -p ${HOME}/.spack

COMPILER_MODULE=$1
MPI_MODULE=$2

compiler_package=$(bash $PARTHENON/scripts/darwin/get_package.sh $COMPILER_MODULE)
compiler_version=$(bash $PARTHENON/scripts/darwin/get_version.sh $COMPILER_MODULE)
mpi_package=$(bash $PARTHENON/scripts/darwin/get_package.sh $MPI_MODULE)
mpi_version=$(bash $PARTHENON/scripts/darwin/get_version.sh $MPI_MODULE)

# Setup spack package yaml
echo "packages:" > ${HOME}/.spack/packages.yaml
echo "  python:" >> ${HOME}/.spack/packages.yaml
echo "    version: ['3:']" >> ${HOME}/.spack/packages.yaml
echo "  openmpi:" >> ${HOME}/.spack/packages.yaml
echo "    externals:" >> ${HOME}/.spack/packages.yaml
echo "    - spec: \"${mpi_package}@${mpi_version}%${compiler_package}@${compiler_version}\"" >> ${HOME}/.spack/packages.yaml
echo "    prefix: $MPI_MODULE" >> ${HOME}/.spack/packages.yaml

ls -a
cd $PARTHENON
