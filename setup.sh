#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile

# Download spack
git clone https://github.com/spack/spack.git

echo $HOME
pwd
ls

# Setup spack package yaml
echo "packages:" > $HOME/.spack/packages.yaml
echo "  openmpi:" >> $HOME/.spack/packages.yaml
echo "    modules:" >> $HOME/.spack/packages.yaml
echo "      openmpi@4.0.2: openmpi/p9/4.0.2-gcc_7.4.0" >> $HOME/.spack/packages.yaml
# Initialize spack env
. spack/share/spack/setup-env.sh

# Install hdf5
spack install hdf5

