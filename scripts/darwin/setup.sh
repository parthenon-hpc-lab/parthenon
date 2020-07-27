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

# Create .spack folder
if [ ! -d ".spack" ]; then
  mkdir -p .spack
fi

# Setup spack package yaml
echo "packages:" > .spack/packages.yaml
echo "  python:" >> .spack/packages.yaml
echo "    version: ['3:']" >> .spack/packages.yaml
echo "  openmpi:" >> .spack/packages.yaml
echo "    modules:" >> .spack/packages.yaml
echo "      openmpi@4.0.0: openmpi/p9/4.0.0-gcc_6.4.0" >> .spack/packages.yaml
echo "      openmpi@4.0.2: openmpi/p9/4.0.2-gcc_7.4.0" >> .spack/packages.yaml
