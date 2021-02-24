#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/.bashrc

# Exit on error
set -e
# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

cmake -S. -Bbuild

cmake --build build

cd build

ctest --output-on-failure

exit 0
