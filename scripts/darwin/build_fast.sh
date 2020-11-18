#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/.bashrc

function exit_on_error() {
  local error_code=$?
  if [[ $error_code != 0 ]]; then
    exit 1
  fi
}

set -e
# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

cmake -S. -Bbuild
#exit_on_error

cmake --build build
#exit_on_error

cd build

ctest 
#exit_on_error

exit 0
