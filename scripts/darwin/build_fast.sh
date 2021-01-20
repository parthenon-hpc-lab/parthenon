#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh

# Exit on error
set -eE

# Export COMMIT CI
#export CI_COMMIT_SHA="b4cdf92c76df3b9b89c705473f2e7dd6f2476895"

trap 'catch $? $LINENO' ERR
catch() {
  echo "Error $1 occurred on $2"
  ./scripts/python/parthenon_metrics_app.py --status "error"
  exit $ERR
}

GITHUB_APP_PEM="$1"
export CI_COMMIT_SHA=$2
export GITHUB_APP_PEM="$GITHUB_APP_PEM"
module load gcc/9.2.0
spack compiler find
spack env activate darwin-ppc64le-gcc9-2021-01-20
echo "build fast $GITHUB_APP_PEM"
./scripts/python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}"  --status "pending"

# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

source /projects/parthenon-int/parthenon-project/.bashrc
cmake -S. -Bbuild

cmake --build build

cd build

ctest --output-on-failure -R performance

exit 0
