#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh

# Exit on error
set -eE

# Export COMMIT CI
SCRIPT=$(realpath "$0")
SOURCE=$(dirname "$SCRIPT")
GITHUB_APP_PEM="$1"
export CI_COMMIT_SHA="$2"
export CI_COMMIT_BRANCH="$3"
CI_JOB_URL="$4"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"

trap 'catch $? $LINENO' ERR
catch() {
  echo "Error $1 occurred on $2"
  "${SOURCE}"/../python/parthenon_metrics_app.py --status "error" --status-context "Parthenon Metrics App" --status-description "CI failed" --status-url "${CI_JOB_URL}"
  wait
  echo "BUILD FAILED"
  # For some reason the ERR variable is not recongized as a numeric type when quoted, 
  # exit requires err be recognized as a numeric type, furthermore the quotes
  # are needed as good practice
  exit "$(($ERR))"
}

module load gcc/9.3.0
spack compiler find
spack env activate darwin-ppc64le-gcc9-2021-02-08
echo "build fast $GITHUB_APP_PEM"
"${SOURCE}"/../python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Building parthenon" --status-url "${CI_JOB_URL}"

# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

source /projects/parthenon-int/parthenon-project/.bashrc
cmake -S. -Bbuild

cmake --build build

cd build

"${SOURCE}"/../python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Running tests" --status-url "${CI_JOB_URL}"
ctest --output-on-failure -R performance

