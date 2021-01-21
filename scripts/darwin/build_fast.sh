#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh

# Exit on error
set -eE

# Export COMMIT CI
#export CI_COMMIT_SHA="b4cdf92c76df3b9b89c705473f2e7dd6f2476895"

SOURCE=$(dirname "$0") 
GITHUB_APP_PEM="$1"
export CI_COMMIT_SHA=$2
export CI_COMMIT_BRANCH=$3
CI_JOB_URL="$4"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"

trap 'catch $? $LINENO' ERR
catch() {
  echo "Error $1 occurred on $2"
  ${SOURCE}/../python/parthenon_metrics_app.py --status "error" --status-url "${CI_JOB_URL}"
  exit $ERR
}

module load gcc/9.2.0
spack compiler find
spack env activate darwin-ppc64le-gcc9-2021-01-20
echo "build fast $GITHUB_APP_PEM"
${SOURCE}/../python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Building parthenon" --status-url "${CI_JOB_URL}"

# Calculate number of available cores
export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

source /projects/parthenon-int/parthenon-project/.bashrc
cmake -S. -Bbuild

cmake --build build

cd build

${SOURCE}/../python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Running tests" --status-url "${CI_JOB_URL}"
ctest --output-on-failure -R performance

exit 0
