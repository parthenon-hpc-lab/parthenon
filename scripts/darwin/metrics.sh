#!/bin/bash

# Load system env only
#source /etc/bashrc
#source /etc/profile
#source /projects/parthenon-int/parthenon-project/.bashrc
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh

# Exit on error
set -e

module load gcc/9.2.0
spack compiler find
spack env activate darwin-ppc64le-gcc9-2021-01-20

SOURCE=${BASH_SOURCE[0]}
GITHUB_APP_PEM="${1}"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"
echo "GITHUB_APP_PEM is $GITHUB_APP_PEM"
BUILD_DIR=${2}
export CI_COMMIT_SHA=${3}
export CI_COMMIT_BRANCH=${4}
#FILE_TO_UPLOAD=${2}
echo "Metrics PEM file $GITHUB_APP_PEM"
touch file_test.txt
${SOURCE}/../python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --analyze "${BUILD_DIR}/tst/regression/outputs"
