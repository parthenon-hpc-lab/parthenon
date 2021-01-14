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
spack env activate darwin-ppc64le-gcc9-2021-01-13

PARTHENON_PEM_FILE=${1}
echo "PARTHENON_PEM_FILE is $PARTHENON_PEM_FILE"
BUILD_DIR=${2}
export CI_COMMIT_SHA=${3}
#FILE_TO_UPLOAD=${2}
touch file_test.txt
./scripts/python/parthenon_metrics_app.py -p ${PARTHENON_PEM_FILE} --analyze "${BUILD_DIR}/tst/regression/outputs"
