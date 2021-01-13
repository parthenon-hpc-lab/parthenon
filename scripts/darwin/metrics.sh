#!/bin/bash

# Load system env only
#source /etc/bashrc
#source /etc/profile
#source /projects/parthenon-int/parthenon-project/.bashrc
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh

# Exit on error
set -e

spack env activate darwin-ppc64le-gcc9-2021-01-13

PARTHENON_PEM_FILE=${1}
#FILE_TO_UPLOAD=${2}
touch file_test.txt
./scripts/python/parthenon_metrics_app.py -p ${PARTHENON_PEM_FILE} --status pending
