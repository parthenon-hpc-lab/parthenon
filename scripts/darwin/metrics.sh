#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/.bashrc

# Exit on error
set -e

PARTHENON_PEM_FILE=${1}
#FILE_TO_UPLOAD=${2}
touch file_test.txt
./scripts/python/parthenon_metrics_app.py -p ${PARTHENON_PEM_FILE} --upload file_test.txt --wiki
