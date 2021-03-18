#!/bin/bash
#=========================================================================================
# (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================


# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh
source /projects/parthenon-int/parthenon-project/.bashrc

# Exit on error
set -eE

# Export COMMIT CI
GITHUB_APP_PEM="$1"
BUILD_DIR="${2}"
export CI_COMMIT_SHA="$3"
export CI_COMMIT_BRANCH="$4"
CI_JOB_URL="$5"
CI_JOB_TOKEN="$6"

# Should be a value of ON or OFF depending on whether we are
# also running metrics on the pr we want to merge with
BUILD_TARGET="$7"

export CI_JOB_TOKEN="$CI_JOB_TOKEN"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"

METRICS_APP="${SOURCE}/../python/parthenon_metrics_app.py"

trap 'catch $? $LINENO' ERR
catch() {
  echo "Error $1 occurred on $2"
  "${METRICS_APP}" --status "error" --status-context "Parthenon Metrics App" --status-description "CI failed" --status-url "${CI_JOB_URL}"
  wait
  echo "BUILD FAILED"
  # For some reason the ERR variable is not recongized as a numeric type when quoted, 
  # exit requires err be recognized as a numeric type, furthermore the quotes
  # are needed as good practice
  exit "$(($ERR))"
}

echo "CI commit branch ${CI_COMMIT_BRANCH}"

module load gcc/9.3.0
spack compiler find

# Always get the latest spack environment
spack_env_latest=$(spack env list | grep darwin-ppc64le-gcc9 | sort | tail -n 1 | tr -d '[:space:]')
spack env activate "${spack_env_latest}"
