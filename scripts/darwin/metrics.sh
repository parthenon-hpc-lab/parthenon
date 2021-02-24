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
source /projects/parthenon-int/parthenon-project/.bashrc
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh

# Exit on error
set -e

module load gcc/9.2.0
spack compiler find
spack env activate darwin-ppc64le-gcc9-2021-02-08

SCRIPT=$(realpath "$0")
SOURCE=$(dirname "$SCRIPT")
GITHUB_APP_PEM="${1}"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"
echo "GITHUB_APP_PEM is $GITHUB_APP_PEM"
BUILD_DIR="${2}"
export CI_COMMIT_SHA="${3}"
export CI_COMMIT_BRANCH="${4}"
echo "Metrics PEM file $GITHUB_APP_PEM"

trap 'catch $? $LINENO' ERR
catch() {
  echo "Error $1 occurred on $2"
  echo "BUILD FAILED"
  # For some reason the ERR variable is not recongized as a numeric type when quoted, 
  # exit requires err be recognized as a numeric type, furthermore the quotes
  # are needed as good practice
  exit "$(($ERR))"
}

if [[ "develop" == "${CI_COMMIT_BRANCH}" ]]; then
  # This is for the case where we are running on a schedule
  target_branch="${CI_COMMIT_BRANCH}"
else
  data=$("${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --get-target-branch --branch "${CI_COMMIT_BRANCH}")

  echo "Get target branch or pr"
  echo "${data}"

  target_branch=$(echo "$data" | grep "Target branch is:" | awk '{print $4}')
fi

"${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --analyze "${BUILD_DIR}/tst/regression/outputs" --create --post-analyze-status --branch "${CI_COMMIT_BRANCH}" --target-branch "$target_branch" --generate-figures-on-analysis
