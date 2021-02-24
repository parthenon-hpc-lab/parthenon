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

# Exit on error
set -eE

# Export COMMIT CI
SCRIPT=$(realpath "$0")
SOURCE=$(dirname "$SCRIPT")
GITHUB_APP_PEM="$1"
BUILD_DIR="${2}"
export CI_COMMIT_SHA="$3"
export CI_COMMIT_BRANCH="$4"
CI_JOB_URL="$5"
CI_JOB_TOKEN="$6"
export CI_JOB_TOKEN="$CI_JOB_TOKEN"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"

trap 'catch $? $LINENO' ERR
catch() {
  echo "Error $1 occurred on $2"
  "${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py --status "error" --status-context "Parthenon Metrics App" --status-description "CI failed" --status-url "${CI_JOB_URL}"
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
spack env activate darwin-ppc64le-gcc9-2021-02-08
data=$("${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --get-target-branch --branch "${CI_COMMIT_BRANCH}")

echo "Get target branch or pr"
echo "${data}"

target_branch=$(echo "$data" | grep "Target branch is:" | awk '{print $4}')

data=$("${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --check-branch-metrics-uptodate --branch "${target_branch}")

echo "Check if the target branch contains metrics data that is uptodate"
echo "${data}"

performance_metrics_uptodate=$(echo "$data" | grep "Performance Metrics are uptodate:" | awk '{print $5}')

echo "Performance Metrics uptodate: ${performance_metrics_uptodate}"

if [[ "$performance_metrics_uptodate" == *"False"* ]]; then

  "${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py --status "pending" --status-context "Parthenon Metrics App" --status-description "Target branch ($target_branch) performance metrics are out of date, building and running for latest commit." --status-url "${CI_JOB_URL}"

  # Calculate number of available cores
  export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

  # Before checking out target branch copy the metrics app
  echo "Copying files parthenon_metrics_app.py and app.py to ${SOURCE}/../../../"
  cp "${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py "${SOURCE}"/../../../
  cp "${SOURCE}"/../python/parthenon_metrics_app/app.py "${SOURCE}"/../../../
  ls "${SOURCE}"/../../../
  git checkout "$target_branch"
#  git pull
#  git log --name-status HEAD^..HEAD

  echo "Copying files parthenon_metrics_app.py and app.py to ${SOURCE}/../python/parthenon_metrics_app/"
  cp "${SOURCE}"/../../../parthenon_metrics_app.py "${SOURCE}"/../python/parthenon_metrics_app/
  cp "${SOURCE}"/../../../app.py "${SOURCE}"/../python/parthenon_metrics_app/
  ls "${SOURCE}"/../python/parthenon_metrics_app/

  source /projects/parthenon-int/parthenon-project/.bashrc
  cmake -S. -Bbuild

  cmake --build build

  cd build

  "${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Running tests for target branch ($target_branch)" --status-url "${CI_JOB_URL}"
  ctest --output-on-failure -R performance

  "${SOURCE}"/../python/parthenon_metrics_app/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --branch "$target_branch" --target-branch "$target_branch" --analyze "${BUILD_DIR}/tst/regression/outputs" --create
fi
