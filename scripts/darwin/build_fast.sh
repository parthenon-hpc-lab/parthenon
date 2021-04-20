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
SCRIPT=$(realpath "$0")
SOURCE=$(dirname "$SCRIPT")
source "${SOURCE}/setup.sh"

if [[ "${BUILD_TARGET}" == "ON" ]]; then

  data=$("${METRICS_APP}" -p "${GITHUB_APP_PEM}" --get-target-branch --branch "${CI_COMMIT_BRANCH}")

  echo "Clean out old parthenon_wiki if it exists"
  if [[ -d "${SOURCE}/../parthenon.wiki" ]]; then
    rm -rf "${SOURCE}/../parthenon.wiki"
  fi

  echo "Get target branch or pr"
  echo "${data}"

  target_branch=$(echo "$data" | grep "Target branch is:" | awk '{print $4}')

  data=$("${METRICS_APP}" -p "${GITHUB_APP_PEM}" --check-branch-metrics-uptodate --branch "${target_branch}")

  echo "Check if the target branch contains metrics data that is uptodate"
  echo "${data}"

  performance_metrics_uptodate=$(echo "$data" | grep "Performance Metrics are uptodate:" | awk '{print $5}')

  echo "Performance Metrics uptodate: ${performance_metrics_uptodate}"

  if [[ "$performance_metrics_uptodate" == *"False"* ]]; then

    "${METRICS_APP}" --status "pending" --status-context "Parthenon Metrics App" --status-description "Target branch ($target_branch) performance metrics are out of date, building and running for latest commit." --status-url "${CI_JOB_URL}"

    # Calculate number of available cores
    export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

    ls "${SOURCE}"/../../../
    git checkout "$target_branch"
    git pull origin "$target_branch"
    git log --name-status HEAD^..HEAD

    cmake -S. -Bbuild -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

    cmake --build build

    cd build

    "${METRICS_APP}" -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Running tests for target branch ($target_branch)" --status-url "${CI_JOB_URL}"
    ctest --output-on-failure -R performance

    "${METRICS_APP}" -p "${GITHUB_APP_PEM}" --branch "$target_branch" --target-branch "$target_branch" --analyze "${BUILD_DIR}/tst/regression/outputs" --create
  fi

else

  echo "build fast $GITHUB_APP_PEM"
   "${METRICS_APP}" -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Building parthenon" --status-url "${CI_JOB_URL}"

  # Calculate number of available cores
  export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

  cmake -S. -Bbuild -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

  cmake --build build

  cd build

  "${METRICS_APP}" -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Running tests" --status-url "${CI_JOB_URL}"
  ctest --output-on-failure

fi
