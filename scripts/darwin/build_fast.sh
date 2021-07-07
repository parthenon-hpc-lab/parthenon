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

# This script is responsible for building and executing tests on parthenon.
#
# There are two possiblities that the script accounts for.
#
# 1. If the CI is being run for a pull request it may be necessary to generate
#    performance metrics for the target branch (the branch you are merging into) if no
#    performance metrics exist for that branch that are up to date. In that scenario,
#    the variable {BUILD_TARGET} will be set to on.
#
# 2. The second scenario is for simply building and running tests for current commit and
#    branch with no need to switch branches.


if [[ "${BUILD_TARGET}" == "ON" ]]; then

  # Here the metrics app will find out from the github server what the target branch is,
  # this is assuming that there is an open pull request for the current branch given by
  # the variable {CI_COMMIT_BRANCH}
  data=$("${METRICS_APP}" -p "${GITHUB_APP_PEM}" --get-target-branch --branch "${CI_COMMIT_BRANCH}")

  # In the case that we have residual wiki files from a failed build we are cleaning them out to avoid errors
  echo "Clean out old parthenon_wiki if it exists"
  if [[ -d "${SOURCE}/../parthenon.wiki" ]]; then
    rm -rf "${SOURCE}/../parthenon.wiki"
  fi

  echo "Get target branch or pr"
  echo "${data}"

  target_branch=$(echo "$data" | grep "Target branch is:" | awk '{print $4}')

  # Here we are checking to see if the metrics for the target branch are up to date.
  # The metrics app wil do this by locally cloneing the remote wiki and searching for
  # a performance metrics file in the wiki that should be associated with the target
  # branch. It will then look to see if those metrics exist for the latest commit.
  data=$("${METRICS_APP}" -p "${GITHUB_APP_PEM}" --check-branch-metrics-uptodate --branch "${target_branch}")

  echo "Check if the target branch contains metrics data that is uptodate"
  echo "${data}"

  performance_metrics_uptodate=$(echo "$data" | grep "Performance Metrics are uptodate:" | awk '{print $5}')

  echo "Performance Metrics uptodate: ${performance_metrics_uptodate}"

  # If the metrics are up to date there is no reason to do anything, but if they are
  # not we need to run performance tests on the target branch and update them.
  if [[ "$performance_metrics_uptodate" == *"False"* ]]; then

    "${METRICS_APP}" --status "pending" --status-context "Parthenon Metrics App" --status-description "Target branch ($target_branch) performance metrics are out of date, building and running for latest commit." --status-url "${CI_JOB_URL}"

    # Calculate number of available cores
    export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

    ls "${SOURCE}"/../../../
    # Switching the local parthenon repository to the target branch building it
    git checkout "$target_branch"
    git pull origin "$target_branch"
    git log --name-status HEAD^..HEAD

    cmake -S. -Bbuild -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

    cmake --build build

    cd build

    "${METRICS_APP}" -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Running tests for target branch ($target_branch)" --status-url "${CI_JOB_URL}"

    # Here we are just running the performance tests without SpaceInstances - (that test was not stable)
    ctest --output-on-failure -R performance -E "SpaceInstances"

    # The command below is now simply updating the performance metrics file for the target branch
    # and then pushing the results to the github server, so that the remote parthenon wiki now has
    # an up todate copy of the performance metrics for the target branch.
    "${METRICS_APP}" -p "${GITHUB_APP_PEM}" --branch "$target_branch" --target-branch "$target_branch" --analyze "${BUILD_DIR}/tst/regression/outputs" --create
  fi

else

  # Here we are doing the same thing as the above logic except there is no need to switch
  # branches. Secondly, uploading and analzyses for the current branch is handled in this
  # case in a separate script.
  echo "build fast $GITHUB_APP_PEM"
   "${METRICS_APP}" -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Building parthenon" --status-url "${CI_JOB_URL}"

  # Calculate number of available cores
  export J=$(( $(nproc --all) )) && echo Using ${J} cores during build

  cmake -S. -Bbuild -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

  cmake --build build

  cd build

  "${METRICS_APP}" -p "${GITHUB_APP_PEM}"  --status "pending" --status-context "Parthenon Metrics App" --status-description "Running tests" --status-url "${CI_JOB_URL}"

  ctest --output-on-failure  -E "SpaceInstances"

fi
