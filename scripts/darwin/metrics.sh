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

if [[ "develop" == "${CI_COMMIT_BRANCH}" ]]; then
  # This is for the case where we are running on a schedule
  target_branch="${CI_COMMIT_BRANCH}"
else
  data=$("${METRICS_APP}" -p "${GITHUB_APP_PEM}" --get-target-branch --branch "${CI_COMMIT_BRANCH}")

  echo "Get target branch or pr"
  echo "${data}"

  target_branch=$(echo "$data" | grep "Target branch is:" | awk '{print $4}')
fi

"${METRICS_APP}" -p "${GITHUB_APP_PEM}" --analyze "${BUILD_DIR}/tst/regression/outputs" --create --post-analyze-status --branch "${CI_COMMIT_BRANCH}" --target-branch "$target_branch" --generate-figures-on-analysis
