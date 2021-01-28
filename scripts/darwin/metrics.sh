#!/bin/bash

# Load system env only
source /etc/bashrc
source /etc/profile
source /projects/parthenon-int/parthenon-project/.bashrc
source /projects/parthenon-int/parthenon-project/spack/share/spack/setup-env.sh

# Exit on error
set -e

module load gcc/9.2.0
spack compiler find
spack env activate darwin-ppc64le-gcc9-2021-01-20

SCRIPT=`realpath $0`
SOURCE=`dirname $SCRIPT`
GITHUB_APP_PEM="${1}"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"
echo "GITHUB_APP_PEM is $GITHUB_APP_PEM"
BUILD_DIR="${2}"
export CI_COMMIT_SHA="${3}"
export CI_COMMIT_BRANCH="${4}"
echo "Metrics PEM file $GITHUB_APP_PEM"

if [[ "develop" == "${CI_COMMIT_BRANCH}" ]]; then
  # This is for the case where we are running on a schedule
  target_branch="${CI_COMMIT_BRANCH}"
else
  data=$(${SOURCE}/../python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --get-target-branch --branch "${CI_COMMIT_BRANCH}")

  echo "Get target branch or pr"
  echo ${data}

  target_branch=$(echo "$data" | grep "Target branch is:" | awk '{print $4}')
fi

${SOURCE}/../python/parthenon_metrics_app.py -p "${GITHUB_APP_PEM}" --analyze "${BUILD_DIR}/tst/regression/outputs" --create --post-analyze-status --branch "${CI_COMMIT_BRANCH}" --target-branch "$target_branch" --generate-figures-on-analysis
