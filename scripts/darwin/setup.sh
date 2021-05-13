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
source "${SOURCE}/base_setup.sh"

METRICS_APP="${PYTHON_SCRIPTS_DIR}/bin/parthenon_metrics_app.py"

trap 'catch $? $LINENO' ERR
catch() {
  echo "Error $1 occurred on $2"
  "${METRICS_APP}" --status "error" --status-context "Parthenon Metrics App" --status-description "CI failed" --status-url "${CI_JOB_URL}"
  wait
  echo "BUILD FAILED"
  # For some reason the ERR variable is not recongized as a numeric type when quoted,
  # exit requires err be recognized as a numeric type, furthermore the quotes
  # are needed as good practice
  exit "$((ERR))"
}

