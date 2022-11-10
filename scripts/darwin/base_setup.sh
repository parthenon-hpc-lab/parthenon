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

# This file is designed to ensure consistent environmental settings
# are available to any script that sources it. It is specifically meant
# to be used on Darwin.
#
# Commonly needed system env variables are sourced, followed by the
# spack env variables specific to the Parthenon project and finally
# the .bashrc in the parthenon-project space needed to automatically setup
# the build dependencies when building Parthenon with cmake.

# Load system env only
source /etc/bashrc
source /etc/profile

# Exit on error
set -eE

# Defining Variables
#
# Below several variables provided to this script are defined and named


GITHUB_APP_PEM="$1"
# shellcheck disable=SC2034
PARTHENON_DIR="$2"
# shellcheck disable=SC2034
BUILD_DIR="${2}/build"
export CI_COMMIT_SHA="$3"
export CI_COMMIT_BRANCH="$4"
# shellcheck disable=SC2034
CI_JOB_URL="$5"
CI_JOB_TOKEN="$6"

# Should be a value of ON or OFF depending on whether we are
# also running metrics on the pr we want to merge with
# shellcheck disable=SC2034
BUILD_TARGET="$7"

PYTHON_SCRIPTS_DIR="$8"
CMAKE_BUILD_TYPE="$9"

# Point to Python install path
export PIP=/projects/parthenon-int/python/amd-epyc/bin/pip

# Machine config file
export MACHINE_CFG=${PARTHENON_DIR}/cmake/Darwin-AMD-Epyc-gpu.cmake

# Here we are adding the location of the python scripts bin folder
# to the path variable. The python scripts directory contains the
# python Parthenon metrics files. This files are used to interact
# with the github server.
export PATH=${PYTHON_SCRIPTS_DIR}/bin:${PATH}
export PYTHONPATH=${PYTHON_SCRIPTS_DIR}/${PYTHONPATH}

export CI_JOB_TOKEN="$CI_JOB_TOKEN"
export GITHUB_APP_PEM="$GITHUB_APP_PEM"

echo "CI commit branch ${CI_COMMIT_BRANCH}"

# Load relevant modules
module use --append /projects/parthenon-int/dependencies/modulefiles
module load gcc/9.4.0 nvhpc/22.7 amd-epyc-gpu/hdf5/1.10.5 cmake/3.22.2
module list
