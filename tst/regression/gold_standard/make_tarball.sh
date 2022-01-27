# ========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: ${0}: VERSION"
  exit 1
fi

outname="parthenon_regression_gold_v$1.tgz"

# check if output tarball already exists
if [[ -f "$outname" ]]; then
  read -p "WARNING: Output file ${outname} already exists. Overwrite? [yN] " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting"
    exit 1
  fi
fi

# check if contents of current_version file matches the version we're creating
curr_v=`cat current_version`
if [[ "${curr_v}" != "${1}" ]]; then
  read -p "WARNING: Contents of 'current_version' is '${curr_v}', but making tarball for version '${1}'. Update 'current_version' file? [yN] " -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting"
    exit 1
  else
    echo "Updating 'current_version' file"
    echo
    echo ${1} > current_version
  fi
fi

# check if version is present in README.md
if ! grep -q "${1}:" README.md; then
  echo "ERROR: README.md does not have entry for version '${1}'. Please update README.md"
  echo "Aborting"
  exit 1
fi

echo
echo "Adding the following files to tarball:"
ls -lah *.phdf *.phdf.xdmf current_version README.md

echo

tar czf "$outname" *.phdf *.phdf.xdmf current_version README.md

echo "Created tarball $outname, SHA-512 hash:"
echo

sha512sum "$outname"
