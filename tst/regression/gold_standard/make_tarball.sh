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

tar czf $outname *.phdf *.phdf.xdmf current_version README.md

echo "Created tarball $outname, SHA-512 hash:"
sha512sum $outname
