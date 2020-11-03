#!/bin/bash

# Works to return the package if provided with a module, where the package is of the form
#
# openmpi/2.1.5-pgi_18.3
# 
# The package name appears first and before a /
MODULE="$1"
if [[ "$1" == "gcc/"* ]]; then
  echo "gcc" 
elif [[ "$1" == "clang/"* ]]; then
  echo "clang"
elif [[ "$1" == "openmpi/"* ]]; then
  echo "openmpi"
else
  echo "No matching package"
fi
