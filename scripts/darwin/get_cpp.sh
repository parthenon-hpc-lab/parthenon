#!/bin/bash

# Works to return the c++ compiler provided a compiler package
#
# gcc 
# 
# Then g++ would be returned 
COMPILER="$1"
if [[ "$COMPILER" == "gcc" ]]; then
  echo "g++" 
elif [[ "$COMPILER" == "clang" ]]; then
  echo "clang++"
elif [[ "$COMPILER" == "openmpi" ]]; then
  echo "mpic++"
else
  echo "No matching package"
fi
