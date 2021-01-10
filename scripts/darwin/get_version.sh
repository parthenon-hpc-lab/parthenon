#!/bin/bash

# Works to get the version of a package given that it is provided a form

# openmpi/2.1.5-pgi_18.3

# Where the version of the package is indicated after the last / and before any - 
string=$1
sub_str=$(printf "%s\n" "${string##*\/}")
echo "${sub_str%%-*}"
