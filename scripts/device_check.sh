#!/bin/bash

# This script is designed to detect the number of gpu devices that are available before the
# ctest suite is run. It does this by using the nvidia-smi. The script is only called if
# parthenon is built with kokkos_ENABLE_CUDA.
#
# The script takes a single argument the number of GPUs that are meant to be used with the 
# tests. The script then checks that the number of GPUs that are actually avaliable on the
# system are enough to satisfy this requirement. 

if [ "$#" -ne 1 ]; then
  printf "You must enter exactly 1 command line arguments, which should indicate
the number of GPUs to be used with the tests.\n"
  exit 1
fi

NUM_GPUS_ASSIGNED_FOR_TEST=$1

if ! command -v nvidia-smi &> /dev/null
then
  printf "CUDA has been enabled but nvidia-smi cannot be found\n"
  exit 1 
fi

output=$(nvidia-smi -L)

found=$(echo $output | grep "GPU 0: ")

GPUS_DETECTED=0
while [ ! -z "$found" ]; do
  let GPUS_DETECTED+=1
  found=$(echo $output | grep "GPU ${GPUS_DETECTED}: ")
done

if [ "$GPUS_DETECTED" -eq "0" ]
then
  printf "CUDA has been enabled but no GPUs have been detected.\n"
  exit 1
elif [ "$NUM_GPUS_ASSIGNED_FOR_TEST" -gt "$GPUS_DETECTED" ]
then
  printf "You are trying to build the parthenon regression 
tests with CUDA enabled kokkos, with the following settings:\n
Number of CUDA devices set to: ${NUM_GPUS_ASSIGNED_FOR_TEST}
Number of CUDA devices available: ${GPUS_DETECTED}\n
The number of gpus detected is less than then the number of devices requested,
consider changing:\n
NUM_GPU_DEVICES_TESTING=${GPUS_DETECTED}
NUM_MPI_PROC_TESTING=${GPUS_DETECTED}\n
Or consider building without CUDA.\n"
  exit 1
elif [ "$NUM_GPUS_ASSIGNED_FOR_TEST" -lt "$GPUS_DETECTED" ]
then
  printf "\nNumber of GPUs detected: $GPUS_DETECTED
Number of GPUs using in tests: $NUM_GPUS_ASSIGNED_FOR_TEST\n
If this is not the intended behavior consider changing NUM_GPU_DEVICES_TESTING 
to a different value. By default it is made equal to NUM_MPI_PROC_TESTING.\n\n"
  exit 0
else
  printf "\nNumber of GPUs detected: $GPUS_DETECTED
Number of GPUs using in tests: $NUM_GPUS_ASSIGNED_FOR_TEST\n\n"
  exit 0
fi
