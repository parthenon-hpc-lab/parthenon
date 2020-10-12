#!/bin/bash

# This script is designed to detect the number of gpu devices that are available before the
# ctest suite is run. It does this by using the nvidia-smi. The script is only called if
# parthenon is built with kokkos_ENABLE_CUDA.
#
# The script takes a single argument the number of GPUs that are meant to be used with the 
# tests. The script then checks that the number of GPUs that are actually avaliable on the
# system are enough to satisfy this requirement. 

if [ "$#" -ne 3 ]; then
  printf "You must enter exactly 3 command line arguments, which should indicate:
1. the number of GPUs per node to be used with the tests,
2. the name of the MPI executable,
3. the number of MPI procs to use.\n"
  exit 0
fi

NUM_GPUS_PER_NODE_REQUESTED=$1
MPI_EXEC_NAME=$2
NUM_MPI_PROCS_REQUESTED=$3

# Only run checks if MPI exec is used, because it is standardized
printf "\n****************************************** Beginning Precheck ******************************************\n\n"
if [[ "${MPI_EXEC_NAME}" = *mpiexec ]]
then

  NODE_COUNT=$(${MPI_EXEC_NAME} --display-allocation bash -c "exit 0" | grep -o ': flags=' | wc -l)

  REMAINDER=$(( $NUM_MPI_PROCS_REQUESTED % $NODE_COUNT ))
  if [ "$REMAINDER" -ne "0" ] 
  then
    printf "The number of MPI procs requested is not evenly divisible by the number of nodes:
Number of MPI Procs ${NUM_MPI_PROCS_REQUESTED}
Number of Nodes allocated ${NODE_COUNT}\n
Skipping remaining checks.\n\n" 
    printf "******************************************  Ending Precheck   ******************************************\n\n"
    exit 0
  fi

  MPI_PROCS_PER_NODE=$(( $NUM_MPI_PROCS_REQUESTED / $NODE_COUNT ))

  if ! command -v nvidia-smi &> /dev/null
  then
    printf "CUDA has been enabled but nvidia-smi cannot be found\n\n"
    printf "******************************************  Ending Precheck   ******************************************\n\n"
    exit 0 
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
    printf "CUDA has been enabled but no GPUs have been detected.\n\n"
    printf "******************************************  Ending Precheck   ******************************************\n\n"
    exit 0
  elif [ "$NUM_GPUS_PER_NODE_REQUESTED" -gt "$GPUS_DETECTED" ]
  then
    printf "You are trying to build the parthenon regression tests with CUDA enabled kokkos, with the following
settings:

Number of CUDA devices per node set to: ${NUM_GPUS_PER_NODE_REQUESTED}
Number of CUDA devices per node available: ${GPUS_DETECTED}

The number of GPUs detected on node is less than then the number of GPUs requested, consider changing:

NUM_GPU_DEVICES_PER_NODE=${GPUS_DETECTED}

Or consider building without CUDA.\n\n"
    printf "******************************************  Ending Precheck   ******************************************\n\n"
    exit 0
  else
    printf "Number of GPUs detected per node: $GPUS_DETECTED
Number of GPUs per node, requested in tests: $NUM_GPUS_PER_NODE_REQUESTED\n"
  fi

  if [ "$NUM_GPUS_PER_NODE_REQUESTED" -gt "$MPI_PROCS_PER_NODE" ]
  then
    printf "\nYou are trying to run parthenon regression tests with:\n
NUM_GPU_DEVICES_PER_NODE=${NUM_GPUS_PER_NODE_REQUESTED}
NUM_MPI_PROC_TESTING=${NUM_MPI_PROCS_REQUESTED}
  
Number of nodes detected: ${NODE_COUNT}
Number of MPI procs per node: ${MPI_PROCS_PER_NODE}

Assigning more than a single GPU to a given MPI proc is not supported. You have a total of ${NUM_GPUS_PER_NODE_REQUESTED} GPU(s) 
requested per node, while you only have a total of ${MPI_PROCS_PER_NODE} MPI proc per node.\n\n"
    printf "******************************************  Ending Precheck   ******************************************\n\n"
    exit 0
  else
    printf "\nNumber of nodes detected: ${NODE_COUNT}
Number of MPI procs, requested in tests: ${NUM_MPI_PROCS_REQUESTED}
Number of MPI procs per node: ${MPI_PROCS_PER_NODE}\n\n"
    printf "******************************************  Ending Precheck   ******************************************\n\n"
  fi

else
  printf "Non standard MPI binary command detected: ${MPI_EXEC_NAME}\nprecheck is not valid.\n\n"
  printf "******************************************  Ending Precheck   ******************************************\n\n"
fi
