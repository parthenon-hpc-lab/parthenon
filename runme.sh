#!/bin/bash

set +x
set +e

module load atp
module load gcc-mixed
ulimit -c unlimited 
#NX=416
NX=32
NXB=16
NLIM=10
#NLIM=20
NLVL=3
#RANKS=96
RANKS=8
#RANKS=768
#export CALI_CONFIG=spot,time.exclusive,profile.mpi
export CALI_CONFIG="runtime-report(output=stdout),time.exclusive,profile.mpi"
#export OMP_PROC_BIND=spread 
#export OMP_PLACES=threads

srun -n ${RANKS}  burgers-benchmark -i ../../../benchmarks/burgers/burgers.pin parthenon/mesh/nx1=${NX} parthenon/mesh/nx2=${NX} parthenon/mesh/nx3=${NX} parthenon/meshblock/nx1=${NXB} parthenon/meshblock/nx2=${NXB} parthenon/meshblock/nx3=${NXB} parthenon/time/nlim=${NLIM} parthenon/mesh/numlevel=${NLVL}
