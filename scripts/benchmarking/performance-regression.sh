#!/bin/bash --login

# This file serves as a sample on how to run automated performance regression.
# In addtion to the following lines for the scheduler, lines with MODIFY should be adapated to local needs. 

########## SBATCH Lines for Resource Request ##########
#SBATCH --time=0:59:00 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1 
#SBATCH --job-name perf-regression
#SBATCH --tasks-per-node 5
#SBATCH --gres=gpu:v100s:1
########## Command Lines to Run ########## 

####################################################################################
### MODIFY commit hash, some text string, and optional extra argument to be passed
####################################################################################
COMMIT=10ddbd7230e23777baac6c7cef72d2d6d320f8f4
COMMENT="additonal infos could go here"
EXTRAARGS="Advection/num_vars=5"

####################################################################################
### MODIFY potentially pick a different base directory, however using some temp
### directory is important as otherwise two jobs running simulatenously may
### use/modify the same source directory (by checking out different commits)
####################################################################################
BASELOGFILE=$(mktemp ~/perf-data/$COMMIT.XXXXXXXX.out)

####################################################################################
### MODIFY path to existing, checked out repo where performance data is stored
####################################################################################
REPOPATH=/path/to/repo/containing/performance/data

####################################################################################
### MODIFY load local modules
####################################################################################
module purge
module load intel/2020a git

LOGFILE=$BASELOGFILE.icc
echo "PERFORMANCE LOG" > $LOGFILE
echo "COMMENT: $COMMENT" >> $LOGFILE
echo "EXTRAARGS: $EXTRAARGS" >> $LOGFILE
module list 2>> $LOGFILE
icpc --version >> $LOGFILE

DIR=$(mktemp -d)
cd $DIR
git clone https://github.com/lanl/parthenon.git
cd parthenon
git checkout $COMMIT
git submodule init
git submodule update

# fix Kokkos version manually for now
cd external/Kokkos
git checkout 3.2.00
cd ../..

mkdir build-$COMMIT
cd build-$COMMIT
cmake -DKokkos_ARCH_SKX=True -DCMAKE_CXX_COMPILER=icpc -DPARTHENON_DISABLE_OPENMP=ON -DPARTHENON_DISABLE_HDF5=ON -DPARTHENON_DISABLE_MPI=ON  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-ipo -qopenmp-simd -qopt-prefetch=4" -DPARTHENON_LINT_DEFAULT=OFF -DBUILD_TESTING=OFF ..
make -j 5 advection-example

export OMP_PROC_BIND=true
export OMP_NUM_THREADS=1

export M=128
for N in 128 64 32 16 8; do
  echo "Doing ICC mx_$M-mb_$N"

  cp $LOGFILE $LOGFILE.adaptive_none-mx_$M-mb_$N

  srun -N 1 -n 1 ./example/advection/advection-example -i ../tst/regression/test_suites/advection_performance/parthinput.advection_performance parthenon/mesh/nx1=$M parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M parthenon/time/nlim=10 parthenon/meshblock/nx1=$N parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N $EXTRAARGS | tee -a $LOGFILE.adaptive_none-mx_$M-mb_$N
  
#need to skip for now, too time consuming
 if [ ! $N -eq 8 ]; then
####################################################################################
### MODIFY set to local precompiled profile lib
####################################################################################
export KOKKOS_PROFILE_LIBRARY=/path/to/kp_space_time_stack_intel2020a.so
  srun -N 1 -n 1 ./example/advection/advection-example -i ../tst/regression/test_suites/advection_performance/parthinput.advection_performance parthenon/mesh/nx1=$M parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M parthenon/time/nlim=10 parthenon/meshblock/nx1=$N parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N $EXTRAARGS | tee -a $LOGFILE.adaptive_none-mx_$M-mb_$N
  unset KOKKOS_PROFILE_LIBRARY
  fi
done

cp $LOGFILE.* $REPOPATH
cd ..
rm -rf build-$COMMIT

####################################################################################
### MODIFY load local modules
####################################################################################
module purge
module load gcccuda/2020a OpenMPI git Python HDF5 Qt5

LOGFILE=$BASELOGFILE.cuda
echo "PERFORMANCE LOG" > $LOGFILE
echo "COMMENT: $COMMENT" >> $LOGFILE
echo "EXTRAARGS: $EXTRAARGS" >> $LOGFILE
module list 2>> $LOGFILE
g++ --version >> $LOGFILE
nvcc --version >> $LOGFILE

mkdir build-$COMMIT
cd build-$COMMIT

cmake -DKokkos_ENABLE_CUDA=True -DKokkos_ARCH_VOLTA70=True -DKokkos_ARCH_SKX=True -DCMAKE_CXX_COMPILER=${PWD}/../external/Kokkos/bin/nvcc_wrapper -DPARTHENON_DISABLE_OPENMP=ON -DPARTHENON_DISABLE_HDF5=ON -DPARTHENON_DISABLE_MPI=ON  -DCMAKE_BUILD_TYPE=Release -DPARTHENON_LINT_DEFAULT=OFF -DBUILD_TESTING=OFF ..
make -j 5 advection-example


for M in 256 128; do
for N in $M $((M/2)) $((M/4)) $((M/8)) $((M/16)); do
  echo "Doing CUDA mx_$M-mb_$N"

  cp $LOGFILE $LOGFILE.adaptive_none-mx_$M-mb_$N

  srun -N 1 -n 1 ./example/advection/advection-example -i ../tst/regression/test_suites/advection_performance/parthinput.advection_performance parthenon/mesh/nx1=$M parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M parthenon/time/nlim=10 parthenon/meshblock/nx1=$N parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N $EXTRAARGS | tee -a $LOGFILE.adaptive_none-mx_$M-mb_$N

#need to skip for now, too time consuming
 if [ ! $N -eq $((M/16)) ]; then
####################################################################################
### MODIFY set to local precompiled profile lib
####################################################################################
export KOKKOS_PROFILE_LIBRARY=/path/to/kp_space_time_stack_gcccuda2020a.so
  srun -N 1 -n 1 ./example/advection/advection-example -i ../tst/regression/test_suites/advection_performance/parthinput.advection_performance parthenon/mesh/nx1=$M parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M parthenon/time/nlim=10 parthenon/meshblock/nx1=$N parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N $EXTRAARGS | tee -a $LOGFILE.adaptive_none-mx_$M-mb_$N
  unset KOKKOS_PROFILE_LIBRARY
  fi
done
done

cp $LOGFILE.* $REPOPATH
cd ..
rm -rf build-$COMMIT

cd $REPOPATH
git pull
git add ${BASELOGFILE##*/}.*
git commit -m "Add test data"
git push
