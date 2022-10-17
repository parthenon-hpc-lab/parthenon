#========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2022 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
# (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#========================================================================================

message(STATUS "Loading machine configuration for AMD Epyc with GPU support.\n"
	"On Darwin. Assumes the following about the environment:\n"
	"\t-partition: amd-epyc-gpu\n"
	"\t-module use --append /projects/parthenon-int/dependencies/modulefiles"
	"\t-modules currently loaded: gcc nvhpc amd-epyc-gpu/hdf5 cmake\n"
	"\t-hdf5 installed in parthenon-int/dependencies/hdf5-1.10.5\n"
	"\t-numpy scipy h5py matplotlib installed in ~/.conda/envs/parthenon-amd-epyc\n")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default release build")
set(PARTHENON_DISABLE_OPENMP ON CACHE BOOL "OpenMP support not yet tested in Parthenon.")

set(CMAKE_CXX_COMPILER "nvc++" CACHE STRING "Default compiler")
set(CMAKE_C_COMPILER "nvcc" CACHE STRING "Default compiler")

set(PARTHENON_DISABLE_HDF5_COMPRESSION ON CACHE BOOL "No HDF5 xompression")

set(Python3_ROOT_DIR "$ENV{HOME}/.conda/envs/parthenon-amd-epyc/bin")
set(Python_ROOT_DIR "$ENV{HOME}/.conda/envs/parthenon-amd-epyc/bin")

set(SERIAL_WITH_MPIEXEC ON CACHE BOOL "Run with mpiexec -n 1 for serial")
set(TEST_MPIEXEC mpirun CACHE STRING "Command to launch MPI applications")
set(TEST_NUMPROC_FLAG "-np" CACHE STRING "Flag to set number of processes")
set(NUM_MPI_PROC_TESTING "4" CACHE STRING "Run tests with4 ranks")
