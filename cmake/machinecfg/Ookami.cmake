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

message(STATUS "Loading machine configuration for Stony Brook's A64FX Ookami.\n"
	"This machine file is configured for the Fujitsu compiler in Clang mode.\n"
	"$ module load cmake fujitsu/compiler/4.5 hdf5/fujitsu/1.12.0\n"
	"This requires Kokkos patch https://github.com/kokkos/kokkos/pull/4745 to compile.\n"
	"Also note that very aggressive optmization flags are used.\n")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default release build")
set(Kokkos_ARCH_A64FX ON CACHE BOOL "CPU architecture")

set(CMAKE_CXX_COMPILER "mpiFCC" CACHE STRING "Default compiler")
set(CMAKE_CXX_FLAGS "-Nclang -ffj-fast-matmul -ffast-math -ffp-contract=fast -ffj-fp-relaxed -ffj-ilfunc -fbuiltin -fomit-frame-pointer -finline-functions -ffj-preex -ffj-zfill -ffj-swp -fopenmp-simd" CACHE STRING "Default opt flags")

set(TEST_MPIEXEC mpirun CACHE STRING "Command to launch MPI applications")
set(TEST_NUMPROC_FLAG "-np" CACHE STRING "Flag to set number of processes")
set(NUM_MPI_PROC_TESTING "4" CACHE STRING "Run tests with4 ranks")
