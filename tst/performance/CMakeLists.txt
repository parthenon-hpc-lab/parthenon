
##========================================================================================
## Parthenon performance portable AMR framework
## Copyright(C) 2020-2022 The Parthenon collaboration
## Licensed under the 3-clause BSD License, see LICENSE file for details
##========================================================================================
## (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
##
## This program was produced under U.S. Government contract 89233218CNA000001 for Los
## Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
## for the U.S. Department of Energy/National Nuclear Security Administration. All rights
## in the program are reserved by Triad National Security, LLC, and the U.S. Department
## of Energy/National Nuclear Security Administration. The Government is granted for
## itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
## license in this material to reproduce, prepare derivative works, distribute copies to
## the public, perform publicly and display publicly, and to permit others to do so.
##========================================================================================

add_executable(performance_tests test_meshblock_data_iterator.cpp)
target_link_libraries(performance_tests PRIVATE Parthenon::parthenon catch2_define Kokkos::kokkos)
lint_target(performance_tests)

catch_discover_tests(performance_tests EXTRA_ARGS "--kokkos-disable-warnings")
