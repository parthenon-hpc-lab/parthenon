//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//! \file show_config.cpp

#include <iostream>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "defs.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void ShowConfig()
//  \brief prints diagnostic messages about the configuration of an Athena++ executable

void ShowConfig() {
  // To match configure.py output: use 2 space indent for option, value output starts on
  // column 30
  std::cout << "This Parthenon library is configured with:" << std::endl;
  std::cout << "  Problem generator:          " << PROBLEM_GENERATOR << std::endl;

  if (SINGLE_PRECISION_ENABLED) {
    std::cout << "  Floating-point precision:   single" << std::endl;
  } else {
    std::cout << "  Floating-point precision:   double" << std::endl;
  }
#ifdef MPI_PARALLEL
  std::cout << "  MPI parallelism:            ON" << std::endl;
#else
  std::cout << "  MPI parallelism:            OFF" << std::endl;
#endif

#ifdef HDF5OUTPUT
  std::cout << "  HDF5 output:                ON" << std::endl;
#else
  std::cout << "  HDF5 output:                OFF" << std::endl;
#endif

  std::cout << "  Compiler:                   " << COMPILED_WITH << std::endl;
  std::cout << "  Compilation command:        " << COMPILER_COMMAND
            << COMPILED_WITH_OPTIONS << std::endl;
  // configure.py output: Doesnt append "Linker flags" in prev. output (excessive space!)

  std::cout << std::endl << "# Kokkos configuration" << std::endl;
  Kokkos::print_configuration(std::cout);
}

} // namespace parthenon
