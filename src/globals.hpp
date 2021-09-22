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
#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_
//! \file globals.hpp
//  \brief namespace containing external global variables

namespace parthenon {
namespace Globals {

struct SparseConfig {
  bool enabled = true;
  double allocation_threshold = 1.0e-12;
  double deallocation_threshold = 1.0e-14;
  int deallocation_count = 5;
};

extern int my_rank, nranks, nghost;

extern SparseConfig sparse_config;

} // namespace Globals
} // namespace parthenon

#endif // GLOBALS_HPP_
