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

#include "basic_types.hpp"

namespace parthenon {
namespace Globals {

struct SparseConfig {
#ifdef ENABLE_SPARSE
  bool enabled = true;
#else
  bool enabled = false;
#endif
  Real allocation_threshold = 1.0e-12;
  Real deallocation_threshold = 1.0e-14;
  int deallocation_count = 5;
};

extern int my_rank, nranks, nghost;

extern SparseConfig sparse_config;

extern Real receive_boundary_buffer_timeout;
extern Real current_task_runtime_sec;

namespace cell_centered_refinement {
// If the info object has more buffers than this, do
// hierarchical parallelism. If it does not, loop over buffers on the
// host and launch kernels manually.
//
// min_num_bufs = 1 implies that the old per-buffer machinery doesn't
// use hierarchical parallelism. This also means that for
// prolongation/restriction over a whole meshblock, hierarchical
// parallelism is not used, which is probably important for
// re-meshing.
//
// min_num_bufs = 6 implies that in a unigrid sim a meshblock pack of
// size 1 would be looped over manually while a pack of size 2 would
// use hierarchical parallelism.
extern int min_num_bufs;
} // namespace cell_centered_refinement

} // namespace Globals
} // namespace parthenon

#endif // GLOBALS_HPP_
