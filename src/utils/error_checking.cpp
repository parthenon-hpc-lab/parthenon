//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include "error_checking.hpp"

#ifdef MPI_PARALLEL
[[noreturn]] void parthenon::ErrorChecking::fail_throws_mpi(int const status,
                                                            char const *const expr,
                                                            char const *const filename,
                                                            int const linenumber) {
  int err_length = 0;
  char err_string_buf[MPI_MAX_ERROR_STRING];
  MPI_Error_string(status, err_string_buf, &err_length);

  std::string const err_string(err_string_buf, err_length);

  std::stringstream ss;
  ss << "MPI failure: `" << expr << "`, MPI error: \"" << err_string << "\"";
  fail_throws(ss, filename, linenumber);
}
#endif

#ifdef HDF5OUTPUT
[[noreturn]] void parthenon::ErrorChecking::fail_throws_hdf5(herr_t err,
                                                             char const *const expr,
                                                             char const *const filename,
                                                             int const linenumber) {
  std::stringstream ss;
  ss << "HDF5 failure: `" << expr << "`, Code: 0x" << std::hex << err;
  fail_throws(ss, filename, linenumber);
}
#endif
