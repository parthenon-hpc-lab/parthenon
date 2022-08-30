//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
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
//! \file io_wrapper.cpp
//  \brief functions that provide wrapper for MPI-IO versus serial input/output

#include "outputs/io_wrapper.hpp"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "defs.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void IOWrapper::Open(const char* fname, FileMode rw)
//  \brief wrapper for {MPI_File_open} versus {std::fopen} including error check

void IOWrapper::Open(const char *const fname, FileMode const rw) {
  std::stringstream msg;

  if (rw == FileMode::read) {
#ifdef MPI_PARALLEL
    // NOLINTNEXTLINE
    if (MPI_File_open(comm_, const_cast<char *>(fname), MPI_MODE_RDONLY, MPI_INFO_NULL,
                      &fh_) != MPI_SUCCESS) // use const_cast to convince the compiler.
#else
    if ((fh_ = std::fopen(fname, "rb")) == nullptr) // NOLINT
#endif
    {
      msg << "### FATAL ERROR in function [IOWrapper:Open]" << std::endl
          << "Input file '" << fname << "' could not be opened" << std::endl;
      PARTHENON_FAIL(msg);
    }

  } else if (rw == FileMode::write) {
#ifdef MPI_PARALLEL
    MPI_File_delete(const_cast<char *>(fname), MPI_INFO_NULL); // truncation
    // NOLINTNEXTLINE
    if (MPI_File_open(comm_, const_cast<char *>(fname), MPI_MODE_WRONLY | MPI_MODE_CREATE,
                      MPI_INFO_NULL, &fh_) != MPI_SUCCESS)
#else
    if ((fh_ = std::fopen(fname, "wb")) == nullptr) // NOLINT
#endif
    {
      msg << "### FATAL ERROR in function [IOWrapper:Open]" << std::endl
          << "Output file '" << fname << "' could not be opened" << std::endl;
      PARTHENON_FAIL(msg);
    }
  } else {
    PARTHENON_FAIL("Unknown filemode for IOWrapper::Open");
  }
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read(void *buf, IOWrapperSizeT size, IOWrapperSizeT count)
//  \brief wrapper for {MPI_File_read} versus {std::fread}

std::ptrdiff_t IOWrapper::Read(void *const buf, IOWrapperSizeT const size,
                               IOWrapperSizeT const count) {
#ifdef MPI_PARALLEL
  MPI_Status status;
  int nread;
  if (MPI_File_read(fh_, buf, count * size, MPI_BYTE, &status) != MPI_SUCCESS) return -1;
  if (MPI_Get_count(&status, MPI_BYTE, &nread) == MPI_UNDEFINED) return -1;
  return nread / size;
#else
  return std::fread(buf, size, count, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count)
//  \brief wrapper for {MPI_File_read_all} versus {std::fread}

std::ptrdiff_t IOWrapper::Read_all(void *const buf, IOWrapperSizeT const size,
                                   IOWrapperSizeT const count) {
#ifdef MPI_PARALLEL
  MPI_Status status;
  int nread;
  if (MPI_File_read_all(fh_, buf, count * size, MPI_BYTE, &status) != MPI_SUCCESS)
    return -1;
  if (MPI_Get_count(&status, MPI_BYTE, &nread) == MPI_UNDEFINED) return -1;
  return nread / size;
#else
  return std::fread(buf, size, count, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Read_at_all(void *buf, IOWrapperSizeT size,
//                             IOWrapperSizeT count, IOWrapperSizeT offset)
//  \brief wrapper for {MPI_File_read_at_all} versus {std::fseek+std::fread}

std::ptrdiff_t IOWrapper::Read_at_all(void *const buf, IOWrapperSizeT const size,
                                      IOWrapperSizeT const count,
                                      IOWrapperSizeT const offset) {
#ifdef MPI_PARALLEL
  MPI_Status status;
  int nread;
  if (MPI_File_read_at_all(fh_, offset, buf, count * size, MPI_BYTE, &status) !=
      MPI_SUCCESS)
    return -1;
  if (MPI_Get_count(&status, MPI_BYTE, &nread) == MPI_UNDEFINED) return -1;
  return nread / size;
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fread(buf, size, count, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Write(const void *buf, IOWrapperSizeT size, IOWrapperSizeT cnt)
//  \brief wrapper for {MPI_File_write} versus {std::fwrite}

std::ptrdiff_t IOWrapper::Write(const void *const buf, IOWrapperSizeT const size,
                                IOWrapperSizeT const cnt) {
#ifdef MPI_PARALLEL
  MPI_Status status;
  int nwrite;
  if (MPI_File_write(fh_, const_cast<void *>(buf), cnt * size, MPI_BYTE, &status) !=
      MPI_SUCCESS)
    return -1;
  if (MPI_Get_count(&status, MPI_BYTE, &nwrite) == MPI_UNDEFINED) return -1;
  return nwrite / size;
#else
  return std::fwrite(buf, size, cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Write_at_all(const void *buf, IOWrapperSizeT size,
//                                  IOWrapperSizeT cnt, IOWrapperSizeT offset)
//  \brief wrapper for {MPI_File_write_at_all} versus {std::fseek+std::fwrite}.

std::ptrdiff_t IOWrapper::Write_at_all(const void *const buf, IOWrapperSizeT const size,
                                       IOWrapperSizeT const cnt,
                                       IOWrapperSizeT const offset) {
#ifdef MPI_PARALLEL
  MPI_Status status;
  int nwrite;
  if (MPI_File_write_at_all(fh_, offset, const_cast<void *>(buf), cnt * size, MPI_BYTE,
                            &status) != MPI_SUCCESS)
    return -1;
  if (MPI_Get_count(&status, MPI_BYTE, &nwrite) == MPI_UNDEFINED) return -1;
  return nwrite / size;
#else
  std::fseek(fh_, offset, SEEK_SET);
  return std::fwrite(buf, size, cnt, fh_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void IOWrapper::Close()
//  \brief wrapper for {MPI_File_close} versus {std::fclose}

void IOWrapper::Close() {
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_File_close(&fh_));
#else
  PARTHENON_REQUIRE_THROWS(0 == std::fclose(fh_), "Could not close file");
#endif
}

//----------------------------------------------------------------------------------------
//! \fn int IOWrapper::Seek(IOWrapperSizeT offset)
//  \brief wrapper for {MPI_File_seek} versus {std::fseek}

void IOWrapper::Seek(IOWrapperSizeT const offset) {
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_File_seek(fh_, offset, MPI_SEEK_SET));
#else
  PARTHENON_REQUIRE_THROWS(0 == std::fseek(fh_, offset, SEEK_SET), "Could not seek file");
#endif
}

//----------------------------------------------------------------------------------------
//! \fn IOWrapperSizeT IOWrapper::GetPosition()
//  \brief wrapper for {MPI_File_get_position} versus {ftell}

IOWrapperSizeT IOWrapper::GetPosition() {
#ifdef MPI_PARALLEL
  MPI_Offset position;
  PARTHENON_MPI_CHECK(MPI_File_get_position(fh_, &position));
  return position;
#else
  return ftell(fh_);
#endif
}

} // namespace parthenon
