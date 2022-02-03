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
#ifndef UTILS_BUFFER_UTILS_HPP_
#define UTILS_BUFFER_UTILS_HPP_
//! \file buffer_utils.hpp
//  \brief prototypes of utility functions to pack/unpack buffers

#include "defs.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {
namespace BufferUtility {

// 2x templated and overloaded functions
// 4D
template <typename T>
void PackData(ParArray4D<T> &src, BufArray1D<T> &buf, int sn, int en, int si, int ei,
              int sj, int ej, int sk, int ek, int &offset, MeshBlock *pmb);
template <typename T>
void PackZero(BufArray1D<T> &buf, int sn, int en, int si, int ei, int sj, int ej, int sk,
              int ek, int &offset, MeshBlock *pmb);
// 3D
template <typename T>
void PackData(ParArray3D<T> &src, BufArray1D<T> &buf, int si, int ei, int sj, int ej,
              int sk, int ek, int &offset, MeshBlock *pmb);
// 4D
template <typename T>
void UnpackData(BufArray1D<T> &buf, ParArray4D<T> &dst, int sn, int en, int si, int ei,
                int sj, int ej, int sk, int ek, int &offset, MeshBlock *pmb);
// 3D
template <typename T>
void UnpackData(BufArray1D<T> &buf, ParArray3D<T> &dst, int si, int ei, int sj, int ej,
                int sk, int ek, int &offset, MeshBlock *pmb);

} // namespace BufferUtility
} // namespace parthenon

#endif // UTILS_BUFFER_UTILS_HPP_
