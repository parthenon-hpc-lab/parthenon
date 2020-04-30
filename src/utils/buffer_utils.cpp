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
//! \file buffer_utils.cpp
//  \brief namespace containing buffer utilities.

#include "utils/buffer_utils.hpp"

#include "athena.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {
namespace BufferUtility {

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void PackData(ParArray4D<T> &src, ParArray1D<T> &buf,
//                     int sn, int en,
//                     int si, int ei, int sj, int ej, int sk, int ek, int &offset,
//                     MeshBlock *pmb)
//  \brief pack a 4D ParArray into a one-dimensional buffer

template <typename T>
void PackData(ParArray4D<T> &src, ParArray1D<T> &buf, int sn, int en, int si, int ei,
              int sj, int ej, int sk, int ek, int &offset, MeshBlock *pmb) {

  int ni = ei + 1 - si;
  int nj = ej + 1 - sj;
  int nk = ek + 1 - sk;
  int nn = en + 1 - sn;

  pmb->par_for(
      "PackData 4D", sn, en, sk, ek, sj, ej, si, ei,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        buf(offset + i - si + ni * (j - sj + nj * (k - sk + nk * (n - sn)))) =
            src(n, k, j, i);
      });
  offset += nn * nk * nj * ni;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void PackData(ParArrayND<T> &src, ParArray1D<T> &buf,
//                      int si, int ei, int sj, int ej, int sk, int ek, int &offset,
//                      MeshBlock *pmb)
//  \brief pack a 3D ParArray into a one-dimensional buffer

template <typename T>
void PackData(ParArray3D<T> &src, ParArray1D<T> &buf, int si, int ei, int sj, int ej,
              int sk, int ek, int &offset, MeshBlock *pmb) {
  for (int k = sk; k <= ek; k++) {
    for (int j = sj; j <= ej; j++) {
#pragma omp simd
      for (int i = si; i <= ei; i++)
        buf[offset++] = src(k, j, i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackData(ParArray1D<T> &buf, ParArray4D<T> &dst,
//                        int sn, int en, int si, int ei, int sj, int ej, int sk, int ek,
//                        int &offset, MeshBlock *pmb)
//  \brief unpack a one-dimensional buffer into a ParArray4D

template <typename T>
void UnpackData(ParArray1D<T> &buf, ParArray4D<T> &dst, int sn, int en, int si, int ei,
                int sj, int ej, int sk, int ek, int &offset, MeshBlock *pmb) {
  for (int n = sn; n <= en; ++n) {
    for (int k = sk; k <= ek; ++k) {
      for (int j = sj; j <= ej; ++j) {
#pragma omp simd
        for (int i = si; i <= ei; ++i)
          dst(n, k, j, i) = buf[offset++];
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn template <typename T> void UnpackData(ParArray1D<T> &buf, ParArray3D<T> &dst,
//                        int si, int ei, int sj, int ej, int sk, int ek, int &offset,
//                        MeshBlock *pmb)
//  \brief unpack a one-dimensional buffer into a 3D ParArray

template <typename T>
void UnpackData(ParArray1D<T> &buf, ParArray3D<T> &dst, int si, int ei, int sj, int ej,
                int sk, int ek, int &offset, MeshBlock *pmb) {
  for (int k = sk; k <= ek; ++k) {
    for (int j = sj; j <= ej; ++j) {
#pragma omp simd
      for (int i = si; i <= ei; ++i)
        dst(k, j, i) = buf[offset++];
    }
  }
  return;
}

// provide explicit instantiation definitions (C++03) to allow the template definitions to
// exist outside of header file (non-inline), but still provide the requisite instances
// for other TUs during linking time (~13x files include "buffer_utils.hpp")

// 13x files include buffer_utils.hpp
template void UnpackData<Real>(ParArray1D<Real> &, ParArray4D<Real> &, int, int, int, int,
                               int, int, int, int, int &, MeshBlock *);
template void UnpackData<Real>(ParArray1D<Real> &, ParArray3D<Real> &, int, int, int, int,
                               int, int, int &, MeshBlock *);

template void PackData<Real>(ParArray4D<Real> &, ParArray1D<Real> &, int, int, int, int,
                             int, int, int, int, int &, MeshBlock *);
template void PackData<Real>(ParArray3D<Real> &, ParArray1D<Real> &, int, int, int, int,
                             int, int, int &, MeshBlock *);

} // namespace BufferUtility
} // namespace parthenon
