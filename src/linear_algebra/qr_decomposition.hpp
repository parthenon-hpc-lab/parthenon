//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef QR_DECOMPOSITION_HPP
#define QR_DECOMPOSITION_HPP

#include "householder.hpp"
#include "matrix.hpp"

// TODO(LFR): Probably just remove this whole file

template <class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void QRDecomposition(matrix_t &A, matrix_t &Q) {
  Vector v(A.nrows()), scratch(A.nrows());
  serial_tm_t tm{};
  for (int i = 0; i < A.nrows() - 1; ++i) {
    build_householder_vector_col(tm, i, i, A, v.data());
    apply_left_householder_transformation(tm, v.data(), scratch.data(), A, i);
    apply_right_householder_transformation(tm, v.data(), scratch.data(), Q, i);
  }
}

#endif // QR_DECOMPOSITION_HPP
