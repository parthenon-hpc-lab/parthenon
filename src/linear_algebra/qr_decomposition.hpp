#ifndef QR_DECOMPOSITION_HPP
#define QR_DECOMPOSITION_HPP

#include "householder.hpp"
#include "matrix.hpp"

template <class matrix_t>
KOKKOS_FORCEINLINE_FUNCTION void QRDecomposition(matrix_t &A, matrix_t &Q) {
  Vector v(A.nrows()), scratch(A.nrows());
  serial_tm_t tm{};
  for (int i = 0; i < A.nrows() - 1; ++i) {
    build_householder_vector_col(tm, i, i, A, v.data());
    apply_left_householder_transformation(tm, v.data(), scratch.data(), A);
    apply_right_householder_transformation(tm, v.data(), scratch.data(), Q);
  }
}

#endif // QR_DECOMPOSITION_HPP
