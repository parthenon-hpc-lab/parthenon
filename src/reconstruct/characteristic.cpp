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
//! \file characteristic.cpp
//  \brief Functions to transform vectors between primitive and characteristic variables

#include "reconstruction.hpp"

namespace parthenon {
//----------------------------------------------------------------------------------------
//! \fn Reconstruction::LeftEigenmatrixDotVector()
//  \brief Computes inner-product of left-eigenmatrix of Roe's matrix A in the primitive
//  variables and an input vector.  This operation converts primitive to characteristic
//  variables.  The result is returned in the input vector, with the components of the
//  characteristic field stored such that vect(1,i) is in the direction of the sweep.
//
//  The order of the components in the input vector should be:
//     (IDN,IVX,IVY,IVZ,[IPR],[IBY,IBZ])
//  and these are permuted according to the direction specified by the input flag "ivx".
//
// REFERENCES:
// - J. Stone, T. Gardiner, P. Teuben, J. Hawley, & J. Simon "Athena: A new code for
//   astrophysical MHD", ApJS, (2008), Appendix A.  Equation numbers refer to this paper.

void Reconstruction::LeftEigenmatrixDotVector(
    const int ivx, const int il, const int iu,
    const AthenaArray<Real> &b1, const AthenaArray<Real> &w, AthenaArray<Real> &vect) {
  throw std::runtime_error(std::string(__func__) + " is not implemented");
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::RightEigenmatrixDotVector()
//  \brief Computes inner-product of right-eigenmatrix of Roe's matrix A in the primitive
//  variables and an input vector.  This operation converts characteristic to primitive
//  variables.  The result is returned in the input vector.
//
//  The order of the components in the input vector (characteristic fields) should be:
//     (IDN,ivx,ivy,ivz,[IPR],[IBY,IBZ])
//  where the lower-case indices indicate that the characteristic field in the direction
//  of the sweep (designated by the input flag "ivx") is stored first.  On output, the
//  components of velocity are in the standard order used for primitive variables.
//
// REFERENCES:
// - J. Stone, T. Gardiner, P. Teuben, J. Hawley, & J. Simon "Athena: A new code for
//   astrophysical MHD", ApJS, (2008), Appendix A.  Equation numbers refer to this paper.

void Reconstruction::RightEigenmatrixDotVector(
    const int ivx, const int il, const int iu,
    const AthenaArray<Real> &b1, const AthenaArray<Real> &w, AthenaArray<Real> &vect) {
  throw std::runtime_error(std::string(__func__) + " is not implemented");
}
}
