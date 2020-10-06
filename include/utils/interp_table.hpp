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
#ifndef UTILS_INTERP_TABLE_HPP_
#define UTILS_INTERP_TABLE_HPP_

//! \file interp_table.hpp
//  \brief defines class InterpTable2D
//  Contains functions that implement an intpolated lookup table

#include "defs.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

class InterpTable2D {
 public:
  InterpTable2D() = default;
  InterpTable2D(const int nvar, const int nx2, const int nx1);

  void SetSize(const int nvar, const int nx2, const int nx1);
  Real interpolate(int nvar, Real x2, Real x1);
  int nvar();
  ParArrayND<Real> data;
  void SetX1lim(Real x1min, Real x1max);
  void SetX2lim(Real x2min, Real x2max);
  void GetX1lim(Real &x1min, Real &x1max);
  void GetX2lim(Real &x2min, Real &x2max);
  void GetSize(int &nvar, int &nx2, int &nx1);

 private:
  int nvar_;
  int nx1_;
  int nx2_;
  Real x1min_;
  Real x1max_;
  Real x1norm_;
  Real x2min_;
  Real x2max_;
  Real x2norm_;
};

} // namespace parthenon

#endif // UTILS_INTERP_TABLE_HPP_
