//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef RECONSTRUCT_RECONSTRUCTION_HPP_
#define RECONSTRUCT_RECONSTRUCTION_HPP_
//! \file reconstruction.hpp
//  \brief defines class Reconstruction, data and functions for spatial reconstruction

#include <memory>

#include "defs.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// Forward declarations
class ParameterInput;

//! \class Reconstruction
//  \brief member functions implement various spatial reconstruction algorithms

class Reconstruction {
 public:
  enum VelocityIndex { IVX = 1, IVY = 2, IVZ = 3 };

  Reconstruction(std::weak_ptr<MeshBlock> pmb, ParameterInput *pin);

  // data
  // switches for reconstruction method variants:
  int xorder; // roughly the formal order of accuracy of overall reconstruction method
  bool characteristic_projection;
  bool uniform[4];
  // (Cartesian reconstruction formulas are used for x3 azimuthal coordinate in both
  // cylindrical and spherical-polar coordinates)

  // related fourth-order solver switches
  const bool correct_ic, correct_err; // used in Mesh::Initialize() and ProblemGenerator()

  // x1-sliced arrays of interpolation coefficients and limiter parameters:
  ParArrayND<Real> c1i, c2i, c3i, c4i, c5i, c6i;  // coefficients for PPM in x1
  ParArrayND<Real> hplus_ratio_i, hminus_ratio_i; // for curvilinear PPMx1
  ParArrayND<Real> c1j, c2j, c3j, c4j, c5j, c6j;  // coefficients for PPM in x2
  ParArrayND<Real> hplus_ratio_j, hminus_ratio_j; // for curvilinear PPMx2
  ParArrayND<Real> c1k, c2k, c3k, c4k, c5k, c6k;  // coefficients for PPM in x3
  ParArrayND<Real> hplus_ratio_k, hminus_ratio_k; // for curvilinear PPMx3

  // functions
  // linear transformations of vectors between primitive and characteristic variables
  void LeftEigenmatrixDotVector(const int ivx, const int il, const int iu,
                                const ParArrayND<Real> &b1, const ParArrayND<Real> &w,
                                ParArrayND<Real> &vect);
  void RightEigenmatrixDotVector(const int ivx, const int il, const int iu,
                                 const ParArrayND<Real> &b1, const ParArrayND<Real> &w,
                                 ParArrayND<Real> &vect);

  // reconstruction functions of various orders in each dimension
  void DonorCellX1(const int k, const int j, const int il, const int iu,
                   const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                   ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void DonorCellX2(const int k, const int j, const int il, const int iu,
                   const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                   ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void DonorCellX3(const int k, const int j, const int il, const int iu,
                   const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                   ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void PiecewiseLinearX1(const int k, const int j, const int il, const int iu,
                         const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                         ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void PiecewiseLinearX2(const int k, const int j, const int il, const int iu,
                         const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                         ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void PiecewiseLinearX3(const int k, const int j, const int il, const int iu,
                         const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                         ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void PiecewiseParabolicX1(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                            ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void PiecewiseParabolicX2(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                            ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void PiecewiseParabolicX3(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &w, const ParArrayND<Real> &bcc,
                            ParArrayND<Real> &wl, ParArrayND<Real> &wr);

  void DonorCellX1(const int k, const int j, const int il, const int iu,
                   const ParArrayND<Real> &q, ParArrayND<Real> &ql, ParArrayND<Real> &qr);

  void DonorCellX2(const int k, const int j, const int il, const int iu,
                   const ParArrayND<Real> &q, ParArrayND<Real> &ql, ParArrayND<Real> &qr);

  void DonorCellX3(const int k, const int j, const int il, const int iu,
                   const ParArrayND<Real> &q, ParArrayND<Real> &ql, ParArrayND<Real> &qr);

  void PiecewiseLinearX1(const int k, const int j, const int il, const int iu,
                         const ParArrayND<Real> &q, ParArrayND<Real> &ql,
                         ParArrayND<Real> &qr);

  void PiecewiseLinearX2(const int k, const int j, const int il, const int iu,
                         const ParArrayND<Real> &q, ParArrayND<Real> &ql,
                         ParArrayND<Real> &qr);

  void PiecewiseLinearX3(const int k, const int j, const int il, const int iu,
                         const ParArrayND<Real> &q, ParArrayND<Real> &ql,
                         ParArrayND<Real> &qr);

  void PiecewiseParabolicX1(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &q, ParArrayND<Real> &ql,
                            ParArrayND<Real> &qr);

  void PiecewiseParabolicX2(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &q, ParArrayND<Real> &ql,
                            ParArrayND<Real> &qr);

  void PiecewiseParabolicX3(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &q, ParArrayND<Real> &ql,
                            ParArrayND<Real> &qr);

 private:
  // ptr to MeshBlock containing this Reconstruction
  std::weak_ptr<MeshBlock> pmy_block_;

  // scratch arrays used in PLM and PPM reconstruction functions
  ParArrayND<Real> scr01_i_, scr02_i_, scr03_i_, scr04_i_, scr05_i_;
  ParArrayND<Real> scr06_i_, scr07_i_, scr08_i_, scr09_i_, scr10_i_;
  ParArrayND<Real> scr11_i_, scr12_i_, scr13_i_, scr14_i_;
  ParArrayND<Real> scr1_ni_, scr2_ni_, scr3_ni_, scr4_ni_, scr5_ni_;
  ParArrayND<Real> scr6_ni_, scr7_ni_, scr8_ni_;

  // Returns shared pointer to a block
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (pmy_block_.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block_.lock();
  }
};

} // namespace parthenon

#endif // RECONSTRUCT_RECONSTRUCTION_HPP_
