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
#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//! \file coordinates.hpp
//  \brief defines abstract base and derived classes for coordinates.  These classes
//  provide data and functions to compute/store coordinate positions and spacing, as well
//  as geometrical factors (areas, volumes, coordinate source terms) for various
//  coordinate systems.

// C headers

// C++ headers
#include <iostream>

// Athena++ headers
#include "athena.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {
// forward declarations
class Mesh;
class MeshBlock;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \class Coordinates
//  \brief abstract base class for all coordinate derived classes

class Coordinates {
 public:
  Coordinates(MeshBlock *pmb, ParameterInput *pin, bool flag = false);
  virtual ~Coordinates() = default;

  // data
  MeshBlock *pmy_block; // ptr to MeshBlock containing this Coordinates
  ParArrayND<Real> dx1f, dx2f, dx3f, x1f, x2f, x3f;    // face   spacing and positions
  ParArrayND<Real> dx1v, dx2v, dx3v, x1v, x2v, x3v;    // volume spacing and positions
  ParArrayND<Real> x1s2, x1s3, x2s1, x2s3, x3s1, x3s2; // area averaged positions for AMR
  // geometry coefficients (only used in SphericalPolar, Cylindrical, Cartesian)
  ParArrayND<Real> h2f, dh2fd1, h31f, h32f, dh31fd1, dh32fd2;
  ParArrayND<Real> h2v, dh2vd1, h31v, h32v, dh31vd1, dh32vd2;

  // functions...
  // ...to compute length of edges
  virtual void Edge1Length(const int k, const int j, const int il, const int iu,
                           ParArrayND<Real> &len);
  virtual void Edge2Length(const int k, const int j, const int il, const int iu,
                           ParArrayND<Real> &len);
  virtual void Edge3Length(const int k, const int j, const int il, const int iu,
                           ParArrayND<Real> &len);
  virtual Real GetEdge1Length(const int k, const int j, const int i);
  virtual Real GetEdge2Length(const int k, const int j, const int i);
  virtual Real GetEdge3Length(const int k, const int j, const int i);
  // ...to compute length connecting cell centers (for non-ideal MHD)
  virtual void VolCenter1Length(const int k, const int j, const int il, const int iu,
                                ParArrayND<Real> &len);
  virtual void VolCenter2Length(const int k, const int j, const int il, const int iu,
                                ParArrayND<Real> &len);
  virtual void VolCenter3Length(const int k, const int j, const int il, const int iu,
                                ParArrayND<Real> &len);
  // ...to compute physical width at cell center
  virtual void CenterWidth1(const int k, const int j, const int il, const int iu,
                            ParArrayND<Real> &dx1);
  virtual void CenterWidth2(const int k, const int j, const int il, const int iu,
                            ParArrayND<Real> &dx2);
  virtual void CenterWidth3(const int k, const int j, const int il, const int iu,
                            ParArrayND<Real> &dx3);

  // ...to compute area of faces
  virtual void Face1Area(const int k, const int j, const int il, const int iu,
                         ParArrayND<Real> &area);
  virtual void Face2Area(const int k, const int j, const int il, const int iu,
                         ParArrayND<Real> &area);
  virtual void Face3Area(const int k, const int j, const int il, const int iu,
                         ParArrayND<Real> &area);
  virtual Real GetFace1Area(const int k, const int j, const int i);
  virtual Real GetFace2Area(const int k, const int j, const int i);
  virtual Real GetFace3Area(const int k, const int j, const int i);
  // ...to compute area of faces joined by cell centers (for non-ideal MHD)
  virtual void VolCenterFace1Area(const int k, const int j, const int il, const int iu,
                                  ParArrayND<Real> &area);
  virtual void VolCenterFace2Area(const int k, const int j, const int il, const int iu,
                                  ParArrayND<Real> &area);
  virtual void VolCenterFace3Area(const int k, const int j, const int il, const int iu,
                                  ParArrayND<Real> &area);

  // ...to compute Laplacian of quantities in the coord system and orthogonal subspaces
  virtual void Laplacian(const ParArrayND<Real> &s, ParArrayND<Real> &delta_s,
                         const int il, const int iu, const int jl, const int ju,
                         const int kl, const int ku, const int nl, const int nu);
  virtual void LaplacianX1(const ParArrayND<Real> &s, ParArrayND<Real> &delta_s,
                           const int n, const int k, const int j, const int il,
                           const int iu);
  virtual void LaplacianX1All(const ParArrayND<Real> &s, ParArrayND<Real> &delta_s,
                              const int nl, const int nu, const int kl, const int ku,
                              const int jl, const int ju, const int il, const int iu);
  virtual void LaplacianX2(const ParArrayND<Real> &s, ParArrayND<Real> &delta_s,
                           const int n, const int k, const int j, const int il,
                           const int iu);
  virtual void LaplacianX2All(const ParArrayND<Real> &s, ParArrayND<Real> &delta_s,
                              const int nl, const int nu, const int kl, const int ku,
                              const int jl, const int ju, const int il, const int iu);
  virtual void LaplacianX3(const ParArrayND<Real> &s, ParArrayND<Real> &delta_s,
                           const int n, const int k, const int j, const int il,
                           const int iu);
  virtual void LaplacianX3All(const ParArrayND<Real> &s, ParArrayND<Real> &delta_s,
                              const int nl, const int nu, const int kl, const int ku,
                              const int jl, const int ju, const int il, const int iu);

  // ...to compute volume of cells
  virtual void CellVolume(const int k, const int j, const int il, const int iu,
                          ParArrayND<Real> &vol);
  virtual Real GetCellVolume(const int k, const int j, const int i);

  // ...to compute geometrical source terms
  virtual void AddCoordTermsDivergence(const Real dt, const ParArrayND<Real> *flux,
                                       const ParArrayND<Real> &prim,
                                       const ParArrayND<Real> &bcc, ParArrayND<Real> &u);

  // In GR, functions...
  // ...to return private variables
  Real GetMass() const { return bh_mass_; }
  Real GetSpin() const { return bh_spin_; }

  // ...to compute metric
  void Metric(Real x1, Real x2, Real x3, ParameterInput *pin, ParArrayND<Real> &g,
              ParArrayND<Real> &g_inv, ParArrayND<Real> &dg_dx1, ParArrayND<Real> &dg_dx2,
              ParArrayND<Real> &dg_dx3);
  virtual void CellMetric(const int k, const int j, const int il, const int iu,
                          ParArrayND<Real> &g, ParArrayND<Real> &gi) {}
  virtual void Face1Metric(const int k, const int j, const int il, const int iu,
                           ParArrayND<Real> &g, ParArrayND<Real> &g_inv) {}
  virtual void Face2Metric(const int k, const int j, const int il, const int iu,
                           ParArrayND<Real> &g, ParArrayND<Real> &g_inv) {}
  virtual void Face3Metric(const int k, const int j, const int il, const int iu,
                           ParArrayND<Real> &g, ParArrayND<Real> &g_inv) {}

  // ...to transform primitives to locally flat space
  virtual void PrimToLocal1(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &b1_vals, ParArrayND<Real> &prim_left,
                            ParArrayND<Real> &prim_right, ParArrayND<Real> &bx) {}
  virtual void PrimToLocal2(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &b2_vals, ParArrayND<Real> &prim_left,
                            ParArrayND<Real> &prim_right, ParArrayND<Real> &bx) {}
  virtual void PrimToLocal3(const int k, const int j, const int il, const int iu,
                            const ParArrayND<Real> &b3_vals, ParArrayND<Real> &prim_left,
                            ParArrayND<Real> &prim_right, ParArrayND<Real> &bx) {}

  // ...to transform fluxes in locally flat space to global frame
  virtual void FluxToGlobal1(const int k, const int j, const int il, const int iu,
                             const ParArrayND<Real> &cons, const ParArrayND<Real> &bbx,
                             ParArrayND<Real> &flux, ParArrayND<Real> &ey,
                             ParArrayND<Real> &ez) {}
  virtual void FluxToGlobal2(const int k, const int j, const int il, const int iu,
                             const ParArrayND<Real> &cons, const ParArrayND<Real> &bbx,
                             ParArrayND<Real> &flux, ParArrayND<Real> &ey,
                             ParArrayND<Real> &ez) {}
  virtual void FluxToGlobal3(const int k, const int j, const int il, const int iu,
                             const ParArrayND<Real> &cons, const ParArrayND<Real> &bbx,
                             ParArrayND<Real> &flux, ParArrayND<Real> &ey,
                             ParArrayND<Real> &ez) {}

  // ...to raise (lower) covariant (contravariant) components of a vector
  virtual void RaiseVectorCell(Real a_0, Real a_1, Real a_2, Real a_3, int k, int j,
                               int i, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {}
  virtual void LowerVectorCell(Real a0, Real a1, Real a2, Real a3, int k, int j, int i,
                               Real *pa_0, Real *pa_1, Real *pa_2, Real *pa_3) {}

 protected:
  bool coarse_flag; // true if this coordinate object is parent (coarse) mesh in AMR
  Mesh *pm;
  int il, iu, jl, ju, kl, ku, ng; // limits of indices of arrays (normal or coarse)
  int nc1, nc2, nc3;              // # cells in each dir of arrays (normal or coarse)
  // Scratch arrays for coordinate factors
  // Format: coord_<type>[<direction>]_<index>[<count>]_
  //   type: vol[ume], area, etc.
  //   direction: 1/2/3 depending on which face, edge, etc. is in play
  //   index: i/j/k indicating which coordinates index array
  //   count: 1/2/... in case multiple arrays are needed for different terms
  ParArrayND<Real> coord_vol_i_, coord_vol_i1_, coord_vol_i2_;
  ParArrayND<Real> coord_vol_j_, coord_vol_j1_, coord_vol_j2_;
  ParArrayND<Real> coord_vol_k1_;
  ParArrayND<Real> coord_vol_kji_;
  ParArrayND<Real> coord_area1_i_, coord_area1_i1_;
  ParArrayND<Real> coord_area1_j_, coord_area1_j1_, coord_area1_j2_;
  ParArrayND<Real> coord_area1_k1_;
  ParArrayND<Real> coord_area1_kji_;
  ParArrayND<Real> coord_area2_i_, coord_area2_i1_, coord_area2_i2_;
  ParArrayND<Real> coord_area2_j_, coord_area2_j1_, coord_area2_j2_;
  ParArrayND<Real> coord_area2_k1_;
  ParArrayND<Real> coord_area2_kji_;
  ParArrayND<Real> coord_area3_i_, coord_area3_i1_, coord_area3_i2_;
  ParArrayND<Real> coord_area3_j1_, coord_area3_j2_;
  ParArrayND<Real> coord_area3_kji_;
  ParArrayND<Real> coord_area1vc_i_, coord_area1vc_j_; // nonidealmhd additions
  ParArrayND<Real> coord_area2vc_i_, coord_area2vc_j_; // nonidealmhd additions
  ParArrayND<Real> coord_area3vc_i_;                   // nonidealmhd addition
  ParArrayND<Real> coord_len1_i1_, coord_len1_i2_;
  ParArrayND<Real> coord_len1_j1_, coord_len1_j2_;
  ParArrayND<Real> coord_len1_kji_;
  ParArrayND<Real> coord_len2_i1_;
  ParArrayND<Real> coord_len2_j1_, coord_len2_j2_;
  ParArrayND<Real> coord_len2_kji_;
  ParArrayND<Real> coord_len3_i1_;
  ParArrayND<Real> coord_len3_j1_, coord_len3_j2_;
  ParArrayND<Real> coord_len3_k1_;
  ParArrayND<Real> coord_len3_kji_;
  ParArrayND<Real> coord_width1_i1_;
  ParArrayND<Real> coord_width1_kji_;
  ParArrayND<Real> coord_width2_i1_;
  ParArrayND<Real> coord_width2_j1_;
  ParArrayND<Real> coord_width2_kji_;
  ParArrayND<Real> coord_width3_j1_, coord_width3_j2_, coord_width3_j3_;
  ParArrayND<Real> coord_width3_k1_;
  ParArrayND<Real> coord_width3_ji1_;
  ParArrayND<Real> coord_width3_kji_;
  ParArrayND<Real> coord_src_j1_, coord_src_j2_;
  ParArrayND<Real> coord_src_kji_;
  ParArrayND<Real> coord_src1_i_;
  ParArrayND<Real> coord_src1_j_;
  ParArrayND<Real> coord_src2_i_;
  ParArrayND<Real> coord_src2_j_;
  ParArrayND<Real> coord_src3_j_;

  // Scratch arrays for physical source terms
  ParArrayND<Real> phy_src1_i_, phy_src2_i_;

  // GR-specific scratch arrays
  ParArrayND<Real> metric_cell_i1_, metric_cell_i2_;
  ParArrayND<Real> metric_cell_j1_, metric_cell_j2_;
  ParArrayND<Real> metric_cell_kji_;
  ParArrayND<Real> metric_face1_i1_, metric_face1_i2_;
  ParArrayND<Real> metric_face1_j1_, metric_face1_j2_;
  ParArrayND<Real> metric_face1_kji_;
  ParArrayND<Real> metric_face2_i1_, metric_face2_i2_;
  ParArrayND<Real> metric_face2_j1_, metric_face2_j2_;
  ParArrayND<Real> metric_face2_kji_;
  ParArrayND<Real> metric_face3_i1_, metric_face3_i2_;
  ParArrayND<Real> metric_face3_j1_, metric_face3_j2_;
  ParArrayND<Real> metric_face3_kji_;
  ParArrayND<Real> trans_face1_i1_, trans_face1_i2_;
  ParArrayND<Real> trans_face1_j1_;
  ParArrayND<Real> trans_face1_ji1_, trans_face1_ji2_, trans_face1_ji3_, trans_face1_ji4_,
      trans_face1_ji5_, trans_face1_ji6_, trans_face1_ji7_;
  ParArrayND<Real> trans_face1_kji_;
  ParArrayND<Real> trans_face2_i1_, trans_face2_i2_;
  ParArrayND<Real> trans_face2_j1_;
  ParArrayND<Real> trans_face2_ji1_, trans_face2_ji2_, trans_face2_ji3_, trans_face2_ji4_,
      trans_face2_ji5_, trans_face2_ji6_;
  ParArrayND<Real> trans_face2_kji_;
  ParArrayND<Real> trans_face3_i1_, trans_face3_i2_;
  ParArrayND<Real> trans_face3_j1_;
  ParArrayND<Real> trans_face3_ji1_, trans_face3_ji2_, trans_face3_ji3_, trans_face3_ji4_,
      trans_face3_ji5_, trans_face3_ji6_;
  ParArrayND<Real> trans_face3_kji_;
  ParArrayND<Real> g_, gi_;

  // GR-specific variables
  Real bh_mass_;
  Real bh_spin_;
};

//----------------------------------------------------------------------------------------
//! \class Cartesian
//  \brief derived class for Cartesian coordinates.  None of the virtual funcs
//  in the Coordinates abstract base class need to be overridden.

class Cartesian : public Coordinates {

 public:
  Cartesian(MeshBlock *pmb, ParameterInput *pin, bool flag);
};
} // namespace parthenon
#endif // COORDINATES_COORDINATES_HPP_
