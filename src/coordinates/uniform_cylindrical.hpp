//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef COORDINATES_UNIFORM_CYLINDRICAL_HPP_
#define COORDINATES_UNIFORM_CYLINDRICAL_HPP_

#include <array>
#include <cassert>

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

class UniformCylindrical {
 public:
  UniformCylindrical() = default;
  UniformCylindrical(const RegionSize &rs, ParameterInput *pin) {
    dx_[0] = (rs.x1max - rs.x1min) / rs.nx1;
    dx_[1] = (rs.x2max - rs.x2min) / rs.nx2;
    dx_[2] = (rs.x3max - rs.x3min) / rs.nx3;
    area_[0] = dx_[1] * dx_[2];
    area_[1] = dx_[0] * dx_[2];
    area_[2] = dx_[0] * dx_[1];
    istart_[0] = Globals::nghost;
    istart_[1] = (rs.nx2 > 1 ? Globals::nghost : 0);
    istart_[2] = (rs.nx3 > 1 ? Globals::nghost : 0);
    xmin_[0] = rs.x1min - istart_[0] * dx_[0];
    xmin_[1] = rs.x2min - istart_[1] * dx_[1];
    xmin_[2] = rs.x3min - istart_[2] * dx_[2];
    nx_[0] = rs.nx1;
    nx_[1] = rs.nx2;
    nx_[2] = rs.nx3;

    std::string coord_type_str = pin->GetOrAddString("parthenon/mesh", "coord", "cylindrical");
    if( coord_type_str != "cylindrical" ){
      if (coord_type_str == "cartesian") {
        PARTHENON_THROW(" Please rebuild with -DCOORDINATE_TYPE=UniformCartesian");
      } else if (coord_type_str == "spherical") {
        PARTHENON_THROW(" Please rebuild with -DCOORDINATE_TYPE=UniformSpherical");
      } else {
        PARTHENON_THROW("Invalid coord input in <parthenon/mesh>.");
      }
    }
  }
  UniformCylindrical(const UniformCylindrical &src, int coarsen)
      : istart_(src.GetStartIndex()) {
    dx_ = src.Dx_();
    xmin_ = src.GetXmin();
    xmin_[0] += istart_[0] * dx_[0] * (1 - coarsen);
    xmin_[1] += istart_[1] * dx_[1] * (1 - coarsen);
    xmin_[2] += istart_[2] * dx_[2] * (1 - coarsen);
    dx_[0] *= coarsen;
    dx_[1] *= (istart_[1] > 0 ? coarsen : 1);
    dx_[2] *= (istart_[2] > 0 ? coarsen : 1);
    area_[0] = dx_[1] * dx_[2];
    area_[1] = dx_[0] * dx_[2];
    area_[2] = dx_[0] * dx_[1];
    nx_[0] = src.nx_[0] / ( coarsen ? 2 : 1 );
    nx_[1] = src.nx_[1] / ( coarsen ? 2 : 1 );
    nx_[2] = src.nx_[2] / ( coarsen ? 2 : 1 );
  }

  //----------------------------------------
  // Dxc: Distance between cell centers
  // Dxc<1> returns distance between cell centroids in r
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int idx) const {
    assert(dir > 0 && dir < 4);
    if( dir == X1DIR ){
      return R_c_(idx+1) - R_c_(idx);
    } else {
      return dx_[dir-1];
    }
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case X1DIR:
      return Dxc<X1DIR>(i);
    case X2DIR:
      return Dxc<X2DIR>(j);
    case X3DIR:
      return Dxc<X3DIR>(k);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION Real DxcFA(const int dir, const int idx) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case X1DIR:
      return Dxc<X1DIR>(idx);
    case X2DIR:
      return Dxc<X2DIR>(idx);
    case X3DIR:
      return Dxc<X3DIR>(idx);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION Real DxcFA(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case X1DIR:
      return Dxc<X1DIR>(i);
    case X2DIR:
      return Dxc<X2DIR>(j);
    case X3DIR:
      return Dxc<X3DIR>(k);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }

  //----------------------------------------
  // Dxf: Distance between cell faces
  //----------------------------------------
  template <int dir, int face, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxf(Args... args) const {
    assert(dir > 0 && dir < 4 && face > 0 && face < 4);
    return dx_[face - 1];
  }
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxf(Args... args) const {
    assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }

  //----------------------------------------
  // Xc: Positions at cell centers
  // Xc<1> returns r of cell centroid
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int idx) const {
    assert(dir > 0 && dir < 4);
    if( dir == X1DIR ){
      return R_c_(idx);
    } else {
      return xmin_[dir - 1] + (idx + 0.5) * dx_[dir - 1];
    }
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case X1DIR:
      return Xc<dir>(i);
    case X2DIR:
      return Xc<dir>(j);
    case X3DIR:
      return Xc<dir>(k);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }


  //----------------------------------------
  // Xf: Positions on Faces
  //----------------------------------------
  template <int dir, int face>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int idx) const {
    assert(dir > 0 && dir < 4 && face > 0 && face < 4);
    // Return position in direction "dir" along index "idx" on face "face"
    if constexpr (dir == face) {
      return xmin_[dir - 1] + idx * dx_[dir - 1];
    } else {
      return Xc<dir>(idx);
    }
  }
  template <int dir, int face>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case X1DIR:
      return Xf<dir, face>(i);
    case X2DIR:
      return Xf<dir, face>(j);
    case X3DIR:
      return Xf<dir, face>(k);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int idx) const {
    assert(dir > 0 && dir < 4);
    // Return position in direction "dir" along index "idx" on face "dir"
    return xmin_[dir - 1] + idx * dx_[dir - 1];
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case X1DIR:
      return Xf<dir>(i);
    case X2DIR:
      return Xf<dir>(j);
    case X3DIR:
      return Xf<dir>(k);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }

  //----------------------------------------
  // Xs: Area averaged positions
  //----------------------------------------
  template <int dir, int side>
  KOKKOS_FORCEINLINE_FUNCTION Real Xs(const int idx) const {
    assert(dir > 0 && dir < 4 && side > 0 && side < 4);
    return Xc<dir>(idx);
  }
  template <int dir, int side>
  KOKKOS_FORCEINLINE_FUNCTION Real Xs(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4 && side > 0 && side < 4);
    return Xc<dir>(k,j,i);
  }

  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real X(const int idx) const {
    using TE = TopologicalElement;
    bool constexpr X1EDGE = el == TE::F1 || el == TE::E2 || el == TE::E3 || el == TE::NN;
    bool constexpr X2EDGE = el == TE::F2 || el == TE::E3 || el == TE::E1 || el == TE::NN;
    bool constexpr X3EDGE = el == TE::F3 || el == TE::E1 || el == TE::E2 || el == TE::NN;
    if constexpr (dir == X1DIR && X1EDGE) {
      return xmin_[dir - 1] + idx * dx_[dir - 1]; // idx - 1/2
    } else if constexpr (dir == X2DIR && X2EDGE) {
      return xmin_[dir - 1] + idx * dx_[dir - 1]; // idx - 1/2
    } else if constexpr (dir == X3DIR && X3EDGE) {
      return xmin_[dir - 1] + idx * dx_[dir - 1]; // idx - 1/2
    } else {
      return xmin_[dir - 1] + (idx + 0.5) * dx_[dir - 1]; // idx
    }
    return 0; // This should never be reached, but w/o it some compilers generate warnings
  }

  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real X(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case 1:
      return X<dir, el>(i);
    case 2:
      return X<dir, el>(j);
    case 3:
      return X<dir, el>(k);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }

  //----------------------------------------
  // CellWidth: Physical width of cells at cell centers
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    if( dir == X2DIR ){
      return Xf<1>(k, j, i)*dx_[1]; //r*dphi
    } else {
      return dx_[dir-1];
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidthFA(const int dir,const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    if( dir == X2DIR ){
      return Xf<1>(k, j, i)*dx_[1]; //r*dphi
    } else {
      return dx_[dir-1];
    }
  }

  //----------------------------------------
  // EdgeLength: Physical length of cell edges
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    if( dir == X2DIR ){
      return Xf<1>(k, j, i)*dx_[1]; //r*dphi
    } else {
      return dx_[dir-1];
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLengthFA(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    if( dir == X2DIR ){
      return Xf<1>(k, j, i)*dx_[1]; //r*dphi
    } else {
      return dx_[dir-1];
    }
  }

  //----------------------------------------
  // FaceArea: Physical area of cell areas
  //----------------------------------------
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir) {
      case X1DIR:
        return X<dir, TE::F1>(i)*area_[0]; //r*dphi*dz
      case X2DIR:
        return area_[1]; //dr*dz
      case X3DIR:
        return Coord_vol_i_(i)*dx_[1]; //d(r^2/2)*dphi
    }
    return NAN; //To appease compiler
  }
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceAreaFA(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir) {
      case X1DIR:
        return X<1, TE::F1>(i)*area_[0]; //r*dphi*dz
      case X2DIR:
        return area_[1]; //dr*dz
      case X3DIR:
        return Coord_vol_i_(i)*dx_[1]; //d(r^2/2)*dphi
    }
    return NAN; //To appease compiler
  }

  //----------------------------------------
  // CellVolume
  //----------------------------------------
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j, const int i) const {
    return Coord_vol_i_(i)*area_[0];
  }

  //----------------------------------------
  // Generalized physical volume
  //----------------------------------------
  template <TopologicalElement el, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Volume(Args... args) const {
    using TE = TopologicalElement;
    if constexpr (el == TE::CC) {
      return CellVolume(args...);
    } else if constexpr (el == TE::F1) {
      return FaceArea<X1DIR>(args...);
    } else if constexpr (el == TE::F2) {
      return FaceArea<X2DIR>(args...);
    } else if constexpr (el == TE::F3) {
      return FaceArea<X3DIR>(args...);
    } else if constexpr (el == TE::E1) {
      return EdgeLength<X1DIR>(args...);
    } else if constexpr (el == TE::E2) {
      return EdgeLength<X2DIR>(args...);
    } else if constexpr (el == TE::E3) {
      return EdgeLength<X3DIR>(args...);
    } else if constexpr (el == TE::NN) {
      return 1.0;
    }
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

  //----------------------------------------
  // Cylindrical+Spherical Conversions
  //----------------------------------------
  // TODO

  //----------------------------------------
  // Geometric Terms (Find better names for these!)
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION
  Real h2v(const int i) const { return R_c_(i); };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h2f(const int i) const { return Xf<1>(i); };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h31v(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h31f(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh2vd1(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh2fd1(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh31vd1(const int i) const { return 0.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh31fd1(const int i) const { return 0.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h32v(const int j) const { return 1.0; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real h32f(const int j) const { return 1.0; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh32vd2(const int j) const { return 0.0; }

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh32fd2(const int j) const { return 0.0; }

  //What is this?
  KOKKOS_FORCEINLINE_FUNCTION
  void GetAcc(const int k, const int j, const int i, const Real &rp,
	      const Real &cosphip, const Real &sinphip, const Real &zp,
	      Real &acc1, Real &acc2, Real &acc3) const
  {
    const Real cosdphi = Cos_phi_c_(j)*cosphip - Sin_phi_c_(j)*sinphip; //cos(x3v(k)-phip)
    const Real sindphi = Sin_phi_c_(j)*cosphip - Cos_phi_c_(j)*sinphip; //sin(x3v(k)-phip)
    //Psi = - 1.0/sqrt(r0^2 + rp^2 - 2r0*rp*cos(phi-phip) + (z-zp)^2)
    acc1 = (R_c_(i) - rp*cosdphi); //acc1 = dPsi/dr, Psi=-1/dist2
    acc2 = rp*sindphi;        //acc2 = dPsi/(r*dphi) 
    acc3 = Xc<3>(k)-zp;   
  }

  //----------------------------------------
  // Position to Cell index
  //----------------------------------------
  KOKKOS_INLINE_FUNCTION
  void Xtoijk(const Real &r, const Real &theta, const Real &z, int &i, int &j, int &k) const {
    i = (nx_[0] != 1) ? static_cast<int>(
        std::floor((r - xmin_[0]) / dx_[0])) +
      istart_[0] : istart_[0] ;
    j = (nx_[1] != 1) ? static_cast<int>(std::floor(
          (theta - xmin_[1]) / dx_[1])) +
      istart_[1]
      : istart_[1];
    k = (nx_[2] != 1)? static_cast<int>(std::floor(
          (z - xmin_[2]) / dx_[2])) +
      istart_[2]
      : istart_[2];
  }

  //----------------------------------------
  // Terms for Source Terms
  //----------------------------------------

  KOKKOS_INLINE_FUNCTION Real Coord_vol_i_(const int i) const {
      const Real rm = xmin_[0] + i * dx_[0];
      const Real rp = xmin_[0] + (i+1) * dx_[0];
      return 0.5*(rp*rp - rm*rm);
  }

  KOKKOS_INLINE_FUNCTION Real CoordSrc1i(const int i) const {
    return Dxf<1,1>(i)/Coord_vol_i_(i);
  }
  KOKKOS_INLINE_FUNCTION Real CoordSrc2i(const int i) const {
    const Real rm = xmin_[0] + i * dx_[0];
    const Real rp = xmin_[0] + (i+1) * dx_[0];
    return Dxf<1,1>(i)/( (rm + rp)*Coord_vol_i_(i));
  }

  KOKKOS_INLINE_FUNCTION
  Real PhySrc1i(const int i) const {
    return 1./( Xc<1>(i)*Xf<1>(i) );
  }

  KOKKOS_INLINE_FUNCTION
  Real PhySrc2i(const int i) const {
    return 1./( Xc<1>(i)*Xf<1>(i+1) );
  }

  const std::array<Real, 3> &GetXmin() const { return xmin_; }
  const std::array<int, 3> &GetStartIndex() const { return istart_; }
  const char *Name() const { return name_; }

 private:
  std::array<int, 3> istart_;
  std::array<Real, 3> xmin_, dx_, area_, nx_;
  constexpr static const char *name_ = "UniformCylindrical";

  const std::array<Real, 3> &Dx_() const { return dx_; }

  //ParArrayND<Real> r_c_, coord_vol_i_, cos_phi_c_, sin_phi_c_;

  KOKKOS_INLINE_FUNCTION Real R_c_(const int i) const {
      const Real rm = xmin_[0] + i * dx_[0];
      const Real rp = xmin_[0] + (i+1) * dx_[0];
      return TWO_3RD*(rp*rp*rp - rm*rm*rm)/(SQR(rp) - SQR(rm));
  }
  KOKKOS_INLINE_FUNCTION Real Cos_phi_c_(const int j) const {
      const Real Phi = xmin_[1] + (j + 0.5) * dx_[1];
      return std::cos(Phi);
  }
  KOKKOS_INLINE_FUNCTION Real Sin_phi_c_(const int j) const {
      const Real Phi = xmin_[1] + (j + 0.5) * dx_[1];
      return std::sin(Phi);
  }

};


} // namespace parthenon

#endif // COORDINATES_UNIFORM_CYLINDRICAL_HPP_
