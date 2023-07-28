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
#ifndef COORDINATES_UNIFORM_SPHERICAL_HPP_
#define COORDINATES_UNIFORM_SPHERICAL_HPP_

#include <array>
#include <cassert>

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

class UniformSpherical {
 public:
  UniformSpherical() = default;
  UniformSpherical(const RegionSize &rs, ParameterInput *pin) {
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

    std::string coord_type_str = pin->GetOrAddString("parthenon/mesh", "coord", "cartesian");
    //REMOVE ME
    //if (coord_type_str == "cartesian") {
    //  coord_type = parthenon::uniformOrthMesh::cartesian;
    //} else 
    if( coord_type_str != "spherical" ){
      if (coord_type_str == "cartesian") {
        PARTHENON_THROW(" Please rebuild with -DCOORDINATE_TYPE=UniformCartesian");
      } else if (coord_type_str == "cylindrical") {
        PARTHENON_THROW(" Please rebuild with -DCOORDINATE_TYPE=UniformCylindrical");
      } else {
        PARTHENON_THROW("Invalid coord input in <parthenon/mesh>.");
      }
    }

    //Initialized cached coordinates
    InitCachedArrays();
  }
  UniformSpherical(const UniformSpherical &src, int coarsen)
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

    //Initialized cached coordinates
    InitCachedArrays();
  }

  //----------------------------------------
  // Dxc: Distance between cell centers
  // Dxc<1> and Dxc<2> returns distance between cell centroids
  //----------------------------------------
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir){
      case X1DIR:
        return r_c_(i+1) - r_c_(i);
      case X2DIR:
        return theta_c_(j+1) - theta_c_(j);
      default:
        return dx_[2];
    }
  }
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real DxcFA(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir){
      case X1DIR:
        return r_c_(i+1) - r_c_(i);
      case X2DIR:
        return theta_c_(j+1) - theta_c_(j);
      default:
        return dx_[2];
    }
  }

  //----------------------------------------
  // Dxf: Distance between cell faces
  //----------------------------------------
  template <int dir, int face, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxf(Args... args) const {
    assert(dir > 0 && dir < 4 && face > 0 && face < 4);
    //(forrestglines): Double check how dir and face are used
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
    switch (dir) {
    case 1:
      return r_c_(idx);
    case 2:
      return theta_c_(idx);
    case 3:
      return xmin_[dir - 1] + (idx + 0.5) * dx_[dir - 1];
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case 1:
      return Xc<dir>(i);
    case 2:
      return Xc<dir>(j);
    case 3:
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
    case 1:
      return Xf<dir, face>(i);
    case 2:
      return Xf<dir, face>(j);
    case 3:
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
    case 1:
      return Xf<dir>(i);
    case 2:
      return Xf<dir>(j);
    case 3:
      return Xf<dir>(k);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
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
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir){
      case X1DIR:
        return Dxf<1>(k,j,i);
      case X2DIR:
        return Xc<1>(k,j,i)*Dxf<2>(k,j,i);//r*dphi;
      case X3DIR:
        return Xc<1>(k,j,i)*sintht_c_(j)*Dxf<3>(k,j,i); //r*sin(th)*dphi
      default:
        PARTHENON_FAIL("Unknown dir");
        return 0; // To appease compiler
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidthFA(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    if( dir == X2DIR ){
      return Xf<1>( k, j, i)*dx_[1]; //r*dphi
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
    switch(dir){
      case X1DIR:
        return Dxf<1>(k,j,i);
      case X2DIR:
        return Xf<1>(k,j,i)*Dxf<2>(k,j,i);//r*dphi;
      case X3DIR:
        return Xf<1>(k,j,i)*sintht_f_(j)*Dxf<3>(k,j,i); //r*sin(th)*dphi
      default:
        PARTHENON_FAIL("Unknown dir");
        return 0; // To appease compiler
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLengthFA(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir){
      case X1DIR:
        return EdgeLength<X1DIR>(k, j, i);
      case X2DIR:
        return EdgeLength<X2DIR>(k, j, i);
      case X3DIR:
        return EdgeLength<X3DIR>(k, j, i);
      default:
        PARTHENON_FAIL("Unknown dir");
        return 0; // To appease compiler
    }
  }

  //----------------------------------------
  // FaceArea: Physical area of cell areas
  //----------------------------------------
  //(How is this different from Area(dir, k, j, i)?
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir) {
      case X1DIR:
        return (Xf<1>(i)*Xf<1>(i)*(costht_f_(j)-costht_f_(j+1))*dx_[2]); //r^2*d[-cos(tht)]*dph
      case X2DIR:
        return (coord_area2_i_(i)*sintht_f_(j)*dx_[2]);//rdr*sin(th)*dph
      case X3DIR:
        return (coord_area2_i_(i)*dx_[1]); //d(r^2/2)*dtheta;
    }

  }
  KOKKOS_FORCEINLINE_FUNCTION Real FaceAreaFA(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    switch(dir){
      case X1DIR:
        return FaceArea<X1DIR>(k, j, i);
      case X2DIR:
        return FaceArea<X2DIR>(k, j, i);
      case X3DIR:
        return FaceArea<X3DIR>(k, j, i);
      default:
        PARTHENON_FAIL("Unknown dir");
        return 0; // To appease compiler
    }
  }

  //----------------------------------------
  // CellVolume
  //----------------------------------------
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j, const int i) const {
    return (coord_vol_i_(i)*coord_area1_j_(j)*dx_[2]);
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
  // Area Averaged Positions (for CT MHD?) AMR
  //----------------------------------------
  // TODO
  // x1s2
  // x1s3
  // x2s1
  // etc.

  //----------------------------------------
  // Coordinate source terms?
  //----------------------------------------
  // TODO
  // coord_src1_i
  // coord_src2_i
  // coord_area1_j
  // coord_area2_j

  //----------------------------------------
  // Geometric Terms (Find better names for these!)
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION
  Real h2v(const int i) const { return r_c_(i); };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h2f(const int i) const { return Xf<1>(i); };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h31v(const int i) const { return r_c_(i); };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h31f(const int i) const { return Xf<1>(i); };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh2vd1(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh2fd1(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh31vd1(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh31fd1(const int i) const { return 1.0; };

  KOKKOS_FORCEINLINE_FUNCTION
  Real h32v(const int j) const { return sintht_c_(j); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real h32f(const int j) const { return sintht_f_(j); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh32vd2(const int j) const { return costht_c_(j); }

  KOKKOS_FORCEINLINE_FUNCTION
  Real dh32fd2(const int j) const { return costht_f_(j); }

  //What is this?
  KOKKOS_FORCEINLINE_FUNCTION
  void GetAcc(const int k, const int j, const int i, const Real &rp,
	      const Real &cosphip, const Real &sinphip, const Real &zp,
	      Real &acc1, Real &acc2, Real &acc3) const
  {
    const Real cosdphi = cosphi_c_(j)*cosphip - sinphi_c_(j)*sinphip; //cos(x3v(k)-phip)
    const Real sindphi = sinphi_c_(j)*cosphip - cosphi_c_(j)*sinphip; //sin(x3v(k)-phip)
    //Psi = - 1.0/sqrt(r0^2 + rp^2 - 2r0*rp*sin(tht)cos(phi-phip) + zp^2- 2*r0*cos(tht)*zp
    acc1 = (r_c_(i) - rp*sintht_c_(j)*cosdphi - costht_c_(j)*zp); //dPsi/dr0
    acc2 = (-rp*costht_c_(j)*cosdphi + sintht_c_(j)*zp); //dPsi/(r0*dtht)
    acc3 = (rp*sindphi);  //dPsi/(r0*sintht*dphi)
  }

  //----------------------------------------
  // Position to Cell index
  //----------------------------------------
  KOKKOS_INLINE_FUNCTION
  void Xtoijk(const Real &r, const Real &theta, const Real &phi, int &i, int &j, int &k) const {
    i = (nx_[0] != 1) ? static_cast<int>(
        std::floor((r - xmin_[0]) / dx_[0])) +
      istart_[0] : istart_[0];
    j = (nx_[1] != 1) ? static_cast<int>(std::floor(
          (theta - xmin_[1]) / dx_[1])) +
      istart_[1]
      : istart_[1];
    k = (nx_[2] != 1)? static_cast<int>(std::floor(
          (phi - xmin_[2]) / dx_[2])) +
      istart_[2]
      : istart_[2];
  }

  const std::array<Real, 3> &GetXmin() const { return xmin_; }
  const std::array<int, 3> &GetStartIndex() const { return istart_; }
  const char *Name() const { return name_; }

 //private:
  std::array<int, 3> istart_, nx_;
  std::array<Real, 3> xmin_, dx_, area_;
  constexpr static const char *name_ = "UniformCylindrical";

  const std::array<Real, 3> &Dx_() const { return dx_; }

  ParArrayND<Real> r_c_, theta_c_, x1s2_, coord_vol_i_, coord_area2_i_, costht_f_, sintht_f_, sintht_c_, costht_c_, coord_area1_j_, cosphi_c_, sinphi_c_;
  void InitCachedArrays() {
    int nx1_tot = nx_[0]+2*Globals::nghost+1;
    r_c_ = ParArrayND<Real>("centroid(r)", nx1_tot-1);
    x1s2_ = ParArrayND<Real>("area1(r)", nx1_tot-1);
    coord_vol_i_ = ParArrayND<Real>("volume(r)", nx1_tot-1);
    coord_area2_i_ = ParArrayND<Real>("area2(r)", nx1_tot-1);
    parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeArraySph_r", 
      parthenon::DevExecSpace(), 0, nx1_tot-2,
      KOKKOS_LAMBDA(const int &i) {
  //for (int i = 0; i < nx1_tot-1; i++) {
      Real rm = xmin_[0] + i * dx_[0];
      Real rp = rm + dx_[0];
      r_c_(i) = 0.75*(std::pow(rp, 4) - std::pow(rm, 4)) /
  (std::pow(rp, 3) - std::pow(rm, 3));
      x1s2_(i) = TWO_3RD*(std::pow(rp,3) - std::pow(rm,3))/(SQR(rp) - SQR(rm));
      coord_vol_i_(i) = (ONE_3RD)*(rp*rp*rp - rm*rm*rm);
      coord_area2_i_(i) = 0.5*(rp*rp - rm*rm);
      });

    int nx2_tot = nx_[1]+1;
    if (istart_[1] > 0) {
      nx2_tot += 2*Globals::nghost;
    }
    costht_f_ = ParArrayND<Real>("cos(theta_f)", nx2_tot);
    sintht_f_ = ParArrayND<Real>("sin(theta_f)", nx2_tot);
    sintht_c_ = ParArrayND<Real>("sin(theta_c)", nx2_tot-1);
    costht_c_ = ParArrayND<Real>("cos(theta_c)", nx2_tot-1);
    coord_area1_j_ = ParArrayND<Real>("dcos(theta)", nx2_tot-1);
    theta_c_     = ParArrayND<Real>("centroid-theta", std::max(2,nx2_tot-1));

    parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeArraySph_th1", 
      parthenon::DevExecSpace(), 0, nx2_tot-1,
      KOKKOS_LAMBDA(const int &j) {
  //for (int j = 0; j < nx2_tot; j++) {
      Real theta = xmin_[1] + j * dx_[1];
      costht_f_(j) = std::cos(theta);
      sintht_f_(j) = std::sin(theta);
      });

    parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeArraySph_th2", 
      parthenon::DevExecSpace(), 0, nx2_tot-2,
      KOKKOS_LAMBDA(const int &j) {
  //for (int j = 0; j < nx2_tot-1; j++) {
      coord_area1_j_(j) = std::abs(costht_f_(j  ) - costht_f_(j+1));
      Real tm = xmin_[1] + j * dx_[1];
      Real tp = tm + dx_[1];
      theta_c_(j) = (((sintht_f_(j+1) - tp*costht_f_(j+1)) -
      (sintht_f_(j  ) - tm*costht_f_(j  )))/
      (costht_f_(j  ) - costht_f_(j+1)));
      sintht_c_(j) = std::sin(theta_c_(j));
      costht_c_(j) = std::cos(theta_c_(j));
      });

    if (nx2_tot==2) {
      theta_c_(1) = theta_c_(0) + dx_[1];
    }

    //phi-direction
    int nx3_tot = nx_[2]+1;
    if (istart_[2] > 0) {
      nx3_tot += 2*Globals::nghost;
    }
    sinphi_c_ = ParArrayND<Real>("sin(phi_c)", nx3_tot-1);
    cosphi_c_ = ParArrayND<Real>("cos(phi_c)", nx3_tot-1);
    parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "initializeArraySph_ph", 
      parthenon::DevExecSpace(), 0, nx3_tot-2,
      KOKKOS_LAMBDA(const int &k) {
  //for (int k = 0; k < nx3_tot-1; k++) {
      Real Phi = xmin_[2] + (k + 0.5) * dx_[2];
      cosphi_c_(k) = std::cos(Phi);
      sinphi_c_(k) = std::sin(Phi);
      });
  }

};

} // namespace parthenon

#endif // COORDINATES_UNIFORM_SPHERICAL_HPP_
