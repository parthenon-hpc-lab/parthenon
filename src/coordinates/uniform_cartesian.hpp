//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef COORDINATES_UNIFORM_CARTESIAN_HPP_
#define COORDINATES_UNIFORM_CARTESIAN_HPP_

#include <array>
#include <cassert>

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

class UniformCartesian {
 public:
  UniformCartesian() = default;
  UniformCartesian(const RegionSize &rs, ParameterInput *pin) {
    dx_[0] = (rs.x1max - rs.x1min) / rs.nx1;
    dx_[1] = (rs.x2max - rs.x2min) / rs.nx2;
    dx_[2] = (rs.x3max - rs.x3min) / rs.nx3;
    area_[0] = dx_[1] * dx_[2];
    area_[1] = dx_[0] * dx_[2];
    area_[2] = dx_[0] * dx_[1];
    cell_volume_ = dx_[0] * dx_[1] * dx_[2];
    istart_[0] = Globals::nghost;
    istart_[1] = (rs.nx2 > 1 ? Globals::nghost : 0);
    istart_[2] = (rs.nx3 > 1 ? Globals::nghost : 0);
    xmin_[0] = rs.x1min - istart_[0] * dx_[0];
    xmin_[1] = rs.x2min - istart_[1] * dx_[1];
    xmin_[2] = rs.x3min - istart_[2] * dx_[2];
  }
  UniformCartesian(const UniformCartesian &src, int coarsen)
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
    cell_volume_ = dx_[0] * dx_[1] * dx_[2];
  }

  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Volume(Args... args) const {
    return cell_volume_;
  }

  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dx(const int dir, Args... args) const {
    assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }

  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Area(const int dir, Args... args) const {
    assert(dir > 0 && dir < 4);
    return area_[dir - 1];
  }

  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(Args... args) const {
    return dx_[dir];
  }

  template <int face, int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxf(Args... args) const {
    return dx_[face];
  }

  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int dir, Args... args) const {
    return Dx(dir);
  }

  template<int dir>
  KOKKOS_FORCEINLINE_FUNCTION
  Real Xc(const int idx) const {
    return xmin_[dir] + (idx + 0.5) * dx_[dir];
  }

  template<int face>
  KOKKOS_FORCEINLINE_FUNCTION
  Real Xf(const int idx) const {
    return Xf<face,face>(idx);
  }

  template<int face, int dir>
  KOKKOS_FORCEINLINE_FUNCTION
  Real Xf(const int idx) const {
    //Return position in direction "dir" along index "idx" on face "dir"
    if constexpr( dir == face ) {
      return xmin_[dir] + (idx + 0.5) * dx_[dir];
    } else {
      return Xc<dir>(idx);
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real x1s2(const int i) const { return Xc<1>(i); }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x1s3(const int i) const { return Xc<1>(i); }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x2s1(const int j) const { return Xc<2>(j); }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x2s3(const int j) const { return Xc<2>(j); }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x3s1(const int k) const { return Xc<3>(k); }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x3s2(const int k) const { return Xc<3>(k); }

  //// k, j, i grid functions
  //KOKKOS_FORCEINLINE_FUNCTION
  //Real Xc<1>(const int k, const int j, const int i) const {
  //  return xmin_[0] + (i + 0.5) * dx_[0];
  //}
  //KOKKOS_FORCEINLINE_FUNCTION
  //Real Xf<1,1>(const int k, const int j, const int i) const { return xmin_[0] + i * dx_[0]; }
  //KOKKOS_FORCEINLINE_FUNCTION
  //Real Xc<2>(const int k, const int j, const int i) const {
  //  return xmin_[1] + (j + 0.5) * dx_[1];
  //}
  //KOKKOS_FORCEINLINE_FUNCTION
  //Real Xf<2,2>(const int k, const int j, const int i) const { return xmin_[1] + j * dx_[1]; }
  //KOKKOS_FORCEINLINE_FUNCTION
  //Real Xc<3>(const int k, const int j, const int i) const {
  //  return xmin_[2] + (k + 0.5) * dx_[2];
  //}
  //KOKKOS_FORCEINLINE_FUNCTION
  //Real Xf<3,3>(const int k, const int j, const int i) const { return xmin_[2] + k * dx_[2]; }

  const std::array<Real, 3> &GetXmin() const { return xmin_; }
  const std::array<int, 3> &GetStartIndex() const { return istart_; }
  const char *Name() const { return name_; }

 private:
  std::array<int, 3> istart_;
  std::array<Real, 3> xmin_, dx_, area_;
  Real cell_volume_;
  constexpr static const char *name_ = "UniformCartesian";

  const std::array<Real, 3> &Dx_() const { return dx_; }
};

} // namespace parthenon

#endif // COORDINATES_UNIFORM_CARTESIAN_HPP_
