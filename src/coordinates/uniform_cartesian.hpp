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
#ifndef COORDINATES_UNIFORM_CARTESIAN_HPP_
#define COORDINATES_UNIFORM_CARTESIAN_HPP_

#include <array>

#include "basic_types.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

class UniformCartesian {
 public:
  UniformCartesian() = default;
  UniformCartesian(const RegionSize &rs, ParameterInput *pin) {
    xmin_[0] = rs.x1min;
    xmin_[1] = rs.x2min;
    xmin_[2] = rs.x3min;
    dx_[0] = (rs.x1max - rs.x1min)/rs.nx1;
    dx_[1] = (rs.x2max - rs.x2min)/rs.nx2;
    dx_[2] = (rs.x3max - rs.x3min)/rs.nx3;
    area_[0] = dx_[1]*dx_[2];
    area_[1] = dx_[0]*dx_[2];
    area_[2] = dx_[0]*dx_[1];
    cell_volume_ = dx_[0]*dx_[1]*dx_[2];
    istart_[0] = NGHOST;
    istart_[1] = (rs.nx2 > 1 ? NGHOST : 0);
    istart_[2] = (rs.nx3 > 1 ? NGHOST : 0);
  }

  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  Real Volume(Args... args) { return cell_volume_; }

  const std::array<Real, 3> &GetDx() { return dx_; }

  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  Real Dx(const int dir, Args... args) { return dx_[dir]; }

  const std::array<Real, 3> &GetArea() { return area_; }

  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION
  Real Area(const int dir, Args... args) { return area_[dir]; }


  KOKKOS_FORCEINLINE_FUNCTION
  Real x1v(const int i) { return xmin_[0] + (i - NGHOST + 0.5) * dx_[0]; }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x1f(const int i) { return xmin_[0] + (i - istart_[0]) * dx_[0]; }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x2v(const int j) { return xmin_[1] + (j - istart_[1] + 0.5) * dx_[1]; }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x2f(const int j) { return xmin_[1] + (j - istart_[1]) * dx_[1]; }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x3v(const int k) { return xmin_[2] + (k - istart_[2] + 0.5) * dx_[2]; }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x3f(const int k) { return xmin_[2] + (k - istart_[2]) * dx_[2]; }

  // k, j, i grid functions
  KOKKOS_FORCEINLINE_FUNCTION
  Real x1v(const int k, const int j, const int i) const {
    return xmin_[0] + (i - istart_[0] + 0.5) * dx_[0];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x1f(const int k, const int j, const int i) const {
    return xmin_[0] + (i - istart_[0]) * dx_[0];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x2v(const int k, const int j, const int i) const {
    return xmin_[1] + (j - istart_[1] + 0.5) * dx_[1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x2f(const int k, const int j, const int i) const {
    return xmin_[1] + (j - istart_[1]) * dx_[1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x3v(const int k, const int j, const int i) const {
    return xmin_[2] + (k - istart_[2] + 0.5) * dx_[2];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real x3f(const int k, const int j, const int i) const {
    return xmin_[2] + (k - istart_[2]) * dx_[2];
  }

  const std::array<Real, 3> &GetXmin() { return xmin_; }

 private:
  std::array<int, 3> istart_;
  std::array<Real, 3> xmin_, dx_, area_;
  Real cell_volume_;
};

} // namespace parthenon

#endif // COORDINATES_UNIFORM_CARTESIAN_HPP_