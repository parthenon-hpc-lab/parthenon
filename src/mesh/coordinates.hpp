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

#ifndef MESH_COORDINATES_HPP_
#define MESH_COORDINATES_HPP_

#include <array>

#include "basic_types.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

class Coordinates {
 public:
  Coordinates() = default;
  Coordinates(std::array<Real, 3> xmin, std::array<Real, 3> dx, std::array<int, 3> istart)
    : xmin_(xmin), dx_(dx), istart_(istart),
      area_({dx[1]*dx[2], dx[0]*dx[2], dx[0]*dx[1]}),
      cell_volume_(dx[0]*dx[1]*dx[2]) { }

  const Real GetVolume() { return cell_volume_; }
  const std::array<Real, 3> GetDx() { return dx_; }
  const Real GetDx(const int dir) { return dx_[dir]; }
  const std::array<Real, 3> GetArea() { return area_; }
  const Real GetArea(const int dir) { return area_[dir]; }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x1v(const int i) { return xmin_[0] + (i-istart_[0]+0.5)*dx_[0]; }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x1f(const int i) { return xmin_[0] + (i-istart_[0])*dx_[0]; }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x2v(const int j) { return xmin_[1] + (j-istart_[1]+0.5)*dx_[1]; }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x2f(const int j) { return xmin_[1] + (j-istart_[1])*dx_[1]; }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x3v(const int k) { return xmin_[2] + (k-istart_[2]+0.5)*dx_[2]; }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x3f(const int k) { return xmin_[2] + (k-istart_[2])*dx_[2]; }

  // k, j, i grid functions
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x1v(const int k, const int j, const int i) {
    return xmin_[0] + (i-istart_[0]+0.5)*dx_[0];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x1f(const int k, const int j, const int i) {
    return xmin_[0] + (i-istart_[0])*dx_[0];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x2v(const int k, const int j, const int i) {
    return xmin_[1] + (j-istart_[1]+0.5)*dx_[1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x2f(const int k, const int j, const int i) {
    return xmin_[1] + (j-istart_[1])*dx_[1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x3v(const int k, const int j, const int i) {
    return xmin_[2] + (k-istart_[2]+0.5)*dx_[2];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  const Real x3f(const int k, const int j, const int i) {
    return xmin_[2] + (k-istart_[2])*dx_[2];
  }

  const std::array<Real, 3> GetXmin() { return xmin_;}

 private:
  std::array<int, 3> istart_;
  std::array<Real, 3> xmin_, dx_, area_;
  Real cell_volume_;
};

} // namespace parthenon

#endif // MESH_COORDINATES_HPP_
