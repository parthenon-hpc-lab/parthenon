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

  //----------------------------------------
  // Dxc: Distance between cell centers
  //----------------------------------------
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(Args... args) const {
    assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
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
  // Xf: Positions at cell centers
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int idx) const {
    assert(dir > 0 && dir < 4);
    return xmin_[dir - 1] + (idx + 0.5) * dx_[dir - 1];
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
      PARTHENON_THROW("Unknown dir");
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
      PARTHENON_THROW("Unknown dir");
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
      PARTHENON_THROW("Unknown dir");
      return 0; // To appease compiler
    }
  }

  //----------------------------------------
  // CellWidth: Width of cells at cell centers
  //----------------------------------------
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(Args... args) const {
    assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real cellWidth(const int dir, Args... args) const {
    assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }

  //----------------------------------------
  // EdgeLength: Length of cell edges
  //----------------------------------------
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(Args... args) const {
    assert(dir > 0 && dir < 4);
    return CellWidth<dir>();
  }
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real edgeLength(const int dir, Args... args) const {
    return cellWidth(dir);
  }

  //----------------------------------------
  // EdgeLength: Length of cell edges
  //----------------------------------------
  template <int dir, class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(Args... args) const {
    assert(dir > 0 && dir < 4);
    return area_[dir - 1];
  }
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real faceArea(const int dir, Args... args) const {
    assert(dir > 0 && dir < 4);
    return area_[dir - 1];
  }

  //----------------------------------------
  // CellVolume
  //----------------------------------------
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(Args... args) const {
    return cell_volume_;
  }

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
