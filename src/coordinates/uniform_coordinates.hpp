//========================================================================================
// (C) (or copyright) 2024-2025. Triad National Security, LLC. All rights reserved.
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
#ifndef COORDINATES_UNIFORM_COORDINATES_HPP_
#define COORDINATES_UNIFORM_COORDINATES_HPP_

#include <array>
#include <cassert>

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

template <typename System>
class UniformCoordinates {
 public:
  UniformCoordinates() = default;
  UniformCoordinates(const RegionSize &rs, ParameterInput *pin) {
    for (auto &dir : {X1DIR, X2DIR, X3DIR}) {
      dx_[dir - 1] = (rs.xmax(dir) - rs.xmin(dir)) / rs.nx(dir);
      istart_[dir - 1] = (!rs.symmetry(dir) ? Globals::nghost : 0);
      xmin_[dir - 1] = rs.xmin(dir) - istart_[dir - 1] * dx_[dir - 1];
    }
    area_[0] = dx_[1] * dx_[2];
    area_[1] = dx_[0] * dx_[2];
    area_[2] = dx_[0] * dx_[1];
    cell_volume_ = dx_[0] * dx_[1] * dx_[2];
  }
  UniformCoordinates(const UniformCoordinates<System> &src, int coarsen)
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
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dx() const {
    static_assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION Real Dx(const int dir) const {
    assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR)
      return static_cast<const System *>(this)->template Dxc<dir>(i);
    else if constexpr (dir == X2DIR)
      return static_cast<const System *>(this)->template Dxc<dir>(j);
    else if constexpr (dir == X3DIR)
      return static_cast<const System *>(this)->template Dxc<dir>(k);
    PARTHENON_FAIL("Unknown dir.");
    return 0.0;
  }
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int dir, const int k, const int j,
                                       const int i) const {
    assert(dir > 0 && dir < 4);
    if (dir == X1DIR)
      return static_cast<const System *>(this)->template Dxc<X1DIR>(k, j, i);
    else if (dir == X2DIR)
      return static_cast<const System *>(this)->template Dxc<X2DIR>(k, j, i);
    else if (dir == X3DIR)
      return static_cast<const System *>(this)->template Dxc<X3DIR>(k, j, i);
    PARTHENON_FAIL("Unknown dir.");
    return 0.0;
  }

  //----------------------------------------
  // Dxf: Distance between cell faces
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxf(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxf(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }

  //----------------------------------------
  // Xc: Positions at cell centroids
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    return xmin_[dir - 1] + (idx + 0.5) * dx_[dir - 1];
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR)
      return static_cast<const System *>(this)->template Xc<dir>(i);
    else if constexpr (dir == X2DIR)
      return static_cast<const System *>(this)->template Xc<dir>(j);
    else if constexpr (dir == X3DIR)
      return static_cast<const System *>(this)->template Xc<dir>(k);
    return 0; // To appease compiler
  }

  //----------------------------------------
  // Xf: Positions on Faces
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    return xmin_[dir - 1] + idx * dx_[dir - 1];
  }
  template <int dir, int face>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    static_assert(face > 0 && face < 4);
    if constexpr (dir == X1DIR && dir == face) {
      return Xf<dir>(i);
    } else if constexpr (dir == X1DIR) {
      return static_cast<const System *>(this)->template Xc<dir>(k, j, i);
    } else if constexpr (dir == X2DIR && dir == face) {
      return Xf<dir>(j);
    } else if constexpr (dir == X2DIR) {
      return static_cast<const System *>(this)->template Xc<dir>(k, j, i);
    } else if constexpr (dir == X3DIR && dir == face) {
      return Xf<dir>(k);
    } else if constexpr (dir == X3DIR) {
      return static_cast<const System *>(this)->template Xc<dir>(k, j, i);
    }
    return 0; // To appease compiler
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    return Xf<dir, dir>(k, j, i);
  }

  // X should not be overwritten in derived classes
  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real X(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR && TopologicalOffsetI(el)) {
      return xmin_[dir - 1] + idx * dx_[dir - 1]; // idx - 1/2
    } else if constexpr (dir == X2DIR && TopologicalOffsetJ(el)) {
      return xmin_[dir - 1] + idx * dx_[dir - 1]; // idx - 1/2
    } else if constexpr (dir == X3DIR && TopologicalOffsetK(el)) {
      return xmin_[dir - 1] + idx * dx_[dir - 1]; // idx - 1/2
    } else {
      return static_cast<const System *>(this)->template Xc<dir>(idx); // idx
    }
    return 0; // This should never be reached, but w/o it some compilers generate warnings
  }
  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real X(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR)
      return X<dir, el>(i);
    else if constexpr (dir == X2DIR)
      return X<dir, el>(j);
    else if constexpr (dir == X3DIR)
      return X<dir, el>(k);
    return 0.0;
  }

  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real Scale(const int k, const int j, const int i) const {
    if constexpr (dir > 0 && dir < 4) return 1.0;
    PARTHENON_FAIL("Unknown dir");
    return 0.0;
  }

  template <TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real Scale(const int dir, const int k, const int j,
                                         const int i) const {
    assert(dir > 0 && dir < 4);
    if (dir == X1DIR)
      return static_cast<const System *>(this)->template Scale<X1DIR, el>(k, j, i);
    else if (dir == X2DIR)
      return static_cast<const System *>(this)->template Scale<X2DIR, el>(k, j, i);
    else if (dir == X3DIR)
      return static_cast<const System *>(this)->template Scale<X3DIR, el>(k, j, i);
    return 0.0;
  }

  //----------------------------------------
  // CellWidth: Width of cells at cell centers
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int k, const int j,
                                             const int i) const {
    assert(dir > 0 && dir < 4);
    return dx_[dir - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int dir, const int k, const int j,
                                             const int i) const {
    assert(dir > 0 && dir < 4);
    if (dir == X1DIR)
      return static_cast<const System *>(this)->template CellWidth<X1DIR>(k, j, i);
    else if (dir == X2DIR)
      return static_cast<const System *>(this)->template CellWidth<X2DIR>(k, j, i);
    else if (dir == X3DIR)
      return static_cast<const System *>(this)->template CellWidth<X3DIR>(k, j, i);
    return 0.0;
  }

  //----------------------------------------
  // EdgeLength: Length of cell edges
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j,
                                              const int i) const {
    assert(dir > 0 && dir < 4);
    return CellWidth<dir>(k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int dir, const int k, const int j,
                                              const int i) const {
    return CellWidth(dir, k, j, i);
  }

  //----------------------------------------
  // FaceArea: Area of cell areas
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    return area_[dir - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int dir, const int k, const int j,
                                            const int i) const {
    assert(dir > 0 && dir < 4);
    switch (dir) {
    case X1DIR:
      return static_cast<const System *>(this)->template FaceArea<X1DIR>(k, j, i);
    case X2DIR:
      return static_cast<const System *>(this)->template FaceArea<X2DIR>(k, j, i);
    case X3DIR:
      return static_cast<const System *>(this)->template FaceArea<X3DIR>(k, j, i);
    default:
      PARTHENON_FAIL("Unknown dir");
      return 0; // To appease compiler
    }
  }

  //----------------------------------------
  // CellVolume
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j,
                                              const int i) const {
    return cell_volume_;
  }

  //----------------------------------------
  // Generalized volume
  //----------------------------------------
  template <TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real Volume(const int k, const int j, const int i) const {
    using TE = TopologicalElement;
    if constexpr (el == TE::CC) {
      return static_cast<const System *>(this)->CellVolume(k, j, i);
    } else if constexpr (el == TE::F1) {
      return static_cast<const System *>(this)->template FaceArea<X1DIR>(k, j, i);
    } else if constexpr (el == TE::F2) {
      return static_cast<const System *>(this)->template FaceArea<X2DIR>(k, j, i);
    } else if constexpr (el == TE::F3) {
      return static_cast<const System *>(this)->template FaceArea<X3DIR>(k, j, i);
    } else if constexpr (el == TE::E1) {
      return static_cast<const System *>(this)->template EdgeLength<X1DIR>(k, j, i);
    } else if constexpr (el == TE::E2) {
      return static_cast<const System *>(this)->template EdgeLength<X2DIR>(k, j, i);
    } else if constexpr (el == TE::E3) {
      return static_cast<const System *>(this)->template EdgeLength<X3DIR>(k, j, i);
    } else if constexpr (el == TE::NN) {
      return 1.0;
    }
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

  KOKKOS_FORCEINLINE_FUNCTION Real Volume(CellLevel cl, TopologicalElement el,
                                          const int k, const int j, const int i) const {
    using TE = TopologicalElement;
    if (cl == CellLevel::same) {
      if (el == TE::CC) {
        return Volume<TE::CC>(k, j, i);
      } else if (el == TE::F1) {
        return Volume<TE::F1>(k, j, i);
      } else if (el == TE::F2) {
        return Volume<TE::F2>(k, j, i);
      } else if (el == TE::F3) {
        return Volume<TE::F3>(k, j, i);
      } else if (el == TE::E1) {
        return Volume<TE::E1>(k, j, i);
      } else if (el == TE::E2) {
        return Volume<TE::E2>(k, j, i);
      } else if (el == TE::E3) {
        return Volume<TE::E3>(k, j, i);
      } else if (el == TE::NN) {
        return 1.0;
      }
    } else if (cl == CellLevel::fine) {
      if (el == TE::CC) {
        return cell_volume_ / 8.0;
      } else if (el == TE::F1) {
        return area_[X1DIR - 1] / 4.0;
      } else if (el == TE::F2) {
        return area_[X2DIR - 1] / 4.0;
      } else if (el == TE::F3) {
        return area_[X3DIR - 1] / 4.0;
      } else if (el == TE::E1) {
        return dx_[X1DIR - 1] / 2.0;
      } else if (el == TE::E2) {
        return dx_[X2DIR - 1] / 2.0;
      } else if (el == TE::E3) {
        return dx_[X3DIR - 1] / 2.0;
      } else if (el == TE::NN) {
        return 1.0;
      }
    }
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement or CellLevel enum.");
    return 0.0;
  }

  const std::array<Real, 3> &GetXmin() const { return xmin_; }
  const std::array<int, 3> &GetStartIndex() const { return istart_; }
  const char *Name() const { return System::name_; }

 private:
  std::array<int, 3> istart_;
  std::array<Real, 3> xmin_, dx_, area_;
  Real cell_volume_;

  const std::array<Real, 3> &Dx_() const { return dx_; }
};

} // namespace parthenon

#endif // COORDINATES_UNIFORM_COORDINATES_HPP_
