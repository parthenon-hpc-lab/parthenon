//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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

#include "uniform_coordinates.hpp"

namespace parthenon {

class UniformSpherical : public UniformCoordinates<UniformSpherical> {
 using base_t = UniformCoordinates<UniformSpherical>;
 public:
  UniformSpherical() = default;
  UniformSpherical(const RegionSize &rs, ParameterInput *pin) 
    : UniformCoordinates<UniformSpherical>(rs, pin) {}
  UniformSpherical(const UniformSpherical &src, int coarsen)
      : UniformCoordinates<UniformSpherical>(src, coarsen) {}
  constexpr static const char *name_ = "UniformSpherical";

  //----------------------------------------
  // Dxc: Distance between cell centers
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    // only the index along dir is relevant in Xc, so offsetting all three is OK
    return Xc<dir>(idx) - Xc<dir>(idx-1);
  }

  //----------------------------------------
  // Xc: Positions at cell centroids
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR) {
      const Real r0 = Xf<X1DIR>(idx);
      const Real r0sq = r0 * r0;
      const Real r1 = Xf<X1DIR>(idx + 1);
      const Real r1sq = r1 * r1;
      return 0.75 * (r1sq * r1sq - r0sq * r0sq) / (r1sq * r1 - r0sq * r0);
    } else if constexpr (dir == X2DIR) {
      const Real th0 = Xf<X2DIR>(idx);
      const Real sth0 = std::sin(th0);
      const Real cth0 = std::cos(th0);
      const Real th1 = Xf<X2DIR>(idx + 1);
      const Real sth1 = std::sin(th1);
      const Real cth1 = std::cos(th1);
      return (th0 * cth0 - th1 * cth1 - sth0 + sth1) / (cth0 - cth1);
    }
    return base_t::Xc<X3DIR>(idx);
  }

  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION
  Real hx(const Real r, const Real th, const Real phi) const {
    if (dir == X1DIR) {
      return 1.0;
    } else if constexpr (dir == X2DIR) {
      return r;
    } else if constexpr (dir == X3DIR) {
      return r * std::sin(th);
    } else {
      PARTHENON_FAIL("Unknown dir");
    }
    return 0.0;
  }

  //----------------------------------------
  // EdgeLength: Length of cell edges
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR) {
      // radial direction is trivial
      return Dx<dir>();
    } else if constexpr (dir == X2DIR) {
      // theta direction
      return Xf<X1DIR>(k, j, i) * Dx<dir>();
    }
    // phi direction
    return Xf<X1DIR>(k, j, i) * std::sin(Xf<X2DIR>(k, j, i)) * Dx<dir>();
  }
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int dir, const int k, const int j, const int i) const {
    assert(dir > 0 && dir < 4);
    if (dir == X1DIR) return EdgeLength<X1DIR>(k, j, i);
    else if (dir == X2DIR) return EdgeLength<X2DIR>(k, j, i);
    return EdgeLength<X3DIR>(k, j, i);
  }

  //----------------------------------------
  // FaceArea: Area of cell areas
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR) {
      Real dOmega = Dx<X3DIR>() * (std::cos(Xf<X2DIR>(k, j, i)) - std::cos(Xf<X2DIR>(k, j+1, i)));
      Real r = Xf<X1DIR>(k, j, i);
      return r * r * dOmega;
    } else if constexpr (dir == X2DIR) {
      Real r0 = Xf<X1DIR>(k, j, i);
      Real r1 = Xf<X1DIR>(k, j, i+1);
      return (r1*r1*r1 - r0*r0*r0) * std::sin(Xf<X2DIR>(k, j, i)) * Dx<X3DIR>() / 3.0;
    }
    Real r0 = Xf<X1DIR>(k, j, i);
    Real r1 = Xf<X1DIR>(k, j, i+1);
    Real dcth = std::cos(Xf<X2DIR>(k, j, i)) - std::cos(Xf<X2DIR>(k, j+1, i));
    return (r1*r1*r1 - r0*r0*r0) * dcth / 3.0;
  }

  //----------------------------------------
  // CellVolume
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j, const int i) const {
    return FaceArea<X3DIR>(k, j, i) * Dx<X3DIR>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real Volume(CellLevel cl, TopologicalElement el, const int k, const int j, const int i) {
    using TE = TopologicalElement;
    if (cl == CellLevel::same) {
      if (el == TE::CC) return CellVolume(k, j, i);
      else if (el == TE::F1) return FaceArea<X1DIR>(k, j, i);
      else if (el == TE::F2) return FaceArea<X2DIR>(k, j, i);
      else if (el == TE::F3) return FaceArea<X3DIR>(k, j, i);
      else if (el == TE::E1) return EdgeLength<X1DIR>(k, j, i);
      else if (el == TE::E2) return EdgeLength<X2DIR>(k, j, i);
      else if (el == TE::E3) return EdgeLength<X3DIR>(k, j, i);
      else if (el == TE::NN) return 1.0;
    } else {
      PARTHENON_FAIL("Have not yet implemented fine fields for UniformSpherical coordinates.");
    }
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

 private:
};

} // namespace parthenon

#endif // COORDINATES_UNIFORM_SPHERICAL_HPP_
