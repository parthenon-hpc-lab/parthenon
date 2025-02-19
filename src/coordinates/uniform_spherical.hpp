//========================================================================================
// (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
#include "utils/error_checking.hpp"

namespace parthenon {

// Spherical coordinates with X1->r, X2->theta (polar angle), and X3->phi

class UniformSpherical : public UniformCoordinates<UniformSpherical> {
  using base_t = UniformCoordinates<UniformSpherical>;

 public:
  using base_t::CellWidth;
  using base_t::Dxc;
  using base_t::FaceArea;
  using base_t::Scale;
  using base_t::Volume;
  using base_t::Xc;
  UniformSpherical() = default;
  UniformSpherical(const RegionSize &rs, ParameterInput *pin)
      : UniformCoordinates<UniformSpherical>(rs, pin) {
    PARTHENON_REQUIRE(rs.xmin(X1DIR) >= 0.0, "Min radius must be >= 0.");
    PARTHENON_REQUIRE(rs.xmin(X2DIR) >= 0.0, "Min theta must be >= 0.0.");
    PARTHENON_REQUIRE(rs.xmax(X2DIR) <= M_PI + 1.e-15, "Max theta must be <= pi");
    PARTHENON_REQUIRE(rs.xmin(X3DIR) >= 0.0, "Min phi must be >= 0.0.");
    PARTHENON_REQUIRE(rs.xmax(X3DIR) <= 2.0 * M_PI + 1.e-15, "Max phi must be <= 2pi");
  }
  UniformSpherical(const UniformSpherical &src, int coarsen)
      : UniformCoordinates<UniformSpherical>(src, coarsen) {}
  constexpr static const char *name_ = "UniformSpherical";

  //----------------------------------------
  // Dxc: Distance between cell centers
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Dxc(const int idx) const {
    static_assert(dir > 0 && dir < 4);
    return Xc<dir>(idx) - Xc<dir>(idx - 1);
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

  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real Scale(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    using TE = TopologicalElement;
    if constexpr (dir == X1DIR) {
      return 1.0;
    } else {
      const Real r = X<X1DIR, el>(k, j, i);
      if constexpr (dir == X2DIR) {
        return std::abs(r);
      } else {
        const Real th = X<X2DIR, el>(k, j, i);
        return std::abs(r * std::sin(th));
      }
    }
    return 0.0;
  }

  //----------------------------------------
  // CellWidth: width of cell through the centroid
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int k, const int j,
                                             const int i) const {
    using TE = TopologicalElement;
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR)
      return Dx<dir>();
    else if constexpr (dir == X2DIR)
      return std::abs(Xc<X1DIR>(i) * Dx<dir>());
    else if constexpr (dir == X3DIR)
      return std::abs(Xc<X1DIR>(i) * std::sin(Xc<X2DIR>(j)) * Dx<dir>());
    return 0.0;
  }

  //----------------------------------------
  // EdgeLength: Length of cell edges
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j,
                                              const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR) {
      // radial direction is trivial
      return Dx<dir>();
    } else if constexpr (dir == X2DIR) {
      // theta direction
      return std::abs(Xf<X1DIR>(k, j, i) * Dx<dir>());
    }
    // phi direction
    return std::abs(Xf<X1DIR>(k, j, i) * std::sin(Xf<X2DIR>(k, j, i)) * Dx<dir>());
  }
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int dir, const int k, const int j,
                                              const int i) const {
    assert(dir > 0 && dir < 4);
    if (dir == X1DIR)
      return EdgeLength<X1DIR>(k, j, i);
    else if (dir == X2DIR)
      return EdgeLength<X2DIR>(k, j, i);
    return EdgeLength<X3DIR>(k, j, i);
  }

  //----------------------------------------
  // FaceArea: Area of cell areas
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR) {
      Real dOmega = Dx<X3DIR>() * std::abs(std::cos(Xf<X2DIR>(k, j, i)) -
                                           std::cos(Xf<X2DIR>(k, j + 1, i)));
      Real r = Xf<X1DIR>(k, j, i);
      return r * r * dOmega;
    } else if constexpr (dir == X2DIR) {
      Real r0 = Xf<X1DIR>(k, j, i);
      Real r1 = Xf<X1DIR>(k, j, i + 1);
      return std::abs(0.5 * (r1 * r1 - r0 * r0) * std::sin(Xf<X2DIR>(k, j, i)) *
                      Dx<X3DIR>());
    }
    Real r0 = Xf<X1DIR>(k, j, i);
    Real r1 = Xf<X1DIR>(k, j, i + 1);
    return std::abs(0.5 * (r1 * r1 - r0 * r0) * Dx<X2DIR>());
  }

  //----------------------------------------
  // CellVolume
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j,
                                              const int i) const {
    Real dcth = std::cos(Xf<X2DIR>(k, j, i)) - std::cos(Xf<X2DIR>(k, j + 1, i));
    Real r0 = Xf<X1DIR>(k, j, i);
    Real r1 = Xf<X1DIR>(k, j, i + 1);
    return (1.0 / 3.0) * std::abs((r1 * r1 * r1 - r0 * r0 * r0) * dcth * Dx<X3DIR>());
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real Volume(CellLevel cl, TopologicalElement el, const int k, const int j,
              const int i) {
    using TE = TopologicalElement;
    if (cl == CellLevel::same) {
      if (el == TE::CC)
        return CellVolume(k, j, i);
      else if (el == TE::F1)
        return FaceArea<X1DIR>(k, j, i);
      else if (el == TE::F2)
        return FaceArea<X2DIR>(k, j, i);
      else if (el == TE::F3)
        return FaceArea<X3DIR>(k, j, i);
      else if (el == TE::E1)
        return EdgeLength<X1DIR>(k, j, i);
      else if (el == TE::E2)
        return EdgeLength<X2DIR>(k, j, i);
      else if (el == TE::E3)
        return EdgeLength<X3DIR>(k, j, i);
      else if (el == TE::NN)
        return 1.0;
    } else {
      PARTHENON_FAIL(
          "Have not yet implemented fine fields for UniformSpherical coordinates.");
    }
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }
};

} // namespace parthenon

#endif // COORDINATES_UNIFORM_SPHERICAL_HPP_
