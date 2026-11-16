//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#include <array>
#include <cmath>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "interface/variable_state.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "prolong_restrict/pr_ops.hpp"

using parthenon::CellLevel;
using parthenon::Coordinates_t;
using parthenon::IndexRange;
using parthenon::ParameterInput;
using parthenon::ParArrayND;
using parthenon::Real;
using parthenon::RegionSize;
using parthenon::VariableState;
using TE = parthenon::TopologicalElement;

namespace {
// Fill the outer faces of a single coarse cell (split into 2x2x2 fine cells) with
// arbitrary fine face values that carry zero net flux, prolongate the internal faces with
// Toth & Roe, and return the maximum relative divergence of the eight fine cells.
Real TothAndRoeMaxDivergence(const std::array<Real, 3> &xmin,
                             const std::array<Real, 3> &xmax) {
  ParameterInput pin;
  const int nghost_save = parthenon::Globals::nghost;
  parthenon::Globals::nghost = 0;
  const std::array<Real, 3> xrat{1.0, 1.0, 1.0};
  Coordinates_t fine_coords(RegionSize(xmin, xmax, xrat, {2, 2, 2}), &pin);
  Coordinates_t coarse_coords(RegionSize(xmin, xmax, xrat, {1, 1, 1}), &pin);
  parthenon::Globals::nghost = nghost_save;

  // Fine face field with layout (face element, l, m, n, k, j, i)
  ParArrayND<Real, VariableState> fine("fine", 3, 1, 1, 1, 3, 3, 3);
  auto fine_h = fine.GetHostMirror();
  auto area = [&](int e, int k, int j, int i) {
    return fine_coords.Volume(CellLevel::same,
                              static_cast<TE>(static_cast<int>(TE::F1) + e), k, j, i);
  };
  auto is_outer = [](int e, int k, int j, int i) {
    const std::array<int, 3> idx{i, j, k};
    for (int d = 0; d < 3; ++d) {
      if (idx[d] > (d == e ? 2 : 1)) return false;
    }
    return idx[e] != 1;
  };

  Real net_flux = 0.0;
  Real flux_scale = 0.0;
  for (int e = 0; e < 3; ++e) {
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
          fine_h(e, 0, 0, 0, k, j, i) = 0.0;
          if (!is_outer(e, k, j, i)) continue;
          const Real b = std::sin(1.0 + 3.1 * e + 1.7 * k + 2.3 * j + 0.9 * i);
          fine_h(e, 0, 0, 0, k, j, i) = b;
          const int ihi = (e == 0) ? i : (e == 1) ? j : k;
          net_flux += (ihi == 2 ? 1.0 : -1.0) * b * area(e, k, j, i);
          flux_scale += std::abs(b * area(e, k, j, i));
        }
      }
    }
  }
  // Make the coarse cell divergence free
  fine_h(0, 0, 0, 0, 0, 0, 2) -= net_flux / area(0, 0, 0, 2);
  fine.DeepCopy(fine_h);

  const IndexRange cb{0, 0};
  const IndexRange fb{0, 1};
  using TR = parthenon::refinement_ops::ProlongateInternalTothAndRoe;
  parthenon::par_for(
      parthenon::loop_pattern_flatrange_tag, "TothAndRoe", parthenon::DevExecSpace(), 0,
      0, KOKKOS_LAMBDA(const int) {
        TR::Do<3, TE::F1, TE::CC>(0, 0, 0, 0, 0, 0, cb, cb, cb, fb, fb, fb, fine_coords,
                                  coarse_coords, nullptr, &fine);
        TR::Do<3, TE::F2, TE::CC>(0, 0, 0, 0, 0, 0, cb, cb, cb, fb, fb, fb, fine_coords,
                                  coarse_coords, nullptr, &fine);
        TR::Do<3, TE::F3, TE::CC>(0, 0, 0, 0, 0, 0, cb, cb, cb, fb, fb, fb, fine_coords,
                                  coarse_coords, nullptr, &fine);
      });
  fine_h = fine.GetHostMirrorAndCopy();

  Real max_div = 0.0;
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const Real div = fine_h(0, 0, 0, 0, k, j, i + 1) * area(0, k, j, i + 1) -
                         fine_h(0, 0, 0, 0, k, j, i) * area(0, k, j, i) +
                         fine_h(1, 0, 0, 0, k, j + 1, i) * area(1, k, j + 1, i) -
                         fine_h(1, 0, 0, 0, k, j, i) * area(1, k, j, i) +
                         fine_h(2, 0, 0, 0, k + 1, j, i) * area(2, k + 1, j, i) -
                         fine_h(2, 0, 0, 0, k, j, i) * area(2, k, j, i);
        max_div = std::max(max_div, std::abs(div) / flux_scale);
      }
    }
  }
  return max_div;
}
} // namespace

TEST_CASE("Toth and Roe prolongation is divergence free", "[Prolongation]") {
  GIVEN("Cubic cells") {
    REQUIRE(TothAndRoeMaxDivergence({0.5, 0.0, 0.0}, {1.5, 1.0, 1.0}) < 1.e-13);
  }
  GIVEN("Non-cubic cells") {
    REQUIRE(TothAndRoeMaxDivergence({0.5, 0.0, 0.0}, {1.5, 2.0, 1.0}) < 1.e-13);
    REQUIRE(TothAndRoeMaxDivergence({0.5, 0.0, 0.0}, {0.8, 2.5, 0.4}) < 1.e-13);
  }
}
