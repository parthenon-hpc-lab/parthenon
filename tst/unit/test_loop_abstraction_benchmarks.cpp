//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <algorithm>
#include <array>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch.hpp>

#include "Kokkos_Core.hpp"

#include "basic_types.hpp"
#include "kokkos_types.hpp"
#include "loop_abstraction.hpp"
#include "utils/indexer.hpp"

namespace {

using LoopTag = plb2::loop_abstraction::loop_tag;
using InnerTag = plb2::loop_abstraction::inner_tag;

template <typename T>
using View5D = Kokkos::View<T *****, Kokkos::LayoutRight>;

using RealView5D = View5D<parthenon::Real>;
using CountsView = Kokkos::View<int *>;

struct TestSpec {
  std::string name;
  int nblocks;
  int nvars;
  int nz;
  int ny;
  int nx;
  int nghost;
  std::vector<int> active_vars;
};

template <LoopTag LOOP_TAG, InnerTag INNER_TAG>
struct Pattern {
  static constexpr LoopTag loop_tag = LOOP_TAG;
  static constexpr InnerTag inner_tag = INNER_TAG;
};

template <class Pattern>
std::string PatternName() {
  std::string name;
  switch (Pattern::loop_tag) {
  case LoopTag::bvoi:
    name = "bvoi";
    break;
  case LoopTag::bovi:
    name = "bovi";
    break;
  case LoopTag::boiv:
    name = "boiv";
    break;
  }
  name += "_";
  switch (Pattern::inner_tag) {
  case InnerTag::logical:
    name += "logical";
    break;
  case InnerTag::memory:
    name += "memory";
    break;
  }
  return name;
}

int MemoryNz(const TestSpec &spec) { return spec.nz + 2 * spec.nghost; }
int MemoryNy(const TestSpec &spec) { return spec.ny + 2 * spec.nghost; }
int MemoryNx(const TestSpec &spec) { return spec.nx + 2 * spec.nghost; }
int LogicalCellCount(const TestSpec &spec) { return spec.nz * spec.ny * spec.nx; }

std::string SpecName(const TestSpec &spec) { return spec.name; }

bool IsLogicalCell(const TestSpec &spec, int k, int j, int i) {
  return k >= spec.nghost && k < spec.nghost + spec.nz && j >= spec.nghost &&
         j < spec.nghost + spec.ny && i >= spec.nghost && i < spec.nghost + spec.nx;
}

parthenon::Indexer3D LogicalIndexer(const TestSpec &spec) {
  return parthenon::Indexer3D({spec.nghost, spec.nghost + spec.nz - 1},
                              {spec.nghost, spec.nghost + spec.ny - 1},
                              {spec.nghost, spec.nghost + spec.nx - 1});
}

parthenon::Indexer3D MemoryIndexer(const TestSpec &spec) {
  return parthenon::Indexer3D({0, MemoryNz(spec) - 1}, {0, MemoryNy(spec) - 1},
                              {0, MemoryNx(spec) - 1});
}

std::vector<int> NinnerCases(const TestSpec &spec) {
  std::vector<int> values;
  const int plane = std::max(1, spec.nx * spec.ny);
  const int logical_cells = LogicalCellCount(spec);
  values.push_back(1);
  values.push_back(std::max(1, spec.nx));
  values.push_back(std::max(1, plane - 1));
  values.push_back(plane);
  values.push_back(plane + 1);
  values.push_back(logical_cells + 3);
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

CountsView MakeActiveCounts(const TestSpec &spec) {
  CountsView counts("active_counts", spec.nblocks);
  auto host = Kokkos::create_mirror_view(counts);
  for (int b = 0; b < spec.nblocks; ++b) {
    host(b) = spec.active_vars.at(b);
  }
  Kokkos::deep_copy(counts, host);
  return counts;
}

RealView5D MakeView(const std::string &label, const TestSpec &spec) {
  return RealView5D(label, spec.nblocks, spec.nvars, MemoryNz(spec), MemoryNy(spec),
                    MemoryNx(spec));
}

void InitializeInput(const TestSpec &spec, RealView5D view) {
  auto host = Kokkos::create_mirror_view(view);
  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.nvars; ++v) {
      for (int k = 0; k < MemoryNz(spec); ++k) {
        for (int j = 0; j < MemoryNy(spec); ++j) {
          for (int i = 0; i < MemoryNx(spec); ++i) {
            host(b, v, k, j, i) = 100000.0 * b + 10000.0 * v + 1000.0 * k + 100.0 * j +
                                  10.0 * i + 1.0;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(view, host);
}

template <class HostView>
void ZeroHostView(const TestSpec &spec, HostView host) {
  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.nvars; ++v) {
      for (int k = 0; k < MemoryNz(spec); ++k) {
        for (int j = 0; j < MemoryNy(spec); ++j) {
          for (int i = 0; i < MemoryNx(spec); ++i) {
            host(b, v, k, j, i) = 0.0;
          }
        }
      }
    }
  }
}

template <class Pattern>
auto RunTouchCountCase(const TestSpec &spec, int ninner) {
  const auto active_counts = MakeActiveCounts(spec);
  auto counts = MakeView("touch_counts", spec);
  Kokkos::deep_copy(counts, 0.0);

  plb2::loop_abstraction::IndexSpace<Pattern::loop_tag, Pattern::inner_tag> idx_space(
      spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);

  plb2::loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    for (int v = 0; v < active_counts(b); ++v) {
      auto out = plb2::loop_abstraction::GetView(idx_range, counts, v);
      plb2::loop_abstraction::inner(idx_range, [&](const auto idx) { out(idx) += 1.0; });
    }
  });

  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);
}

template <class Pattern, std::size_t SX, std::size_t SY, std::size_t SZ>
auto RunStencilCase(const TestSpec &spec, int ninner, const std::array<int, SX> &dx,
                    const std::array<int, SY> &dy, const std::array<int, SZ> &dz) {
  const auto active_counts = MakeActiveCounts(spec);
  auto input = MakeView("stencil_input", spec);
  auto output = MakeView("stencil_output", spec);
  InitializeInput(spec, input);
  Kokkos::deep_copy(output, 0.0);

  plb2::loop_abstraction::IndexSpace<Pattern::loop_tag, Pattern::inner_tag> idx_space(
      spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);

  plb2::loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    for (int v = 0; v < active_counts(b); ++v) {
      auto center = plb2::loop_abstraction::GetView(idx_range, input, v);
      auto out = plb2::loop_abstraction::GetView(idx_range, output, v);

      std::array<decltype(center), SX> x_views{};
      std::array<decltype(center), SY> y_views{};
      std::array<decltype(center), SZ> z_views{};

      for (int ix = 0; ix < SX; ++ix) {
        x_views[ix] = plb2::loop_abstraction::GetView(idx_range, input, v, {0, 0, dx[ix]});
      }
      for (int iy = 0; iy < SY; ++iy) {
        y_views[iy] = plb2::loop_abstraction::GetView(idx_range, input, v, {0, dy[iy], 0});
      }
      for (int iz = 0; iz < SZ; ++iz) {
        z_views[iz] = plb2::loop_abstraction::GetView(idx_range, input, v, {dz[iz], 0, 0});
      }

      plb2::loop_abstraction::inner(idx_range, [&](const auto idx) {
        parthenon::Real value = 17.0 * center(idx) + 19.0 * b + 23.0 * v;
        for (int ix = 0; ix < SX; ++ix) {
          value += (2.0 + ix) * x_views[ix](idx);
        }
        for (int iy = 0; iy < SY; ++iy) {
          value -= (3.0 + iy) * y_views[iy](idx);
        }
        for (int iz = 0; iz < SZ; ++iz) {
          value += (5.0 + 2.0 * iz) * z_views[iz](idx);
        }
        out(idx) = value;
      });
    }
  });

  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
}

template <class Pattern>
void CheckLogicalTouchesExactlyOnce(const TestSpec &spec, int ninner) {
  const auto host = RunTouchCountCase<Pattern>(spec, ninner);

  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.nvars; ++v) {
      for (int k = 0; k < MemoryNz(spec); ++k) {
        for (int j = 0; j < MemoryNy(spec); ++j) {
          for (int i = 0; i < MemoryNx(spec); ++i) {
            const bool active_cell = v < spec.active_vars[b] && IsLogicalCell(spec, k, j, i);
            CAPTURE(SpecName(spec), PatternName<Pattern>(), ninner, b, v, k, j, i);
            if (active_cell) {
              REQUIRE(host(b, v, k, j, i) == 1.0);
            } else if (v >= spec.active_vars[b]) {
              REQUIRE(host(b, v, k, j, i) == 0.0);
            }
          }
        }
      }
    }
  }
}

template <class Pattern>
void CheckLogicalPatternDoesNotTouchHalo(const TestSpec &spec, int ninner) {
  static_assert(Pattern::inner_tag == InnerTag::logical);
  const auto host = RunTouchCountCase<Pattern>(spec, ninner);

  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.active_vars[b]; ++v) {
      for (int k = 0; k < MemoryNz(spec); ++k) {
        for (int j = 0; j < MemoryNy(spec); ++j) {
          for (int i = 0; i < MemoryNx(spec); ++i) {
            if (!IsLogicalCell(spec, k, j, i)) {
              CAPTURE(SpecName(spec), PatternName<Pattern>(), ninner, b, v, k, j, i);
              REQUIRE(host(b, v, k, j, i) == 0.0);
            }
          }
        }
      }
    }
  }
}

template <class Pattern>
void CheckMemoryTouchShape(const TestSpec &spec, int ninner) {
  static_assert(Pattern::inner_tag == InnerTag::memory);
  const auto actual = RunTouchCountCase<Pattern>(spec, ninner);
  auto expected_device = MakeView("expected_touch_shape", spec);
  auto expected = Kokkos::create_mirror_view(expected_device);
  ZeroHostView(spec, expected);

  const auto logical = LogicalIndexer(spec);
  const auto memory = MemoryIndexer(spec);
  const int logical_cells = LogicalCellCount(spec);
  const int nouter = logical_cells / ninner + (logical_cells % ninner != 0);

  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.active_vars[b]; ++v) {
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * ninner;
        const int logical_end = std::min((o + 1) * ninner - 1, logical_cells - 1);
        const auto [ks, js, is] = logical(logical_start);
        const auto [ke, je, ie] = logical(logical_end);
        const int flat_start = memory.GetFlatIdx(ks, js, is);
        const int flat_end = memory.GetFlatIdx(ke, je, ie);
        for (int flat = flat_start; flat <= flat_end; ++flat) {
          const auto [k, j, i] = memory(flat);
          expected(b, v, k, j, i) += 1.0;
        }
      }
    }
  }

  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.nvars; ++v) {
      for (int k = 0; k < MemoryNz(spec); ++k) {
        for (int j = 0; j < MemoryNy(spec); ++j) {
          for (int i = 0; i < MemoryNx(spec); ++i) {
            CAPTURE(SpecName(spec), PatternName<Pattern>(), ninner, b, v, k, j, i);
            REQUIRE(actual(b, v, k, j, i) == expected(b, v, k, j, i));
          }
        }
      }
    }
  }
}

template <class HostView>
void CheckInactiveVarsRemainZero(const TestSpec &spec, HostView host, const std::string &label) {
  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = spec.active_vars[b]; v < spec.nvars; ++v) {
      for (int k = 0; k < MemoryNz(spec); ++k) {
        for (int j = 0; j < MemoryNy(spec); ++j) {
          for (int i = 0; i < MemoryNx(spec); ++i) {
            CAPTURE(SpecName(spec), label, b, v, k, j, i);
            REQUIRE(host(b, v, k, j, i) == 0.0);
          }
        }
      }
    }
  }
}

template <class HostViewA, class HostViewB>
void CheckLogicalInteriorMatches(const TestSpec &spec, HostViewA reference, HostViewB candidate,
                                 const std::string &ref_label,
                                 const std::string &candidate_label) {
  for (int b = 0; b < spec.nblocks; ++b) {
    for (int v = 0; v < spec.active_vars[b]; ++v) {
      for (int k = 0; k < MemoryNz(spec); ++k) {
        for (int j = 0; j < MemoryNy(spec); ++j) {
          for (int i = 0; i < MemoryNx(spec); ++i) {
            if (IsLogicalCell(spec, k, j, i)) {
              CAPTURE(SpecName(spec), ref_label, candidate_label, b, v, k, j, i);
              REQUIRE(candidate(b, v, k, j, i) == reference(b, v, k, j, i));
            }
          }
        }
      }
    }
  }
}

template <class Pattern, std::size_t SX, std::size_t SY, std::size_t SZ>
void CheckNinnerIndependence(const TestSpec &spec, const std::array<int, SX> &dx,
                             const std::array<int, SY> &dy, const std::array<int, SZ> &dz) {
  const auto ninners = NinnerCases(spec);
  const auto reference = RunStencilCase<Pattern>(spec, ninners.front(), dx, dy, dz);
  CheckInactiveVarsRemainZero(spec, reference, PatternName<Pattern>());

  for (std::size_t idx = 1; idx < ninners.size(); ++idx) {
    const auto candidate = RunStencilCase<Pattern>(spec, ninners[idx], dx, dy, dz);
    CheckInactiveVarsRemainZero(spec, candidate, PatternName<Pattern>());
    CheckLogicalInteriorMatches(spec, reference, candidate, PatternName<Pattern>(),
                                PatternName<Pattern>() + "_ninner_" +
                                    std::to_string(ninners[idx]));
  }
}

template <std::size_t SX, std::size_t SY, std::size_t SZ, class ReferencePattern,
          class CandidatePattern>
void CheckPatternsAgree(const TestSpec &spec, int ninner, const std::array<int, SX> &dx,
                        const std::array<int, SY> &dy, const std::array<int, SZ> &dz) {
  const auto reference = RunStencilCase<ReferencePattern>(spec, ninner, dx, dy, dz);
  const auto candidate = RunStencilCase<CandidatePattern>(spec, ninner, dx, dy, dz);
  CheckInactiveVarsRemainZero(spec, candidate, PatternName<CandidatePattern>());
  CheckLogicalInteriorMatches(spec, reference, candidate, PatternName<ReferencePattern>(),
                              PatternName<CandidatePattern>());
}

void CheckAllPatternsLogicalTouchesExactlyOnce(const TestSpec &spec, int ninner) {
  CheckLogicalTouchesExactlyOnce<Pattern<LoopTag::bvoi, InnerTag::logical>>(spec, ninner);
  CheckLogicalTouchesExactlyOnce<Pattern<LoopTag::bvoi, InnerTag::memory>>(spec, ninner);
  CheckLogicalTouchesExactlyOnce<Pattern<LoopTag::bovi, InnerTag::logical>>(spec, ninner);
  CheckLogicalTouchesExactlyOnce<Pattern<LoopTag::bovi, InnerTag::memory>>(spec, ninner);
  CheckLogicalTouchesExactlyOnce<Pattern<LoopTag::boiv, InnerTag::logical>>(spec, ninner);
}

void CheckAllPatternsNinnerIndependenceCenterOnly(const TestSpec &spec) {
  constexpr std::array<int, 0> none{};
  CheckNinnerIndependence<Pattern<LoopTag::bvoi, InnerTag::logical>>(spec, none, none, none);
  CheckNinnerIndependence<Pattern<LoopTag::bvoi, InnerTag::memory>>(spec, none, none, none);
  CheckNinnerIndependence<Pattern<LoopTag::bovi, InnerTag::logical>>(spec, none, none, none);
  CheckNinnerIndependence<Pattern<LoopTag::bovi, InnerTag::memory>>(spec, none, none, none);
  CheckNinnerIndependence<Pattern<LoopTag::boiv, InnerTag::logical>>(spec, none, none, none);
}

void CheckAllPatternsNinnerIndependenceMixedStencil(const TestSpec &spec,
                                                    const std::array<int, 3> &x_offsets,
                                                    const std::array<int, 2> &y_offsets,
                                                    const std::array<int, 2> &z_offsets) {
  CheckNinnerIndependence<Pattern<LoopTag::bvoi, InnerTag::logical>>(spec, x_offsets, y_offsets,
                                                                      z_offsets);
  CheckNinnerIndependence<Pattern<LoopTag::bvoi, InnerTag::memory>>(spec, x_offsets, y_offsets,
                                                                     z_offsets);
  CheckNinnerIndependence<Pattern<LoopTag::bovi, InnerTag::logical>>(spec, x_offsets, y_offsets,
                                                                      z_offsets);
  CheckNinnerIndependence<Pattern<LoopTag::bovi, InnerTag::memory>>(spec, x_offsets, y_offsets,
                                                                     z_offsets);
  CheckNinnerIndependence<Pattern<LoopTag::boiv, InnerTag::logical>>(spec, x_offsets, y_offsets,
                                                                      z_offsets);
}

std::vector<TestSpec> CoverageSpecs() {
  return {
      {"base", 2, 3, 2, 3, 4, 1, {3, 2}},
      {"zero_ghost_with_zero_active_block", 2, 3, 2, 3, 4, 0, {0, 3}},
      {"wide_ghost", 2, 4, 2, 2, 3, 2, {4, 1}},
      {"degenerate_z", 2, 3, 1, 3, 4, 1, {3, 0}},
      {"degenerate_y", 2, 3, 2, 1, 4, 1, {1, 3}},
      {"degenerate_x", 2, 3, 2, 3, 1, 1, {2, 2}},
      {"single_cell", 2, 3, 1, 1, 1, 1, {0, 1}},
  };
}

TEST_CASE("benchmark loop abstraction covers logical cells exactly once across chunking cases",
          "[unit]") {
  if constexpr (!plb2::loop_abstraction::impl::use_raw_for_v) {
    SUCCEED("CPU/raw-loop-only test skipped for non-host default execution space");
    return;
  }

  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec)) {
      CheckAllPatternsLogicalTouchesExactlyOnce(spec, ninner);
    }
  }
}

TEST_CASE("benchmark logical-tag patterns do not touch halos", "[unit]") {
  if constexpr (!plb2::loop_abstraction::impl::use_raw_for_v) {
    SUCCEED("CPU/raw-loop-only test skipped for non-host default execution space");
    return;
  }

  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec)) {
      CheckLogicalPatternDoesNotTouchHalo<Pattern<LoopTag::bvoi, InnerTag::logical>>(spec,
                                                                                       ninner);
      CheckLogicalPatternDoesNotTouchHalo<Pattern<LoopTag::bovi, InnerTag::logical>>(spec,
                                                                                       ninner);
      CheckLogicalPatternDoesNotTouchHalo<Pattern<LoopTag::boiv, InnerTag::logical>>(spec,
                                                                                       ninner);
    }
  }
}

TEST_CASE("benchmark memory-tag patterns touch the expected flat memory spans", "[unit]") {
  if constexpr (!plb2::loop_abstraction::impl::use_raw_for_v) {
    SUCCEED("CPU/raw-loop-only test skipped for non-host default execution space");
    return;
  }

  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec)) {
      CheckMemoryTouchShape<Pattern<LoopTag::bvoi, InnerTag::memory>>(spec, ninner);
      CheckMemoryTouchShape<Pattern<LoopTag::bovi, InnerTag::memory>>(spec, ninner);
    }
  }
}

TEST_CASE("benchmark loop abstraction patterns agree on logical interiors for multiple stencils",
          "[unit]") {
  if constexpr (!plb2::loop_abstraction::impl::use_raw_for_v) {
    SUCCEED("CPU/raw-loop-only test skipped for non-host default execution space");
    return;
  }

  using Reference = Pattern<LoopTag::boiv, InnerTag::logical>;
  constexpr std::array<int, 0> none{};
  constexpr std::array<int, 3> x_offsets{-1, 0, 1};
  constexpr std::array<int, 2> y_offsets{-1, 1};
  constexpr std::array<int, 2> z_offsets{-1, 1};

  for (const auto &spec : CoverageSpecs()) {
    const int ninner = NinnerCases(spec).front();
    CheckPatternsAgree<0, 0, 0, Reference, Pattern<LoopTag::bvoi, InnerTag::logical>>(
        spec, ninner, none, none, none);
    CheckPatternsAgree<0, 0, 0, Reference, Pattern<LoopTag::bvoi, InnerTag::memory>>(
        spec, ninner, none, none, none);
    CheckPatternsAgree<0, 0, 0, Reference, Pattern<LoopTag::bovi, InnerTag::logical>>(
        spec, ninner, none, none, none);
    CheckPatternsAgree<0, 0, 0, Reference, Pattern<LoopTag::bovi, InnerTag::memory>>(
        spec, ninner, none, none, none);

    if (spec.nghost > 0) {
      CheckPatternsAgree<3, 0, 0, Reference, Pattern<LoopTag::bvoi, InnerTag::logical>>(
          spec, ninner, x_offsets, none, none);
      CheckPatternsAgree<3, 0, 0, Reference, Pattern<LoopTag::bvoi, InnerTag::memory>>(
          spec, ninner, x_offsets, none, none);
      CheckPatternsAgree<3, 0, 0, Reference, Pattern<LoopTag::bovi, InnerTag::logical>>(
          spec, ninner, x_offsets, none, none);
      CheckPatternsAgree<3, 0, 0, Reference, Pattern<LoopTag::bovi, InnerTag::memory>>(
          spec, ninner, x_offsets, none, none);

      CheckPatternsAgree<0, 2, 2, Reference, Pattern<LoopTag::bvoi, InnerTag::logical>>(
          spec, ninner, none, y_offsets, z_offsets);
      CheckPatternsAgree<0, 2, 2, Reference, Pattern<LoopTag::bvoi, InnerTag::memory>>(
          spec, ninner, none, y_offsets, z_offsets);
      CheckPatternsAgree<0, 2, 2, Reference, Pattern<LoopTag::bovi, InnerTag::logical>>(
          spec, ninner, none, y_offsets, z_offsets);
      CheckPatternsAgree<0, 2, 2, Reference, Pattern<LoopTag::bovi, InnerTag::memory>>(
          spec, ninner, none, y_offsets, z_offsets);

      CheckPatternsAgree<3, 2, 2, Reference, Pattern<LoopTag::bvoi, InnerTag::logical>>(
          spec, ninner, x_offsets, y_offsets, z_offsets);
      CheckPatternsAgree<3, 2, 2, Reference, Pattern<LoopTag::bvoi, InnerTag::memory>>(
          spec, ninner, x_offsets, y_offsets, z_offsets);
      CheckPatternsAgree<3, 2, 2, Reference, Pattern<LoopTag::bovi, InnerTag::logical>>(
          spec, ninner, x_offsets, y_offsets, z_offsets);
      CheckPatternsAgree<3, 2, 2, Reference, Pattern<LoopTag::bovi, InnerTag::memory>>(
          spec, ninner, x_offsets, y_offsets, z_offsets);
    }
  }
}

TEST_CASE("benchmark loop abstraction interior output is independent of ninner", "[unit]") {
  if constexpr (!plb2::loop_abstraction::impl::use_raw_for_v) {
    SUCCEED("CPU/raw-loop-only test skipped for non-host default execution space");
    return;
  }

  constexpr std::array<int, 3> x_offsets{-1, 0, 1};
  constexpr std::array<int, 2> y_offsets{-1, 1};
  constexpr std::array<int, 2> z_offsets{-1, 1};

  for (const auto &spec : CoverageSpecs()) {
    if (spec.nghost == 0) {
      CheckAllPatternsNinnerIndependenceCenterOnly(spec);
    } else {
      CheckAllPatternsNinnerIndependenceMixedStencil(spec, x_offsets, y_offsets, z_offsets);
    }
  }
}

} // namespace
