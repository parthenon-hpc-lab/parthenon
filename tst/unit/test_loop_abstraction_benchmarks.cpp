//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <array>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "Kokkos_Core.hpp"

#include "basic_types.hpp"
#include "kokkos_types.hpp"
#include "loop_abstraction.hpp"

namespace {

using LoopTag = plb2::loop_abstraction::loop_tag;
using InnerTag = plb2::loop_abstraction::inner_tag;
using View5D = Kokkos::View<parthenon::Real *****, Kokkos::LayoutRight>;
using CountsView = Kokkos::View<int *>;

constexpr int kNblocks = 2;
constexpr int kNvars = 3;
constexpr int kNz = 2;
constexpr int kNy = 3;
constexpr int kNx = 4;
constexpr int kNghost = 1;
constexpr int kMemoryNz = kNz + 2 * kNghost;
constexpr int kMemoryNy = kNy + 2 * kNghost;
constexpr int kMemoryNx = kNx + 2 * kNghost;
constexpr std::array<int, kNblocks> kActiveVars{3, 2};

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

CountsView MakeActiveCounts() {
  CountsView counts("active_counts", kNblocks);
  auto host = Kokkos::create_mirror_view(counts);
  for (int b = 0; b < kNblocks; ++b) {
    host(b) = kActiveVars[b];
  }
  Kokkos::deep_copy(counts, host);
  return counts;
}

View5D MakeView(const std::string &label) {
  return View5D(label, kNblocks, kNvars, kMemoryNz, kMemoryNy, kMemoryNx);
}

void InitializeInput(View5D view) {
  auto host = Kokkos::create_mirror_view(view);
  for (int b = 0; b < kNblocks; ++b) {
    for (int v = 0; v < kNvars; ++v) {
      for (int k = 0; k < kMemoryNz; ++k) {
        for (int j = 0; j < kMemoryNy; ++j) {
          for (int i = 0; i < kMemoryNx; ++i) {
            host(b, v, k, j, i) =
                10000.0 * b + 1000.0 * v + 100.0 * k + 10.0 * j + static_cast<double>(i);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(view, host);
}

template <class Pattern>
void RunTouchCountKernel(View5D counts, const CountsView &active_counts, int ninner) {
  plb2::loop_abstraction::index_space_t<Pattern::loop_tag, Pattern::inner_tag> idx_space(
      kNblocks, kNx, kNy, kNz, kNghost, ninner);

  plb2::loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    for (int v = 0; v < active_counts(b); ++v) {
      auto out = idx_range.view(counts, v);
      plb2::loop_abstraction::inner(idx_range, [&](const auto idx) { out(idx) += 1.0; });
    }
  });
}

template <class Pattern>
void RunAgreementKernel(const View5D &input, View5D output, const CountsView &active_counts,
                        int ninner) {
  plb2::loop_abstraction::index_space_t<Pattern::loop_tag, Pattern::inner_tag> idx_space(
      kNblocks, kNx, kNy, kNz, kNghost, ninner);

  plb2::loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    for (int v = 0; v < active_counts(b); ++v) {
      auto center = idx_range.view(input, v);
      auto xp = idx_range.view(input, v, {0, 0, 1});
      auto ym = idx_range.view(input, v, {0, -1, 0});
      auto zp = idx_range.view(input, v, {1, 0, 0});
      auto out = idx_range.view(output, v);
      plb2::loop_abstraction::inner(idx_range, [&](const auto idx) {
        out(idx) = 3.0 * center(idx) + 5.0 * xp(idx) - 2.0 * ym(idx) + 7.0 * zp(idx) +
                   11.0 * b + 13.0 * v;
      });
    }
  });
}

template <class Pattern>
void CheckTouchedExactlyOnce() {
  const auto active_counts = MakeActiveCounts();
  auto counts = MakeView("touch_counts");
  Kokkos::deep_copy(counts, 0.0);

  RunTouchCountKernel<Pattern>(counts, active_counts, 5);

  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);
  for (int b = 0; b < kNblocks; ++b) {
    for (int v = 0; v < kNvars; ++v) {
      for (int k = 0; k < kMemoryNz; ++k) {
        const bool logical_k = k >= kNghost && k < kNghost + kNz;
        for (int j = 0; j < kMemoryNy; ++j) {
          const bool logical_j = j >= kNghost && j < kNghost + kNy;
          for (int i = 0; i < kMemoryNx; ++i) {
            const bool logical_i = i >= kNghost && i < kNghost + kNx;
            const bool active_cell = v < kActiveVars[b] && logical_k && logical_j && logical_i;
            CAPTURE(PatternName<Pattern>(), b, v, k, j, i);
            if (active_cell) {
              REQUIRE(host(b, v, k, j, i) == 1.0);
            } else if (v >= kActiveVars[b]) {
              REQUIRE(host(b, v, k, j, i) == 0.0);
            } else if constexpr (Pattern::inner_tag == InnerTag::logical) {
              REQUIRE(host(b, v, k, j, i) == 0.0);
            }
          }
        }
      }
    }
  }
}

template <class Pattern>
auto RunAgreementCase(int ninner) {
  const auto active_counts = MakeActiveCounts();
  auto input = MakeView("input");
  auto output = MakeView("output");
  InitializeInput(input);
  Kokkos::deep_copy(output, 0.0);
  RunAgreementKernel<Pattern>(input, output, active_counts, ninner);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
}

template <class ReferencePattern, class CandidatePattern>
void CheckPatternsAgree(int ninner) {
  auto reference = RunAgreementCase<ReferencePattern>(ninner);
  auto candidate = RunAgreementCase<CandidatePattern>(ninner);

  for (int b = 0; b < kNblocks; ++b) {
    for (int v = 0; v < kNvars; ++v) {
      for (int k = 0; k < kMemoryNz; ++k) {
        const bool logical_k = k >= kNghost && k < kNghost + kNz;
        for (int j = 0; j < kMemoryNy; ++j) {
          const bool logical_j = j >= kNghost && j < kNghost + kNy;
          for (int i = 0; i < kMemoryNx; ++i) {
            const bool logical_i = i >= kNghost && i < kNghost + kNx;
            const bool active_cell = v < kActiveVars[b] && logical_k && logical_j && logical_i;
            CAPTURE(PatternName<ReferencePattern>(), PatternName<CandidatePattern>(), b, v, k, j,
                    i);
            if (active_cell) {
              REQUIRE(candidate(b, v, k, j, i) == reference(b, v, k, j, i));
            } else if (v >= kActiveVars[b]) {
              REQUIRE(candidate(b, v, k, j, i) == 0.0);
            } else if constexpr (CandidatePattern::inner_tag == InnerTag::logical) {
              REQUIRE(candidate(b, v, k, j, i) == 0.0);
            }
          }
        }
      }
    }
  }
}

} // namespace

TEST_CASE("benchmark loop abstraction touches each logical cell once", "[unit]") {
  if constexpr (!plb2::loop_abstraction::impl::use_raw_for_v) {
    SUCCEED("CPU/raw-loop-only test skipped for non-host default execution space");
    return;
  }

  CheckTouchedExactlyOnce<Pattern<LoopTag::bvoi, InnerTag::logical>>();
  CheckTouchedExactlyOnce<Pattern<LoopTag::bvoi, InnerTag::memory>>();
  CheckTouchedExactlyOnce<Pattern<LoopTag::bovi, InnerTag::logical>>();
  CheckTouchedExactlyOnce<Pattern<LoopTag::bovi, InnerTag::memory>>();
  CheckTouchedExactlyOnce<Pattern<LoopTag::boiv, InnerTag::logical>>();
}

TEST_CASE("benchmark loop abstraction patterns agree on CPU raw loops", "[unit]") {
  if constexpr (!plb2::loop_abstraction::impl::use_raw_for_v) {
    SUCCEED("CPU/raw-loop-only test skipped for non-host default execution space");
    return;
  }

  using Reference = Pattern<LoopTag::boiv, InnerTag::logical>;

  CheckPatternsAgree<Reference, Pattern<LoopTag::bvoi, InnerTag::logical>>(5);
  CheckPatternsAgree<Reference, Pattern<LoopTag::bvoi, InnerTag::memory>>(5);
  CheckPatternsAgree<Reference, Pattern<LoopTag::bovi, InnerTag::logical>>(5);
  CheckPatternsAgree<Reference, Pattern<LoopTag::bovi, InnerTag::memory>>(5);
}
