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

// This file was made in part with generative AI.

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Kokkos_Core.hpp"

#include "basic_types.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "kokkos_types.hpp"
#include "loop_abstraction/loop_abstraction.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "pack/sparse_pack/sparse_pack.hpp"

namespace {

// loop_abstraction now lives under parthenon; keep the short name for the test body.
namespace loop_abstraction = parthenon::loop_abstraction;

using Real = double;
using loop_abstraction::default_loop_backend_v;
using loop_abstraction::Index3;
using loop_abstraction::IndexSpace;
using loop_abstraction::inner_tag;
using loop_abstraction::InnerIndexRange;
using loop_abstraction::loop_backend;
using loop_abstraction::loop_tag;
using loop_abstraction::impl::ForceCapture;
using parthenon::BlockList_t;
using parthenon::DevExecSpace;
using parthenon::IndexDomain;
using parthenon::MeshBlock;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::StateDescriptor;
using parthenon::TopologicalElement;
using parthenon::TopologicalType;

constexpr int kNVars = 3;

struct ProblemSpec {
  int nblocks;
  int nx;
  int ny;
  int nz;
  int nghost;
};

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
using PatternIndexSpace = IndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;

BlockList_t MakeBlockList(const std::shared_ptr<StateDescriptor> pkg, const int NBLOCKS,
                          const int NSIDE, const int NDIM) {
  BlockList_t block_list;
  block_list.reserve(NBLOCKS);
  for (int i = 0; i < NBLOCKS; ++i) {
    auto pmb = std::make_shared<MeshBlock>(NSIDE, NDIM);
    auto &pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);
    block_list.push_back(pmb);
  }
  return block_list;
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
constexpr std::string_view PatternName() {
  if constexpr (LOOP_TAG == loop_tag::bvoi && INNER_TAG == inner_tag::logical_flat) {
    return "bvoi/logical_flat";
  } else if constexpr (LOOP_TAG == loop_tag::bvoi && // NOLINT(readability/braces)
                       INNER_TAG == inner_tag::logical_coords) {
    return "bvoi/logical_coords";
  } else if constexpr (LOOP_TAG == loop_tag::bvoi && INNER_TAG == inner_tag::memory) {
    return "bvoi/memory";
  } else if constexpr (LOOP_TAG == loop_tag::bovi && // NOLINT(readability/braces)
                       INNER_TAG == inner_tag::logical_flat) {
    return "bovi/logical_flat";
  } else if constexpr (LOOP_TAG == loop_tag::bovi && // NOLINT(readability/braces)
                       INNER_TAG == inner_tag::logical_coords) {
    return "bovi/logical_coords";
  } else if constexpr (LOOP_TAG == loop_tag::bovi && INNER_TAG == inner_tag::memory) {
    return "bovi/memory";
  } else if constexpr (LOOP_TAG == loop_tag::boiv && // NOLINT(readability/braces)
                       INNER_TAG == inner_tag::logical_flat) {
    return "boiv/logical_flat";
  } else if constexpr (LOOP_TAG == loop_tag::boiv && // NOLINT(readability/braces)
                       INNER_TAG == inner_tag::logical_coords) {
    return "boiv/logical_coords";
  } else {
    return "unknown";
  }
}

template <inner_tag INNER_TAG>
constexpr bool UsesMemorySpan() {
  return INNER_TAG == inner_tag::memory;
}

template <class ViewType>
void ZeroView(ViewType &view) {
  Kokkos::deep_copy(view, Real{0});
}

template <class ViewType>
auto MirrorToHost(const ViewType &view) {
  auto host = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host, view);
  return host;
}

// Device-side mismatch counter used to move correctness checks out of kernels.
// Catch2's REQUIRE/INFO/Approx cannot run on device, so loop-abstraction bodies
// atomic-increment this counter on a failed comparison instead. The test then fences
// and REQUIREs the host-side total is zero. Capture by value into the kernel and call
// note() inside the body; call total() on the host afterwards.
struct MismatchCounter {
  Kokkos::View<int> view;
  MismatchCounter() : view("mismatch") { Kokkos::deep_copy(view, 0); }

  KOKKOS_INLINE_FUNCTION void note(bool wrong) const {
    if (wrong) Kokkos::atomic_add(&view(), 1);
  }

  int total() const {
    Kokkos::fence();
    int out = 0;
    Kokkos::deep_copy(out, view);
    return out;
  }
};

// Device-safe analog of Catch2's Approx inequality (relative tolerance). Returns true
// when a and b differ by more than a small multiple of their magnitude.
KOKKOS_INLINE_FUNCTION bool NotApprox(const Real a, const Real b) {
  const Real diff = a > b ? a - b : b - a;
  const Real mag_a = a < 0 ? -a : a;
  const Real mag_b = b < 0 ? -b : b;
  const Real mag = mag_a > mag_b ? mag_a : mag_b;
  return diff > 1.0e-8 * (1.0 + mag);
}

KOKKOS_INLINE_FUNCTION Real EncodeValue(const int b, const int v, const int k,
                                        const int j, const int i) {
  return 1.0e6 * static_cast<Real>(b) + 1.0e5 * static_cast<Real>(v) +
         1.0e3 * static_cast<Real>(k) + 10.0 * static_cast<Real>(j) +
         static_cast<Real>(i) + 1.0;
}

KOKKOS_INLINE_FUNCTION Real ScratchExpectedValue(const int b, const int k, const int j,
                                                 const int i) {
  Real out = 0.0;
  for (int v = 0; v < kNVars; ++v) {
    out += EncodeValue(b, v, k, j, i);
  }
  return out;
}

KOKKOS_INLINE_FUNCTION Real ShapedScratchValue(const int b, const int c0, const int c1,
                                               const int k, const int j, const int i) {
  return EncodeValue(b, 3 * c0 + c1, k, j, i);
}

template <class IndexSpaceType>
KOKKOS_INLINE_FUNCTION bool IsLogicalCell(const IndexSpaceType &idx_space, const int k,
                                          const int j, const int i) {
  const auto &logical = idx_space.GetLogicalIndexer();
  return k >= logical.template StartIdx<0>() && k <= logical.template EndIdx<0>() &&
         j >= logical.template StartIdx<1>() && j <= logical.template EndIdx<1>() &&
         i >= logical.template StartIdx<2>() && i <= logical.template EndIdx<2>();
}

template <class IndexSpaceType>
auto MakeOutput(const IndexSpaceType &idx_space) {
  const auto &memory = idx_space.GetMemoryIndexer();
  const int nk = memory.template EndIdx<0>() - memory.template StartIdx<0>() + 1;
  const int nj = memory.template EndIdx<1>() - memory.template StartIdx<1>() + 1;
  const int ni = memory.template EndIdx<2>() - memory.template StartIdx<2>() + 1;
  return parthenon::ParArray5D<Real>("unit_out", idx_space.GetNBlocks(), kNVars, nk, nj,
                                     ni);
}

template <class IndexSpaceType>
void CheckLogicalContract(const IndexSpaceType &idx_space,
                          const parthenon::HostArray5D<Real> &host) {
  const auto &logical = idx_space.GetLogicalIndexer();
  const auto &memory = idx_space.GetMemoryIndexer();
  for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
    for (int v = 0; v < kNVars; ++v) {
      for (int flat = 0; flat < static_cast<int>(logical.size()); ++flat) {
        const auto [k, j, i] = logical(flat);
        REQUIRE(host(b, v, k, j, i) == Approx(EncodeValue(b, v, k, j, i)));
      }
      if constexpr (!UsesMemorySpan<IndexSpaceType::inner_tag_v>()) {
        for (int k = memory.template StartIdx<0>(); k <= memory.template EndIdx<0>();
             ++k) {
          for (int j = memory.template StartIdx<1>(); j <= memory.template EndIdx<1>();
               ++j) {
            for (int i = memory.template StartIdx<2>(); i <= memory.template EndIdx<2>();
                 ++i) {
              if (!IsLogicalCell(idx_space, k, j, i)) {
                REQUIRE(host(b, v, k, j, i) == Approx(0.0));
              }
            }
          }
        }
      }
    }
  }
}

template <class IndexSpaceType>
void CheckParity(const parthenon::HostArray5D<Real> &lhs,
                 const parthenon::HostArray5D<Real> &rhs,
                 const IndexSpaceType &idx_space) {
  const auto &memory = idx_space.GetMemoryIndexer();
  for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
    for (int v = 0; v < kNVars; ++v) {
      for (int k = memory.template StartIdx<0>(); k <= memory.template EndIdx<0>(); ++k) {
        for (int j = memory.template StartIdx<1>(); j <= memory.template EndIdx<1>();
             ++j) {
          for (int i = memory.template StartIdx<2>(); i <= memory.template EndIdx<2>();
               ++i) {
            REQUIRE(lhs(b, v, k, j, i) == Approx(rhs(b, v, k, j, i)));
          }
        }
      }
    }
  }
}

std::vector<int> NinnerCases(const int logical_cells) {
  std::vector<int> cases{1, std::max(1, logical_cells - 1), logical_cells,
                         logical_cells + 1};
  std::sort(cases.begin(), cases.end());
  cases.erase(std::unique(cases.begin(), cases.end()), cases.end());
  return cases;
}

constexpr std::array<ProblemSpec, 3> CoverageSpecs() {
  return {ProblemSpec{2, 3, 2, 2, 1}, ProblemSpec{1, 1, 1, 1, 1},
          ProblemSpec{2, 4, 3, 2, 2}};
}

struct PackViewSpec {
  int nblocks;
  int ncell;
  int nghost;
};

constexpr std::array<PackViewSpec, 3> PackViewCoverageSpecs() {
  return {PackViewSpec{2, 3, 2}, PackViewSpec{1, 1, 2}, PackViewSpec{2, 4, 2}};
}

std::vector<int> PackViewNinnerCases(const int logical_cells) {
  std::vector<int> cases{1, std::max(1, logical_cells - 1), logical_cells,
                         logical_cells + 1};
  std::sort(cases.begin(), cases.end());
  cases.erase(std::unique(cases.begin(), cases.end()), cases.end());
  return cases;
}

struct v1 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v1(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v1"; }
  static constexpr bool is_sparse() { return false; }
};

struct v2 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v2(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v2"; }
  static constexpr bool is_sparse() { return false; }
};

struct v5 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v5(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v5"; }
  static constexpr bool is_sparse() { return false; }
};

struct vf : public parthenon::variable_names::base_w_tt_t<false, TopologicalType::Face> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION vf(Ts &&...args)
      : parthenon::variable_names::base_w_tt_t<false, TopologicalType::Face>(
            std::forward<Ts>(args)...) {}
  static std::string name() { return "vf"; }
  static constexpr bool is_sparse() { return false; }
};

KOKKOS_INLINE_FUNCTION Real PackViewSourceValue(const int b, const int src_var,
                                                const int k, const int j, const int i) {
  return 1.0e6 * static_cast<Real>(b) + 1.0e5 * static_cast<Real>(src_var + 1) +
         1.0e3 * static_cast<Real>(k) + 10.0 * static_cast<Real>(j) +
         static_cast<Real>(i);
}

KOKKOS_INLINE_FUNCTION Real PackViewExpectedValue(const int b, const int v, const int k,
                                                  const int j, const int i) {
  return 2.0e6 * static_cast<Real>(b) + 2.0e5 * static_cast<Real>(v + 1) +
         1.0e3 * static_cast<Real>(k) + 10.0 * static_cast<Real>(j) +
         static_cast<Real>(i) + 1.0;
}

KOKKOS_INLINE_FUNCTION Real FacePackViewExpectedValue(const int b,
                                                      const TopologicalElement te,
                                                      const int k, const int j,
                                                      const int i) {
  return 3.0e6 * static_cast<Real>(b) + 1.0e5 * static_cast<Real>(te) +
         1.0e3 * static_cast<Real>(k) + 10.0 * static_cast<Real>(j) +
         static_cast<Real>(i) + 2.0;
}

struct plus_j_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return n == 0 ? 0 : 1; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct minus_i_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n == 0 ? -1 : 0; }
};

struct plus_i_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n == 0 ? 0 : 1; }
};

struct minus_j_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return n == 0 ? -1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

// Offsets {(-1,0,2), (0,0,0), (0,0,2)}, sorted lexicographically. The (0,0,2) point is
// the projection of (-1,0,2) onto the i-j plane; including it makes the halo closed
// under projection so it is valid in a reduced-dimension run as well as in 3D. In 3D
// (where this halo is exercised) the (0,0,2) copy is simply an extra produced cell.
struct plus_two_i_minus_k_halo_t {
  static constexpr int npoints = 3;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return n == 0 ? -1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n == 1 ? 0 : 2; }
};

struct k_triplet_halo_t {
  static constexpr int npoints = 3;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) {
    return n == 0 ? -1 : (n == 1 ? 0 : 1);
  }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct unsorted_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n == 0 ? 0 : -1; }
};

struct duplicate_identity_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct missing_identity_halo_t {
  static constexpr int npoints = 1;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 1; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

static_assert(
    loop_abstraction::impl::HaloSatisfiesContract<loop_abstraction::halo::none_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<plus_j_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<minus_i_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<minus_j_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<plus_two_i_minus_k_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<k_triplet_halo_t>());
static_assert(!loop_abstraction::impl::HaloSatisfiesContract<unsorted_halo_t>());
static_assert(
    !loop_abstraction::impl::HaloSatisfiesContract<duplicate_identity_halo_t>());
static_assert(!loop_abstraction::impl::HaloSatisfiesContract<missing_identity_halo_t>());

// A halo with a bare diagonal / off-axis offset whose projection onto a degenerate
// direction is not itself a declared offset. minus_i is in the i-j plane but its k
// component is zero, so it is projection-closed; this one has a nonzero k that, when
// projected out in 2D, lands on (0,0,-1) which is absent -> not closed.
struct not_projection_closed_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return n == 0 ? -1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n == 0 ? -1 : 0; }
};

// Projection-closure: identity-only and in-plane/on-axis halos are closed under every
// degeneration; k_triplet is closed because both non-identity k-offsets project to the
// identity; plus_two_i_minus_k is closed by construction (it includes (0,0,2)); the
// bare-diagonal halo above is not.
static_assert(
    loop_abstraction::impl::HaloIsProjectionClosed<loop_abstraction::halo::none_t>());
static_assert(loop_abstraction::impl::HaloIsProjectionClosed<plus_j_halo_t>());
static_assert(loop_abstraction::impl::HaloIsProjectionClosed<minus_i_halo_t>());
static_assert(loop_abstraction::impl::HaloIsProjectionClosed<k_triplet_halo_t>());
static_assert(
    loop_abstraction::impl::HaloIsProjectionClosed<plus_two_i_minus_k_halo_t>());
static_assert(
    !loop_abstraction::impl::HaloIsProjectionClosed<not_projection_closed_halo_t>());

// HaloReducedRange picks the contiguous [begin, end) run of offsets that survive in a
// reduced-dimension run. ndim follows the Parthenon convention (i active for ndim>=1,
// j for >=2, k for >=3). Verified at compile time so the drop logic is pinned down
// independent of any mesh construction.
//
// plus_j = {(0,0,0),(0,1,0)}: kept whole in 3D/2D, +j dropped in 1D (only identity).
static_assert(loop_abstraction::HaloReducedRange<plus_j_halo_t>(3).begin == 0 &&
              loop_abstraction::HaloReducedRange<plus_j_halo_t>(3).end == 2);
static_assert(loop_abstraction::HaloReducedRange<plus_j_halo_t>(2).begin == 0 &&
              loop_abstraction::HaloReducedRange<plus_j_halo_t>(2).end == 2);
static_assert(loop_abstraction::HaloReducedRange<plus_j_halo_t>(1).begin == 0 &&
              loop_abstraction::HaloReducedRange<plus_j_halo_t>(1).end == 1);
// k_triplet = {(-1,0,0),(0,0,0),(1,0,0)}: whole in 3D; in 2D only the identity (the
// middle offset) survives -> the contiguous run [1,2).
static_assert(loop_abstraction::HaloReducedRange<k_triplet_halo_t>(3).begin == 0 &&
              loop_abstraction::HaloReducedRange<k_triplet_halo_t>(3).end == 3);
static_assert(loop_abstraction::HaloReducedRange<k_triplet_halo_t>(2).begin == 1 &&
              loop_abstraction::HaloReducedRange<k_triplet_halo_t>(2).end == 2);
static_assert(loop_abstraction::HaloReducedRange<k_triplet_halo_t>(1).begin == 1 &&
              loop_abstraction::HaloReducedRange<k_triplet_halo_t>(1).end == 2);

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
parthenon::HostArray5D<Real> RunAutoIndexBody(const ProblemSpec &spec, const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto out = MakeOutput(idx_space);
  ZeroView(out);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        for (int v = 0; v < kNVars; ++v) {
          loop_abstraction::inner(idx_range, [&](auto idx) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            out(b, v, k, j, i) += EncodeValue(b, v, k, j, i);
          });
        }
      });

  Kokkos::fence();
  return MirrorToHost(out);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
parthenon::HostArray5D<Real> RunKjiBody(const ProblemSpec &spec, const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto out = MakeOutput(idx_space);
  ZeroView(out);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        for (int v = 0; v < kNVars; ++v) {
          loop_abstraction::inner(idx_range, [&](const int k, const int j, const int i) {
            out(b, v, k, j, i) += EncodeValue(b, v, k, j, i);
          });
        }
      });

  Kokkos::fence();
  return MirrorToHost(out);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunContractCase(const ProblemSpec &spec, const int ninner, const char *body_name,
                     const bool kji_body) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", body=" << body_name);

  const auto default_out = kji_body ? RunKjiBody<LOOP_TAG, INNER_TAG>(spec, ninner)
                                    : RunAutoIndexBody<LOOP_TAG, INNER_TAG>(spec, ninner);

  CheckLogicalContract(PatternIndexSpace<LOOP_TAG, INNER_TAG>(
                           spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner),
                       default_out);

  // Raw-vs-Kokkos parity is only meaningful where the raw backend can run (host):
  // on a device build the raw backend would drive host loops over device memory. So
  // only cross-check the two backends when raw is the default.
  if constexpr (default_loop_backend_v == loop_backend::raw) {
    const auto kokkos_out =
        kji_body
            ? RunKjiBody<LOOP_TAG, INNER_TAG, loop_backend::kokkos>(spec, ninner)
            : RunAutoIndexBody<LOOP_TAG, INNER_TAG, loop_backend::kokkos>(spec, ninner);
    CheckParity(default_out, kokkos_out,
                PatternIndexSpace<LOOP_TAG, INNER_TAG>(spec.nblocks, spec.nx, spec.ny,
                                                       spec.nz, spec.nghost, ninner));
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunPatternMatrix(const char *body_name, const bool kji_body) {
  for (const auto &spec : CoverageSpecs()) {
    const auto cases = NinnerCases(spec.nx * spec.ny * spec.nz);
    for (const int ninner : cases) {
      RunContractCase<LOOP_TAG, INNER_TAG>(spec, ninner, body_name, kji_body);
    }
  }
}

// Is (k,j,i) in the halo-extended logical set S_halo = S ∪ shift(S, h) for all
// halo offsets h? Only offsets kept in a reduced-dimension run count (those pointing
// into a degenerate direction are dropped by the abstraction), so this iterates the
// same [begin, end) run the production code uses.
template <class HaloType, class IndexSpaceType>
bool InHaloLogicalSet(const IndexSpaceType &idx_space, const int k, const int j,
                      const int i) {
  const auto hrange = loop_abstraction::HaloReducedRange<HaloType>(idx_space.GetNdim());
  for (int n = hrange.begin; n < hrange.end; ++n) {
    // (k,j,i) is a shifted image of a logical cell p under offset h_n iff
    // p = (k,j,i) - h_n is itself a logical cell.
    if (IsLogicalCell(idx_space, k - HaloType::dk(n), j - HaloType::dj(n),
                      i - HaloType::di(n))) {
      return true;
    }
  }
  return false;
}

// bvoi-specific regression: a full-block AddHalo traversal should cover each point in
// the block's halo-extended set exactly once. This is not a general halo contract for
// bovi/boiv, where neighboring outer ranges may overlap.
template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG, bool USE_KOKKOS>
void RunHaloProducerSingleTouchCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner
                  << ", producer-single-touch=" << typeid(HaloType).name()
                  << ", backend=" << (USE_KOKKOS ? "kokkos" : "raw"));

  using IndexSpaceType =
      PatternIndexSpace<LOOP_TAG, INNER_TAG,
                        USE_KOKKOS ? loop_backend::kokkos : loop_backend::raw>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto touches = MakeOutput(idx_space);
  ZeroView(touches);

  // Backend is fixed by the IndexSpace template argument above; the public outer()
  // dispatches accordingly.
  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
        for (int v = 0; v < kNVars; ++v) {
          loop_abstraction::inner(halo_range, [&](auto idx) {
            const auto [k, j, i] = halo_range.GetKJI(idx);
            touches(b, v, k, j, i) += 1.0;
          });
          idx_range.TeamBarrier();
        }
      });

  Kokkos::fence();
  const auto host = MirrorToHost(touches);

  const auto &memory = idx_space.GetMemoryIndexer();
  for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
    for (int v = 0; v < kNVars; ++v) {
      for (int k = memory.template StartIdx<0>(); k <= memory.template EndIdx<0>(); ++k) {
        for (int j = memory.template StartIdx<1>(); j <= memory.template EndIdx<1>();
             ++j) {
          for (int i = memory.template StartIdx<2>(); i <= memory.template EndIdx<2>();
               ++i) {
            INFO("b=" << b << ", v=" << v << ", k=" << k << ", j=" << j << ", i=" << i);
            const bool in_set = InHaloLogicalSet<HaloType>(idx_space, k, j, i);
            if (in_set) {
              // Every cell of the halo-extended logical set: touched exactly once.
              REQUIRE(host(b, v, k, j, i) == Approx(1.0));
            } else if constexpr (!UsesMemorySpan<INNER_TAG>()) {
              // Logical inner tags must not touch anything outside S_halo. Memory
              // tags may legitimately touch ghost cells in the contiguous span, so
              // only assert the no-extra-touch bound for logical tags.
              REQUIRE(host(b, v, k, j, i) == Approx(0.0));
            }
          }
        }
      }
    }
  }
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunHaloProducerSingleTouchPatternMatrix(const ProblemSpec &spec,
                                             const std::vector<int> &ninner_cases) {
  for (const int ninner : ninner_cases) {
    // Always exercise the kokkos backend (valid on host and device). Additionally
    // run the raw backend only on a host build; on a device build the raw backend
    // would drive host loops over device memory, which is invalid.
    RunHaloProducerSingleTouchCase<HaloType, LOOP_TAG, INNER_TAG, true>(spec, ninner);
    if constexpr (default_loop_backend_v == loop_backend::raw) {
      RunHaloProducerSingleTouchCase<HaloType, LOOP_TAG, INNER_TAG, false>(spec, ninner);
    }
  }
}

struct ScopedNghost {
  const int old;
  ScopedNghost() : old(parthenon::Globals::nghost) {}
  ~ScopedNghost() { parthenon::Globals::nghost = old; }
};

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunPackViewCase(const PackViewSpec &spec, const int ninner, const bool kji_body) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner
                  << ", body=" << (kji_body ? "kji" : "auto"));

  ScopedNghost guard;
  parthenon::Globals::nghost = spec.nghost;

  const std::vector<int> scalar_shape{spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost};
  Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
  auto pkg = std::make_shared<StateDescriptor>("PackView package");
  pkg->AddField<v1>(m);
  pkg->AddField<v2>(m);
  pkg->AddField<v5>(m);

  BlockList_t block_list = MakeBlockList(pkg, spec.nblocks, spec.ncell, 3);
  MeshData<Real> mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);

  std::array<std::string, 3> var_names{v1::name(), v2::name(), v5::name()};
  const auto ib = block_list[0]->cellbounds.GetBoundsI(IndexDomain::entire);
  const auto jb = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::entire);
  const auto kb = block_list[0]->cellbounds.GetBoundsK(IndexDomain::entire);

  for (int b = 0; b < spec.nblocks; ++b) {
    auto &pmb = block_list[b];
    auto &pmbd = pmb->meshblock_data.Get();
    for (int v = 0; v < static_cast<int>(var_names.size()); ++v) {
      auto var = pmbd->Get(var_names[v]);
      auto var4 = var.data.template Get<4>();
      const int num_components = var.GetDim(4);
      parthenon::par_for(
          parthenon::loop_pattern_mdrange_tag, "initialize pack view data",
          parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(int k, int j, int i) {
            for (int c = 0; c < num_components; ++c) {
              var4(c, k, j, i) = PackViewSourceValue(b, v, k, j, i);
            }
          });
    }
  }

  Kokkos::fence();

  using IndexSpaceType = IndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell, spec.nghost,
                           ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(pkg.get());
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        auto pack_view = loop_abstraction::make_pack_view(idx_range, sparse_pack);
        if (kji_body) {
          loop_abstraction::inner(idx_range, [&](const int k, const int j, const int i) {
            pack_view(v1(), k, j, i) = PackViewExpectedValue(b, 0, k, j, i);
            pack_view(v2(), k, j, i) = PackViewExpectedValue(b, 1, k, j, i);
            pack_view(v5(), k, j, i) = PackViewExpectedValue(b, 2, k, j, i);
          });
        } else {
          loop_abstraction::inner(idx_range, [&](auto idx) {
            if constexpr (std::is_same_v<std::decay_t<decltype(idx)>, int>) {
              const auto [k, j, i] = idx_range.GetKJI(idx);
              pack_view(v1(), idx) = PackViewExpectedValue(b, 0, k, j, i);
              pack_view(v2(), idx) = PackViewExpectedValue(b, 1, k, j, i);
              pack_view(v5(), idx) = PackViewExpectedValue(b, 2, k, j, i);
            } else {
              const auto [k, j, i] = idx_range.GetKJI(idx);
              pack_view(v1(), idx) = PackViewExpectedValue(b, 0, k, j, i);
              pack_view(v2(), idx) = PackViewExpectedValue(b, 1, k, j, i);
              pack_view(v5(), idx) = PackViewExpectedValue(b, 2, k, j, i);
            }
          });
        }
      });

  Kokkos::fence();

  int nwrong = 0;
  const auto ib_int = block_list[0]->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb_int = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb_int = block_list[0]->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "check pack view", parthenon::DevExecSpace(),
      0, sparse_pack.GetNBlocks() - 1, kb_int.s, kb_int.e, jb_int.s, jb_int.e, ib_int.s,
      ib_int.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
        if (sparse_pack(b, v1(), k, j, i) != PackViewExpectedValue(b, 0, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, v2(), k, j, i) != PackViewExpectedValue(b, 1, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, v5(), k, j, i) != PackViewExpectedValue(b, 2, k, j, i)) {
          ++ltot;
        }
      },
      nwrong);
  REQUIRE(nwrong == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunFacePackViewCase(const PackViewSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", topological=face");

  ScopedNghost guard;
  parthenon::Globals::nghost = spec.nghost;

  const std::vector<int> scalar_shape{spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost};
  Metadata m_cell({Metadata::Independent}, scalar_shape);
  Metadata m_face;
  if constexpr (INNER_TAG == inner_tag::logical_coords) {
    m_face = Metadata({Metadata::Face, Metadata::Independent});
  } else {
    m_face = Metadata({Metadata::Face, Metadata::Independent, Metadata::CellMemAligned});
  }
  auto pkg = std::make_shared<StateDescriptor>("FacePackView package");
  pkg->AddField<v1>(m_cell);
  pkg->AddField<vf>(m_face);
  pkg->AddField<v2>(m_cell);

  BlockList_t block_list = MakeBlockList(pkg, spec.nblocks, spec.ncell, 3);
  MeshData<Real> mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);

  using TE = TopologicalElement;
  using IndexSpaceType = IndexSpace<LOOP_TAG, INNER_TAG>;
  const auto memory_te = INNER_TAG == inner_tag::logical_coords ? TE::NN : TE::CC;
  IndexSpaceType idx_space(loop_abstraction::NInner(ninner), IndexDomain::interior, 0,
                           spec.nblocks, &mesh_data, TE::NN, memory_te);
  auto desc = parthenon::MakePackDescriptor<v1, vf, v2>(pkg.get());
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        auto pack_view = loop_abstraction::make_pack_view(idx_range, sparse_pack);
        loop_abstraction::inner(idx_range, [&](auto idx) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          pack_view(v1(), idx) = PackViewExpectedValue(b, 0, k, j, i);
          pack_view(TE::F1, vf(), idx) = FacePackViewExpectedValue(b, TE::F1, k, j, i);
          pack_view(TE::F2, vf(), idx) = FacePackViewExpectedValue(b, TE::F2, k, j, i);
          pack_view(TE::F3, vf(), idx) = FacePackViewExpectedValue(b, TE::F3, k, j, i);
          pack_view(v2(), idx) = PackViewExpectedValue(b, 1, k, j, i);
        });
      });

  Kokkos::fence();

  int nwrong = 0;
  const auto ib_int = block_list[0]->cellbounds.GetBoundsI(IndexDomain::interior, TE::NN);
  const auto jb_int = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::interior, TE::NN);
  const auto kb_int = block_list[0]->cellbounds.GetBoundsK(IndexDomain::interior, TE::NN);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "check face pack view",
      parthenon::DevExecSpace(), 0, sparse_pack.GetNBlocks() - 1, kb_int.s, kb_int.e,
      jb_int.s, jb_int.e, ib_int.s, ib_int.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
        if (sparse_pack(b, v1(), k, j, i) != PackViewExpectedValue(b, 0, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, TE::F1, vf(), k, j, i) !=
            FacePackViewExpectedValue(b, TE::F1, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, TE::F2, vf(), k, j, i) !=
            FacePackViewExpectedValue(b, TE::F2, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, TE::F3, vf(), k, j, i) !=
            FacePackViewExpectedValue(b, TE::F3, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, v2(), k, j, i) != PackViewExpectedValue(b, 1, k, j, i)) {
          ++ltot;
        }
      },
      nwrong);
  REQUIRE(nwrong == 0);
}

// Write a known pattern into the pack's flux arrays through make_flux_pack_view, then
// read it back through sparse_pack.flux() on the interior. Exercises flux_pack_view_t
// end to end (dense write, all inner tags, all ninner). dir is fixed per case.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunFluxViewCase(const PackViewSpec &spec, const int ninner, const int dir) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", flux dir=" << dir);

  ScopedNghost guard;
  parthenon::Globals::nghost = spec.nghost;

  const std::vector<int> scalar_shape{spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost};
  Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
  auto pkg = std::make_shared<StateDescriptor>("FluxView package");
  pkg->AddField<v1>(m);
  pkg->AddField<v2>(m);
  pkg->AddField<v5>(m);

  BlockList_t block_list = MakeBlockList(pkg, spec.nblocks, spec.ncell, 3);
  MeshData<Real> mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);

  using IndexSpaceType = IndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell, spec.nghost,
                           ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(pkg.get(), {},
                                                        {parthenon::PDOpt::WithFluxes});
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        auto flux_view =
            loop_abstraction::make_flux_pack_view(idx_range, sparse_pack, dir);
        loop_abstraction::inner(idx_range, [&](auto idx) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          flux_view(v1(), idx) = PackViewExpectedValue(b, 0, k, j, i);
          flux_view(v2(), idx) = PackViewExpectedValue(b, 1, k, j, i);
          flux_view(v5(), idx) = PackViewExpectedValue(b, 2, k, j, i);
        });
      });

  Kokkos::fence();

  int nwrong = 0;
  const auto ib_int = block_list[0]->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb_int = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb_int = block_list[0]->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "check flux view", parthenon::DevExecSpace(),
      0, sparse_pack.GetNBlocks() - 1, kb_int.s, kb_int.e, jb_int.s, jb_int.e, ib_int.s,
      ib_int.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
        if (sparse_pack.flux(b, dir, v1(), k, j, i) !=
            PackViewExpectedValue(b, 0, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack.flux(b, dir, v2(), k, j, i) !=
            PackViewExpectedValue(b, 1, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack.flux(b, dir, v5(), k, j, i) !=
            PackViewExpectedValue(b, 2, k, j, i)) {
          ++ltot;
        }
      },
      nwrong);
  REQUIRE(nwrong == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunFluxViewPatternMatrix() {
  for (const auto &spec : PackViewCoverageSpecs()) {
    for (const int ninner : PackViewNinnerCases(spec.ncell * spec.ncell * spec.ncell)) {
      for (const int dir : {1, 2, 3}) {
        RunFluxViewCase<LOOP_TAG, INNER_TAG>(spec, ninner, dir);
      }
    }
  }
}

// Single-variable (anonymous) state view: write a known pattern through make_var_view,
// then read it back through sparse_pack() on the interior. Exercises var_view_t end to
// end, covering BOTH index forms: v1 via a typed index (v1()) and v2/v5 via a raw int
// index resolved through GetIndex.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunVarViewCase(const PackViewSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner);

  ScopedNghost guard;
  parthenon::Globals::nghost = spec.nghost;

  const std::vector<int> scalar_shape{spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost};
  Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
  auto pkg = std::make_shared<StateDescriptor>("VarView package");
  pkg->AddField<v1>(m);
  pkg->AddField<v2>(m);
  pkg->AddField<v5>(m);

  BlockList_t block_list = MakeBlockList(pkg, spec.nblocks, spec.ncell, 3);
  MeshData<Real> mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);

  using IndexSpaceType = IndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell, spec.nghost,
                           ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(pkg.get());
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        // typed index
        auto var1 = loop_abstraction::make_var_view(idx_range, sparse_pack, v1());
        // raw int index (resolved through GetIndex's integral overload)
        auto var2 = loop_abstraction::make_var_view(idx_range, sparse_pack,
                                                    sparse_pack.GetIndex(b, v2()));
        auto var5 = loop_abstraction::make_var_view(idx_range, sparse_pack,
                                                    sparse_pack.GetIndex(b, v5()));
        loop_abstraction::inner(idx_range, [&](auto idx) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          var1(idx) = PackViewExpectedValue(b, 0, k, j, i);
          var2(idx) = PackViewExpectedValue(b, 1, k, j, i);
          var5(idx) = PackViewExpectedValue(b, 2, k, j, i);
        });
      });

  Kokkos::fence();

  int nwrong = 0;
  const auto ib_int = block_list[0]->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb_int = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb_int = block_list[0]->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "check var view", parthenon::DevExecSpace(), 0,
      sparse_pack.GetNBlocks() - 1, kb_int.s, kb_int.e, jb_int.s, jb_int.e, ib_int.s,
      ib_int.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
        if (sparse_pack(b, v1(), k, j, i) != PackViewExpectedValue(b, 0, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, v2(), k, j, i) != PackViewExpectedValue(b, 1, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack(b, v5(), k, j, i) != PackViewExpectedValue(b, 2, k, j, i)) {
          ++ltot;
        }
      },
      nwrong);
  REQUIRE(nwrong == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunVarViewPatternMatrix() {
  for (const auto &spec : PackViewCoverageSpecs()) {
    for (const int ninner : PackViewNinnerCases(spec.ncell * spec.ncell * spec.ncell)) {
      RunVarViewCase<LOOP_TAG, INNER_TAG>(spec, ninner);
    }
  }
}

// Single-variable (anonymous) flux view: write a known pattern into one variable's flux
// array through make_flux_view, read it back through sparse_pack.flux(). Exercises
// flux_view_t end to end, covering both a typed index (v1()) and raw int indices.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunVarFluxViewCase(const PackViewSpec &spec, const int ninner, const int dir) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", flux dir=" << dir);

  ScopedNghost guard;
  parthenon::Globals::nghost = spec.nghost;

  const std::vector<int> scalar_shape{spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost,
                                      spec.ncell + 2 * spec.nghost};
  Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
  auto pkg = std::make_shared<StateDescriptor>("VarFluxView package");
  pkg->AddField<v1>(m);
  pkg->AddField<v2>(m);
  pkg->AddField<v5>(m);

  BlockList_t block_list = MakeBlockList(pkg, spec.nblocks, spec.ncell, 3);
  MeshData<Real> mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);

  using IndexSpaceType = IndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell, spec.nghost,
                           ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(pkg.get(), {},
                                                        {parthenon::PDOpt::WithFluxes});
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        auto flux1 = loop_abstraction::make_flux_view(idx_range, sparse_pack, dir, v1());
        auto flux2 = loop_abstraction::make_flux_view(idx_range, sparse_pack, dir,
                                                      sparse_pack.GetIndex(b, v2()));
        auto flux5 = loop_abstraction::make_flux_view(idx_range, sparse_pack, dir,
                                                      sparse_pack.GetIndex(b, v5()));
        loop_abstraction::inner(idx_range, [&](auto idx) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          flux1(idx) = PackViewExpectedValue(b, 0, k, j, i);
          flux2(idx) = PackViewExpectedValue(b, 1, k, j, i);
          flux5(idx) = PackViewExpectedValue(b, 2, k, j, i);
        });
      });

  Kokkos::fence();

  int nwrong = 0;
  const auto ib_int = block_list[0]->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb_int = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb_int = block_list[0]->cellbounds.GetBoundsK(IndexDomain::interior);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "check var flux view",
      parthenon::DevExecSpace(), 0, sparse_pack.GetNBlocks() - 1, kb_int.s, kb_int.e,
      jb_int.s, jb_int.e, ib_int.s, ib_int.e,
      KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
        if (sparse_pack.flux(b, dir, v1(), k, j, i) !=
            PackViewExpectedValue(b, 0, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack.flux(b, dir, v2(), k, j, i) !=
            PackViewExpectedValue(b, 1, k, j, i)) {
          ++ltot;
        }
        if (sparse_pack.flux(b, dir, v5(), k, j, i) !=
            PackViewExpectedValue(b, 2, k, j, i)) {
          ++ltot;
        }
      },
      nwrong);
  REQUIRE(nwrong == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunVarFluxViewPatternMatrix() {
  for (const auto &spec : PackViewCoverageSpecs()) {
    for (const int ninner : PackViewNinnerCases(spec.ncell * spec.ncell * spec.ncell)) {
      for (const int dir : {1, 2, 3}) {
        RunVarFluxViewCase<LOOP_TAG, INNER_TAG>(spec, ninner, dir);
      }
    }
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunPackViewPatternMatrix(const std::string &body_name, const bool kji_body) {
  for (const auto &spec : PackViewCoverageSpecs()) {
    for (const int ninner : PackViewNinnerCases(spec.ncell * spec.ncell * spec.ncell)) {
      const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
      INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.ncell
                      << " nghost=" << spec.nghost << ", ninner=" << ninner
                      << ", body=" << body_name);
      RunPackViewCase<LOOP_TAG, INNER_TAG>(spec, ninner, kji_body);
    }
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunPackViewPatternMatrix() {
  RunPackViewPatternMatrix<LOOP_TAG, INNER_TAG>("auto", false);
  RunPackViewPatternMatrix<LOOP_TAG, INNER_TAG>("kji", true);
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
void RunScratchRoundtripCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx << "x"
                  << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", halo=" << typeid(HaloType).name()
                  << ", backend=" << (BACKEND == loop_backend::raw ? "raw" : "kokkos"));

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real, HaloType>();
  idx_space.template AddPerPointScratch<Real, HaloType>();
  idx_space.template AddPerPointScratch<Real, HaloType>();
  idx_space.template AddPerPointScratch<Real, HaloType, 2, 3>();
  auto di = idx_space.GetDelta(parthenon::X1DIR);
  auto dj = idx_space.GetDelta(parthenon::X2DIR);
  auto dk = idx_space.GetDelta(parthenon::X3DIR);
  using offset_t = decltype(di);
  std::array<offset_t, HaloType::npoints> offsets{};
  for (int n = 0; n < HaloType::npoints; ++n) {
    offsets[n] = HaloType::di(n) * di + HaloType::dj(n) * dj + HaloType::dk(n) * dk;
  }

  MismatchCounter wrong;

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        ForceCapture(idx_space, offsets);
        const auto scratch_range = loop_abstraction::AddHalo<HaloType>(idx_range);

        auto scratch_a = loop_abstraction::GetPerPointScratch<Real>(scratch_range);
        auto scratch_b = loop_abstraction::GetPerPointScratch<Real>(scratch_range);
        auto scratch_c = loop_abstraction::GetPerPointScratch<Real>(scratch_range);
        auto scratch_shaped =
            loop_abstraction::GetPerPointScratch<Real, 2, 3>(scratch_range);

        loop_abstraction::inner(scratch_range, [&](auto idx) {
          const auto [k, j, i] = scratch_range.GetKJI(idx);
          scratch_a(idx) = 0.0;
          scratch_b(idx) = 0.0;
          scratch_c(idx) = 0.0;
          for (int c0 = 0; c0 < 2; ++c0) {
            for (int c1 = 0; c1 < 3; ++c1) {
              scratch_shaped(c0, c1, idx) = ShapedScratchValue(b, c0, c1, k, j, i);
            }
          }
        });
        scratch_range.TeamBarrier();

        for (int v = 0; v < kNVars; ++v) {
          loop_abstraction::inner(scratch_range, [&](auto idx) {
            const auto [k, j, i] = scratch_range.GetKJI(idx);
            scratch_a(idx) += EncodeValue(b, v, k, j, i);
            scratch_b(idx) += EncodeValue(b, v, k, j, i);
            scratch_c(idx) += EncodeValue(b, v, k, j, i);
          });
          scratch_range.TeamBarrier();
        }

        loop_abstraction::inner(idx_range, [&](auto idx) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          if constexpr (INNER_TAG == inner_tag::memory &&
                        !std::is_same_v<HaloType, loop_abstraction::halo::none_t>) {
            if (!IsLogicalCell(idx_space, k, j, i)) {
              return;
            }
          }
          // Only offsets kept in a reduced-dimension run are produced (those pointing
          // into a degenerate direction are dropped), so verify over the same
          // [begin, end) run the abstraction uses.
          const auto hrange =
              loop_abstraction::HaloReducedRange<HaloType>(idx_space.GetNdim());
          for (int n = hrange.begin; n < hrange.end; ++n) {
            const int kk = k + HaloType::dk(n);
            const int jj = j + HaloType::dj(n);
            const int ii = i + HaloType::di(n);
            const Real expected = ScratchExpectedValue(b, kk, jj, ii);
            const auto shifted = idx + offsets[n];
            wrong.note(NotApprox(scratch_a(shifted), expected));
            wrong.note(NotApprox(scratch_b(Index3{kk, jj, ii}), expected));
            wrong.note(NotApprox(scratch_c(kk, jj, ii), expected));
            for (int c0 = 0; c0 < 2; ++c0) {
              for (int c1 = 0; c1 < 3; ++c1) {
                const Real shaped_expected = ShapedScratchValue(b, c0, c1, kk, jj, ii);
                wrong.note(NotApprox(scratch_shaped(c0, c1, Index3{kk, jj, ii}),
                                     shaped_expected));
                wrong.note(NotApprox(scratch_shaped(c0, c1, shifted), shaped_expected));
                wrong.note(
                    NotApprox(scratch_shaped(c0, c1, kk, jj, ii), shaped_expected));
              }
            }
          }
        });
      });

  REQUIRE(wrong.total() == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
void RunScratchCase(const ProblemSpec &spec, const int ninner) {
  RunScratchRoundtripCase<loop_abstraction::halo::none_t, LOOP_TAG, INNER_TAG, BACKEND>(
      spec, ninner);
}

// Exercise scratch.Zero(): fill with garbage, Zero(), accumulate, verify; then
// reuse the same buffer (Zero() again, re-accumulate) to confirm a buffer can be
// zeroed and reused rather than needing a fresh zero-initialized allocation. The
// BACKEND parameter selects the raw HostScratch1D::Zero() or the team-parallel
// TeamScratch1D::Zero().
template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
void RunScratchZeroCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx << "x"
                  << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", Zero()");

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real>();

  MismatchCounter wrong;

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        auto scratch = loop_abstraction::GetPerPointScratch<Real>(idx_range);

        for (int pass = 0; pass < 2; ++pass) {
          // Dirty the whole buffer, then Zero() it.
          loop_abstraction::inner(idx_range, [&](auto idx) { scratch(idx) = -7.0; });
          idx_range.TeamBarrier();
          scratch.Zero();
          idx_range.TeamBarrier();

          for (int v = 0; v < kNVars; ++v) {
            loop_abstraction::inner(idx_range, [&](auto idx) {
              const auto [k, j, i] = idx_range.GetKJI(idx);
              scratch(idx) += EncodeValue(b, v, k, j, i);
            });
            idx_range.TeamBarrier();
          }

          loop_abstraction::inner(idx_range, [&](auto idx) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            wrong.note(NotApprox(scratch(idx), ScratchExpectedValue(b, k, j, i)));
          });
        }
      });

  REQUIRE(wrong.total() == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
void RunScratchZeroPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchZeroCase<LOOP_TAG, INNER_TAG, BACKEND>(spec, ninner);
    }
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
void RunScratchPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchCase<LOOP_TAG, INNER_TAG, BACKEND>(spec, ninner);
    }
  }
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
void RunScratchHaloCase(const ProblemSpec &spec, const int ninner) {
  RunScratchRoundtripCase<HaloType, LOOP_TAG, INNER_TAG, BACKEND>(spec, ninner);
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
void RunScratchHaloPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchHaloCase<HaloType, LOOP_TAG, INNER_TAG, BACKEND>(spec, ninner);
    }
  }
}

template <class HaloType, inner_tag INNER_TAG, parthenon::CoordinateDirection DIR,
          int SIGN>
void RunBoivScratchDeltaCase(const ProblemSpec &spec) {
  using IndexSpaceType = PatternIndexSpace<loop_tag::boiv, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost);
  idx_space.template AddPerPointScratch<Real, HaloType>();
  const auto delta = idx_space.GetDelta(DIR);

  MismatchCounter wrong;

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
        auto scratch = loop_abstraction::GetPerPointScratch<Real>(halo_range);

        loop_abstraction::inner(halo_range, [&](auto idx) {
          const auto [k, j, i] = halo_range.GetKJI(idx);
          scratch(idx) = EncodeValue(b, 0, k, j, i);
        });
        idx_range.TeamBarrier();
        loop_abstraction::inner(idx_range, [&](auto idx) {
          const auto shifted = idx + SIGN * delta;
          const auto [k, j, i] = idx_range.GetKJI(idx);
          int kk = k;
          int jj = j;
          int ii = i;
          if constexpr (DIR == parthenon::X1DIR) {
            ii += SIGN;
          } else if constexpr (DIR == parthenon::X2DIR) {
            jj += SIGN;
          } else if constexpr (DIR == parthenon::X3DIR) {
            kk += SIGN;
          }
          wrong.note(NotApprox(scratch(shifted), EncodeValue(b, 0, kk, jj, ii)));
        });
      });

  REQUIRE(wrong.total() == 0);
}

template <inner_tag INNER_TAG>
void RunBoivScratchMixedDeltaCase(const ProblemSpec &spec) {
  using IndexSpaceType = PatternIndexSpace<loop_tag::boiv, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost);
  idx_space.template AddPerPointScratch<Real, plus_two_i_minus_k_halo_t>();
  const auto dx1 = idx_space.GetDelta(parthenon::X1DIR);
  const auto dx3 = idx_space.GetDelta(parthenon::X3DIR);

  MismatchCounter wrong;

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        const auto halo_range =
            loop_abstraction::AddHalo<plus_two_i_minus_k_halo_t>(idx_range);
        auto scratch = loop_abstraction::GetPerPointScratch<Real>(halo_range);

        loop_abstraction::inner(halo_range, [&](auto idx) {
          const auto [k, j, i] = halo_range.GetKJI(idx);
          scratch(idx) = EncodeValue(b, 0, k, j, i);
        });
        idx_range.TeamBarrier();

        loop_abstraction::inner(idx_range, [&](auto idx) {
          const auto shifted = idx + 2 * dx1 - dx3;
          const auto [k, j, i] = idx_range.GetKJI(idx);
          wrong.note(NotApprox(scratch(shifted), EncodeValue(b, 0, k - 1, j, i + 2)));
        });
      });

  REQUIRE(wrong.total() == 0);
}

// --------------------------------------------------------------------------------------
// Reduction coverage. outer_reduce/inner_reduce fold a single Kokkos reducer over the
// logical cells of the space. The value reduced is EncodeValue(b, 0, k, j, i); the host
// reference iterates the same logical cells so results are exact.
// --------------------------------------------------------------------------------------

// Host reference reductions over every logical cell of every block.
template <class IndexSpaceType>
Real ReferenceSum(const IndexSpaceType &idx_space) {
  const auto &logical = idx_space.GetLogicalIndexer();
  Real total = 0.0;
  for (int b = 0; b < idx_space.GetNBlocks(); ++b)
    for (int flat = 0; flat < static_cast<int>(logical.size()); ++flat) {
      const auto [k, j, i] = logical(flat);
      total += EncodeValue(b, 0, k, j, i);
    }
  return total;
}

template <class IndexSpaceType>
Real ReferenceMin(const IndexSpaceType &idx_space) {
  const auto &logical = idx_space.GetLogicalIndexer();
  Real best = std::numeric_limits<Real>::max();
  for (int b = 0; b < idx_space.GetNBlocks(); ++b)
    for (int flat = 0; flat < static_cast<int>(logical.size()); ++flat) {
      const auto [k, j, i] = logical(flat);
      best = std::min(best, EncodeValue(b, 0, k, j, i));
    }
  return best;
}

template <class IndexSpaceType>
Real ReferenceMax(const IndexSpaceType &idx_space) {
  const auto &logical = idx_space.GetLogicalIndexer();
  Real best = std::numeric_limits<Real>::lowest();
  for (int b = 0; b < idx_space.GetNBlocks(); ++b)
    for (int flat = 0; flat < static_cast<int>(logical.size()); ++flat) {
      const auto [k, j, i] = logical(flat);
      best = std::max(best, EncodeValue(b, 0, k, j, i));
    }
  return best;
}

// Sum EncodeValue over the whole space with outer_reduce + a single inner_reduce.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
Real RunReduceSum(const ProblemSpec &spec, const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  using reduce_t = loop_abstraction::Reduction<Kokkos::Sum<Real>>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  reduce_t::value_t result = 0.0;
  loop_abstraction::outer_reduce(
      idx_space,
      KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b,
                    const reduce_t::handle_t &handle) {
        loop_abstraction::inner_reduce(idx_range, handle,
                                       [&](auto idx, reduce_t::value_t &v) {
                                         const auto [k, j, i] = idx_range.GetKJI(idx);
                                         v += EncodeValue(b, 0, k, j, i);
                                       });
      },
      reduce_t::reducer_t(result));
  Kokkos::fence();
  return result;
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
Real RunReduceMin(const ProblemSpec &spec, const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  using reduce_t = loop_abstraction::Reduction<Kokkos::Min<Real>>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  reduce_t::value_t result = 0.0;
  loop_abstraction::outer_reduce(
      idx_space,
      KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b,
                    const reduce_t::handle_t &handle) {
        loop_abstraction::inner_reduce(idx_range, handle,
                                       [&](auto idx, reduce_t::value_t &v) {
                                         const auto [k, j, i] = idx_range.GetKJI(idx);
                                         v = Kokkos::min(v, EncodeValue(b, 0, k, j, i));
                                       });
      },
      reduce_t::reducer_t(result));
  Kokkos::fence();
  return result;
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
Real RunReduceMax(const ProblemSpec &spec, const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  using reduce_t = loop_abstraction::Reduction<Kokkos::Max<Real>>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  reduce_t::value_t result = 0.0;
  loop_abstraction::outer_reduce(
      idx_space,
      KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b,
                    const reduce_t::handle_t &handle) {
        loop_abstraction::inner_reduce(idx_range, handle,
                                       [&](auto idx, reduce_t::value_t &v) {
                                         const auto [k, j, i] = idx_range.GetKJI(idx);
                                         v = Kokkos::max(v, EncodeValue(b, 0, k, j, i));
                                       });
      },
      reduce_t::reducer_t(result));
  Kokkos::fence();
  return result;
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunReducePatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
      INFO("pattern=" << pattern_name << ", ninner=" << ninner);
      PatternIndexSpace<LOOP_TAG, INNER_TAG> ref(spec.nblocks, spec.nx, spec.ny, spec.nz,
                                                 spec.nghost, ninner);
      REQUIRE(RunReduceSum<LOOP_TAG, INNER_TAG>(spec, ninner) ==
              Approx(ReferenceSum(ref)));
      REQUIRE(RunReduceMin<LOOP_TAG, INNER_TAG>(spec, ninner) ==
              Approx(ReferenceMin(ref)));
      REQUIRE(RunReduceMax<LOOP_TAG, INNER_TAG>(spec, ninner) ==
              Approx(ReferenceMax(ref)));
    }
  }
}

// Interleave a plain inner (fill scratch) with an inner_reduce (reduce over scratch)
// inside a single outer_reduce region. Reduces the per-cell sum-over-vars, so the
// expected total is ReferenceSum scaled by summing ScratchExpectedValue.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
Real RunReduceScratchInterleave(const ProblemSpec &spec, const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  using reduce_t = loop_abstraction::Reduction<Kokkos::Sum<Real>>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real>();
  reduce_t::value_t result = 0.0;
  loop_abstraction::outer_reduce(
      idx_space,
      KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b,
                    const reduce_t::handle_t &handle) {
        auto scratch = loop_abstraction::GetPerPointScratch<Real>(idx_range);
        scratch.Zero();
        idx_range.TeamBarrier();
        // Variable (component) loop OUTSIDE the inner cell loop, matching the
        // per-cell-accumulator pattern in the docs: each iteration is a full inner
        // sweep that adds one component's contribution, with a barrier between sweeps.
        for (int v = 0; v < kNVars; ++v) {
          loop_abstraction::inner(idx_range, [&](auto idx) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            scratch(idx) += EncodeValue(b, v, k, j, i);
          });
          idx_range.TeamBarrier();
        }
        loop_abstraction::inner_reduce(
            idx_range, handle, [&](auto idx, reduce_t::value_t &v) { v += scratch(idx); });
      },
      reduce_t::reducer_t(result));
  Kokkos::fence();
  return result;
}

template <class IndexSpaceType>
Real ReferenceScratchSum(const IndexSpaceType &idx_space) {
  const auto &logical = idx_space.GetLogicalIndexer();
  Real total = 0.0;
  for (int b = 0; b < idx_space.GetNBlocks(); ++b)
    for (int flat = 0; flat < static_cast<int>(logical.size()); ++flat) {
      const auto [k, j, i] = logical(flat);
      total += ScratchExpectedValue(b, k, j, i);
    }
  return total;
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunReduceScratchInterleavePatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
      INFO("pattern=" << pattern_name << ", ninner=" << ninner);
      PatternIndexSpace<LOOP_TAG, INNER_TAG> ref(spec.nblocks, spec.nx, spec.ny, spec.nz,
                                                 spec.nghost, ninner);
      REQUIRE(RunReduceScratchInterleave<LOOP_TAG, INNER_TAG>(spec, ninner) ==
              Approx(ReferenceScratchSum(ref)));
    }
  }
}

// Two inner_reduce calls in one region, each covering the same cells, must double the
// single-inner_reduce sum (both join into the same handle/accumulator).
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
Real RunReduceTwoInner(const ProblemSpec &spec, const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  using reduce_t = loop_abstraction::Reduction<Kokkos::Sum<Real>>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  reduce_t::value_t result = 0.0;
  loop_abstraction::outer_reduce(
      idx_space,
      KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b,
                    const reduce_t::handle_t &handle) {
        loop_abstraction::inner_reduce(idx_range, handle,
                                       [&](auto idx, reduce_t::value_t &v) {
                                         const auto [k, j, i] = idx_range.GetKJI(idx);
                                         v += EncodeValue(b, 0, k, j, i);
                                       });
        loop_abstraction::inner_reduce(idx_range, handle,
                                       [&](auto idx, reduce_t::value_t &v) {
                                         const auto [k, j, i] = idx_range.GetKJI(idx);
                                         v += EncodeValue(b, 0, k, j, i);
                                       });
      },
      reduce_t::reducer_t(result));
  Kokkos::fence();
  return result;
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunReduceTwoInnerPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
      INFO("pattern=" << pattern_name << ", ninner=" << ninner);
      PatternIndexSpace<LOOP_TAG, INNER_TAG> ref(spec.nblocks, spec.nx, spec.ny, spec.nz,
                                                 spec.nghost, ninner);
      REQUIRE(RunReduceTwoInner<LOOP_TAG, INNER_TAG>(spec, ninner) ==
              Approx(2.0 * ReferenceSum(ref)));
    }
  }
}

} // namespace

// The TEST_CASEs below enumerate every valid (loop_tag, inner_tag) pair. The
// boiv/memory combination is intentionally absent throughout: it is rejected at
// compile time by a static_assert in IndexSpace (boiv walks one logical cell at a
// time, so a contiguous memory-span inner traversal is not a meaningful contract).
TEST_CASE("loop abstraction logical contracts with auto index bodies",
          "[loop_abstraction][contract]") {
  RunPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>("auto", false);
  RunPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>("auto", false);
  RunPatternMatrix<loop_tag::bvoi, inner_tag::memory>("auto", false);
  RunPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>("auto", false);
  RunPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>("auto", false);
  RunPatternMatrix<loop_tag::bovi, inner_tag::memory>("auto", false);
  RunPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>("auto", false);
  RunPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>("auto", false);
}

TEST_CASE("loop abstraction logical contracts with kji bodies",
          "[loop_abstraction][contract]") {
  RunPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>("kji", true);
  RunPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>("kji", true);
  RunPatternMatrix<loop_tag::bvoi, inner_tag::memory>("kji", true);
  RunPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>("kji", true);
  RunPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>("kji", true);
  RunPatternMatrix<loop_tag::bovi, inner_tag::memory>("kji", true);
  RunPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>("kji", true);
  RunPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>("kji", true);
}

TEST_CASE("loop abstraction scratch roundtrip", "[loop_abstraction][contract][scratch]") {
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunScratchPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunScratchPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction scratch Zero", "[loop_abstraction][contract][scratch]") {
  RunScratchZeroPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunScratchZeroPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunScratchZeroPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunScratchZeroPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunScratchZeroPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunScratchZeroPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunScratchZeroPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunScratchZeroPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction scratch Zero kokkos",
          "[loop_abstraction][contract][scratch]") {
  RunScratchZeroPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat,
                              loop_backend::kokkos>();
  RunScratchZeroPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords,
                              loop_backend::kokkos>();
  RunScratchZeroPatternMatrix<loop_tag::bvoi, inner_tag::memory, loop_backend::kokkos>();
  RunScratchZeroPatternMatrix<loop_tag::bovi, inner_tag::logical_flat,
                              loop_backend::kokkos>();
  RunScratchZeroPatternMatrix<loop_tag::bovi, inner_tag::logical_coords,
                              loop_backend::kokkos>();
  RunScratchZeroPatternMatrix<loop_tag::bovi, inner_tag::memory, loop_backend::kokkos>();
  RunScratchZeroPatternMatrix<loop_tag::boiv, inner_tag::logical_flat,
                              loop_backend::kokkos>();
  RunScratchZeroPatternMatrix<loop_tag::boiv, inner_tag::logical_coords,
                              loop_backend::kokkos>();
}

TEST_CASE("loop abstraction scratch roundtrip kokkos",
          "[loop_abstraction][contract][scratch]") {
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat,
                          loop_backend::kokkos>();
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords,
                          loop_backend::kokkos>();
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::memory, loop_backend::kokkos>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::logical_flat,
                          loop_backend::kokkos>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::logical_coords,
                          loop_backend::kokkos>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::memory, loop_backend::kokkos>();
  RunScratchPatternMatrix<loop_tag::boiv, inner_tag::logical_flat,
                          loop_backend::kokkos>();
  RunScratchPatternMatrix<loop_tag::boiv, inner_tag::logical_coords,
                          loop_backend::kokkos>();
}

TEST_CASE("loop abstraction scratch halo roundtrip",
          "[loop_abstraction][contract][scratch][halo]") {
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::logical_flat>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::logical_coords>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::memory>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::logical_flat>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::logical_coords>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::memory>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_flat>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction boiv scratch halo GetDelta access",
          "[loop_abstraction][contract][scratch][halo]") {
  constexpr ProblemSpec spec{2, 3, 3, 3, 2};

  RunBoivScratchDeltaCase<minus_i_halo_t, inner_tag::logical_flat, parthenon::X1DIR, -1>(
      spec);
  RunBoivScratchDeltaCase<minus_i_halo_t, inner_tag::logical_coords, parthenon::X1DIR,
                          -1>(spec);
  RunBoivScratchDeltaCase<plus_i_halo_t, inner_tag::logical_flat, parthenon::X1DIR, 1>(
      spec);
  RunBoivScratchDeltaCase<plus_i_halo_t, inner_tag::logical_coords, parthenon::X1DIR, 1>(
      spec);
  RunBoivScratchDeltaCase<minus_j_halo_t, inner_tag::logical_flat, parthenon::X2DIR, -1>(
      spec);
  RunBoivScratchDeltaCase<minus_j_halo_t, inner_tag::logical_coords, parthenon::X2DIR,
                          -1>(spec);
  RunBoivScratchMixedDeltaCase<inner_tag::logical_flat>(spec);
  RunBoivScratchMixedDeltaCase<inner_tag::logical_coords>(spec);
}

TEST_CASE("loop abstraction boiv scratch halo kokkos roundtrip",
          "[loop_abstraction][contract][scratch][halo]") {
  constexpr ProblemSpec spec{2, 3, 3, 3, 2};
  RunScratchHaloCase<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_flat,
                     loop_backend::kokkos>(spec, spec.nx * spec.ny);
  RunScratchHaloCase<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_coords,
                     loop_backend::kokkos>(spec, spec.nx * spec.ny);
}

TEST_CASE("loop abstraction halo producer single touch",
          "[loop_abstraction][contract][halo]") {
  // Cover partial chunks (ninner not a multiple of a plane) since that is where a
  // chunk's halo can overlap the next chunk's, and k-directed halos since the
  // k-sweep is where reconstruction reuse (the flux z-sweep) needs single-touch.
  const ProblemSpec spec{2, 3, 2, 2, 1};
  const int plane = spec.nx * spec.ny;           // 6
  const int cells = spec.nx * spec.ny * spec.nz; // 12
  const std::vector<int> ninner_cases{1, plane - 1, plane, plane + 1, cells - 1, cells};

  RunHaloProducerSingleTouchPatternMatrix<plus_j_halo_t, loop_tag::bvoi,
                                          inner_tag::logical_flat>(spec, ninner_cases);
  RunHaloProducerSingleTouchPatternMatrix<plus_j_halo_t, loop_tag::bvoi,
                                          inner_tag::logical_coords>(spec, ninner_cases);
  RunHaloProducerSingleTouchPatternMatrix<plus_j_halo_t, loop_tag::bvoi,
                                          inner_tag::memory>(spec, ninner_cases);

  RunHaloProducerSingleTouchPatternMatrix<k_triplet_halo_t, loop_tag::bvoi,
                                          inner_tag::logical_flat>(spec, ninner_cases);
  RunHaloProducerSingleTouchPatternMatrix<k_triplet_halo_t, loop_tag::bvoi,
                                          inner_tag::logical_coords>(spec, ninner_cases);
  RunHaloProducerSingleTouchPatternMatrix<k_triplet_halo_t, loop_tag::bvoi,
                                          inner_tag::memory>(spec, ninner_cases);
}

// Genuine reduced-dimension regression, built through the md-based IndexSpace
// constructor (the only faithful way to get a real degenerate direction: a 2D mesh
// gives the k direction a memory extent of 1 and no ghosts, and sets mesh->ndim = 2).
// A k-directed halo (k_triplet) must collapse to just its identity offset so the
// producer touches every logical cell exactly once and never reads a nonexistent
// k-plane. Uses the bvoi/logical single-touch contract (see RunHaloProducerSingleTouch
// rationale). Also the only coverage of the md-based constructor itself.
template <class HaloType, inner_tag INNER_TAG>
void RunHalo2DMeshSingleTouchCase(int nblocks, int nside, int nghost) {
  ScopedNghost guard;
  parthenon::Globals::nghost = nghost;

  const std::vector<int> scalar_shape{nside + 2 * nghost, nside + 2 * nghost, 1};
  Metadata m({Metadata::Independent}, scalar_shape);
  auto pkg = std::make_shared<StateDescriptor>("Halo2D package");
  pkg->AddField<v1>(m);

  BlockList_t block_list = MakeBlockList(pkg, nblocks, nside, /*NDIM=*/2);
  MeshData<Real> mesh_data("base");
  mesh_data.Initialize(block_list, nullptr);

  using IndexSpaceType = IndexSpace<loop_tag::bvoi, INNER_TAG>;
  IndexSpaceType idx_space(
      loop_abstraction::NInner(loop_abstraction::chunk_shape::ij_slab),
      IndexDomain::interior, /*halo=*/0, nblocks, &mesh_data, TopologicalElement::CC);
  REQUIRE(idx_space.GetNdim() == 2);

  auto touches = MakeOutput(idx_space);
  ZeroView(touches);

  loop_abstraction::outer(
      idx_space, KOKKOS_LAMBDA(const InnerIndexRange<IndexSpaceType> &idx_range, int b) {
        const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
        loop_abstraction::inner(halo_range, [&](auto idx) {
          const auto [k, j, i] = halo_range.GetKJI(idx);
          touches(b, 0, k, j, i) += 1.0;
        });
      });

  Kokkos::fence();
  const auto host = MirrorToHost(touches);

  const auto &memory = idx_space.GetMemoryIndexer();
  for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
    for (int k = memory.template StartIdx<0>(); k <= memory.template EndIdx<0>(); ++k) {
      for (int j = memory.template StartIdx<1>(); j <= memory.template EndIdx<1>(); ++j) {
        for (int i = memory.template StartIdx<2>(); i <= memory.template EndIdx<2>();
             ++i) {
          INFO("b=" << b << ", k=" << k << ", j=" << j << ", i=" << i);
          const bool in_set = InHaloLogicalSet<HaloType>(idx_space, k, j, i);
          if (in_set) {
            REQUIRE(host(b, 0, k, j, i) == Approx(1.0));
          } else if constexpr (!UsesMemorySpan<INNER_TAG>()) {
            REQUIRE(host(b, 0, k, j, i) == Approx(0.0));
          }
        }
      }
    }
  }
}

TEST_CASE("loop abstraction halo 2D mesh k-halo single touch",
          "[loop_abstraction][contract][halo]") {
  RunHalo2DMeshSingleTouchCase<k_triplet_halo_t, inner_tag::logical_flat>(2, 3, 2);
  RunHalo2DMeshSingleTouchCase<k_triplet_halo_t, inner_tag::logical_coords>(2, 3, 2);
  RunHalo2DMeshSingleTouchCase<k_triplet_halo_t, inner_tag::memory>(2, 3, 2);
  // An in-plane halo is unaffected by the k degeneration: it is kept whole.
  RunHalo2DMeshSingleTouchCase<plus_j_halo_t, inner_tag::logical_flat>(2, 3, 2);
}

TEST_CASE("loop abstraction pack view contracts",
          "[loop_abstraction][contract][pack_view]") {
  RunPackViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunPackViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunPackViewPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunPackViewPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunPackViewPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunPackViewPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunPackViewPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunPackViewPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction face pack view access",
          "[loop_abstraction][contract][pack_view]") {
  const PackViewSpec spec{2, 3, 2};
  const int nn_cells = (spec.ncell + 1) * (spec.ncell + 1) * (spec.ncell + 1);
  for (const int ninner : PackViewNinnerCases(nn_cells)) {
    RunFacePackViewCase<loop_tag::bvoi, inner_tag::logical_flat>(spec, ninner);
    RunFacePackViewCase<loop_tag::bvoi, inner_tag::logical_coords>(spec, ninner);
    RunFacePackViewCase<loop_tag::bvoi, inner_tag::memory>(spec, ninner);
    RunFacePackViewCase<loop_tag::bovi, inner_tag::logical_flat>(spec, ninner);
    RunFacePackViewCase<loop_tag::bovi, inner_tag::logical_coords>(spec, ninner);
    RunFacePackViewCase<loop_tag::bovi, inner_tag::memory>(spec, ninner);
    RunFacePackViewCase<loop_tag::boiv, inner_tag::logical_flat>(spec, ninner);
    RunFacePackViewCase<loop_tag::boiv, inner_tag::logical_coords>(spec, ninner);
  }
}

TEST_CASE("loop abstraction flux view contracts",
          "[loop_abstraction][contract][pack_view]") {
  RunFluxViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunFluxViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunFluxViewPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunFluxViewPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunFluxViewPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunFluxViewPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunFluxViewPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunFluxViewPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction var view contracts",
          "[loop_abstraction][contract][pack_view]") {
  RunVarViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunVarViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunVarViewPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunVarViewPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunVarViewPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunVarViewPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunVarViewPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunVarViewPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction var flux view contracts",
          "[loop_abstraction][contract][pack_view]") {
  RunVarFluxViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunVarFluxViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunVarFluxViewPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunVarFluxViewPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunVarFluxViewPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunVarFluxViewPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunVarFluxViewPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunVarFluxViewPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

// Sum/Min/Max over every valid (loop_tag, inner_tag) pair. The memory inner tag
// degenerates to logical_flat for reductions, so it must match the logical reference
// (ghost cells excluded) -- covered here by using the same ReferenceSum/Min/Max.
TEST_CASE("loop abstraction reductions", "[loop_abstraction][reduction]") {
  RunReducePatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunReducePatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunReducePatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunReducePatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunReducePatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunReducePatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunReducePatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunReducePatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

// Interleave a plain inner (fill scratch) with an inner_reduce over that scratch in one
// region. Scratch is supported for every tag (stack-allocated for boiv, Kokkos team
// scratch otherwise), so this runs the full pattern matrix.
TEST_CASE("loop abstraction reduction scratch interleave",
          "[loop_abstraction][reduction][scratch]") {
  RunReduceScratchInterleavePatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunReduceScratchInterleavePatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunReduceScratchInterleavePatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunReduceScratchInterleavePatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunReduceScratchInterleavePatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunReduceScratchInterleavePatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunReduceScratchInterleavePatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunReduceScratchInterleavePatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

// Two inner_reduce calls joining into the same handle sum to twice a single pass.
TEST_CASE("loop abstraction reduction two inner regions",
          "[loop_abstraction][reduction]") {
  RunReduceTwoInnerPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunReduceTwoInnerPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunReduceTwoInnerPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunReduceTwoInnerPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunReduceTwoInnerPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunReduceTwoInnerPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunReduceTwoInnerPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunReduceTwoInnerPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}
