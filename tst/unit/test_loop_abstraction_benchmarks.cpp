//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <algorithm>
#include <array>
#include <optional>
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
using parthenon::BlockList_t;
using parthenon::DevExecSpace;
using parthenon::IndexDomain;
using loop_abstraction::Index3;
using loop_abstraction::IndexSpace;
using loop_abstraction::default_loop_backend_v;
using loop_abstraction::inner_tag;
using loop_abstraction::loop_backend;
using loop_abstraction::loop_tag;
using parthenon::MeshBlock;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::StateDescriptor;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::par_for;
using parthenon::par_reduce;

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
  } else if constexpr (LOOP_TAG == loop_tag::bvoi &&
                       INNER_TAG == inner_tag::logical_coords) {
    return "bvoi/logical_coords";
  } else if constexpr (LOOP_TAG == loop_tag::bvoi && INNER_TAG == inner_tag::memory) {
    return "bvoi/memory";
  } else if constexpr (LOOP_TAG == loop_tag::bovi &&
                       INNER_TAG == inner_tag::logical_flat) {
    return "bovi/logical_flat";
  } else if constexpr (LOOP_TAG == loop_tag::bovi &&
                       INNER_TAG == inner_tag::logical_coords) {
    return "bovi/logical_coords";
  } else if constexpr (LOOP_TAG == loop_tag::bovi && INNER_TAG == inner_tag::memory) {
    return "bovi/memory";
  } else if constexpr (LOOP_TAG == loop_tag::boiv &&
                       INNER_TAG == inner_tag::logical_flat) {
    return "boiv/logical_flat";
  } else if constexpr (LOOP_TAG == loop_tag::boiv &&
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
  MismatchCounter() : view("loop_abstraction_mismatch") { Kokkos::deep_copy(view, 0); }

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

KOKKOS_INLINE_FUNCTION Real ScratchExpectedValue(const int b, const int k,
                                                 const int j, const int i) {
  Real out = 0.0;
  for (int v = 0; v < kNVars; ++v) {
    out += EncodeValue(b, v, k, j, i);
  }
  return out;
}

KOKKOS_INLINE_FUNCTION Real ShapedScratchValue(const int b, const int c0,
                                               const int c1, const int k,
                                               const int j, const int i) {
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
KOKKOS_INLINE_FUNCTION bool IsMemoryCell(const IndexSpaceType &idx_space, const int k,
                                         const int j, const int i) {
  const auto &memory = idx_space.GetMemoryIndexer();
  return k >= memory.template StartIdx<0>() && k <= memory.template EndIdx<0>() &&
         j >= memory.template StartIdx<1>() && j <= memory.template EndIdx<1>() &&
         i >= memory.template StartIdx<2>() && i <= memory.template EndIdx<2>();
}

template <class IndexSpaceType>
auto MakeOutput(const IndexSpaceType &idx_space) {
  const auto &memory = idx_space.GetMemoryIndexer();
  const int nk = memory.template EndIdx<0>() - memory.template StartIdx<0>() + 1;
  const int nj = memory.template EndIdx<1>() - memory.template StartIdx<1>() + 1;
  const int ni = memory.template EndIdx<2>() - memory.template StartIdx<2>() + 1;
  return parthenon::ParArray5D<Real>("loop_abstraction_unit_out", idx_space.GetNBlocks(),
                                     kNVars, nk, nj, ni);
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

struct plus_two_i_minus_k_halo_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return n == 0 ? -1 : 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return n == 0 ? 2 : 0; }
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

static_assert(loop_abstraction::impl::HaloSatisfiesContract<loop_abstraction::halo::none_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<plus_j_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<minus_i_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<minus_j_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<plus_two_i_minus_k_halo_t>());
static_assert(loop_abstraction::impl::HaloSatisfiesContract<k_triplet_halo_t>());
static_assert(!loop_abstraction::impl::HaloSatisfiesContract<unsorted_halo_t>());
static_assert(!loop_abstraction::impl::HaloSatisfiesContract<duplicate_identity_halo_t>());
static_assert(!loop_abstraction::impl::HaloSatisfiesContract<missing_identity_halo_t>());

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
parthenon::HostArray5D<Real> RunAutoIndexBody(const ProblemSpec &spec,
                                              const int ninner) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto out = MakeOutput(idx_space);
  ZeroView(out);

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
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

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
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

  const auto default_out =
      kji_body ? RunKjiBody<LOOP_TAG, INNER_TAG>(spec, ninner)
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

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunHaloContractCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", halo="
                 << typeid(HaloType).name());

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto out = MakeOutput(idx_space);
  ZeroView(out);

  // In-kernel checks accumulate into device counters (REQUIRE cannot run on device);
  // the host asserts they are zero after the fence. span_wrong flags a malformed halo
  // span structure; neighbor_wrong flags a produced halo value that does not match.
  MismatchCounter span_wrong;
  MismatchCounter neighbor_wrong;

  // Validate the halo span structure for the k-directed case
  // when the current base chunk is less than ni * nj.
  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);

    if constexpr (std::is_same_v<HaloType, k_triplet_halo_t> &&
                  LOOP_TAG == loop_tag::bovi) {
      const auto &logical = idx_space.GetLogicalIndexer();
      const int ni = logical.template EndIdx<2>() - logical.template StartIdx<2>() + 1;
      const int nj = logical.template EndIdx<1>() - logical.template StartIdx<1>() + 1;
      int base_ninner = 0;
      for (int r = 0; r < idx_range.nregions; ++r) {
        base_ninner += idx_range.flat_end[r] - idx_range.flat_start[r] + 1;
      }
      if (base_ninner < ni * nj) {
        span_wrong.note(halo_range.nregions != HaloType::npoints);
        int total_flat = 0;
        for (int r = 0; r < halo_range.nregions; ++r) {
          span_wrong.note(halo_range.flat_start[r] > halo_range.flat_end[r]);
          if (r > 0) {
            span_wrong.note(halo_range.flat_start[r] <= halo_range.flat_end[r - 1]);
          }
          total_flat += halo_range.flat_end[r] - halo_range.flat_start[r] + 1;
        }
        span_wrong.note(total_flat != HaloType::npoints * base_ninner);
      }
    }

    for (int v = 0; v < kNVars; ++v) {
      loop_abstraction::inner(halo_range, [&](auto idx) {
        const auto [k, j, i] = halo_range.GetKJI(idx);
        out(b, v, k, j, i) = EncodeValue(b, v, k, j, i);
      });
    }

    for (int v = 0; v < kNVars; ++v) {
      loop_abstraction::inner(idx_range, [&](auto idx) {
        const auto [k, j, i] = idx_range.GetKJI(idx);
        // For memory-tag inner loops the visited span may include ghost cells. The
        // producer only covers the halo-extended logical set S_halo, so a halo
        // neighbor is only guaranteed written when its source base cell (k,j,i) is a
        // logical cell -- the neighbor of a ghost base cell need not be in S_halo.
        if (UsesMemorySpan<INNER_TAG>() && !IsLogicalCell(idx_space, k, j, i)) {
          return;
        }
        for (int n = 0; n < HaloType::npoints; ++n) {
          const int kk = k + HaloType::dk(n);
          const int jj = j + HaloType::dj(n);
          const int ii = i + HaloType::di(n);
          neighbor_wrong.note(
              NotApprox(out(b, v, kk, jj, ii), EncodeValue(b, v, kk, jj, ii)));
        }
      });
    }

    for (int v = 0; v < kNVars; ++v) {
      loop_abstraction::inner(halo_range, [&](auto idx) {
        const auto [k, j, i] = halo_range.GetKJI(idx);
        out(b, v, k, j, i) = 0.0;
      });
    }
  });

  Kokkos::fence();
  REQUIRE(span_wrong.total() == 0);
  REQUIRE(neighbor_wrong.total() == 0);

  const auto host = MirrorToHost(out);
  for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
    for (int v = 0; v < kNVars; ++v) {
      const auto &memory = idx_space.GetMemoryIndexer();
      for (int k = memory.template StartIdx<0>(); k <= memory.template EndIdx<0>();
           ++k) {
        for (int j = memory.template StartIdx<1>(); j <= memory.template EndIdx<1>();
             ++j) {
          for (int i = memory.template StartIdx<2>(); i <= memory.template EndIdx<2>();
               ++i) {
            REQUIRE(host(b, v, k, j, i) == Approx(0.0));
          }
        }
      }
    }
  }
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG, bool USE_KOKKOS>
parthenon::HostArray5D<Real> RunHaloTouchBackend(const ProblemSpec &spec,
                                                 const int ninner) {
  using IndexSpaceType =
      PatternIndexSpace<LOOP_TAG, INNER_TAG,
                        USE_KOKKOS ? loop_backend::kokkos : loop_backend::raw>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto touches = MakeOutput(idx_space);
  ZeroView(touches);

  auto run_outer = [&](auto &&body) {
    if constexpr (USE_KOKKOS) {
      loop_abstraction::impl::outer_kokkos(idx_space, std::forward<decltype(body)>(body));
    } else {
      loop_abstraction::outer(idx_space, std::forward<decltype(body)>(body));
    }
  };

  run_outer(KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);

    for (int v = 0; v < kNVars; ++v) {
      loop_abstraction::inner(halo_range, [&](auto idx) {
        const auto [k, j, i] = halo_range.GetKJI(idx);
        touches(b, v, k, j, i) += 1.0;
      });

      loop_abstraction::inner(idx_range, [&](auto idx) {
        const auto [k, j, i] = idx_range.GetKJI(idx);
        for (int n = 0; n < HaloType::npoints; ++n) {
          const int kk = k + HaloType::dk(n);
          const int jj = j + HaloType::dj(n);
          const int ii = i + HaloType::di(n);
          // Memory-tag inner loops may visit ghost cells whose halo neighbor lies
          // outside the allocated memory range; skip those, matching the guarded
          // neighbor accesses elsewhere in this file.
          if (IsMemoryCell(idx_space, kk, jj, ii)) {
            touches(b, v, kk, jj, ii) += 1.0;
          }
        }
      });
    }
  });

  Kokkos::fence();
  return MirrorToHost(touches);
}

// Is (k,j,i) in the halo-extended logical set S_halo = S ∪ shift(S, h) for all
// halo offsets h? This is the set the producer inner(AddHalo<H>) must cover, each
// cell exactly once.
template <class HaloType, class IndexSpaceType>
bool InHaloLogicalSet(const IndexSpaceType &idx_space, const int k, const int j,
                      const int i) {
  for (int n = 0; n < HaloType::npoints; ++n) {
    // (k,j,i) is a shifted image of a logical cell p under offset h_n iff
    // p = (k,j,i) - h_n is itself a logical cell.
    if (IsLogicalCell(idx_space, k - HaloType::dk(n), j - HaloType::dj(n),
                      i - HaloType::di(n))) {
      return true;
    }
  }
  return false;
}

// Contract: the producer pattern `inner(AddHalo<H>(idx_range))` must touch every
// cell of the halo-extended logical set S_halo exactly once, and nothing else.
// This is the invariant that accumulating (+=) bodies depend on; the earlier
// halo tests only checked raw-vs-kokkos parity or used assignment, so a uniform
// double-touch would slip through. Runs a single backend so a self-consistent
// over-count in both backends is still caught.
template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG, bool USE_KOKKOS>
void RunHaloProducerSingleTouchCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", producer-single-touch="
                 << typeid(HaloType).name()
                 << ", backend=" << (USE_KOKKOS ? "kokkos" : "raw"));

  using IndexSpaceType =
      PatternIndexSpace<LOOP_TAG, INNER_TAG,
                        USE_KOKKOS ? loop_backend::kokkos : loop_backend::raw>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto touches = MakeOutput(idx_space);
  ZeroView(touches);

  auto run_outer = [&](auto &&body) {
    if constexpr (USE_KOKKOS) {
      loop_abstraction::impl::outer_kokkos(idx_space, std::forward<decltype(body)>(body));
    } else {
      loop_abstraction::outer(idx_space, std::forward<decltype(body)>(body));
    }
  };

  run_outer(KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
    for (int v = 0; v < kNVars; ++v) {
      loop_abstraction::inner(halo_range, [&](auto idx) {
        const auto [k, j, i] = halo_range.GetKJI(idx);
        touches(b, v, k, j, i) += 1.0;
      });
    }
  });

  Kokkos::fence();
  const auto host = MirrorToHost(touches);

  const auto &memory = idx_space.GetMemoryIndexer();
  for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
    for (int v = 0; v < kNVars; ++v) {
      for (int k = memory.template StartIdx<0>(); k <= memory.template EndIdx<0>();
           ++k) {
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
    RunHaloProducerSingleTouchCase<HaloType, LOOP_TAG, INNER_TAG, false>(spec, ninner);
    if constexpr (default_loop_backend_v == loop_backend::raw) {
      RunHaloProducerSingleTouchCase<HaloType, LOOP_TAG, INNER_TAG, true>(spec, ninner);
    }
  }
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunHaloParityCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", halo-parity="
                 << typeid(HaloType).name());
  if constexpr (default_loop_backend_v == loop_backend::raw) {
    const auto raw =
        RunHaloTouchBackend<HaloType, LOOP_TAG, INNER_TAG, false>(spec, ninner);
    const auto kokkos =
        RunHaloTouchBackend<HaloType, LOOP_TAG, INNER_TAG, true>(spec, ninner);
    CheckParity(raw, kokkos,
                PatternIndexSpace<LOOP_TAG, INNER_TAG>(spec.nblocks, spec.nx, spec.ny,
                                                       spec.nz, spec.nghost, ninner));
  }
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunHaloParityPatternMatrix(const ProblemSpec &spec,
                                const std::vector<int> &ninner_cases) {
  for (const int ninner : ninner_cases) {
    RunHaloParityCase<HaloType, LOOP_TAG, INNER_TAG>(spec, ninner);
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunHaloPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    const auto cases = NinnerCases(spec.nx * spec.ny * spec.nz);
    for (const int ninner : cases) {
      RunHaloContractCase<plus_j_halo_t, LOOP_TAG, INNER_TAG>(spec, ninner);
    }
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunKTripletHaloPatternMatrix() {
  constexpr ProblemSpec spec{2, 3, 2, 2, 1};
  for (const int ninner : {1, 5}) {
    RunHaloContractCase<k_triplet_halo_t, LOOP_TAG, INNER_TAG>(spec, ninner);
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
      parthenon::par_for(parthenon::loop_pattern_mdrange_tag,
                         "initialize pack view data", parthenon::DevExecSpace(), kb.s,
                         kb.e, jb.s, jb.e, ib.s, ib.e,
              KOKKOS_LAMBDA(int k, int j, int i) {
                for (int c = 0; c < num_components; ++c) {
                  var4(c, k, j, i) = PackViewSourceValue(b, v, k, j, i);
                }
              });
    }
  }

  Kokkos::fence();

  using IndexSpaceType = IndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell,
                           spec.nghost, ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(pkg.get());
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
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
  parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "check pack view",
                        parthenon::DevExecSpace(), 0, sparse_pack.GetNBlocks() - 1,
                        kb_int.s, kb_int.e, jb_int.s, jb_int.e, ib_int.s, ib_int.e,
             KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
               if (sparse_pack(b, v1(), k, j, i) !=
                   PackViewExpectedValue(b, 0, k, j, i)) {
                 ++ltot;
               }
               if (sparse_pack(b, v2(), k, j, i) !=
                   PackViewExpectedValue(b, 1, k, j, i)) {
                 ++ltot;
               }
               if (sparse_pack(b, v5(), k, j, i) !=
                   PackViewExpectedValue(b, 2, k, j, i)) {
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
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell,
                           spec.nghost, ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(
      pkg.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    auto flux_view = loop_abstraction::make_flux_pack_view(idx_range, sparse_pack, dir);
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
  parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "check flux view",
                        parthenon::DevExecSpace(), 0, sparse_pack.GetNBlocks() - 1,
                        kb_int.s, kb_int.e, jb_int.s, jb_int.e, ib_int.s, ib_int.e,
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
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell,
                           spec.nghost, ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(pkg.get());
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    // typed index
    auto var1 = loop_abstraction::make_var_view(idx_range, sparse_pack, v1());
    // raw int index (resolved through GetIndex's integral overload)
    auto var2 =
        loop_abstraction::make_var_view(idx_range, sparse_pack, sparse_pack.GetIndex(b, v2()));
    auto var5 =
        loop_abstraction::make_var_view(idx_range, sparse_pack, sparse_pack.GetIndex(b, v5()));
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
  parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "check var view",
                        parthenon::DevExecSpace(), 0, sparse_pack.GetNBlocks() - 1,
                        kb_int.s, kb_int.e, jb_int.s, jb_int.e, ib_int.s, ib_int.e,
             KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
               if (sparse_pack(b, v1(), k, j, i) !=
                   PackViewExpectedValue(b, 0, k, j, i)) {
                 ++ltot;
               }
               if (sparse_pack(b, v2(), k, j, i) !=
                   PackViewExpectedValue(b, 1, k, j, i)) {
                 ++ltot;
               }
               if (sparse_pack(b, v5(), k, j, i) !=
                   PackViewExpectedValue(b, 2, k, j, i)) {
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
  IndexSpaceType idx_space(spec.nblocks, spec.ncell, spec.ncell, spec.ncell,
                           spec.nghost, ninner);
  auto desc = parthenon::MakePackDescriptor<v1, v2, v5>(
      pkg.get(), {}, {parthenon::PDOpt::WithFluxes});
  auto sparse_pack = desc.GetPack(&mesh_data);

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
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
  parthenon::par_reduce(parthenon::loop_pattern_mdrange_tag, "check var flux view",
                        parthenon::DevExecSpace(), 0, sparse_pack.GetNBlocks() - 1,
                        kb_int.s, kb_int.e, jb_int.s, jb_int.e, ib_int.s, ib_int.e,
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

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner);

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real>();
  idx_space.template AddPerPointScratch<Real>();
  idx_space.template AddPerPointScratch<Real>();

  MismatchCounter wrong;

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    auto scratch_a = loop_abstraction::GetPerPointScratch<Real>(idx_range);
    auto scratch_b = loop_abstraction::GetPerPointScratch<Real>(idx_range);
    auto scratch_c = loop_abstraction::GetPerPointScratch<Real>(idx_range);

    loop_abstraction::inner(idx_range, [&](auto idx) {
      scratch_a(idx) = 0.0;
      scratch_b(idx) = 0.0;
      scratch_c(idx) = 0.0;
    });

    for (int v = 0; v < kNVars; ++v) {
      loop_abstraction::inner(idx_range, [&](auto idx) {
        const auto [k, j, i] = idx_range.GetKJI(idx);
        scratch_a(idx) += EncodeValue(b, v, k, j, i);
        scratch_b(idx) += EncodeValue(b, v, k, j, i);
        scratch_c(idx) += EncodeValue(b, v, k, j, i);
      });
    }

    loop_abstraction::inner(idx_range, [&](auto idx) {
      const auto [k, j, i] = idx_range.GetKJI(idx);
      const Real expected = ScratchExpectedValue(b, k, j, i);
      wrong.note(NotApprox(scratch_a(idx), expected));
      wrong.note(NotApprox(scratch_b(idx), expected));
      wrong.note(NotApprox(scratch_c(idx), expected));
    });
  });

  REQUIRE(wrong.total() == 0);
}

// Exercise scratch.Zero(): fill with garbage, Zero(), accumulate, verify; then
// reuse the same buffer (Zero() again, re-accumulate) to confirm a buffer can be
// zeroed and reused rather than needing a fresh zero-initialized allocation.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchZeroCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", Zero()");

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real>();

  MismatchCounter wrong;

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
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
      }

      loop_abstraction::inner(idx_range, [&](auto idx) {
        const auto [k, j, i] = idx_range.GetKJI(idx);
        wrong.note(NotApprox(scratch(idx), ScratchExpectedValue(b, k, j, i)));
      });
    }
  });

  REQUIRE(wrong.total() == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchZeroPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchZeroCase<LOOP_TAG, INNER_TAG>(spec, ninner);
    }
  }
}

// Kokkos-backend counterpart, exercising the team-parallel TeamScratch1D::Zero().
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchZeroCaseKokkos(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", Zero(), backend=kokkos");

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real>();

  MismatchCounter wrong;

  loop_abstraction::impl::outer_kokkos(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    auto scratch = loop_abstraction::GetPerPointScratch<Real>(idx_range);

    for (int pass = 0; pass < 2; ++pass) {
      loop_abstraction::impl::inner_kokkos(idx_range,
                                           [&](auto idx) { scratch(idx) = -7.0; });
      idx_range.TeamBarrier();
      scratch.Zero();
      idx_range.TeamBarrier();

      for (int v = 0; v < kNVars; ++v) {
        loop_abstraction::impl::inner_kokkos(idx_range, [&](auto idx) {
          const auto [k, j, i] = idx_range.GetKJI(idx);
          scratch(idx) += EncodeValue(b, v, k, j, i);
        });
      }

      loop_abstraction::impl::inner_kokkos(idx_range, [&](auto idx) {
        const auto [k, j, i] = idx_range.GetKJI(idx);
        wrong.note(NotApprox(scratch(idx), ScratchExpectedValue(b, k, j, i)));
      });
    }
  });

  REQUIRE(wrong.total() == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchZeroPatternMatrixKokkos() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchZeroCaseKokkos<LOOP_TAG, INNER_TAG>(spec, ninner);
    }
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchCaseKokkos(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", backend=kokkos");

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real>();
  idx_space.template AddPerPointScratch<Real>();
  idx_space.template AddPerPointScratch<Real>();

  MismatchCounter wrong;

  loop_abstraction::impl::outer_kokkos(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    auto scratch_a = loop_abstraction::GetPerPointScratch<Real>(idx_range);
    auto scratch_b = loop_abstraction::GetPerPointScratch<Real>(idx_range);
    auto scratch_c = loop_abstraction::GetPerPointScratch<Real>(idx_range);

    loop_abstraction::impl::inner_kokkos(idx_range, [&](auto idx) {
      scratch_a(idx) = 0.0;
      scratch_b(idx) = 0.0;
      scratch_c(idx) = 0.0;
    });

    for (int v = 0; v < kNVars; ++v) {
      loop_abstraction::impl::inner_kokkos(idx_range, [&](auto idx) {
        const auto [k, j, i] = idx_range.GetKJI(idx);
        scratch_a(idx) += EncodeValue(b, v, k, j, i);
        scratch_b(idx) += EncodeValue(b, v, k, j, i);
        scratch_c(idx) += EncodeValue(b, v, k, j, i);
      });
    }

    loop_abstraction::impl::inner_kokkos(idx_range, [&](auto idx) {
      const auto [k, j, i] = idx_range.GetKJI(idx);
      const Real expected = ScratchExpectedValue(b, k, j, i);
      wrong.note(NotApprox(scratch_a(idx), expected));
      wrong.note(NotApprox(scratch_b(idx), expected));
      wrong.note(NotApprox(scratch_c(idx), expected));
    });
  });

  REQUIRE(wrong.total() == 0);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchCase<LOOP_TAG, INNER_TAG>(spec, ninner);
    }
  }
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchPatternMatrixKokkos() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchCaseKokkos<LOOP_TAG, INNER_TAG>(spec, ninner);
    }
  }
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchHaloCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", halo=" << typeid(HaloType).name());

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real, HaloType>();
  auto di = idx_space.GetDelta(parthenon::X1DIR);
  auto dj = idx_space.GetDelta(parthenon::X2DIR);
  auto dk = idx_space.GetDelta(parthenon::X3DIR);
  using offset_t = decltype(di);
  std::array<offset_t, HaloType::npoints> offsets{};
  for (int n = 0; n < HaloType::npoints; ++n) {
    offsets[n] = HaloType::di(n) * di + HaloType::dj(n) * dj + HaloType::dk(n) * dk;
  }

  MismatchCounter wrong;

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
    auto scratch = loop_abstraction::GetPerPointScratch<Real>(halo_range);

    loop_abstraction::inner(halo_range, [&](auto idx) {
      const auto [k, j, i] = halo_range.GetKJI(idx);
      scratch(idx) = EncodeValue(b, 0, k, j, i);
    });

    loop_abstraction::inner(idx_range, [&](auto idx) {
      const auto [k, j, i] = idx_range.GetKJI(idx);
      for (int n = 0; n < HaloType::npoints; ++n) {
        const int kk = k + HaloType::dk(n);
        const int jj = j + HaloType::dj(n);
        const int ii = i + HaloType::di(n);
        // Memory-tag ranges may include halo-of-ghost cells. Only verify
        // offsets that remain inside the allocated memory span.
        if constexpr (INNER_TAG == inner_tag::memory) {
          if (IsMemoryCell(idx_space, k, j, i)) {
            continue;
          }
        }
        const auto shifted = idx + offsets[n];
        wrong.note(NotApprox(scratch(shifted), EncodeValue(b, 0, kk, jj, ii)));
        wrong.note(NotApprox(scratch(Index3{kk, jj, ii}), EncodeValue(b, 0, kk, jj, ii)));
      }
    });
  });

  REQUIRE(wrong.total() == 0);
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchHaloCaseKokkos(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", halo=" << typeid(HaloType).name()
                  << ", backend=kokkos");

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  idx_space.template AddPerPointScratch<Real, HaloType>();
  auto di = idx_space.GetDelta(parthenon::X1DIR);
  auto dj = idx_space.GetDelta(parthenon::X2DIR);
  auto dk = idx_space.GetDelta(parthenon::X3DIR);
  using offset_t = decltype(di);
  std::array<offset_t, HaloType::npoints> offsets{};
  for (int n = 0; n < HaloType::npoints; ++n) {
    offsets[n] = HaloType::di(n) * di + HaloType::dj(n) * dj + HaloType::dk(n) * dk;
  }

  MismatchCounter wrong;

  loop_abstraction::impl::outer_kokkos(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
    auto scratch = loop_abstraction::GetPerPointScratch<Real>(halo_range);

    loop_abstraction::impl::inner_kokkos(halo_range, [&](auto idx) {
      const auto [k, j, i] = halo_range.GetKJI(idx);
      scratch(idx) = EncodeValue(b, 0, k, j, i);
    });

    loop_abstraction::impl::inner_kokkos(idx_range, [&](auto idx) {
      const auto [k, j, i] = idx_range.GetKJI(idx);
      for (int n = 0; n < HaloType::npoints; ++n) {
        const int kk = k + HaloType::dk(n);
        const int jj = j + HaloType::dj(n);
        const int ii = i + HaloType::di(n);
        if constexpr (INNER_TAG == inner_tag::memory) {
          if (IsMemoryCell(idx_space, k, j, i)) {
            continue;
          }
        }
        const auto shifted = idx + offsets[n];
        wrong.note(NotApprox(scratch(shifted), EncodeValue(b, 0, kk, jj, ii)));
        wrong.note(NotApprox(scratch(Index3{kk, jj, ii}), EncodeValue(b, 0, kk, jj, ii)));
      }
    });
  });

  REQUIRE(wrong.total() == 0);
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchHaloPatternMatrix() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchHaloCase<HaloType, LOOP_TAG, INNER_TAG>(spec, ninner);
    }
  }
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunScratchHaloPatternMatrixKokkos() {
  for (const auto &spec : CoverageSpecs()) {
    for (const int ninner : NinnerCases(spec.nx * spec.ny * spec.nz)) {
      RunScratchHaloCaseKokkos<HaloType, LOOP_TAG, INNER_TAG>(spec, ninner);
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

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
    auto scratch = loop_abstraction::GetPerPointScratch<Real>(halo_range);

    loop_abstraction::inner(halo_range, [&](auto idx) {
      const auto [k, j, i] = halo_range.GetKJI(idx);
      scratch(idx) = EncodeValue(b, 0, k, j, i);
    });

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

template <loop_tag LOOP_TAG, inner_tag INNER_TAG, loop_backend BACKEND>
void RunShapedScratchCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner
                  << ", backend="
                  << (BACKEND == loop_backend::raw ? "raw" : "kokkos"));

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost,
                           ninner);
  idx_space.template AddPerPointScratch<Real, 2, 3>();

  MismatchCounter wrong;

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    auto scratch = loop_abstraction::GetPerPointScratch<Real, 2, 3>(idx_range);

    loop_abstraction::inner(idx_range, [&](auto idx) {
      const auto [k, j, i] = idx_range.GetKJI(idx);
      for (int c0 = 0; c0 < 2; ++c0) {
        for (int c1 = 0; c1 < 3; ++c1) {
          scratch(c0, c1, idx) = ShapedScratchValue(b, c0, c1, k, j, i);
        }
      }
    });

    idx_range.TeamBarrier();

    loop_abstraction::inner(idx_range, [&](auto idx) {
      const auto [k, j, i] = idx_range.GetKJI(idx);
      for (int c0 = 0; c0 < 2; ++c0) {
        for (int c1 = 0; c1 < 3; ++c1) {
          const Real expected = ShapedScratchValue(b, c0, c1, k, j, i);
          wrong.note(NotApprox(scratch(c0, c1, idx), expected));
          wrong.note(NotApprox(scratch(c0, c1, k, j, i), expected));
        }
      }
    });
  });

  REQUIRE(wrong.total() == 0);
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND>
void RunShapedScratchHaloCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", spec=" << spec.nblocks << "x" << spec.nx
                  << "x" << spec.ny << "x" << spec.nz << " nghost=" << spec.nghost
                  << ", ninner=" << ninner << ", halo=" << typeid(HaloType).name()
                  << ", backend="
                  << (BACKEND == loop_backend::raw ? "raw" : "kokkos"));

  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG, BACKEND>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost,
                           ninner);
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

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range = loop_abstraction::AddHalo<HaloType>(idx_range);
    auto scratch = loop_abstraction::GetPerPointScratch<Real, 2, 3>(halo_range);

    loop_abstraction::inner(halo_range, [&](auto idx) {
      const auto [k, j, i] = halo_range.GetKJI(idx);
      for (int c0 = 0; c0 < 2; ++c0) {
        for (int c1 = 0; c1 < 3; ++c1) {
          scratch(c0, c1, idx) = ShapedScratchValue(b, c0, c1, k, j, i);
        }
      }
    });

    halo_range.TeamBarrier();

    loop_abstraction::inner(idx_range, [&](auto idx) {
      const auto [k, j, i] = idx_range.GetKJI(idx);
      for (int n = 0; n < HaloType::npoints; ++n) {
        const int kk = k + HaloType::dk(n);
        const int jj = j + HaloType::dj(n);
        const int ii = i + HaloType::di(n);
        if constexpr (INNER_TAG == inner_tag::memory) {
          if (!IsMemoryCell(idx_space, kk, jj, ii)) {
            continue;
          }
        }
        for (int c0 = 0; c0 < 2; ++c0) {
          for (int c1 = 0; c1 < 3; ++c1) {
            const Real expected = ShapedScratchValue(b, c0, c1, kk, jj, ii);
            wrong.note(NotApprox(scratch(c0, c1, Index3{kk, jj, ii}), expected));
            wrong.note(NotApprox(scratch(c0, c1, idx + offsets[n]), expected));
            wrong.note(NotApprox(scratch(c0, c1, kk, jj, ii), expected));
          }
        }
      }
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

  loop_abstraction::outer(idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
    const auto halo_range =
        loop_abstraction::AddHalo<plus_two_i_minus_k_halo_t>(idx_range);
    auto scratch = loop_abstraction::GetPerPointScratch<Real>(halo_range);

    loop_abstraction::inner(halo_range, [&](auto idx) {
      const auto [k, j, i] = halo_range.GetKJI(idx);
      scratch(idx) = EncodeValue(b, 0, k, j, i);
    });

    loop_abstraction::inner(idx_range, [&](auto idx) {
      const auto shifted = idx + 2 * dx1 - dx3;
      const auto [k, j, i] = idx_range.GetKJI(idx);
      wrong.note(NotApprox(scratch(shifted), EncodeValue(b, 0, k - 1, j, i + 2)));
    });
  });

  REQUIRE(wrong.total() == 0);
}

} // namespace

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

TEST_CASE("loop abstraction scratch roundtrip",
          "[loop_abstraction][contract][scratch]") {
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunScratchPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunScratchPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunScratchPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunScratchPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction scratch Zero",
          "[loop_abstraction][contract][scratch]") {
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
  RunScratchZeroPatternMatrixKokkos<loop_tag::bvoi, inner_tag::logical_flat>();
  RunScratchZeroPatternMatrixKokkos<loop_tag::bvoi, inner_tag::logical_coords>();
  RunScratchZeroPatternMatrixKokkos<loop_tag::bvoi, inner_tag::memory>();
  RunScratchZeroPatternMatrixKokkos<loop_tag::bovi, inner_tag::logical_flat>();
  RunScratchZeroPatternMatrixKokkos<loop_tag::bovi, inner_tag::logical_coords>();
  RunScratchZeroPatternMatrixKokkos<loop_tag::bovi, inner_tag::memory>();
  RunScratchZeroPatternMatrixKokkos<loop_tag::boiv, inner_tag::logical_flat>();
  RunScratchZeroPatternMatrixKokkos<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction scratch roundtrip kokkos",
          "[loop_abstraction][contract][scratch]") {
  RunScratchPatternMatrixKokkos<loop_tag::bvoi, inner_tag::logical_flat>();
  RunScratchPatternMatrixKokkos<loop_tag::bvoi, inner_tag::logical_coords>();
  RunScratchPatternMatrixKokkos<loop_tag::bvoi, inner_tag::memory>();
  RunScratchPatternMatrixKokkos<loop_tag::bovi, inner_tag::logical_flat>();
  RunScratchPatternMatrixKokkos<loop_tag::bovi, inner_tag::logical_coords>();
  RunScratchPatternMatrixKokkos<loop_tag::bovi, inner_tag::memory>();
  RunScratchPatternMatrixKokkos<loop_tag::boiv, inner_tag::logical_flat>();
  RunScratchPatternMatrixKokkos<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction scratch halo roundtrip",
          "[loop_abstraction][contract][scratch][halo]") {
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::logical_flat>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bvoi,
                              inner_tag::logical_coords>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::memory>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::logical_flat>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bovi,
                              inner_tag::logical_coords>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::memory>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::boiv,
                              inner_tag::logical_flat>();
  RunScratchHaloPatternMatrix<plus_j_halo_t, loop_tag::boiv,
                              inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction boiv scratch halo GetDelta access",
          "[loop_abstraction][contract][scratch][halo]") {
  constexpr ProblemSpec spec{2, 3, 3, 3, 2};

  RunBoivScratchDeltaCase<minus_i_halo_t, inner_tag::logical_flat,
                          parthenon::X1DIR, -1>(spec);
  RunBoivScratchDeltaCase<minus_i_halo_t, inner_tag::logical_coords,
                          parthenon::X1DIR, -1>(spec);
  RunBoivScratchDeltaCase<plus_i_halo_t, inner_tag::logical_flat,
                          parthenon::X1DIR, 1>(spec);
  RunBoivScratchDeltaCase<plus_i_halo_t, inner_tag::logical_coords,
                          parthenon::X1DIR, 1>(spec);
  RunBoivScratchDeltaCase<minus_j_halo_t, inner_tag::logical_flat,
                          parthenon::X2DIR, -1>(spec);
  RunBoivScratchDeltaCase<minus_j_halo_t, inner_tag::logical_coords,
                          parthenon::X2DIR, -1>(spec);
  RunBoivScratchMixedDeltaCase<inner_tag::logical_flat>(spec);
  RunBoivScratchMixedDeltaCase<inner_tag::logical_coords>(spec);
}

TEST_CASE("loop abstraction boiv scratch halo kokkos roundtrip",
          "[loop_abstraction][contract][scratch][halo]") {
  constexpr ProblemSpec spec{2, 3, 3, 3, 2};
  RunScratchHaloCaseKokkos<plus_j_halo_t, loop_tag::boiv,
                           inner_tag::logical_flat>(spec, spec.nx * spec.ny);
  RunScratchHaloCaseKokkos<plus_j_halo_t, loop_tag::boiv,
                           inner_tag::logical_coords>(spec, spec.nx * spec.ny);
}

TEST_CASE("loop abstraction shaped scratch roundtrip",
          "[loop_abstraction][contract][scratch][shaped]") {
  constexpr ProblemSpec spec{2, 3, 2, 2, 1};
  constexpr int ninner = spec.nx * spec.ny;

  RunShapedScratchCase<loop_tag::bvoi, inner_tag::logical_flat,
                       loop_backend::raw>(spec, ninner);
  RunShapedScratchCase<loop_tag::bvoi, inner_tag::logical_coords,
                       loop_backend::raw>(spec, ninner);
  RunShapedScratchCase<loop_tag::bvoi, inner_tag::memory, loop_backend::raw>(
      spec, ninner);
  RunShapedScratchCase<loop_tag::bovi, inner_tag::logical_flat,
                       loop_backend::raw>(spec, ninner);
  RunShapedScratchCase<loop_tag::bovi, inner_tag::logical_coords,
                       loop_backend::raw>(spec, ninner);
  RunShapedScratchCase<loop_tag::bovi, inner_tag::memory, loop_backend::raw>(
      spec, ninner);
  RunShapedScratchCase<loop_tag::boiv, inner_tag::logical_flat,
                       loop_backend::raw>(spec, ninner);
  RunShapedScratchCase<loop_tag::boiv, inner_tag::logical_coords,
                       loop_backend::raw>(spec, ninner);

  RunShapedScratchCase<loop_tag::bvoi, inner_tag::logical_flat,
                       loop_backend::kokkos>(spec, ninner);
  RunShapedScratchCase<loop_tag::bvoi, inner_tag::logical_coords,
                       loop_backend::kokkos>(spec, ninner);
  RunShapedScratchCase<loop_tag::bvoi, inner_tag::memory,
                       loop_backend::kokkos>(spec, ninner);
  RunShapedScratchCase<loop_tag::bovi, inner_tag::logical_flat,
                       loop_backend::kokkos>(spec, ninner);
  RunShapedScratchCase<loop_tag::bovi, inner_tag::logical_coords,
                       loop_backend::kokkos>(spec, ninner);
  RunShapedScratchCase<loop_tag::bovi, inner_tag::memory,
                       loop_backend::kokkos>(spec, ninner);
  RunShapedScratchCase<loop_tag::boiv, inner_tag::logical_flat,
                       loop_backend::kokkos>(spec, ninner);
  RunShapedScratchCase<loop_tag::boiv, inner_tag::logical_coords,
                       loop_backend::kokkos>(spec, ninner);
}

TEST_CASE("loop abstraction shaped scratch halo roundtrip",
          "[loop_abstraction][contract][scratch][halo][shaped]") {
  constexpr ProblemSpec spec{2, 3, 3, 3, 2};
  constexpr int ninner = spec.nx * spec.ny;

  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bvoi,
                           inner_tag::logical_flat, loop_backend::raw>(spec,
                                                                       ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bvoi,
                           inner_tag::logical_coords, loop_backend::raw>(spec,
                                                                         ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bovi,
                           inner_tag::logical_flat, loop_backend::raw>(spec,
                                                                       ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bovi,
                           inner_tag::logical_coords, loop_backend::raw>(spec,
                                                                         ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::boiv,
                           inner_tag::logical_flat, loop_backend::raw>(spec,
                                                                       ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::boiv,
                           inner_tag::logical_coords, loop_backend::raw>(spec,
                                                                         ninner);

  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bvoi,
                           inner_tag::logical_flat, loop_backend::kokkos>(spec,
                                                                          ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bvoi,
                           inner_tag::logical_coords, loop_backend::kokkos>(
      spec, ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bovi,
                           inner_tag::logical_flat, loop_backend::kokkos>(spec,
                                                                          ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::bovi,
                           inner_tag::logical_coords, loop_backend::kokkos>(
      spec, ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::boiv,
                           inner_tag::logical_flat, loop_backend::kokkos>(spec,
                                                                          ninner);
  RunShapedScratchHaloCase<plus_j_halo_t, loop_tag::boiv,
                           inner_tag::logical_coords, loop_backend::kokkos>(
      spec, ninner);
}

TEST_CASE("loop abstraction halo producer-consumer contracts",
          "[loop_abstraction][contract][halo]") {
  RunHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunHaloPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunHaloPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunHaloPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunHaloPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction k halo disjoint span contracts",
          "[loop_abstraction][contract][halo]") {
  RunKTripletHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunKTripletHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunKTripletHaloPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
  RunKTripletHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunKTripletHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunKTripletHaloPatternMatrix<loop_tag::bovi, inner_tag::memory>();
  RunKTripletHaloPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunKTripletHaloPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction halo kokkos parity",
          "[loop_abstraction][contract][halo]") {
  const ProblemSpec spec{2, 3, 2, 2, 1};
  const std::vector<int> plus_j_cases{1, 11, 12, 13};
  const std::vector<int> k_triplet_cases{1, 5};

  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::logical_flat>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::logical_coords>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bvoi, inner_tag::memory>(spec,
                                                                               plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::logical_flat>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::logical_coords>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::memory>(spec,
                                                                               plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_flat>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_coords>(
      spec, plus_j_cases);

  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bvoi, inner_tag::logical_flat>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bvoi, inner_tag::logical_coords>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bvoi, inner_tag::memory>(spec,
                                                                                   k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bovi, inner_tag::logical_flat>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bovi, inner_tag::logical_coords>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bovi, inner_tag::memory>(spec,
                                                                                   k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::boiv, inner_tag::logical_flat>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::boiv, inner_tag::logical_coords>(
      spec, k_triplet_cases);
}

TEST_CASE("loop abstraction halo producer single touch",
          "[loop_abstraction][contract][halo]") {
  // Cover partial chunks (ninner not a multiple of a plane) since that is where a
  // chunk's halo can overlap the next chunk's, and k-directed halos since the
  // k-sweep is where reconstruction reuse (the flux z-sweep) needs single-touch.
  const ProblemSpec spec{2, 3, 2, 2, 1};
  const int plane = spec.nx * spec.ny;              // 6
  const int cells = spec.nx * spec.ny * spec.nz;    // 12
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
