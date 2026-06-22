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

using Real = double;
using parthenon::BlockList_t;
using parthenon::DevExecSpace;
using parthenon::IndexDomain;
using loop_abstraction::Index3;
using loop_abstraction::IndexSpace;
using loop_abstraction::inner_tag;
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

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
using PatternIndexSpace = IndexSpace<LOOP_TAG, INNER_TAG>;

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

KOKKOS_INLINE_FUNCTION Real EncodeValue(const int b, const int v, const int k,
                                        const int j, const int i) {
  return 1.0e6 * static_cast<Real>(b) + 1.0e5 * static_cast<Real>(v) +
         1.0e3 * static_cast<Real>(k) + 10.0 * static_cast<Real>(j) +
         static_cast<Real>(i) + 1.0;
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

struct k_triplet_halo_t {
  static constexpr int npoints = 3;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) {
    return n == 0 ? -1 : (n == 1 ? 0 : 1);
  }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
parthenon::HostArray5D<Real> RunAutoIndexBody(const ProblemSpec &spec, const int ninner,
                                              const bool use_kokkos) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto out = MakeOutput(idx_space);
  ZeroView(out);

  if (use_kokkos) {
    loop_abstraction::impl::outer_kokkos(
        idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
          for (int v = 0; v < kNVars; ++v) {
            loop_abstraction::impl::inner_kokkos(
                idx_range, KOKKOS_LAMBDA(auto idx) {
                  if constexpr (std::is_same_v<std::decay_t<decltype(idx)>, int>) {
                    const auto [k, j, i] = idx_range.GetKJI(idx);
                    out(b, v, k, j, i) += EncodeValue(b, v, k, j, i);
                  } else {
                    out(b, v, idx.k, idx.j, idx.i) +=
                        EncodeValue(b, v, idx.k, idx.j, idx.i);
                  }
                });
          }
        });
  } else {
    loop_abstraction::outer(idx_space, [&](const auto &idx_range, int b) {
      for (int v = 0; v < kNVars; ++v) {
        loop_abstraction::inner(idx_range, [&](auto idx) {
          if constexpr (std::is_same_v<std::decay_t<decltype(idx)>, int>) {
            const auto [k, j, i] = idx_range.GetKJI(idx);
            out(b, v, k, j, i) += EncodeValue(b, v, k, j, i);
          } else {
            out(b, v, idx.k, idx.j, idx.i) += EncodeValue(b, v, idx.k, idx.j, idx.i);
          }
        });
      }
    });
  }

  Kokkos::fence();
  return MirrorToHost(out);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
parthenon::HostArray5D<Real> RunKjiBody(const ProblemSpec &spec, const int ninner,
                                        const bool use_kokkos) {
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
  IndexSpaceType idx_space(spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner);
  auto out = MakeOutput(idx_space);
  ZeroView(out);

  if (use_kokkos) {
    loop_abstraction::impl::outer_kokkos(
        idx_space, KOKKOS_LAMBDA(const auto &idx_range, int b) {
          for (int v = 0; v < kNVars; ++v) {
            loop_abstraction::impl::inner_kokkos(
                idx_range, KOKKOS_LAMBDA(const int k, const int j, const int i) {
                  out(b, v, k, j, i) += EncodeValue(b, v, k, j, i);
                });
          }
        });
  } else {
    loop_abstraction::outer(idx_space, [&](const auto &idx_range, int b) {
      for (int v = 0; v < kNVars; ++v) {
        loop_abstraction::inner(idx_range, [&](const int k, const int j, const int i) {
          out(b, v, k, j, i) += EncodeValue(b, v, k, j, i);
        });
      }
    });
  }

  Kokkos::fence();
  return MirrorToHost(out);
}

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunContractCase(const ProblemSpec &spec, const int ninner, const char *body_name,
                     const bool kji_body) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", body=" << body_name);

  const auto default_out =
      kji_body ? RunKjiBody<LOOP_TAG, INNER_TAG>(spec, ninner, false)
               : RunAutoIndexBody<LOOP_TAG, INNER_TAG>(spec, ninner, false);
  const auto kokkos_out = kji_body
                              ? RunKjiBody<LOOP_TAG, INNER_TAG>(spec, ninner, true)
                              : RunAutoIndexBody<LOOP_TAG, INNER_TAG>(spec, ninner, true);

  CheckParity(default_out, kokkos_out,
              PatternIndexSpace<LOOP_TAG, INNER_TAG>(spec.nblocks, spec.nx, spec.ny,
                                                     spec.nz, spec.nghost, ninner));

  CheckLogicalContract(PatternIndexSpace<LOOP_TAG, INNER_TAG>(
                           spec.nblocks, spec.nx, spec.ny, spec.nz, spec.nghost, ninner),
                       default_out);
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

  // Validate the halo span structure for the k-directed case
  // when the current base chunk is less than ni * nj.
  loop_abstraction::outer(idx_space, [&](const auto &idx_range, int b) {
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
        REQUIRE(halo_range.nregions == HaloType::npoints);
        int total_flat = 0;
        for (int r = 0; r < halo_range.nregions; ++r) {
          REQUIRE(halo_range.flat_start[r] <= halo_range.flat_end[r]);
          if (r > 0) {
            REQUIRE(halo_range.flat_start[r] > halo_range.flat_end[r - 1]);
          }
          total_flat += halo_range.flat_end[r] - halo_range.flat_start[r] + 1;
        }
        REQUIRE(total_flat == HaloType::npoints * base_ninner);
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
        INFO("b=" << b << ", v=" << v << ", k=" << k << ", j=" << j << ", i=" << i);
        for (int n = 0; n < HaloType::npoints; ++n) {
          const int kk = k + HaloType::dk(n);
          const int jj = j + HaloType::dj(n);
          const int ii = i + HaloType::di(n);
          REQUIRE(out(b, v, kk, jj, ii) == Approx(EncodeValue(b, v, kk, jj, ii)));
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
  using IndexSpaceType = PatternIndexSpace<LOOP_TAG, INNER_TAG>;
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

  run_outer([&](const auto &idx_range, int b) {
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
          touches(b, v, kk, jj, ii) += 1.0;
        }
      });
    }
  });

  Kokkos::fence();
  return MirrorToHost(touches);
}

template <class HaloType, loop_tag LOOP_TAG, inner_tag INNER_TAG>
void RunHaloParityCase(const ProblemSpec &spec, const int ninner) {
  const auto pattern_name = PatternName<LOOP_TAG, INNER_TAG>();
  INFO("pattern=" << pattern_name << ", ninner=" << ninner << ", halo-parity="
                 << typeid(HaloType).name());
  if constexpr (loop_abstraction::impl::use_raw_for_v) {
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
          pack_view(v1(), idx) = PackViewExpectedValue(b, 0, idx.k, idx.j, idx.i);
          pack_view(v2(), idx) = PackViewExpectedValue(b, 1, idx.k, idx.j, idx.i);
          pack_view(v5(), idx) = PackViewExpectedValue(b, 2, idx.k, idx.j, idx.i);
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

TEST_CASE("loop abstraction halo producer-consumer contracts",
          "[loop_abstraction][contract][halo]") {
  RunHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
  RunHaloPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
  RunHaloPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction k halo disjoint span contracts",
          "[loop_abstraction][contract][halo]") {
  RunKTripletHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
  RunKTripletHaloPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
  RunKTripletHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
  RunKTripletHaloPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
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
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::logical_flat>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::bovi, inner_tag::logical_coords>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_flat>(
      spec, plus_j_cases);
  RunHaloParityPatternMatrix<plus_j_halo_t, loop_tag::boiv, inner_tag::logical_coords>(
      spec, plus_j_cases);

  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bvoi, inner_tag::logical_flat>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bvoi, inner_tag::logical_coords>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bovi, inner_tag::logical_flat>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::bovi, inner_tag::logical_coords>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::boiv, inner_tag::logical_flat>(
      spec, k_triplet_cases);
  RunHaloParityPatternMatrix<k_triplet_halo_t, loop_tag::boiv, inner_tag::logical_coords>(
      spec, k_triplet_cases);
}

TEST_CASE("loop abstraction pack view contracts on bvoi logical_flat",
          "[loop_abstraction][contract][pack_view][bvoi][logical_flat]") {
  RunPackViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_flat>();
}

TEST_CASE("loop abstraction pack view contracts on bvoi logical_coords",
          "[loop_abstraction][contract][pack_view][bvoi][logical_coords]") {
  RunPackViewPatternMatrix<loop_tag::bvoi, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction pack view contracts on bvoi memory",
          "[loop_abstraction][contract][pack_view][bvoi][memory]") {
  RunPackViewPatternMatrix<loop_tag::bvoi, inner_tag::memory>();
}

TEST_CASE("loop abstraction pack view contracts on bovi logical_flat",
          "[loop_abstraction][contract][pack_view][bovi][logical_flat]") {
  RunPackViewPatternMatrix<loop_tag::bovi, inner_tag::logical_flat>();
}

TEST_CASE("loop abstraction pack view contracts on bovi logical_coords",
          "[loop_abstraction][contract][pack_view][bovi][logical_coords]") {
  RunPackViewPatternMatrix<loop_tag::bovi, inner_tag::logical_coords>();
}

TEST_CASE("loop abstraction pack view contracts on bovi memory",
          "[loop_abstraction][contract][pack_view][bovi][memory]") {
  RunPackViewPatternMatrix<loop_tag::bovi, inner_tag::memory>();
}

TEST_CASE("loop abstraction pack view contracts on boiv logical_flat",
          "[loop_abstraction][contract][pack_view][boiv][logical_flat]") {
  RunPackViewPatternMatrix<loop_tag::boiv, inner_tag::logical_flat>();
}

TEST_CASE("loop abstraction pack view contracts on boiv logical_coords",
          "[loop_abstraction][contract][pack_view][boiv][logical_coords]") {
  RunPackViewPatternMatrix<loop_tag::boiv, inner_tag::logical_coords>();
}
