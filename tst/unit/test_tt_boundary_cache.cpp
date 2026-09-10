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

// Step 6b: the per-MeshTTData boundary index-map cache. Builds the cache over a real
// periodic mesh and checks that every boundary yields a congruent, one-to-one cell map
// (src interior -> dst ghost) with a channel registered in the mesh channel map.

#include <memory>
#include <set>
#include <sstream>

#include <catch2/catch.hpp>

#include "application_input.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tensors/tt_boundary_comm.hpp"
#include "tensors/tt_container.hpp"
#include "tensors/tt_field_metadata.hpp"
#include "tensors/tt_pack.hpp"
#include "utils/loop_utils.hpp"

using parthenon::ApplicationInput;
using parthenon::BuildBoundaryTensors;
using parthenon::BuildTTBoundaryCache;
using parthenon::DefaultTTraits;
using parthenon::DevExecSpace;
using parthenon::IndexDomain;
using parthenon::Mesh;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::StateDescriptor;
using parthenon::TTFieldMetadata;
namespace tensor2 = parthenon::tensor2;

namespace {

constexpr int kNGhost = 2;
constexpr int kNTheta = 4;
constexpr int kNPhi = 3;

Packages_t MakePackages() {
  Packages_t packages;
  auto pkg = std::make_shared<StateDescriptor>("tt_boundary_cache_test");
  // FillGhost so ForEachBoundary walks the field (as regular comm requires).
  pkg->AddTTField("I", TTFieldMetadata({kNTheta, kNPhi},
                                       Metadata({Metadata::Cell, Metadata::Independent,
                                                 Metadata::FillGhost})));
  packages.Add(pkg);
  return packages;
}

// A 2x2-block periodic mesh (uniform: identity transforms, same-level neighbors).
std::shared_ptr<Mesh> MakeMesh(ApplicationInput *app_in, Packages_t &packages) {
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  is << "nghost = " << kNGhost << "\n";
  is << "nx1 = 8\nx1min = 0.0\nx1max = 1.0\nix1_bc = periodic\nox1_bc = periodic\n";
  is << "nx2 = 8\nx2min = 0.0\nx2max = 1.0\nix2_bc = periodic\nox2_bc = periodic\n";
  is << "nx3 = 1\nx3min = 0.0\nx3max = 1.0\nix3_bc = outflow\nox3_bc = outflow\n";
  is << "<parthenon/meshblock>\nnx1 = 4\nnx2 = 4\nnx3 = 1\n";
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  return std::make_shared<Mesh>(pin.get(), app_in, packages, 0);
}

} // namespace

TEST_CASE("TT boundary cache maps interior cells to ghost cells", "[TTField][mesh][MPI]") {
  parthenon::Globals::nghost = kNGhost;
  auto app_in = std::make_shared<ApplicationInput>();
  auto packages = MakePackages();
  auto mesh = MakeMesh(app_in.get(), packages);

  auto partition = mesh->GetDefaultBlockPartitions()[0];
  auto md = mesh->tt_data.Add("base", partition);

  // Whole-block spatial extents (entire domain incl. ghosts) for classifying flat indices.
  auto pmb0 = md->GetBlockData(0)->GetBlockPointer();
  const auto &cb = pmb0->cellbounds;
  const int is = cb.is(IndexDomain::interior), ie = cb.ie(IndexDomain::interior);
  const int js = cb.js(IndexDomain::interior), je = cb.je(IndexDomain::interior);
  auto is_interior = [&](int i, int j) {
    return i >= is && i <= ie && j >= js && j <= je;
  };

  auto &cache = md->GetBoundaryCache();
  BuildTTBoundaryCache(md, &cache);
  const auto &bi = cache.bnd_info_h;

  GIVEN("The built boundary cache") {
    THEN("It holds one entry per boundary and records the current epoch") {
      REQUIRE(bi.extent(0) > 0);
      REQUIRE(cache.channels.size() == bi.extent(0));
      REQUIRE(cache.epoch == mesh->tt_comm_map.GetCurrentEpoch());
    }

    THEN("Each boundary is a congruent, one-to-one interior->ghost map with a channel") {
      for (std::size_t b = 0; b < bi.extent(0); ++b) {
        const auto &info = bi(b);
        const int n = static_cast<int>(info.send.size());
        REQUIRE(n > 0);
        REQUIRE(info.recv.size() == static_cast<std::size_t>(n));

        // The recorded channel index is valid and points at a live channel.
        REQUIRE(info.channel_idx >= 0);
        REQUIRE(info.channel_idx < static_cast<int>(cache.channels.size()));
        REQUIRE(cache.channels[info.channel_idx] != nullptr);

        std::set<int> dst_seen;
        for (int e = 0; e < n; ++e) {
          const auto [ts, us, vs, ks, js_, is_] = info.send(e);
          const auto [tr, ur, vr, kr, jr, ir] = info.recv(e);
          // Sender contributes from its interior; receiver deposits into its ghosts.
          REQUIRE(is_interior(is_, js_));
          REQUIRE_FALSE(is_interior(ir, jr));
          // One-to-one: no destination cell written twice.
          const int dst_flat = (kr * cache.nj + jr) * cache.ni + ir;
          REQUIRE(dst_seen.insert(dst_flat).second);
        }
      }
    }
  }
}

TEST_CASE("BuildBoundaryTensors gathers interior cells into the addend ghost layer",
          "[TTField][mesh][MPI]") {
  parthenon::Globals::nghost = kNGhost;
  auto app_in = std::make_shared<ApplicationInput>();
  auto packages = MakePackages();
  auto mesh = MakeMesh(app_in.get(), packages);

  auto partition = mesh->GetDefaultBlockPartitions()[0];
  auto md = mesh->tt_data.Add("base", partition);

  auto &cache = md->GetBoundaryCache();
  BuildTTBoundaryCache(md, &cache);
  const int nbound = static_cast<int>(cache.bnd_info_h.extent(0));
  REQUIRE(nbound > 0);

  // Seed each block's spatial core with a per-block, per-cell fingerprint so the gather
  // can be checked value-by-value: sc(0, idx, r) = 10000*b + idx + 1 (strictly positive,
  // so 0 unambiguously means "unwritten").
  {
    std::vector<std::shared_ptr<tensor2::TensorTrain>> src;
    for (int b = 0; b < md->NumBlocks(); ++b)
      src.push_back(md->GetBlockData(b)->Get("I"));
    auto pack = tensor2::TensorTrainHostPackT<DefaultTTraits>::FromSharedPtrs(src)
                    .MakeDevicePack();
    parthenon::par_for(
        parthenon::loop_pattern_flatrange_tag, "SeedCores", DevExecSpace(), 0,
        pack.GetNBlocks() - 1, KOKKOS_LAMBDA(const int b) {
          auto &sc = pack(b, 0, 0);
          for (int r = 0; r < sc.RR(); ++r)
            for (int idx = 0; idx < sc.DD(); ++idx)
              sc(0, idx, r) = 10000.0 * b + idx + 1.0;
        });
    Kokkos::fence();
  }

  // Capture the source train per boundary in the same ForEachBoundary order that
  // BuildBoundaryTensors uses, so addend e can be compared directly against its source.
  std::vector<std::shared_ptr<tensor2::TensorTrain>> srcs;
  parthenon::loops::ForEachBoundary<parthenon::BoundaryType::any>(
      md, [&](auto, auto, const parthenon::NeighborBlock &, auto v) { srcs.push_back(v); });
  REQUIRE(static_cast<int>(srcs.size()) == nbound);

  auto addends = BuildBoundaryTensors(md, cache);
  REQUIRE(static_cast<int>(addends.size()) == nbound);

  // For each boundary: the addend's spatial core equals the source value at each dst cell
  // (from the corresponding src cell) and is zero everywhere else.
  auto bnd_info = cache.bnd_info;
  const int ni = cache.ni, nj = cache.nj;
  for (int e = 0; e < nbound; ++e) {
    std::vector<std::shared_ptr<tensor2::TensorTrain>> pair{addends[e], srcs[e]};
    auto pack = tensor2::TensorTrainHostPackT<DefaultTTraits>::FromSharedPtrs(pair)
                    .MakeDevicePack();

    int nwrong = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_flatrange_tag, "CheckAddend", DevExecSpace(), 0, 0,
        KOKKOS_LAMBDA(const int, int &lwrong) {
          auto &sc = pack(0, 0, 0);  // addend
          auto &sc_src = pack(1, 0, 0); // source
          const auto &send = bnd_info(e).send;
          const auto &recv = bnd_info(e).recv;
          const int ncell = static_cast<int>(send.size());
          for (int r = 0; r < sc.RR(); ++r) {
            int nnonzero = 0;
            for (int idx = 0; idx < sc.DD(); ++idx)
              nnonzero += (sc(0, idx, r) != 0.0);
            int nmatched = 0;
            for (int c = 0; c < ncell; ++c) {
              const auto [ts, us, vs, ks, js, is] = send(c);
              const auto [tr, ur, vr, kr, jr, ir] = recv(c);
              const int src_idx = (ks * nj + js) * ni + is;
              const int dst_idx = (kr * nj + jr) * ni + ir;
              const double expect = sc_src(0, src_idx, r);
              lwrong += (sc(0, dst_idx, r) != expect);
              nmatched += (sc(0, dst_idx, r) == expect);
            }
            // No stray non-zero cells outside the mapped dst set.
            lwrong += (nnonzero != nmatched);
          }
        },
        nwrong);
    INFO("boundary " << e);
    REQUIRE(nwrong == 0);
  }
}
