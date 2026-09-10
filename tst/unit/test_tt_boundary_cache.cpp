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
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tensors/tt_boundary_comm.hpp"
#include "tensors/tt_container.hpp"
#include "tensors/tt_field_metadata.hpp"

using parthenon::ApplicationInput;
using parthenon::BuildTTBoundaryCache;
using parthenon::IndexDomain;
using parthenon::Mesh;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::StateDescriptor;
using parthenon::TTFieldMetadata;

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
