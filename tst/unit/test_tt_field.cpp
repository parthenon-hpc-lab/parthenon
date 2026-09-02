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

#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "tensors/tt_container.hpp"
#include "tensors/tt_field_metadata.hpp"
#include "tensors/tt_operations.hpp"
#include "tensors/tt_pack.hpp"

using parthenon::BlockList_t;
using parthenon::BlockListPartition;
using parthenon::GridIdentifier;
using parthenon::IndexDomain;
using parthenon::MeshBlock;
using parthenon::MeshBlockTTData;
using parthenon::MeshTTData;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ResolvePackages;
using parthenon::StateDescriptor;
using parthenon::TTFieldMetadata;

TEST_CASE("Tensor-train fields register and resolve", "[TTField]") {
  GIVEN("A package with a registered tensor-train field") {
    constexpr int NTHETA = 8;
    constexpr int NPHI = 16;
    auto pkg = std::make_shared<StateDescriptor>("tt_test");
    TTFieldMetadata d({NTHETA, NPHI}, Metadata({Metadata::Cell, Metadata::Independent}));
    pkg->AddTTField("I", d);

    THEN("The field is present with the expected structure") {
      REQUIRE(pkg->TTFieldPresent("I"));
      const auto &md = pkg->GetTTFieldMetadata("I");
      REQUIRE(md.NCores() == 3);
      REQUIRE(md.phys_dims.size() == 2);
      REQUIRE(md.phys_dims[0] == NTHETA);
      REQUIRE(md.phys_dims[1] == NPHI);
    }

    WHEN("Packages are resolved") {
      Packages_t packages;
      packages.Add(pkg);
      auto resolved = ResolvePackages(packages);

      THEN("The resolved descriptor carries the TT field") {
        REQUIRE(resolved->TTFieldPresent("I"));
        REQUIRE(resolved->AllTTFields().size() == 1);
      }
    }

    WHEN("A duplicate TT field name is registered in a second package") {
      auto pkg2 = std::make_shared<StateDescriptor>("tt_test2");
      pkg2->AddTTField("I", d);
      Packages_t packages;
      packages.Add(pkg);
      packages.Add(pkg2);
      THEN("Resolution raises an error") {
        REQUIRE_THROWS(ResolvePackages(packages));
      }
    }
  }
}

TEST_CASE("Tensor-train containers build trains from a block", "[TTField]") {
  GIVEN("A resolved package and a MeshBlock") {
    constexpr int NTHETA = 8;
    constexpr int NPHI = 16;
    constexpr int NSIDE = 4;
    constexpr int NDIM = 2;

    auto pkg = std::make_shared<StateDescriptor>("tt_test");
    pkg->AddTTField(
        "I", TTFieldMetadata({NTHETA, NPHI}, Metadata({Metadata::Cell, Metadata::Independent})));
    Packages_t packages;
    packages.Add(pkg);
    auto resolved = ResolvePackages(packages);

    auto pmb = std::make_shared<MeshBlock>(NSIDE, NDIM);
    pmb->resolved_packages = resolved;
    // The lightweight MeshBlock ctor does not run the full setup, so initialize
    // the base TT container explicitly (the real ctor does this in meshblock.cpp).
    pmb->tt_block_data.Get()->Initialize(pmb);

    const int nspace = pmb->cellbounds.GetTotal(IndexDomain::entire);

    THEN("The base container holds a train with the derived spatial core dim") {
      auto &ttdata = pmb->tt_block_data.Get();
      REQUIRE(ttdata->Contains("I"));
      auto train = ttdata->Get("I");
      REQUIRE(train->NCores() == 3);
      REQUIRE(train->GetPhysicalDimension(0) == nspace);
      REQUIRE(train->GetPhysicalDimension(1) == NTHETA);
      REQUIRE(train->GetPhysicalDimension(2) == NPHI);
    }

    THEN("GetBounds* mirror the block's cellbounds") {
      auto &ttdata = pmb->tt_block_data.Get();
      auto ib = ttdata->GetBoundsI(parthenon::IndexDomain::interior);
      auto ib_ref = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
      REQUIRE(ib.s == ib_ref.s);
      REQUIRE(ib.e == ib_ref.e);
      auto jb = ttdata->GetBoundsJ(parthenon::IndexDomain::entire);
      auto jb_ref = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
      REQUIRE(jb.s == jb_ref.s);
      REQUIRE(jb.e == jb_ref.e);
    }

    WHEN("A named stage is created as a deep copy") {
      auto &base = pmb->tt_block_data.Get();
      auto &stage = pmb->tt_block_data.Add("stage1", base);

      THEN("The stage has an independent train object") {
        REQUIRE(stage->Contains("I"));
        REQUIRE(stage->Get("I").get() != base->Get("I").get());
        REQUIRE(stage->Get("I")->NCores() == base->Get("I")->NCores());
      }
    }

    WHEN("A named stage is created shallow") {
      auto &base = pmb->tt_block_data.Get();
      auto &stage = pmb->tt_block_data.AddShallow("shallow1", base);

      THEN("The stage aliases the base train object") {
        REQUIRE(stage->Get("I").get() == base->Get("I").get());
      }
    }
  }
}

TEST_CASE("MeshTTData assembles over a block partition", "[TTField]") {
  GIVEN("A partition of blocks each holding a base TT container") {
    constexpr int NTHETA = 8;
    constexpr int NPHI = 16;
    constexpr int NSIDE = 4;
    constexpr int NDIM = 2;
    constexpr int NBLOCKS = 3;

    auto pkg = std::make_shared<StateDescriptor>("tt_test");
    pkg->AddTTField(
        "I", TTFieldMetadata({NTHETA, NPHI}, Metadata({Metadata::Cell, Metadata::Independent})));
    Packages_t packages;
    packages.Add(pkg);
    auto resolved = ResolvePackages(packages);

    BlockList_t block_list;
    for (int i = 0; i < NBLOCKS; ++i) {
      auto pmb = std::make_shared<MeshBlock>(NSIDE, NDIM);
      pmb->resolved_packages = resolved;
      pmb->tt_block_data.Get()->Initialize(pmb);
      block_list.push_back(pmb);
    }
    constexpr int kPartition = 7;
    auto part = std::make_shared<BlockListPartition>(kPartition, GridIdentifier::leaf(),
                                                     block_list, nullptr);

    WHEN("A MeshTTData base stage is initialized from the partition") {
      MeshTTData md("base");
      md.Initialize(part);

      THEN("It exposes one block container per block, each with the field") {
        REQUIRE(md.NumBlocks() == NBLOCKS);
        for (int b = 0; b < NBLOCKS; ++b) {
          REQUIRE(md.GetBlockData(b)->Contains("I"));
          // The MeshTTData stage aliases each block's own base container.
          REQUIRE(md.GetBlockData(b).get() ==
                  block_list[b]->tt_block_data.Get().get());
        }
      }

      THEN("It carries the partition's grid identity for symmetry with MeshData") {
        REQUIRE(md.partition == kPartition);
        REQUIRE(md.grid.type == GridIdentifier::leaf().type);
      }
    }

    WHEN("Host-pack tensor-train ops run over the partition") {
      MeshTTData in("base");
      in.Initialize(part);
      MeshTTData out("stage1");
      out.Initialize(std::make_shared<MeshTTData>(in));

      // Record input ranks (all-ones for a freshly-created field).
      std::vector<int> in_r0(NBLOCKS), in_r1(NBLOCKS);
      for (int b = 0; b < NBLOCKS; ++b) {
        auto t = in.GetBlockData(b)->Get("I");
        in_r0[b] = t->GetCoreHost(0).RR();
        in_r1[b] = t->GetCoreHost(1).RR();
      }

      // Build host packs directly from the mesh containers (the mesh -> host
      // pack -> op path an application uses). FromContainer gathers all fields.
      using parthenon::tensor2::TensorTrainHostPack;
      auto in_pack = TensorTrainHostPack::FromContainer(in);
      auto out_pack = TensorTrainHostPack::FromContainer(out);

      THEN("NonDestructiveSum(in, in, out) yields combined ranks in out") {
        parthenon::tensor2::NonDestructiveSum(in_pack, in_pack, out_pack);
        for (int b = 0; b < NBLOCKS; ++b) {
          auto t = out.GetBlockData(b)->Get("I");
          REQUIRE(t->GetCoreHost(0).RR() == in_r0[b] + in_r0[b]);
          REQUIRE(t->GetCoreHost(1).RR() == in_r1[b] + in_r1[b]);
          // Inputs are untouched (out is a distinct stage).
          REQUIRE(in.GetBlockData(b)->Get("I")->GetCoreHost(0).RR() == in_r0[b]);
        }

        AND_THEN("RoundGramSVD compresses the summed field back down") {
          parthenon::tensor2::RoundGramSVD(out_pack, 1.e-12);
          for (int b = 0; b < NBLOCKS; ++b) {
            auto t = out.GetBlockData(b)->Get("I");
            // in+in is rank-deficient, so rounding cannot exceed the summed rank
            // and should not error.
            REQUIRE(t->GetCoreHost(0).RR() <= in_r0[b] + in_r0[b]);
            REQUIRE(t->GetCoreHost(0).RR() >= 1);
          }
        }
      }
    }
  }
}

// Field tags for the multi-field pack test.
namespace tags {
struct A : public parthenon::tensor2::tt_var_base_t {
  static std::string name() { return "A"; }
};
struct B : public parthenon::tensor2::tt_var_base_t {
  static std::string name() { return "B"; }
};
} // namespace tags

TEST_CASE("Multi-field packs support integer and tag indexing", "[TTField]") {
  GIVEN("A container holding two same-shape tensor-train fields") {
    constexpr int NTHETA = 8;
    constexpr int NPHI = 16;
    constexpr int NSIDE = 4;
    constexpr int NDIM = 2;
    constexpr int NBLOCKS = 2;

    auto pkg = std::make_shared<StateDescriptor>("tt_test");
    Metadata cell({Metadata::Cell, Metadata::Independent});
    pkg->AddTTField("A", TTFieldMetadata({NTHETA, NPHI}, cell));
    pkg->AddTTField("B", TTFieldMetadata({NTHETA, NPHI}, cell));
    Packages_t packages;
    packages.Add(pkg);
    auto resolved = ResolvePackages(packages);

    BlockList_t block_list;
    for (int i = 0; i < NBLOCKS; ++i) {
      auto pmb = std::make_shared<MeshBlock>(NSIDE, NDIM);
      pmb->resolved_packages = resolved;
      pmb->tt_block_data.Get()->Initialize(pmb);
      block_list.push_back(pmb);
    }
    auto part = std::make_shared<BlockListPartition>(0, GridIdentifier::leaf(),
                                                     block_list, nullptr);
    MeshTTData md("base");
    md.Initialize(part);

    WHEN("An untagged host pack is built over all fields") {
      using parthenon::tensor2::TensorTrainHostPack;
      auto host = TensorTrainHostPack::FromContainer(md);

      THEN("It reports the container's field count") {
        REQUIRE(host.NumVars() == 2);
        REQUIRE(host.NumBlocks() == NBLOCKS);
      }
    }

    WHEN("An untagged host pack is built from an explicit name list") {
      using parthenon::tensor2::TensorTrainHostPack;
      auto host = TensorTrainHostPack::FromNames(md, {"B"});

      THEN("It gathers only the named fields, in the given order") {
        REQUIRE(host.NumVars() == 1);
        REQUIRE(host.NumBlocks() == NBLOCKS);
        REQUIRE(&host(0, 0) == md.GetBlockData(0)->Get("B").get());
      }
    }

    WHEN("A tagged host pack is built for a specific field set") {
      using parthenon::tensor2::TensorTrainHostPackFor;
      auto host = TensorTrainHostPackFor<tags::A, tags::B>::FromContainer(md);

      THEN("The host pack is accessible by tag (block leads)") {
        REQUIRE(host.template VarIndex<tags::A>() == 0);
        REQUIRE(host.template VarIndex<tags::B>() == 1);
        // Tag and integer accessors reach the same train.
        REQUIRE(&host(0, tags::A{}) == &host(0, 0));
        REQUIRE(&host(0, tags::B{}) == &host(0, 1));
      }

      THEN("Its device pack resolves tags to var slots (block leads)") {
        auto pack = host.MakeDevicePack();
        REQUIRE(pack.GetNVars() == 2);
        REQUIRE(pack.template VarIndex<tags::A>() == 0);
        REQUIRE(pack.template VarIndex<tags::B>() == 1);
      }
    }
  }
}
