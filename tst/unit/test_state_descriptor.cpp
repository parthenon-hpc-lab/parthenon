//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"

using parthenon::Metadata;
using parthenon::MetadataFlag;
using parthenon::Packages_t;
using parthenon::Real;
using parthenon::ResolvePackages;
using parthenon::StateDescriptor;

TEST_CASE("Test reqendency resolution in StateDescriptor", "[StateDescriptor]") {
  GIVEN("Some empty state descriptors and metadata") {
    // metadata
    using FlagVec = std::vector<MetadataFlag>;
    FlagVec priv = {Metadata::Independent, Metadata::FillGhost, Metadata::Private};
    FlagVec prov = {Metadata::Independent, Metadata::FillGhost, Metadata::Provides};
    FlagVec req = {Metadata::Independent, Metadata::FillGhost, Metadata::Requires};
    FlagVec over = {Metadata::Derived, Metadata::OneCopy, Metadata::Overridable};
    Metadata m_private(priv);
    Metadata m_provides(prov);
    Metadata m_requires(req);
    Metadata m_overridable(over);
    Metadata m_swarmval({Metadata::Real});
    std::vector<int> sparse_ids = {0, 3, 7, 11};
    std::vector<Metadata> m_sparse_private(sparse_ids.size());
    std::vector<Metadata> m_sparse_provides(sparse_ids.size());
    std::vector<Metadata> m_sparse_reqends(sparse_ids.size());
    std::vector<Metadata> m_sparse_overridable(sparse_ids.size());
    for (int i = 0; i < sparse_ids.size(); i++) {
      int id = sparse_ids[i]; // sparse metadata flag automatically added
      m_sparse_private[i] = Metadata(priv, id);
      m_sparse_provides[i] = Metadata(prov, id);
      m_sparse_reqends[i] = Metadata(req, id);
      m_sparse_overridable[i] = Metadata(over, id);
    }
    // packages
    Packages_t packages;
    auto pkg1 = std::make_shared<StateDescriptor>("package1");
    auto pkg2 = std::make_shared<StateDescriptor>("package2");
    auto pkg3 = std::make_shared<StateDescriptor>("package3");
    packages["package1"] = pkg1;
    packages["package2"] = pkg2;
    packages["package3"] = pkg3;

    WHEN("We add two non-sparse variables of the same name") {
      pkg1->AddField("dense", m_provides);
      THEN("The method returns false and the variable is not added") {
        REQUIRE(!(pkg1->AddField("dense", m_provides)));
      }
    }

    // TODO(JMM): This will simplify once we have dense on block
    WHEN("We try to add the same sparse id twice") {
      pkg1->AddField("sparse", m_sparse_provides[0]);
      THEN("The method returns false and the variable is not added") {
        REQUIRE(!(pkg1->AddField("sparse", m_sparse_provides[0])));
      }
    }
    // TODO(JMM): This will simplify once we have dense on block
    WHEN("We try to add sparse variables with the same name but different metadata") {
      pkg1->AddField("sparse", m_sparse_provides[0]);
      THEN("An error is thrown") {
        REQUIRE_THROWS(pkg1->AddField("sparse", m_sparse_overridable[3]));
      }
    }

    // no need to check this case for sparse/swarm as it's the same code path
    WHEN("We add the same dense provides variable to two different packages") {
      pkg1->AddField("dense", m_provides);
      pkg2->AddField("dense", m_provides);
      THEN("Resolution raises an error") { REQUIRE_THROWS(ResolvePackages(packages)); }
    }

    WHEN("We add the same dense private variable to two different packages") {
      pkg1->AddField("dense", m_private);
      pkg2->AddField("dense", m_private);
      THEN("We can safely resolve the conflict") {
        auto pkg3 = ResolvePackages(packages);
        AND_THEN("The names are privately namespaced") {
          REQUIRE(pkg3->FieldPresent("package1::dense"));
          REQUIRE(pkg3->FieldPresent("package2::dense"));
          REQUIRE(!(pkg3->FieldPresent("dense")));
        }
      }
    }
    // TODO(JMM): This will simplify once we have dense on block
    WHEN("We add the same sparse private variable to two different packages") {
      pkg1->AddField("sparse", m_sparse_private[2]);
      pkg2->AddField("sparse", m_sparse_private[3]);
      THEN("We can safely resolve the conflict") {
        auto pkg3 = ResolvePackages(packages);
        AND_THEN("The names are privately namespaced") {
          REQUIRE(pkg3->SparsePresent("package1::sparse"));
          REQUIRE(pkg3->SparsePresent("package2::sparse"));
          REQUIRE(!(pkg3->SparsePresent("sparse")));
        }
        AND_THEN("The appropriate sparse metadata was added") {
          auto &m1 = pkg3->FieldMetadata("package1::sparse", 0);
          auto &m2 = pkg3->FieldMetadata("package2::sparse", 0);
          REQUIRE(m1.GetSparseId() == sparse_ids[2]);
          REQUIRE(m2.GetSparseId() == sparse_ids[3]);
        }
      }
    }

    WHEN("We add the same private swarm to two different packages") {
      pkg1->AddSwarm("swarm", m_private);
      pkg2->AddSwarm("swarm", m_private);
      pkg1->AddSwarmValue("value1", "swarm", m_swarmval);
      pkg2->AddSwarmValue("value2", "swarm", m_swarmval);
      THEN("We can safely resolve the conflicts") {
        auto pkg3 = ResolvePackages(packages);
        AND_THEN("The names are privatley namespaced") {
          REQUIRE(pkg3->SwarmPresent("package1::swarm"));
          REQUIRE(pkg3->SwarmPresent("package2::swarm"));
          REQUIRE(!(pkg3->SwarmPresent("swarm")));
        }
        AND_THEN("The swarm values were added appropriately") {
          REQUIRE(pkg3->SwarmValuePresent("value1", "package1::swarm"));
          REQUIRE(pkg3->SwarmValuePresent("value2", "package2::swarm"));
        }
      }
    }

    // Do not need to repeat this for sparse/swarm as it is the same
    // code path
    WHEN("We add a Requires variable but do not provide it") {
      pkg1->AddField("dense1", m_requires);
      pkg2->AddField("dense2", m_requires);
      THEN("Resolving conflicts raises an error") {
        REQUIRE_THROWS(ResolvePackages(packages));
      }
    }
    WHEN("We add a dense requires variable and a provides variable") {
      pkg1->AddField("dense", m_requires);
      pkg2->AddField("dense", m_provides);
      THEN("We can safely resolve the conflicts") {
        auto pkg3 = ResolvePackages(packages);
        AND_THEN("The provides package is available") {
          REQUIRE(pkg3->FieldPresent("dense"));
          REQUIRE(pkg3->FieldMetadata("dense") == m_provides);
        }
      }
    }

    WHEN("We add an overridable variable and nothing else") {
      pkg1->AddField("dense", m_overridable);
      for (int i = 0; i < sparse_ids.size(); i++) {
        pkg2->AddField("sparse", m_sparse_overridable[i]);
      }
      REQUIRE(pkg2->AllSparseFields().at("sparse").size() == sparse_ids.size());
      pkg3->AddSwarm("swarm", m_overridable);
      pkg3->AddSwarmValue("value1", "swarm", m_swarmval);
      pkg3->AddSwarmValue("value2", "swarm", m_swarmval);
      THEN("We can safely resolve conflicts") {
        auto pkg4 = ResolvePackages(packages);
        AND_THEN("The overridable variables are retained") {
          REQUIRE(pkg4->FieldPresent("dense"));
          REQUIRE(pkg4->SparsePresent("sparse"));
          REQUIRE(pkg4->SwarmPresent("swarm"));
          REQUIRE(pkg4->SwarmValuePresent("value1", "swarm"));
          REQUIRE(pkg4->SwarmValuePresent("value2", "swarm"));
          REQUIRE(pkg4->AllSparseFields().at("sparse").size() == sparse_ids.size());
        }
      }
    }
  }
}
