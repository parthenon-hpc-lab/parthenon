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
using FlagVec = std::vector<MetadataFlag>;

TEST_CASE("Test Add/Get in Packages_t", "[Packages_t]") {
  GIVEN("A Packages_t object and a few packages") {
    Packages_t packages;
    auto pkg1 = std::make_shared<StateDescriptor>("package1");
    auto pkg2 = std::make_shared<StateDescriptor>("package2");
    THEN("We can add a package") {
      packages.Add(pkg1);
      AND_THEN("The package is available and named correctly") {
        auto &pkg = packages.Get("package1");
        REQUIRE(pkg->label() == "package1");
      }
      AND_THEN("Requesting a package not added throws an error") {
        REQUIRE_THROWS(packages.Get("package2"));
      }
      AND_THEN("Adding a different package with the same name throws an error") {
        auto pkg3 = std::make_shared<StateDescriptor>("package1");
        REQUIRE_THROWS(packages.Add(pkg3));
      }
    }
  }
}

TEST_CASE("Test Associate in StateDescriptor", "[StateDescriptor]") {
  GIVEN("Some flags and state descriptors") {
    FlagVec foo = {Metadata::Independent, Metadata::FillGhost};
    StateDescriptor state("state");
    WHEN("We add some fields with and without associated vars in metadata") {
      state.AddField("foo", Metadata(foo));
      state.AddField("bar", Metadata(foo, "foo"));
      state.AddField("baz", Metadata(foo));
      THEN("The associations are correct") {
        REQUIRE(state.FieldMetadata("foo").getAssociated() == "foo");
        REQUIRE(state.FieldMetadata("bar").getAssociated() == "foo");
        REQUIRE(state.FieldMetadata("baz").getAssociated() == "baz");
      }
    }
  }
}

TEST_CASE("Test reqendency resolution in StateDescriptor", "[StateDescriptor]") {
  GIVEN("Some empty state descriptors and metadata") {
    // metadata
    FlagVec priv = {Metadata::Independent, Metadata::FillGhost, Metadata::Private};
    FlagVec prov = {Metadata::Independent, Metadata::FillGhost, Metadata::Provides};
    FlagVec req = {Metadata::Independent, Metadata::FillGhost, Metadata::Requires};
    FlagVec over = {Metadata::Derived, Metadata::OneCopy, Metadata::Overridable};
    FlagVec priv_sparse = {Metadata::Sparse, Metadata::Independent, Metadata::FillGhost,
                           Metadata::Private};
    FlagVec prov_sparse = {Metadata::Sparse, Metadata::Independent, Metadata::FillGhost,
                           Metadata::Provides};
    FlagVec req_sparse = {Metadata::Sparse, Metadata::Independent, Metadata::FillGhost,
                          Metadata::Requires};
    FlagVec over_sparse = {Metadata::Sparse, Metadata::Derived, Metadata::OneCopy,
                           Metadata::Overridable};
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
      m_sparse_private[i] = Metadata(priv_sparse, id);
      m_sparse_provides[i] = Metadata(prov_sparse, id);
      m_sparse_reqends[i] = Metadata(req_sparse, id);
      m_sparse_overridable[i] = Metadata(over_sparse, id);
    }
    // packages
    Packages_t packages;
    auto pkg1 = std::make_shared<StateDescriptor>("package1");
    auto pkg2 = std::make_shared<StateDescriptor>("package2");
    auto pkg3 = std::make_shared<StateDescriptor>("package3");
    packages.Add(pkg1);
    packages.Add(pkg2);
    packages.Add(pkg3);

    WHEN("We add metadata with a sparse ID but the sparse flag unset") {
      THEN("We raise an error") {
        REQUIRE_THROWS(pkg1->AddField("sparse", Metadata(prov, 10)));
      }
    }

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
          REQUIRE(pkg3->SparsePresent("package1::sparse", sparse_ids[2]));
          REQUIRE(pkg3->SparsePresent("package2::sparse", sparse_ids[3]));
          REQUIRE(!(pkg3->SparsePresent("package1::sparse", sparse_ids[3])));
          REQUIRE(!(pkg3->SparsePresent("package2::sparse", sparse_ids[2])));
        }
      }
    }
    // TODO(JMM): This will simplify once we have dense on block
    WHEN("We add multiple provides sparse ids to the same package") {
      for (int i = 0; i < sparse_ids.size(); i++) {
        pkg1->AddField("sparse", m_sparse_provides[i]);
      }
      THEN("We can safely resolve packages") {
        auto pkg4 = ResolvePackages(packages);
        AND_THEN("The sparse variable is present") {
          for (int i = 0; i < sparse_ids.size(); i++) {
            pkg4->FieldMetadata("sparse", i).SparseEqual(m_sparse_provides[0]);
          }
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

    WHEN("We add a provides variable and several overridable/requires variables") {
      pkg1->AddField("dense", m_provides);
      pkg2->AddField("dense", m_overridable);
      pkg3->AddField("dense", m_overridable);

      pkg1->AddSwarm("swarm", m_overridable);
      pkg1->AddSwarmValue("overridable", "swarm", m_swarmval);
      pkg2->AddSwarm("swarm", m_provides);
      pkg2->AddSwarmValue("provides", "swarm", m_swarmval);
      pkg3->AddSwarm("swarm", m_requires);

      for (int i = 0; i < sparse_ids.size(); i++) {
        pkg1->AddField("sparse", m_sparse_overridable[i]);
        pkg2->AddField("sparse", m_sparse_overridable[i]);
        pkg3->AddField("sparse", m_sparse_provides[i]);
        REQUIRE(pkg1->SparsePresent("sparse", sparse_ids[i]));
        REQUIRE(pkg2->SparsePresent("sparse", sparse_ids[i]));
        REQUIRE(pkg3->SparsePresent("sparse", sparse_ids[i]));
      }

      THEN("We can safely resolve conflicts") {
        auto pkg4 = ResolvePackages(packages);
        AND_THEN("The provides variables take precedence.") {
          REQUIRE(pkg4->FieldPresent("dense"));
          REQUIRE(pkg4->FieldMetadata("dense") == m_provides);
          REQUIRE(pkg4->SwarmPresent("swarm"));
          REQUIRE(pkg4->SwarmMetadata("swarm") == m_provides);
          REQUIRE(pkg4->SwarmValuePresent("provides", "swarm"));
          REQUIRE(!(pkg4->SwarmValuePresent("overridable", "swarm")));
          REQUIRE(pkg4->SparsePresent("sparse"));
          for (int i = 0; i < sparse_ids.size(); i++) {
            REQUIRE(pkg4->SparsePresent("sparse", sparse_ids[i]));
          }
          for (int i = 0; i < sparse_ids.size(); i++) {
            REQUIRE(pkg4->FieldMetadata("sparse", sparse_ids[i]) == m_sparse_provides[i]);
          }
        }
      }
    }
  }
}
