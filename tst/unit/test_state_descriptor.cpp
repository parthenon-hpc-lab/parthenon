//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include <set>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/sparse_pool.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "prolong_restrict/pr_ops.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

using parthenon::Coordinates_t;
using parthenon::IndexRange;
using parthenon::Metadata;
using parthenon::MetadataFlag;
using parthenon::Packages_t;
using parthenon::ParArrayND;
using parthenon::Real;
using parthenon::ResolvePackages;
using parthenon::SparsePool;
using parthenon::StateDescriptor;
using FlagVec = std::vector<MetadataFlag>;
using parthenon::VariableState;

// Some fake ops classes
struct MyProlongOp {
  template <int DIM, parthenon::TopologicalElement EL = parthenon::TopologicalElement::C>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    return; // stub
  }
};
struct MyRestrictOp {
  template <int DIM, parthenon::TopologicalElement EL = parthenon::TopologicalElement::C>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int ck, const int cj, const int ci,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *pcoarse,
     const ParArrayND<Real, VariableState> *pfine) {
    return; // stub
  }
};

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

TEST_CASE("Test dependency resolution in StateDescriptor", "[StateDescriptor]") {
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
    Metadata m_sparse_private(priv_sparse);
    Metadata m_sparse_provides(prov_sparse);
    Metadata m_sparse_requires(req_sparse);
    Metadata m_sparse_overridable(over_sparse);

    // packages
    Packages_t packages;
    auto pkg1 = std::make_shared<StateDescriptor>("package1");
    auto pkg2 = std::make_shared<StateDescriptor>("package2");
    auto pkg3 = std::make_shared<StateDescriptor>("package3");
    packages.Add(pkg1);
    packages.Add(pkg2);
    packages.Add(pkg3);

    WHEN("We add two non-sparse variables of the same name") {
      pkg1->AddField("dense", m_provides);
      THEN("The method returns false and the variable is not added") {
        REQUIRE(!(pkg1->AddField("dense", m_provides)));
      }
    }

    WHEN("We try to add the same sparse id twice") {
      pkg1->AddSparsePool("sparse", m_sparse_provides, std::vector<int>{sparse_ids[0]});
      THEN("The method returns false and the variable is not added") {
        REQUIRE(!(pkg1->AddSparsePool("sparse", m_sparse_provides,
                                      std::vector<int>{sparse_ids[0]})));
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

    WHEN("We add the same sparse private variable to two different packages") {
      pkg1->AddSparsePool("sparse", m_sparse_private, std::vector<int>{sparse_ids[2]});
      pkg2->AddSparsePool("sparse", m_sparse_private, std::vector<int>{sparse_ids[3]});
      THEN("We can safely resolve the conflict") {
        auto pkg3 = ResolvePackages(packages);
        AND_THEN("The names are privately namespaced") {
          REQUIRE(pkg3->SparseBaseNamePresent("package1::sparse"));
          REQUIRE(pkg3->SparseBaseNamePresent("package2::sparse"));
          REQUIRE(!(pkg3->SparseBaseNamePresent("sparse")));
        }
        AND_THEN("The appropriate sparse metadata was added") {
          REQUIRE(pkg3->FieldPresent("package1::sparse", sparse_ids[2]));
          REQUIRE(pkg3->FieldPresent("package2::sparse", sparse_ids[3]));
        }
      }
    }

    WHEN("We add multiple provides sparse ids to the same package") {
      pkg1->AddSparsePool("sparse", m_sparse_provides, sparse_ids);
      THEN("We can safely resolve packages") {
        auto pkg4 = ResolvePackages(packages);
        AND_THEN("The sparse variable is present") {
          for (int i = 0; i < sparse_ids.size(); i++) {
            REQUIRE(pkg4->FieldMetadata("sparse", sparse_ids[i]) == (m_sparse_provides));
          }
        }
      }
    }

    WHEN("We add the same dense/sparse provides in two packages") {
      pkg1->AddField("foo", m_provides);
      pkg2->AddSparsePool("foo", m_sparse_provides, sparse_ids);
      THEN("Resolution raises an error") { REQUIRE_THROWS(ResolvePackages(packages)); }
    }

    WHEN("We add the same dense provides as a sparse field provides in two packages") {
      pkg1->AddField("foo_7", m_provides);
      pkg2->AddSparsePool("foo", m_sparse_provides, sparse_ids);
      THEN("Resolution raises an error") { REQUIRE_THROWS(ResolvePackages(packages)); }
    }

    WHEN("We add the same private swarm to two different packages") {
      pkg1->AddSwarm("myswarm", m_private);
      pkg2->AddSwarm("myswarm", m_private);
      pkg1->AddSwarmValue("value1", "myswarm", m_swarmval);
      pkg2->AddSwarmValue("value2", "myswarm", m_swarmval);
      THEN("We can safely resolve the conflicts") {
        auto pkg3 = ResolvePackages(packages);
        AND_THEN("The names are privately namespaced") {
          REQUIRE(pkg3->SwarmPresent("package1::myswarm"));
          REQUIRE(pkg3->SwarmPresent("package2::myswarm"));
          REQUIRE(!(pkg3->SwarmPresent("myswarm")));
        }
        AND_THEN("The swarm values were added appropriately") {
          REQUIRE(pkg3->SwarmValuePresent("value1", "package1::myswarm"));
          REQUIRE(pkg3->SwarmValuePresent("value2", "package2::myswarm"));
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
      pkg2->AddSparsePool("sparse", m_sparse_overridable, sparse_ids);
      pkg3->AddSwarm("myswarm", m_overridable);
      pkg3->AddSwarmValue("value1", "myswarm", m_swarmval);
      pkg3->AddSwarmValue("value2", "myswarm", m_swarmval);
      THEN("We can safely resolve conflicts") {
        auto pkg4 = ResolvePackages(packages);
        AND_THEN("The overridable variables are retained") {
          REQUIRE(pkg4->FieldPresent("dense"));
          for (const int sid : sparse_ids) {
            REQUIRE(pkg4->FieldPresent("sparse", sid));
          }
          REQUIRE(pkg4->SwarmPresent("myswarm"));
          REQUIRE(pkg4->SwarmValuePresent("value1", "myswarm"));
          REQUIRE(pkg4->SwarmValuePresent("value2", "myswarm"));
        }
      }
    }

    WHEN("We add a provides variable and several overridable/requires variables") {
      pkg1->AddField("dense", m_provides);
      pkg2->AddField("dense", m_overridable);
      pkg3->AddField("dense", m_overridable);

      pkg1->AddSwarm("myswarm", m_overridable);
      pkg1->AddSwarmValue("overridable", "myswarm", m_swarmval);
      pkg2->AddSwarm("myswarm", m_provides);
      pkg2->AddSwarmValue("provides", "myswarm", m_swarmval);
      pkg3->AddSwarm("myswarm", m_requires);

      pkg1->AddSparsePool("sparse", m_sparse_overridable, sparse_ids);
      pkg2->AddSparsePool("sparse", m_sparse_overridable, sparse_ids);
      pkg3->AddSparsePool("sparse", m_sparse_provides, sparse_ids);

      pkg2->AddSparsePool("sparse_a", m_sparse_provides, sparse_ids);
      pkg2->AddSparsePool("sparse_b", m_sparse_provides, "sparse_a", sparse_ids);
      pkg2->AddSparsePool("sparse_c", m_sparse_provides, "sparse_b", sparse_ids);

      for (const int sid : sparse_ids) {
        REQUIRE(pkg1->FieldPresent("sparse", sid));
        REQUIRE(pkg2->FieldPresent("sparse", sid));
        REQUIRE(pkg3->FieldPresent("sparse", sid));
      }

      THEN("We can safely resolve conflicts") {
        Metadata m_provides_swarm(m_provides);
        // This is set automatically when adding a Swarm if not already set
        m_provides_swarm.Set(Metadata::Swarm);
        auto pkg4 = ResolvePackages(packages);
        AND_THEN("The provides variables take precedence.") {
          REQUIRE(pkg4->FieldPresent("dense"));
          REQUIRE(pkg4->FieldMetadata("dense") == m_provides);
          REQUIRE(pkg4->SwarmPresent("myswarm"));
          REQUIRE(pkg4->SwarmMetadata("myswarm") == m_provides_swarm);
          REQUIRE(pkg4->SwarmValuePresent("provides", "myswarm"));
          REQUIRE(!(pkg4->SwarmValuePresent("overridable", "myswarm")));
          REQUIRE(pkg4->SparseBaseNamePresent("sparse"));
          for (const int sid : sparse_ids) {
            REQUIRE(pkg4->FieldPresent("sparse", sid));
          }
          for (const int sid : sparse_ids) {
            REQUIRE(pkg4->FieldMetadata("sparse", sid) == m_sparse_provides);
          }
        }
        AND_THEN("The correct sparse allocation control is resolved.") {
          auto controlled_vars_a = pkg4->GetControlledVariables("sparse_a_3");
          REQUIRE(std::count(controlled_vars_a.begin(), controlled_vars_a.end(),
                             "sparse_a_3") == 1);
          REQUIRE(std::count(controlled_vars_a.begin(), controlled_vars_a.end(),
                             "sparse_b_3") == 1);
          REQUIRE(controlled_vars_a.size() == 2);

          auto controlled_vars_b = pkg4->GetControlledVariables("sparse_b_3");
          REQUIRE(std::count(controlled_vars_b.begin(), controlled_vars_b.end(),
                             "sparse_c_3") == 1);
          REQUIRE(controlled_vars_b.size() == 1);

          auto controlled_vars_c = pkg4->GetControlledVariables("sparse_c_3");
          REQUIRE(controlled_vars_c.size() == 0);

          auto controlled_vars_dense = pkg4->GetControlledVariables("dense");
          REQUIRE(std::count(controlled_vars_dense.begin(), controlled_vars_dense.end(),
                             "dense") == 1);
          REQUIRE(controlled_vars_dense.size() == 1);
        }
      }
    }

    WHEN("We register a dense variable custom prolongation/restriction") {
      pkg1->AddField("dense", m_provides);
      WHEN("We register a sparse variable with custom prolongation/restriction") {
        auto m_sparse_provides_ = m_sparse_provides;
        m_sparse_provides_.RegisterRefinementOps<MyProlongOp, MyRestrictOp>();
        pkg2->AddSparsePool("sparse", m_sparse_provides_, sparse_ids);
        THEN("We can perform dependency resolution") {
          auto pkg3 = ResolvePackages(packages);
          AND_THEN("The two relevant prolongation restriction operators exist and have "
                   "unique ids") {
            const auto my_funcs =
                parthenon::refinement::RefinementFunctions_t::RegisterOps<MyProlongOp,
                                                                          MyRestrictOp>();
            const auto cell_funcs =
                parthenon::refinement::RefinementFunctions_t::RegisterOps<
                    parthenon::refinement_ops::ProlongateSharedMinMod,
                    parthenon::refinement_ops::Restrict>();
            REQUIRE(pkg3->NumRefinementFuncs() == 2);
            REQUIRE((pkg3->RefinementFuncID(my_funcs)) !=
                    (pkg3->RefinementFuncID(cell_funcs)));
          }
        }
      }
      // TODO(JMM): This test should go away when issue #844 is resolved
      WHEN("We register a var that needs prolongation/restriction without ops") {
        Metadata m_dangerous(
            FlagVec{Metadata::Sparse, Metadata::Derived, Metadata::OneCopy});
        m_dangerous.Set(Metadata::FillGhost);
        THEN("Dependency resolution captures an ill-formed variable") {
          REQUIRE_THROWS(pkg2->AddSparsePool("sparse", m_dangerous, sparse_ids));
        }
      }
    }
  }
}

TEST_CASE("Test SparsePool interface", "[StateDescriptor]") {
  GIVEN("Some metadata") {
    Metadata dense({Metadata::Independent, Metadata::WithFluxes});
    Metadata sparse_vec({Metadata::Independent, Metadata::Sparse, Metadata::Vector},
                        std::vector<int>{3});

    THEN("We can create a SparsePool with sparse metadata") {
      SparsePool pool("sparse", sparse_vec);
      AND_THEN("We can add sparse indices to the pool") {
        const auto m2 = pool.Add(2);
        REQUIRE(m2 == sparse_vec);
        REQUIRE(m2.IsSet(Metadata::Vector));
        REQUIRE(!m2.IsSet(Metadata::Tensor));

        const int sparse_id = 5;
        const std::vector<int> shape = {2, 2, 4};
        const auto m5 = pool.Add(sparse_id, shape, Metadata::Tensor);

        const std::set<MetadataFlag> expected_flags{
            Metadata::Independent, Metadata::Sparse,   Metadata::Tensor,
            Metadata::None,        Metadata::Provides, Metadata::Real};
        const auto &flags = m5.Flags();
        const std::set<MetadataFlag> actual_flags(flags.begin(), flags.end());
        REQUIRE(expected_flags == actual_flags);

        REQUIRE(m5 != sparse_vec); // because shape and Vector/Tensor flag are different
        REQUIRE(m5.Shape().size() == shape.size());
        REQUIRE(!m5.IsSet(Metadata::Vector));
        REQUIRE(m5.IsSet(Metadata::Tensor));

        const auto mm17 = pool.Add(-17, {1}, Metadata::None, {"foo"});
        REQUIRE(!mm17.IsSet(Metadata::Vector));
        REQUIRE(!mm17.IsSet(Metadata::Tensor));
        REQUIRE(mm17.getComponentLabels() == std::vector<std::string>{"foo"});

        AND_THEN("We can't add the same sparse ID twice") { REQUIRE_THROWS(pool.Add(2)); }
      }

      AND_THEN("We can't add InvalidSparseID") {
        REQUIRE_THROWS(pool.Add(parthenon::InvalidSparseID));
      }
    }

    THEN("Trying to create a SparsePool with dense metadata throws an error") {
      REQUIRE_THROWS(SparsePool("dense", dense));
    }
  }

  GIVEN("An empty state descriptor") {
    auto pkg = std::make_shared<StateDescriptor>("pkg");
    Metadata meta_sparse({Metadata::Sparse});

    THEN("We can add sparse pools in different ways") {
      SparsePool pool1("pool1", meta_sparse);
      pool1.Add(0);
      pool1.Add(55);
      REQUIRE(pkg->AddSparsePool(pool1));

      const std::vector<int> sparse_ids_2{1, 55, 100};
      REQUIRE(pkg->AddSparsePool("pool2", meta_sparse, sparse_ids_2));

      std::vector<std::vector<int>> shapes;
      shapes.push_back({3});
      shapes.push_back({2, 2});
      const std::vector<int> sparse_ids_3{0, 100};
      REQUIRE(pkg->AddSparsePool(
          "pool3", meta_sparse, sparse_ids_3, shapes,
          std::vector<MetadataFlag>{Metadata::Vector, Metadata::Tensor}));

      REQUIRE(pkg->FieldPresent("pool1", 0));
      REQUIRE(pkg->FieldPresent("pool1", 55));
      REQUIRE(pkg->FieldPresent("pool2", 1));
      REQUIRE(pkg->FieldPresent("pool2", 55));
      REQUIRE(pkg->FieldPresent("pool2", 100));
      REQUIRE(pkg->FieldPresent("pool3", 0));
      REQUIRE(pkg->FieldPresent("pool3", 100));

      AND_THEN("We can't add a SparsePool with wrong number of Vector/Tensor flags") {
        REQUIRE_THROWS(pkg->AddSparsePool(
            "pool4", meta_sparse, std::vector<int>{0, 100}, shapes,
            std::vector<MetadataFlag>{Metadata::Vector, Metadata::Tensor,
                                      Metadata::Tensor}));
      }
    }

    THEN("We can't add fields/pools with the same names") {
      REQUIRE(pkg->AddField("dense", Metadata()));
      REQUIRE_FALSE(pkg->AddField("dense", Metadata()));
      REQUIRE_FALSE(pkg->AddSparsePool("dense", meta_sparse, std::vector<int>{1, 2}));

      REQUIRE(pkg->AddSparsePool("sparse", meta_sparse, std::vector<int>{1, 2}));
      REQUIRE_FALSE(pkg->AddSparsePool("sparse", meta_sparse, std::vector<int>{4, 5}));
      REQUIRE_FALSE(pkg->AddField("sparse", Metadata()));
      REQUIRE_FALSE(pkg->AddField("sparse_1", Metadata()));
      REQUIRE_FALSE(pkg->AddField("sparse_2", Metadata()));

      // this is ok, but probably not a good idea
      REQUIRE(pkg->AddField("sparse_3", Metadata()));

      // check the other way around
      REQUIRE(pkg->AddField("fake_sparse_27", Metadata()));
      REQUIRE(pkg->AddSparsePool("fake_sparse", meta_sparse, std::vector<int>{10, 12}));
      REQUIRE_FALSE(
          pkg->AddSparsePool("fake_sparse", meta_sparse, std::vector<int>{13, 27, 9}));

      REQUIRE(pkg->AddField("fake2_sparse_27", Metadata()));
      REQUIRE_THROWS(
          pkg->AddSparsePool("fake2_sparse", meta_sparse, std::vector<int>{13, 27, 9}));
    }
  }
}
