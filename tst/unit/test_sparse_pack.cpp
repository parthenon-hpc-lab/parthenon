//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"
#include "loop_abstraction.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "pack/sparse_pack/sparse_pack.hpp"
#include "pack/sparse_pack/pack_view.hpp"

// TODO(jcd): can't call the MeshBlock constructor without mesh_refinement.hpp???
#include "mesh/mesh_refinement.hpp"

using parthenon::BlockList_t;
using parthenon::DevExecSpace;
using parthenon::IndexDomain;
using parthenon::loop_pattern_mdrange_tag;
using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshData;
using parthenon::Metadata;
using parthenon::PackIndexMap;
using parthenon::par_for;
using parthenon::Real;
using parthenon::StateDescriptor;

namespace {
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

struct v1 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v1(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v1"; }
  static constexpr bool is_sparse() {return false;}
};

struct v3 : public parthenon::variable_names::base_t<false, 3> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v3(Ts &&...args)
      : parthenon::variable_names::base_t<false, 3>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v3"; }
  // This isn't made from a sparse pool, be we allocate and deallocate by hand below
  static constexpr bool is_sparse() {return true;}
};

struct v5 : public parthenon::variable_names::base_t<false> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v5(Ts &&...args)
      : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v5"; }
  static constexpr bool is_sparse() {return false;}
};

using parthenon::variable_names::ANYDIM;
struct v7 : public parthenon::variable_names::base_t<false, ANYDIM, 3> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION v7(Ts &&...args)
      : parthenon::variable_names::base_t<false, ANYDIM, 3>(std::forward<Ts>(args)...) {}
  static std::string name() { return "v7"; }
  static constexpr bool is_sparse() {return false;}
};

} // namespace

TEST_CASE("Test behavior of sparse packs", "[SparsePack]") {
  constexpr int N = 6;
  constexpr int NDIM = 3;
  constexpr int NBLOCKS = 9;

  GIVEN("A tensor variable on a mesh") {
    const std::vector<int> tensor_shape{N, N, N, 3, 3};
    Metadata m_tensor({Metadata::Independent}, tensor_shape);
    auto pkg = std::make_shared<StateDescriptor>("Test package");
    pkg->AddField<v7>(m_tensor);
    BlockList_t block_list = MakeBlockList(pkg, NBLOCKS, N, NDIM);

    MeshData<Real> mesh_data("base");
    mesh_data.Initialize(block_list, nullptr);

    WHEN("We initialize the independent variables by hand and deallocate one") {
      auto ib = block_list[0]->cellbounds.GetBoundsI(IndexDomain::entire);
      auto jb = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::entire);
      auto kb = block_list[0]->cellbounds.GetBoundsK(IndexDomain::entire);
      for (int b = 0; b < NBLOCKS; ++b) {
        auto &pmb = block_list[b];
        auto &pmbd = pmb->meshblock_data.Get();
        auto var = pmbd->Get("v7");
        auto var5 = var.data.Get<5>();
        int slower_rank = var5.GetDim(5);
        int faster_rank = var5.GetDim(4);
        par_for(
            loop_pattern_mdrange_tag, "initializev7", DevExecSpace(), kb.s, kb.e, jb.s,
            jb.e, ib.s, ib.e, KOKKOS_LAMBDA(int k, int j, int i) {
              for (int l = 0; l < slower_rank; ++l) {
                for (int m = 0; m < faster_rank; ++m) {
                  Real n = m + 1e1 * l;
                  var5(l, m, k, j, i) = n;
                }
              }
            });
      }
      THEN("A sparse pack can correctly index into tensor types") {
        auto desc = parthenon::MakePackDescriptor<v7>(pkg.get());
        auto sparse_pack = desc.GetPack(&mesh_data);
        int nwrong = 0;
        int nl = tensor_shape[4];
        int nm = tensor_shape[3];
        par_reduce(
            loop_pattern_mdrange_tag, "check vector", DevExecSpace(), 0,
            sparse_pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              // 0-th is ANYDIM, 1st is 3.
              for (int l = 0; l < nl; ++l) {
                for (int m = 0; m < nm; ++m) {
                  Real n = m + 1e1 * l;
                  if (sparse_pack(b, v7(l, m), k, j, i) != n) {
                    ltot += 1;
                  }
                }
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);

        AND_THEN("A sparse pack can correctly output variable names") {
          REQUIRE(sparse_pack.LabelHost(0, 0) == "v7");
        }
      }
    }
  }

  GIVEN("A set of meshblocks and meshblock and mesh data") {
    const std::vector<int> scalar_shape{N, N, N};
    const std::vector<int> vector_shape{N, N, N, 3};

    Metadata m({Metadata::Independent, Metadata::WithFluxes}, scalar_shape);
    Metadata m_vector({Metadata::Independent, Metadata::WithFluxes, Metadata::Vector},
                      vector_shape);
    auto pkg = std::make_shared<StateDescriptor>("Test package");
    pkg->AddField(v1::name(), m);
    pkg->AddField(v3::name(), m_vector);
    pkg->AddField(v5::name(), m);
    BlockList_t block_list = MakeBlockList(pkg, NBLOCKS, N, NDIM);

    MeshData<Real> mesh_data("base");
    mesh_data.Initialize(block_list, nullptr);

    WHEN("We initialize the independent variables by hand and deallocate one") {
      auto ib = block_list[0]->cellbounds.GetBoundsI(IndexDomain::entire);
      auto jb = block_list[0]->cellbounds.GetBoundsJ(IndexDomain::entire);
      auto kb = block_list[0]->cellbounds.GetBoundsK(IndexDomain::entire);
      const std::vector<std::string> all_indep{"v1", "v3", "v5"};
      for (int b = 0; b < NBLOCKS; ++b) {
        auto &pmb = block_list[b];
        auto &pmbd = pmb->meshblock_data.Get();
        for (int v = 0; v < all_indep.size(); ++v) {
          auto &vnam = all_indep[v];
          auto var = pmbd->Get(vnam);
          auto var4 = var.data.Get<4>();
          int num_components = var.GetDim(4);
          par_for(
              loop_pattern_mdrange_tag, "initialize " + vnam, DevExecSpace(), kb.s, kb.e,
              jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(int k, int j, int i) {
                for (int c = 0; c < num_components; ++c) {
                  Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                  var4(c, k, j, i) = n;
                }
              });
        }
      }

      // Deallocate a variable on an arbitrary block
      block_list[2]->DeallocateSparse("v3");

      THEN("A sparse pack can be loaded on this data and report the bounds for block 2 "
           "appropriately.") {
        auto desc =
            parthenon::MakePackDescriptor<v3, v5>(pkg.get(), {Metadata::WithFluxes});
        auto pack = desc.GetPack(&mesh_data);
        int lo = pack.GetLowerBoundHost(2);
        int hi = pack.GetUpperBoundHost(2);
        REQUIRE(lo == 0); // lo = 0 because always start at 0 on a block
        REQUIRE(hi == 0); // hi is scalar. Only one value.
      }

      THEN("A sparse pack correctly loads this data and can report existence and "
           "nonexistence for variables on different blocks.") {
        auto desc = parthenon::MakePackDescriptor<v1, v3, v5>(pkg.get());
        auto pack = desc.GetPack(&mesh_data);
        REQUIRE(pack.ContainsHost(2, v1()));
        REQUIRE(!pack.ContainsHost(2, v3()));
        REQUIRE(pack.ContainsHost(2, v5()));
        REQUIRE(!pack.ContainsHost(2, v1(), v3(), v5()));
        REQUIRE(pack.ContainsHost<v1, v5>(2));
        REQUIRE(pack.GetSizeHost(2, v1()) == 1);
        REQUIRE(pack.GetSizeHost(2, v3()) == 0);
        REQUIRE(pack.GetSizeHost(1, v3()) == 3);
      }

      THEN("A sparse pack correctly loads this data and can be read from v3 on all "
           "blocks") {
        // Create a pack use type variables
        auto desc =
            parthenon::MakePackDescriptor<v5, v3>(pkg.get(), {Metadata::WithFluxes});
        auto sparse_pack = desc.GetPack(&mesh_data);

        auto desc_notype = parthenon::MakePackDescriptor(
            pkg.get(), std::vector<std::string>{"v5", "v3"}, {Metadata::WithFluxes});
        auto sparse_pack_notype = desc_notype.GetPack(&mesh_data);
        auto pack_map = desc_notype.GetMap();
        parthenon::PackIdx iv3(pack_map["v3"]);

        // Make sure that we have only cached one pack, since these should be the
        // same base pack
        REQUIRE(mesh_data.GetSparsePackCache().size() == 1);

        const int v = 1; // v3 is the second variable in the loop above so v = 1 there
        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check vector", DevExecSpace(), 0,
            sparse_pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(b, v3());
              int hi = sparse_pack.GetUpperBound(b, v3());
              auto sparse_pack_view = parthenon::make_sparse_pack_view(sparse_pack, b, 0);
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                if (n != sparse_pack(b, lo + c, k, j, i)) ltot += 1;
                if (n != sparse_pack(b, v3(c), k, j, i)) ltot += 1;
                if (n != sparse_pack_view(v3(c), k, j, i)) ltot += 1;
              }
              lo = sparse_pack_notype.GetLowerBound(b, iv3);
              hi = sparse_pack_notype.GetUpperBound(b, iv3);
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                if (n != sparse_pack_notype(b, lo + c, k, j, i)) ltot += 1;
                if (n != sparse_pack_notype(b, iv3 + c, k, j, i)) ltot += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A sparse pack can be wrapped in a compile-time indexed view pack") {
        auto desc = parthenon::MakePackDescriptor<v1, v3, v5>(pkg.get());
        auto sparse_pack = desc.GetPack(&mesh_data);

        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check view pack", DevExecSpace(), 0,
            sparse_pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              auto pack_view = parthenon::make_pack_view(sparse_pack, b);

              const auto ref_v1 = sparse_pack(b, v1(), k, j, i);
              const auto ref_v5 = sparse_pack(b, v5(), k, j, i);

              if (pack_view(v1(), k, j, i) != ref_v1) {
                ltot += 1;
              }
              if (pack_view(v5(), k, j, i) != ref_v5) {
                ltot += 1;
              }

              // The wrapper should remain a live view into the pack data.
              const Real saved = pack_view(v1(), k, j, i);
              pack_view(v1(), k, j, i) = saved + 7.0;
              if (sparse_pack(b, v1(), k, j, i) != saved + 7.0) {
                ltot += 1;
              }
              pack_view(v1(), k, j, i) = saved;
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A sparse pack can be wrapped by the loop abstraction pack view") {
        using IS = plb2::loop_abstraction::IndexSpace<
            plb2::loop_abstraction::loop_tag::bovi,
            plb2::loop_abstraction::inner_tag::memory>;
        auto desc = parthenon::MakePackDescriptor<v1, v3, v5>(pkg.get());
        auto sparse_pack = desc.GetPack(&mesh_data);
        IS idx_space(1, N, N, N, 0);
        auto idx_range = plb2::loop_abstraction::FlatRange(
            idx_space, 0, 0, idx_space.GetLogicalIndexer().size() - 1);
        auto pack_view =
            plb2::loop_abstraction::make_pack_view(idx_range, sparse_pack, 0);

        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check loop pack view", DevExecSpace(), kb.s, kb.e,
            jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int k, int j, int i, int &ltot) {
              if (pack_view(v1(), k, j, i) != sparse_pack(0, v1(), k, j, i)) {
                ltot += 1;
              }
              if (pack_view(v5(), k, j, i) != sparse_pack(0, v5(), k, j, i)) {
                ltot += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A boiv sparse pack view can translate flat indices back to coordinates") {
        using namespace plb2::loop_abstraction;
        using IS = IndexSpace<loop_tag::boiv, inner_tag::logical_flat>;
        auto desc = parthenon::MakePackDescriptor<v1, v3, v5>(pkg.get());
        auto sparse_pack = desc.GetPack(&mesh_data);
        IS idx_space(1, N, N, N, 0);

        outer(idx_space, KOKKOS_LAMBDA(const auto &current_range, int b) {
              auto pack_view = make_pack_view(current_range, sparse_pack, b);
              inner(current_range, [&](const int idx) {
                    const auto [k, j, i] = current_range.GetKJI(idx);
                    REQUIRE(pack_view(v1(), idx) == sparse_pack(b, v1(), k, j, i));
                    REQUIRE(pack_view(v5(), idx) == sparse_pack(b, v5(), k, j, i));
                  });
            });
      }

      THEN("A flattened sparse pack can correctly load this data in a unified outer "
           "index space") {
        using parthenon::PDOpt;
        using parthenon::variable_names::any_nonautoflux;
        auto desc = parthenon::MakePackDescriptor<any_nonautoflux>(
            pkg.get(), {}, {PDOpt::WithFluxes, PDOpt::Flatten});
        auto sparse_pack = desc.GetPack(&mesh_data);
        REQUIRE(sparse_pack.GetNBlocks() == 1);
        // v3 is deallocated on one block.
        REQUIRE(sparse_pack.GetMaxNumberOfVars() == 5 * NBLOCKS - 3);
        REQUIRE(sparse_pack.GetLowerBoundHost(0) == 0);
        // upper bound is inclusive
        REQUIRE(sparse_pack.GetUpperBoundHost(0) == 5 - 1);
        REQUIRE(sparse_pack.GetSize() == 5 * NBLOCKS - 3);
        AND_THEN("A flattened sparse pack starting with v3 has sensible lower/upper "
                 "bounds on the block where we deallocate") {
          auto desc = parthenon::MakePackDescriptor<v3, v5>(
              pkg.get(), {}, {PDOpt::WithFluxes, PDOpt::Flatten});
          auto pack = desc.GetPack(&mesh_data);

          int lo = pack.GetLowerBoundHost(2);
          int hi = pack.GetUpperBoundHost(2);
          REQUIRE(lo == 4 - 1 + 4 + 1); // lo = index in flat pack where block 2 starts.
                                        // v3 and v5 = 4 total var components
          REQUIRE(hi == lo); // hi = index in flat pack where block 2 ends. Only v3
                             // present, so only 1 var
          AND_THEN("The flattened sparse pack can access vars correctly") {
            const int nblocks_and_vars = pack.GetMaxNumberOfVars();
            int nwrong = 0;
            par_reduce(
                loop_pattern_mdrange_tag, "test flat", DevExecSpace(), 0,
                nblocks_and_vars - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                KOKKOS_LAMBDA(int v, int k, int j, int i, int &ltot) {
                  int n = i + 1e1 * j + 1e2 * k;
                  if (n != (static_cast<int>(pack(v, k, j, i)) % 1000)) {
                    ltot += 1;
                  }
                },
                nwrong);
            REQUIRE(nwrong == 0);
          }
        }
      }

      THEN("A sparse pack correctly loads this data and can be read from v3 on a single "
           "block") {
        auto desc = parthenon::MakePackDescriptor<v5, v3>(pkg.get());
        auto sparse_pack = desc.GetPack(block_list[0]->meshblock_data.Get().get());
        const int v = 1; // v3 is the second variable in the loop above so v = 1 there
        int nwrong = 0;
        int b = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check vector", DevExecSpace(), kb.s, kb.e, jb.s,
            jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(b, v3());
              int hi = sparse_pack.GetUpperBound(b, v3());
              // Make sure we can pull out pointers to the variables
              auto [pv3, pv5] = sparse_pack.GetPtrs(b, parthenon::TopologicalElement::CC,
                                                    k, j, i, v3(), v5());
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e4 * c + 1e5 * v + 1e3 * b;
                if (n != sparse_pack(b, lo + c, k, j, i)) ltot += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A sparse pack correctly reads based on a regex variable") {
        auto desc =
            parthenon::MakePackDescriptor<parthenon::variable_names::any_nonautoflux>(
                pkg.get());
        auto sparse_pack = desc.GetPack(&mesh_data);

        auto desc_notype = MakePackDescriptor(
            pkg.get(), std::vector<std::pair<std::string, bool>>{
                           {"^(?!" + parthenon::internal_fluxname +
                                parthenon::internal_varname_seperator + ").+",
                            true}});
        auto sparse_pack_notype = desc_notype.GetPack(&mesh_data);
        auto pack_map = desc_notype.GetMap();
        parthenon::PackIdx iall(pack_map[".*"]);

        int nwrong = 0;
        par_reduce(
            loop_pattern_mdrange_tag, "check all", DevExecSpace(), 0, NBLOCKS - 1, kb.s,
            kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &ltot) {
              int lo = sparse_pack.GetLowerBound(
                  b, parthenon::variable_names::any_nonautoflux());
              int hi = sparse_pack.GetUpperBound(
                  b, parthenon::variable_names::any_nonautoflux());
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e3 * b;
                if (std::abs(n - std::fmod(sparse_pack(b, lo + c, k, j, i), 1e4)) >
                    1.e-12)
                  ltot += 1;
              }
              lo = sparse_pack_notype.GetLowerBound(b, iall);
              hi = sparse_pack_notype.GetUpperBound(b, iall);
              for (int c = 0; c <= hi - lo; ++c) {
                Real n = i + 1e1 * j + 1e2 * k + 1e3 * b;
                if (std::abs(n - std::fmod(sparse_pack_notype(b, lo + c, k, j, i), 1e4)) >
                    1.e-12)
                  ltot += 1;
                sparse_pack_notype(b, lo + c, k, j, i) = 0.0;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }

      THEN("A sparse pack built with a subset of blocks is the right size") {
        auto desc =
            parthenon::MakePackDescriptor<parthenon::variable_names::any_nonautoflux>(
                pkg.get());
        std::vector<bool> include_blocks(NBLOCKS);
        for (int i = 0; i < NBLOCKS; i++)
          include_blocks[i] = (i % 2 == 0);
        auto sparse_pack = desc.GetPack(&mesh_data, include_blocks);
        REQUIRE(sparse_pack.GetNBlocks() == NBLOCKS / 2 + 1);
      }

      THEN("Sparse packs built with a subset of blocks are correctly stored in the "
           "cache") {
        auto desc =
            parthenon::MakePackDescriptor<parthenon::variable_names::any_nonautoflux>(
                pkg.get());

        std::vector<bool> include_blocks(NBLOCKS);
        // This should be a new pack in the cache
        for (int i = 0; i < NBLOCKS; i++)
          include_blocks[i] = (i % 2 == 0);
        auto sparse_pack = desc.GetPack(&mesh_data, include_blocks);

        // This should be a new pack in the cache, since it has the
        // same descriptor but a different set of blocks
        for (int i = 0; i < NBLOCKS; i++)
          include_blocks[i] = (i % 2 == 1);
        auto sparse_pack2 = desc.GetPack(&mesh_data, include_blocks);

        // This should be the same as the first pack (in both block
        // list and pack descriptor), so doesn't result in anything
        // new in the cache. Also provides an example of defining a
        // block selector functor
        int b = 0;
        auto block_selector = [&b](MeshBlockData<Real> *pmbd) {
          bool even = (b % 2 == 0);
          b++;
          return even;
        };
        auto sparse_pack3 = desc.GetPack(&mesh_data, block_selector);

        // so there should only be two packs in the cache
        REQUIRE(mesh_data.GetSparsePackCache().size() == 2);
      }
    }
  }
}
