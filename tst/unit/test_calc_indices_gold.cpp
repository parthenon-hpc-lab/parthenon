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

// Golden-master safety net for CalcIndices.
//
// CalcIndices is a pure, deterministic, integer-valued function of
// (LogicalLocation, RegionSize, NeighborBlock, field concept, TopologicalElement,
// IndexRangeType, prores). This test pins its full output -- both the 6D box AND the
// 27-entry ownership mask embedded in the returned indexer -- plus a mesh/neighbor
// fingerprint, over a fixed set of meshes, so that any behavioral drift during the
// upcoming boundary-communication refactor is caught immediately with a precise,
// per-tuple diff.
//
// Meshes:
//   - dim_1 / dim_2 / dim_3: 1D/2D/3D statically-refined periodic meshes with multigrid
//     enabled, so every block's leaf neighbor list AND its five multigrid neighbor lists
//     (coarser/finer/same/self/composite-finer) are covered.
//   - forest_2tree: a 2D two-tree forest whose second tree is glued on with a rotated/
//     flipped orientation, so the logical coordinate transformation between trees is
//     non-trivial -- exercising the lcoord_trans branches of CalcIndices that the
//     single-tree periodic meshes never reach.
//
// The enumeration is over unique neighbor *relationships* (deduped across all neighbor
// lists, since the list is not an input to CalcIndices) crossed with concept x element x
// range-type x prores. Flux-correction paths only narrow the element set for a leaf
// neighbor, so they are a strict subset of what is already pinned here.
//
// Gold storage: the keys/boxes/ownership are stored in an HDF5 file
// (data/calc_indices_gold_v<VER>.h5), which keeps the file small and portable across
// platforms. Following the regression gold-standard convention, this file is NOT
// committed to the repository -- it is downloaded from a GitHub release asset and
// verified by hash at configure time (see CALC_INDICES_GOLD_* in the top-level
// CMakeLists.txt and tst/unit/data/README.md). Two conditions make this test a
// no-op-with-warning rather than a failure: built without HDF5 (ENABLE_HDF5 off), or the
// gold file absent (offline build / new version not yet published). Regenerate + publish
// a new version per the steps in tst/unit/data/README.md; regenerate locally with
//     PARTHENON_REGEN_GOLD=1 mpirun -np 1 <path>/unit_tests "[CalcIndices]"

#include <catch2/catch.hpp>

#include "config.hpp"

#ifndef ENABLE_HDF5

TEST_CASE("CalcIndices golden master", "[CalcIndices][bvals][MPI]") {
  WARN("CalcIndices golden-master test skipped: built without HDF5 (ENABLE_HDF5 off).");
}

#else // ENABLE_HDF5

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>

#include "application_input.hpp"
#include "basic_types.hpp"
#include "bvals/comms/bnd_info.hpp"
#include "bvals/neighbor_block.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/forest/forest_node.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "outputs/parthenon_hdf5_types.hpp"
#include "utils/error_checking.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
// CalcIndices has external linkage but is declared only in bnd_info.cpp. Forward-declare
// it here (without the default lcoord_trans argument, which is supplied at the call site)
// so the test can call the exact production routine under test.
SpatiallyMaskedIndexer6D
CalcIndices(const NeighborBlock &nb, MeshBlock *pmb,
            const std::shared_ptr<Variable<Real>> &v, TopologicalElement el,
            IndexRangeType ir_type, bool prores,
            const forest::LogicalCoordinateTransformation &lcoord_trans);
} // namespace parthenon

namespace {

using parthenon::ApplicationInput;
using parthenon::IndexRangeType;
using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::NeighborBlock;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::SpatiallyMaskedIndexer6D;
using parthenon::TopologicalElement;
using parthenon::Variable;
namespace HDF5 = parthenon::HDF5;
using HDF5::H5D;
using HDF5::H5F;
using HDF5::H5S;
using HDF5::H5T;

// Number of ghost zones used for every mesh in this test. Set on Globals before any mesh
// is constructed because the Mesh constructor (unlike ParthenonManager) does not set it.
constexpr int kNGhost = 2;

// What identifies one CalcIndices evaluation, stored as small integers (not a string).
//
// CalcIndices reads only mesh-derived quantities from (pmb, nb) -- both are functions of
// the mesh topology, which is pinned separately by the fingerprint (layer 1). So, given
// the mesh, the neighbor *relationship* is fully identified by
//   (gid_main, block_coarsenings_main, gid_neighbor, block_coarsenings_neighbor, offsets)
// and the box/ownership output is a function of that relationship crossed with
//   (concept, element, range-type, prores).
// The neighbor-list a NeighborBlock came from (leaf vs the five gmg lists) is NOT an
// input to CalcIndices, so relationships are deduped across all lists -- the same
// relationship appearing in several lists yields identical output and is pinned once.
//
// Key columns (all int32):
constexpr int kKeyLen = 11;
enum KeyCol {
  KC_GID_MAIN = 0,
  KC_BC_MAIN,
  KC_GID_NB,
  KC_BC_NB,
  KC_OX1,
  KC_OX2,
  KC_OX3,
  KC_CONCEPT, // index into the concept list
  KC_ELEMENT, // index into the element list
  KC_RANGE,   // index into the range-type list
  KC_PRORES   // 0 or 1
};

constexpr int kBoxLen = 12;
constexpr int kOwnLen = 27;
struct Record {
  std::array<std::int32_t, kKeyLen> key;
  std::array<std::int32_t, kBoxLen> box;
  std::array<std::uint8_t, kOwnLen> own;
};

//----------------------------------------------------------------------------------------
// A named field "concept" -- a standalone Variable<Real> whose metadata exercises one of
// the axes CalcIndices actually reads (tensor shape via GetDim(4..6) and the
// Fine/CommunicateOne/Flux flags). No field data is allocated: the Variable constructor
// only computes dims_, it does not call Allocate().
struct Concept {
  std::string name;
  std::shared_ptr<Variable<Real>> var;
};

//----------------------------------------------------------------------------------------
// Build the smallest static-refinement periodic deck that exercises same-level
// face/edge/corner neighbors, at least one c2f and one f2c interface, and periodic
// wrap-around, for the requested dimensionality. The mesh is 2 base blocks wide in each
// active direction (8 cells / 4-cell blocks) and one corner block is refined one level.
std::shared_ptr<ParameterInput> MakePin(const int dim) {
  const bool active2 = dim >= 2;
  const bool active3 = dim >= 3;
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  is << "refinement = static\n";
  // Enable multigrid so the constructor also populates the GMG neighbor lists
  // (coarser/finer/same/self/composite-finer). These are still built entirely in the
  // constructor -- no Initialize() and no field/buffer allocation is required.
  is << "multigrid = true\n";
  is << "nghost = " << kNGhost << "\n";
  is << "nx1 = 8\n";
  is << "x1min = 0.0\n";
  is << "x1max = 1.0\n";
  is << "ix1_bc = periodic\n";
  is << "ox1_bc = periodic\n";
  is << "nx2 = " << (active2 ? 8 : 1) << "\n";
  is << "x2min = 0.0\n";
  is << "x2max = 1.0\n";
  is << "ix2_bc = " << (active2 ? "periodic" : "outflow") << "\n";
  is << "ox2_bc = " << (active2 ? "periodic" : "outflow") << "\n";
  is << "nx3 = " << (active3 ? 8 : 1) << "\n";
  is << "x3min = 0.0\n";
  is << "x3max = 1.0\n";
  is << "ix3_bc = " << (active3 ? "periodic" : "outflow") << "\n";
  is << "ox3_bc = " << (active3 ? "periodic" : "outflow") << "\n";
  is << "<parthenon/meshblock>\n";
  is << "nx1 = 4\n";
  is << "nx2 = " << (active2 ? 4 : 1) << "\n";
  is << "nx3 = " << (active3 ? 4 : 1) << "\n";
  is << "<parthenon/static_refinement0>\n";
  is << "level = 1\n";
  is << "x1min = 0.0\n";
  is << "x1max = 0.5\n";
  if (active2) {
    is << "x2min = 0.0\n";
    is << "x2max = 0.5\n";
  }
  if (active3) {
    is << "x3min = 0.0\n";
    is << "x3max = 0.5\n";
  }
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  return pin;
}

//----------------------------------------------------------------------------------------
// Register zero fields so the mesh allocates no per-block field data.
Packages_t MakePackages() {
  Packages_t packages;
  packages.Add(std::make_shared<parthenon::StateDescriptor>("calc_indices_gold_test"));
  return packages;
}

//----------------------------------------------------------------------------------------
// Construct the mesh. Deliberately does NOT call mesh->Initialize(...), so no
// communication buffers are built; neighbor lists are still fully populated by the
// constructor via SetMeshBlockNeighbors.
std::shared_ptr<Mesh> MakeMesh(const int dim, ApplicationInput *app_in,
                               Packages_t &packages) {
  auto pin = MakePin(dim);
  return std::make_shared<Mesh>(pin.get(), app_in, packages, 0);
}

//----------------------------------------------------------------------------------------
// A minimal deck for the forest mesh: just the block size and (user) boundary names the
// forest constructor reads. The forest geometry itself is defined programmatically below.
std::shared_ptr<ParameterInput> MakeForestPin() {
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  is << "nghost = " << kNGhost << "\n";
  is << "nx1 = 8\n";
  is << "nx2 = 8\n";
  is << "nx3 = 1\n";
  is << "<parthenon/meshblock>\n";
  is << "nx1 = 4\n";
  is << "nx2 = 4\n";
  is << "nx3 = 1\n";
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  return pin;
}

//----------------------------------------------------------------------------------------
// Build a 2D two-tree forest in which the second tree is glued to the first along a
// shared edge with a rotated/flipped node ordering, so the logical coordinate
// transformation between the trees is non-trivial (dir_connection reorders and/or
// dir_flip is set). This exercises the lcoord_trans branches of CalcIndices (the
// BoundaryExteriorRecv transform and the InverseTransform of the element) that the
// single-tree periodic deck meshes never reach. This constructor does NOT enable
// multigrid, so this mesh contributes only leaf neighbor relationships.
//
// Face node ordering convention (see forest_topology.hpp):
//   2---3      X0: 0->1
//   |   |      X1: 0->2
//   0---1
// Tree 0 occupies the unit square with nodes at its corners. Tree 1 shares tree 0's
// right edge (nodes 1,3). By listing tree 1's nodes so that the shared edge runs along
// tree 1's X1 axis (rather than X0) and in reversed order, the derived transform both
// swaps the logical directions and flips one -- a rotation+flip.
std::shared_ptr<Mesh> MakeForestMesh(ApplicationInput *app_in, Packages_t &packages) {
  namespace forest = parthenon::forest;
  using forest::Node;
  using ar3_t = std::array<Real, 3>;

  // Shared nodes between the two trees are 1 and 3 (tree 0's right edge).
  std::unordered_map<int, std::shared_ptr<Node>> n;
  n[0] = Node::create(0, {0.0, 0.0});
  n[1] = Node::create(1, {1.0, 0.0});
  n[2] = Node::create(2, {0.0, 1.0});
  n[3] = Node::create(3, {1.0, 1.0});
  // Extra nodes for tree 1, placed to the right.
  n[4] = Node::create(4, {2.0, 0.0});
  n[5] = Node::create(5, {2.0, 1.0});

  forest::ForestDefinition forest_def;
  // Tree 0: standard orientation on the unit square.
  forest_def.AddFace(0, {n[0], n[1], n[2], n[3]}, ar3_t{0.0, 0.0, 0.0},
                     ar3_t{1.0, 1.0, 1.0});
  // Tree 1: node slots {0,1,2,3} = {n3, n1, n5, n4}. Face slot layout (see
  // forest_topology.hpp) is slot0=(-,-), slot1=(+,-), slot2=(-,+), slot3=(+,+), with the
  // X0 edge along slots 0->1 and the X1 edge along slots 0->2. The edge shared with tree
  // 0 is {n1,n3}: on tree 0 it runs along X1 (its +x face), but here n3 and n1 sit in
  // slots 0 and 1, so on tree 1 it runs along X0 -- the logical directions are swapped.
  // The slot0->slot1 order (n3->n1) is opposite tree 0's edge order, adding a flip.
  // Together this yields a non-trivial (rotation + flip) logical coordinate
  // transformation.
  forest_def.AddFace(1, {n[3], n[1], n[5], n[4]}, ar3_t{1.0, 0.0, 0.0},
                     ar3_t{2.0, 1.0, 1.0});

  // Close off the remaining outer edges with user BCs so the forest is well-posed.
  using edge_t = forest::Edge;
  forest_def.AddBC(edge_t({n[0], n[1]}));
  forest_def.AddBC(edge_t({n[0], n[2]}));
  forest_def.AddBC(edge_t({n[2], n[3]}));
  forest_def.AddBC(edge_t({n[1], n[4]}));
  forest_def.AddBC(edge_t({n[4], n[5]}));
  forest_def.AddBC(edge_t({n[3], n[5]}));

  auto pin = MakeForestPin();
  return std::make_shared<Mesh>(pin.get(), app_in, packages, forest_def);
}

// True if any cross-tree neighbor of any block carries a non-identity logical coordinate
// transformation (a genuine rotation/flip). Used to assert the forest mesh actually
// exercises the transform paths, independent of internal orientation conventions.
bool HasNonTrivialTransform(const std::shared_ptr<Mesh> &mesh) {
  const parthenon::forest::LogicalCoordinateTransformation identity;
  for (const auto &pmb : mesh->block_list) {
    for (const auto &nb : pmb->GetNeighbors()) {
      const auto &t = nb.lcoord_trans;
      if (t.dir_connection != identity.dir_connection || t.dir_flip != identity.dir_flip)
        return true;
    }
  }
  return false;
}

//----------------------------------------------------------------------------------------
// Build one standalone Variable<Real> per field concept CalcIndices distinguishes. A
// valid (non-expired) MeshBlock weak_ptr is required only so the mesh-tied metadata can
// resolve its array dimensions; no allocation is performed. The dims/flags read by
// CalcIndices are independent of which block is used here.
std::vector<Concept> MakeConcepts(const std::shared_ptr<MeshBlock> &pmb) {
  std::vector<Concept> concepts;
  auto make = [&](const std::string &name, const Metadata &m) {
    concepts.push_back({name, std::make_shared<Variable<Real>>(
                                  name, m, parthenon::InvalidSparseID, pmb)});
  };

  using F = Metadata;
  make("cc", Metadata({F::Cell, F::Independent, F::FillGhost}));
  make("cc_tensor",
       Metadata({F::Cell, F::Independent, F::FillGhost}, std::vector<int>{3}));
  make("fine", Metadata({F::Cell, F::Independent, F::FillGhost, F::Fine}));
  make("comm_one", Metadata({F::Cell, F::Independent, F::FillGhost, F::CommunicateOne}));

  // Flux concept: the flux metadata derived from a WithFluxes cell variable carries the
  // Flux (and Face) flags, driving CalcIndices' flux branch.
  Metadata with_fluxes({F::Cell, F::Independent, F::WithFluxes});
  auto flux_m = with_fluxes.GetSPtrFluxMetadata();
  concepts.push_back({"flux", std::make_shared<Variable<Real>>(
                                  "flux", *flux_m, parthenon::InvalidSparseID, pmb)});
  return concepts;
}

//----------------------------------------------------------------------------------------
// The enumeration axes, in a fixed order. The index into these tables is what is stored
// in the corresponding key column, so the order must stay stable across regenerations.
using TE = TopologicalElement;
constexpr std::array<TE, 8> kElements{TE::CC, TE::F1, TE::F2, TE::F3,
                                      TE::E1, TE::E2, TE::E3, TE::NN};
constexpr std::array<IndexRangeType, 4> kRangeTypes{
    IndexRangeType::BoundaryInteriorSend, IndexRangeType::BoundaryExteriorRecv,
    IndexRangeType::InteriorSend, IndexRangeType::InteriorRecv};

std::string PadInt(long v, int width) {
  const bool neg = v < 0;
  std::string digits = std::to_string(neg ? -v : v);
  while (static_cast<int>(digits.size()) < width)
    digits = "0" + digits;
  return (neg ? "-" : "+") + digits;
}

std::string ElName(TopologicalElement el) {
  using TE = TopologicalElement;
  switch (el) {
  case TE::CC:
    return "CC";
  case TE::F1:
    return "F1";
  case TE::F2:
    return "F2";
  case TE::F3:
    return "F3";
  case TE::E1:
    return "E1";
  case TE::E2:
    return "E2";
  case TE::E3:
    return "E3";
  case TE::NN:
    return "NN";
  }
  return "??";
}

std::string IrName(IndexRangeType ir) {
  switch (ir) {
  case IndexRangeType::BoundaryInteriorSend:
    return "BndIntSend";
  case IndexRangeType::BoundaryExteriorRecv:
    return "BndExtRecv";
  case IndexRangeType::InteriorSend:
    return "IntSend";
  case IndexRangeType::InteriorRecv:
    return "IntRecv";
  }
  return "??";
}

// Stable human-readable descriptor of a neighbor, used only in the topology fingerprint
// (layer 1). Independent of iteration order.
std::string NeighborKey(const NeighborBlock &nb) {
  const auto &t = nb.lcoord_trans;
  std::ostringstream os;
  os << "off=(" << PadInt(nb.offsets(parthenon::X1DIR), 1) << ","
     << PadInt(nb.offsets(parthenon::X2DIR), 1) << ","
     << PadInt(nb.offsets(parthenon::X3DIR), 1) << ") lev=" << PadInt(nb.loc.level(), 2)
     << " nloc=(" << PadInt(nb.loc.lx1(), 3) << "," << PadInt(nb.loc.lx2(), 3) << ","
     << PadInt(nb.loc.lx3(), 3) << ") ngid=" << PadInt(nb.gid, 4) << " ncoarsen="
     << nb.block_coarsenings
     // Include the logical coordinate transformation: it is an input to CalcIndices for
     // receive ranges and is non-trivial for the multi-tree forest mesh.
     << " lct_dir=(" << t.dir_connection[0] << "," << t.dir_connection[1] << ","
     << t.dir_connection[2] << ") lct_flip=(" << t.dir_flip[0] << "," << t.dir_flip[1]
     << "," << t.dir_flip[2] << ")";
  return os.str();
}

// Flatten the returned indexer's 6D box into 12 ints (s0,e0,...,s5,e5).
std::array<std::int32_t, kBoxLen> BoxOf(const SpatiallyMaskedIndexer6D &idx) {
  return {idx.StartIdx<0>(), idx.EndIdx<0>(), idx.StartIdx<1>(), idx.EndIdx<1>(),
          idx.StartIdx<2>(), idx.EndIdx<2>(), idx.StartIdx<3>(), idx.EndIdx<3>(),
          idx.StartIdx<4>(), idx.EndIdx<4>(), idx.StartIdx<5>(), idx.EndIdx<5>()};
}

// Flatten the ownership mask embedded in the returned indexer into 27 bytes in a fixed
// (ox1,ox2,ox3) order. This is genuine CalcIndices output (see
// GetIndexRangeMaskFromOwnership) and is exercised for the receive range-types.
std::array<std::uint8_t, kOwnLen> OwnOf(const SpatiallyMaskedIndexer6D &idx) {
  const auto &own = idx.GetOwnership();
  std::array<std::uint8_t, kOwnLen> out{};
  int n = 0;
  for (int ox1 = -1; ox1 <= 1; ++ox1)
    for (int ox2 = -1; ox2 <= 1; ++ox2)
      for (int ox3 = -1; ox3 <= 1; ++ox3)
        out[n++] = own(ox1, ox2, ox3) ? 1 : 0;
  return out;
}

std::string BoxToString(const std::array<std::int32_t, kBoxLen> &b) {
  std::ostringstream os;
  for (int d = 0; d < kBoxLen; d += 2)
    os << "[" << PadInt(b[d], 3) << "," << PadInt(b[d + 1], 3) << "]";
  return os.str();
}

std::string OwnToString(const std::array<std::uint8_t, kOwnLen> &o) {
  std::string s;
  for (auto v : o)
    s += v ? "1" : "0";
  return s;
}

// Render an integer key row back to the human-readable descriptor used in mismatch
// messages. concept_names indexes KC_CONCEPT.
std::string KeyToString(const std::array<std::int32_t, kKeyLen> &k,
                        const std::vector<std::string> &concept_names) {
  std::ostringstream os;
  os << "gid=" << PadInt(k[KC_GID_MAIN], 4) << " bc=" << k[KC_BC_MAIN]
     << " | ngid=" << PadInt(k[KC_GID_NB], 4) << " nbc=" << k[KC_BC_NB] << " off=("
     << PadInt(k[KC_OX1], 1) << "," << PadInt(k[KC_OX2], 1) << "," << PadInt(k[KC_OX3], 1)
     << ") | " << concept_names[k[KC_CONCEPT]] << " | "
     << ElName(kElements[k[KC_ELEMENT]]) << " | " << IrName(kRangeTypes[k[KC_RANGE]])
     << " | prores=" << k[KC_PRORES];
  return os.str();
}

//----------------------------------------------------------------------------------------
// A labeled neighbor list. The label distinguishes the leaf same-level list (the one the
// regular boundary-comm and flux-correction paths walk) from the five multigrid lists.
struct NeighborView {
  std::string label;
  const std::vector<NeighborBlock> &list;
};

// All neighbor lists a block carries. Regular boundary comms and flux correction iterate
// only "leaf"; the GMG (multigrid) transfer operators iterate the five gmg_* lists. Any
// list that is empty for a given block simply contributes no records.
std::vector<NeighborView> NeighborViews(const MeshBlock *pmb) {
  return {
      {"leaf", pmb->GetNeighbors()},
      {"gmg_coarser", pmb->GetGMGCoarserNeighbors()},
      {"gmg_finer", pmb->GetGMGFinerNeighbors()},
      {"gmg_same", pmb->GetGMGSameNeighbors()},
      {"gmg_self", pmb->GetGMGSelfNeighbors()},
      {"gmg_composite_finer", pmb->GetGMGCompositeFinerNeighbors()},
  };
}

//----------------------------------------------------------------------------------------
// Every unique block reachable in the mesh, including the internal/coarse blocks that
// live only on the multigrid grids (and so are absent from the leaf block_list). Sorted
// by (block_coarsenings, gid, loc) -- a stable, unique key -- for deterministic order.
std::vector<std::shared_ptr<MeshBlock>> AllBlocks(const std::shared_ptr<Mesh> &mesh) {
  std::map<const MeshBlock *, std::shared_ptr<MeshBlock>> uniq;
  for (const auto &pmb : mesh->block_list)
    uniq[pmb.get()] = pmb;
  // The multigrid grids carry the internal/coarse blocks. They are only built when the
  // mesh was constructed with multigrid enabled (the deck meshes); the forest mesh has
  // none, and GetMultigridBlockPartitions would assert.
  if (mesh->multigrid) {
    for (int lvl = mesh->GetGMGMinLevel(); lvl <= mesh->GetGMGMaxLevel(); ++lvl) {
      for (const auto &part : mesh->GetMultigridBlockPartitions(lvl)) {
        for (const auto &pmb : part->block_list)
          uniq[pmb.get()] = pmb;
      }
    }
  }
  std::vector<std::shared_ptr<MeshBlock>> blocks;
  blocks.reserve(uniq.size());
  for (auto &[ptr, pmb] : uniq)
    blocks.push_back(pmb);
  std::sort(blocks.begin(), blocks.end(), [](const auto &a, const auto &b) {
    auto key = [](const std::shared_ptr<MeshBlock> &p) {
      return std::make_tuple(p->block_coarsenings, p->gid, p->loc.level(), p->loc.lx1(),
                             p->loc.lx2(), p->loc.lx3());
    };
    return key(a) < key(b);
  });
  return blocks;
}

//----------------------------------------------------------------------------------------
// The mesh/neighbor fingerprint (layer 1). If this drifts, the box gold is meaningless
// and must be regenerated deliberately. Kept as human-readable text -- it is small and
// topology is what a reviewer actually wants to eyeball.
std::string BuildFingerprint(const std::shared_ptr<Mesh> &mesh,
                             const std::string &label) {
  std::vector<std::string> lines;
  for (const auto &pmb : AllBlocks(mesh)) {
    std::ostringstream head;
    head << "mesh=" << label << " gid=" << PadInt(pmb->gid, 4) << " loc=("
         << PadInt(pmb->loc.level(), 2) << "," << PadInt(pmb->loc.lx1(), 3) << ","
         << PadInt(pmb->loc.lx2(), 3) << "," << PadInt(pmb->loc.lx3(), 3)
         << ") ncoarsen=" << pmb->block_coarsenings;
    lines.push_back(head.str());

    std::vector<std::string> nbr_lines;
    for (const auto &view : NeighborViews(pmb.get())) {
      for (const auto &nb : view.list) {
        nbr_lines.push_back("  [" + view.label + "] " + NeighborKey(nb));
      }
    }
    std::sort(nbr_lines.begin(), nbr_lines.end());
    for (auto &l : nbr_lines)
      lines.push_back(l);
  }
  std::ostringstream out;
  for (const auto &l : lines)
    out << l << "\n";
  return out.str();
}

// One unique neighbor relationship: a (main block, neighbor block) pair. Two
// relationships with equal (gid_main, bc_main, gid_nb, bc_nb, offsets) produce identical
// CalcIndices output regardless of which neighbor list they were found in, so we keep one
// per triple.
struct Relationship {
  std::shared_ptr<MeshBlock> pmb;
  NeighborBlock nb;
  std::array<std::int32_t, 7> rel_key; // the relationship columns of a Record key
};

// Gather the unique relationships across the leaf list and all five multigrid lists.
std::vector<Relationship> UniqueRelationships(const std::shared_ptr<Mesh> &mesh) {
  std::map<std::array<std::int32_t, 7>, Relationship> uniq;
  for (const auto &pmb : AllBlocks(mesh)) {
    for (const auto &view : NeighborViews(pmb.get())) {
      for (const auto &nb : view.list) {
        std::array<std::int32_t, 7> rel{static_cast<std::int32_t>(pmb->gid),
                                        static_cast<std::int32_t>(pmb->block_coarsenings),
                                        static_cast<std::int32_t>(nb.gid),
                                        static_cast<std::int32_t>(nb.block_coarsenings),
                                        nb.offsets(parthenon::X1DIR),
                                        nb.offsets(parthenon::X2DIR),
                                        nb.offsets(parthenon::X3DIR)};
        uniq.emplace(rel, Relationship{pmb, nb, rel});
      }
    }
  }
  std::vector<Relationship> out;
  out.reserve(uniq.size());
  for (auto &[k, r] : uniq)
    out.push_back(r);
  return out;
}

//----------------------------------------------------------------------------------------
// The full CalcIndices matrix (layer 2) for one dimension: box + ownership per tuple over
// unique relationships x concepts x elements x range-types x prores. Records are sorted
// by their integer key so the ordering is independent of iteration order.
std::vector<Record> BuildRecords(const std::shared_ptr<Mesh> &mesh,
                                 std::vector<std::string> *concept_names) {
  std::vector<Record> records;
  bool captured_names = false;
  for (const auto &rel : UniqueRelationships(mesh)) {
    const auto concepts = MakeConcepts(rel.pmb);
    if (!captured_names) {
      concept_names->clear();
      for (const auto &c : concepts)
        concept_names->push_back(c.name);
      captured_names = true;
    }
    for (std::int32_t ci = 0; ci < static_cast<std::int32_t>(concepts.size()); ++ci) {
      for (std::int32_t ei = 0; ei < static_cast<std::int32_t>(kElements.size()); ++ei) {
        for (std::int32_t ri = 0; ri < static_cast<std::int32_t>(kRangeTypes.size());
             ++ri) {
          for (std::int32_t pr = 0; pr < 2; ++pr) {
            // Set lcoord_trans.ncell exactly as BndInfo::BndInfo does before calling
            // CalcIndices: it is the number of cells along the first dimension of the
            // array CalcIndices will index into (the coarse buffer when the neighbor is
            // coarser, otherwise the main array), and it is only consumed by the flipped
            // branch of the transform. The forest constructor leaves ncell unset, so
            // failing to set it here feeds CalcIndices an uninitialized value.
            auto lct = rel.nb.lcoord_trans;
            const bool nb_coarser = rel.nb.loc.level() < rel.pmb->loc.level() ||
                                    rel.nb.block_coarsenings > rel.pmb->block_coarsenings;
            lct.ncell = nb_coarser ? concepts[ci].var->GetCoarseDim(1)
                                   : concepts[ci].var->GetDim(1);
            auto idx =
                parthenon::CalcIndices(rel.nb, rel.pmb.get(), concepts[ci].var,
                                       kElements[ei], kRangeTypes[ri], pr != 0, lct);
            std::array<std::int32_t, kKeyLen> key{};
            for (int c = 0; c < 7; ++c)
              key[c] = rel.rel_key[c];
            key[KC_CONCEPT] = ci;
            key[KC_ELEMENT] = ei;
            key[KC_RANGE] = ri;
            key[KC_PRORES] = pr;
            records.push_back({key, BoxOf(idx), OwnOf(idx)});
          }
        }
      }
    }
  }
  std::sort(records.begin(), records.end(),
            [](const Record &a, const Record &b) { return a.key < b.key; });
  return records;
}

//----------------------------------------------------------------------------------------
// HDF5 gold I/O. The build system passes the data directory and the pinned gold version;
// the file itself (calc_indices_gold_v<VER>.h5) is downloaded from a release asset by the
// top-level CMakeLists.txt, not stored in the repository.
#ifndef CALC_INDICES_GOLD_DIR
#error "CALC_INDICES_GOLD_DIR must be defined by the build system"
#endif
#ifndef CALC_INDICES_GOLD_VER
#error "CALC_INDICES_GOLD_VER must be defined by the build system"
#endif

#define CALC_INDICES_STR_(x) #x
#define CALC_INDICES_STR(x) CALC_INDICES_STR_(x)

std::string GoldPath() {
  return std::string(CALC_INDICES_GOLD_DIR) + "/calc_indices_gold_v" +
         CALC_INDICES_STR(CALC_INDICES_GOLD_VER) + ".h5";
}

bool RegenRequested() { return std::getenv("PARTHENON_REGEN_GOLD") != nullptr; }

// Write one dimension's records (boxes + ownership + keys) as a group in the gold file,
// plus the fingerprint text as a dataset and the enum-structure hash as an attribute.
// Create a gzip-compressed, chunked dataset creation property list for a [n, ncol]
// dataset. The data is highly redundant, so deflate shrinks it dramatically.
HDF5::H5P DeflateProps(hsize_t n, hsize_t ncol) {
  HDF5::H5P props = HDF5::H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_CREATE));
  // Chunk over a block of rows (bounded so tiny dims don't make an oversized chunk).
  hsize_t chunk[2] = {std::min<hsize_t>(n, 8192), ncol};
  PARTHENON_HDF5_CHECK(H5Pset_chunk(props, 2, chunk));
  PARTHENON_HDF5_CHECK(H5Pset_deflate(props, 6));
  return props;
}

// Each row's identifying key is stored as compact int32 columns (see KeyCol) alongside
// its box/ownership, so the gold is self-describing: compare pairs rows by matching keys,
// not by position. The keys are ~15 small repetitive integers per row, which gzip
// crushes.
void WriteDimGroup(hid_t file, const std::string &group_name,
                   const std::vector<Record> &records, const std::string &fingerprint) {
  const hsize_t n = records.size();
  HDF5::H5G group = HDF5::MakeGroup(file, group_name);

  // Keys: [n, kKeyLen] int32, gzip-compressed.
  {
    std::vector<std::int32_t> flat(n * kKeyLen);
    for (hsize_t i = 0; i < n; ++i)
      for (int d = 0; d < kKeyLen; ++d)
        flat[i * kKeyLen + d] = records[i].key[d];
    hsize_t dims[2] = {n, kKeyLen};
    H5S space = H5S::FromHIDCheck(H5Screate_simple(2, dims, NULL));
    HDF5::H5P props = DeflateProps(n, kKeyLen);
    H5D dset = H5D::FromHIDCheck(H5Dcreate(group, "keys", H5T_NATIVE_INT32, space,
                                           H5P_DEFAULT, props, H5P_DEFAULT));
    PARTHENON_HDF5_CHECK(
        H5Dwrite(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()));
  }
  // Boxes: [n, 12] int32, gzip-compressed.
  {
    std::vector<std::int32_t> flat(n * kBoxLen);
    for (hsize_t i = 0; i < n; ++i)
      for (int d = 0; d < kBoxLen; ++d)
        flat[i * kBoxLen + d] = records[i].box[d];
    hsize_t dims[2] = {n, kBoxLen};
    H5S space = H5S::FromHIDCheck(H5Screate_simple(2, dims, NULL));
    HDF5::H5P props = DeflateProps(n, kBoxLen);
    H5D dset = H5D::FromHIDCheck(H5Dcreate(group, "boxes", H5T_NATIVE_INT32, space,
                                           H5P_DEFAULT, props, H5P_DEFAULT));
    PARTHENON_HDF5_CHECK(
        H5Dwrite(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()));
  }
  // Ownership: [n, 27] uint8, gzip-compressed.
  {
    std::vector<std::uint8_t> flat(n * kOwnLen);
    for (hsize_t i = 0; i < n; ++i)
      for (int d = 0; d < kOwnLen; ++d)
        flat[i * kOwnLen + d] = records[i].own[d];
    hsize_t dims[2] = {n, kOwnLen};
    H5S space = H5S::FromHIDCheck(H5Screate_simple(2, dims, NULL));
    HDF5::H5P props = DeflateProps(n, kOwnLen);
    H5D dset = H5D::FromHIDCheck(H5Dcreate(group, "ownership", H5T_NATIVE_UINT8, space,
                                           H5P_DEFAULT, props, H5P_DEFAULT));
    PARTHENON_HDF5_CHECK(
        H5Dwrite(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data()));
  }
  // Fingerprint: scalar variable-length string
  {
    const char *fp = fingerprint.c_str();
    H5S space = H5S::FromHIDCheck(H5Screate(H5S_SCALAR));
    H5T str_type = H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
    PARTHENON_HDF5_CHECK(H5Tset_size(str_type, H5T_VARIABLE));
    H5D dset = H5D::FromHIDCheck(H5Dcreate(group, "fingerprint", str_type, space,
                                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    PARTHENON_HDF5_CHECK(H5Dwrite(dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fp));
  }
}

// Read back one dimension's keys/box/ownership values plus the fingerprint from a group
// in the gold file.
struct GoldData {
  std::vector<std::array<std::int32_t, kKeyLen>> keys;
  std::vector<std::array<std::int32_t, kBoxLen>> boxes;
  std::vector<std::array<std::uint8_t, kOwnLen>> owns;
  std::string fingerprint;
};

GoldData ReadDimGroup(hid_t file, const std::string &group_name) {
  HDF5::H5G group =
      HDF5::H5G::FromHIDCheck(H5Gopen(file, group_name.c_str(), H5P_DEFAULT));

  H5D key_dset = H5D::FromHIDCheck(H5Dopen(group, "keys", H5P_DEFAULT));
  H5S key_space = H5S::FromHIDCheck(H5Dget_space(key_dset));
  hsize_t key_dims[2];
  H5Sget_simple_extent_dims(key_space, key_dims, NULL);
  const hsize_t n = key_dims[0];
  std::vector<std::int32_t> key_flat(n * kKeyLen);
  PARTHENON_HDF5_CHECK(H5Dread(key_dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                               key_flat.data()));

  H5D box_dset = H5D::FromHIDCheck(H5Dopen(group, "boxes", H5P_DEFAULT));
  std::vector<std::int32_t> box_flat(n * kBoxLen);
  PARTHENON_HDF5_CHECK(H5Dread(box_dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                               box_flat.data()));

  H5D own_dset = H5D::FromHIDCheck(H5Dopen(group, "ownership", H5P_DEFAULT));
  std::vector<std::uint8_t> own_flat(n * kOwnLen);
  PARTHENON_HDF5_CHECK(H5Dread(own_dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                               own_flat.data()));

  GoldData out;
  out.keys.resize(n);
  out.boxes.resize(n);
  out.owns.resize(n);
  for (hsize_t i = 0; i < n; ++i) {
    for (int d = 0; d < kKeyLen; ++d)
      out.keys[i][d] = key_flat[i * kKeyLen + d];
    for (int d = 0; d < kBoxLen; ++d)
      out.boxes[i][d] = box_flat[i * kBoxLen + d];
    for (int d = 0; d < kOwnLen; ++d)
      out.owns[i][d] = own_flat[i * kOwnLen + d];
  }

  H5T str_type = H5T::FromHIDCheck(H5Tcopy(H5T_C_S1));
  PARTHENON_HDF5_CHECK(H5Tset_size(str_type, H5T_VARIABLE));
  H5D fp_dset = H5D::FromHIDCheck(H5Dopen(group, "fingerprint", H5P_DEFAULT));
  char *fp = nullptr;
  H5S fp_space = H5S::FromHIDCheck(H5Dget_space(fp_dset));
  PARTHENON_HDF5_CHECK(H5Dread(fp_dset, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &fp));
  out.fingerprint = fp ? fp : "";
  PARTHENON_HDF5_CHECK(H5Dvlen_reclaim(str_type, fp_space, H5P_DEFAULT, &fp));

  return out;
}

//----------------------------------------------------------------------------------------
// Per-mesh bundle of everything we pin. `group` is the HDF5 group name / mesh label.
struct DimGold {
  std::string group;
  std::vector<Record> records;
  std::string fingerprint;
  std::vector<std::string> concept_names; // indexed by KC_CONCEPT, for messages
};

// Compare a freshly-generated dimension against the gold, failing on the first divergence
// with a precise, per-tuple message (key + gold vs test box/ownership). Rows are paired
// by matching their stored integer keys, so the comparison is order-independent and any
// change to the enumeration surfaces as a missing/extra key rather than a silent
// misalign.
void CompareDim(hid_t file, const DimGold &gen,
                const std::vector<std::string> &concept_names) {
  const GoldData gold = ReadDimGroup(file, gen.group);

  // Layer 1: the fingerprint is the tripwire -- if topology drifted, the box gold is
  // meaningless, so check it before the boxes.
  if (gold.fingerprint != gen.fingerprint) {
    std::istringstream gs(gold.fingerprint), ns(gen.fingerprint);
    std::string gl, nl;
    int line = 0;
    while (true) {
      const bool ghas = static_cast<bool>(std::getline(gs, gl));
      const bool nhas = static_cast<bool>(std::getline(ns, nl));
      ++line;
      if (!ghas && !nhas) break;
      if (ghas != nhas || gl != nl) {
        FAIL("Mesh fingerprint drift (mesh=" << gen.group << ") at line " << line
                                             << ":\n  gold: " << (ghas ? gl : "<eof>")
                                             << "\n  test: " << (nhas ? nl : "<eof>"));
      }
    }
  }

  // Layer 2: box + ownership, paired by key. Both gold and generated records are sorted
  // by key, so equal key sequences line up index-for-index; verify that and diff on
  // mismatch.
  REQUIRE(gold.keys.size() == gen.records.size());
  for (std::size_t i = 0; i < gen.records.size(); ++i) {
    const auto &t = gen.records[i];
    if (gold.keys[i] != t.key) {
      FAIL("Enumeration changed (mesh="
           << gen.group << ") at record " << i
           << ": gold and test keys differ. Regenerate the gold deliberately with "
              "PARTHENON_REGEN_GOLD=1.\n  key(gold): "
           << KeyToString(gold.keys[i], concept_names)
           << "\n  key(test): " << KeyToString(t.key, concept_names));
    }
    if (gold.boxes[i] != t.box || gold.owns[i] != t.own) {
      FAIL("CalcIndices gold mismatch (mesh="
           << gen.group << ") at record " << i
           << ":\n  key: " << KeyToString(t.key, concept_names) << "\n  box(gold): "
           << BoxToString(gold.boxes[i]) << "\n  box(test): " << BoxToString(t.box)
           << "\n  own(gold): " << OwnToString(gold.owns[i])
           << "\n  own(test): " << OwnToString(t.own));
    }
  }
  SUCCEED();
}

} // namespace

//----------------------------------------------------------------------------------------
TEST_CASE("CalcIndices golden master", "[CalcIndices][bvals][MPI]") {
  parthenon::Globals::nghost = kNGhost;
  auto app_in = std::make_shared<ApplicationInput>();
  auto packages = MakePackages();

  std::vector<DimGold> golds;
  auto add_mesh = [&](const std::string &label, const std::shared_ptr<Mesh> &mesh) {
    DimGold g;
    g.group = label;
    g.records = BuildRecords(mesh, &g.concept_names);
    g.fingerprint = BuildFingerprint(mesh, label);
    golds.push_back(std::move(g));
  };
  for (const int dim : {1, 2, 3})
    add_mesh("dim_" + std::to_string(dim), MakeMesh(dim, app_in.get(), packages));

  // A 2D two-tree forest with a rotated/flipped second tree, exercising the non-trivial
  // logical-coordinate-transformation paths of CalcIndices. Assert the transform is
  // actually non-trivial so this coverage cannot silently degrade to an identity map.
  {
    auto forest_mesh = MakeForestMesh(app_in.get(), packages);
    REQUIRE(HasNonTrivialTransform(forest_mesh));
    add_mesh("forest_2tree", forest_mesh);
  }

  if (RegenRequested()) {
    H5F file = H5F::FromHIDCheck(
        H5Fcreate(GoldPath().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    for (const auto &g : golds)
      WriteDimGroup(file, g.group, g.records, g.fingerprint);
    WARN("Regenerated gold file '" << GoldPath() << "'.");
    return;
  }

  // The gold file is downloaded from a release asset at configure time (see
  // CALC_INDICES_GOLD_* in the top-level CMakeLists.txt). If it is absent -- offline
  // build, download disabled, or a new version not yet published -- skip rather than
  // fail, since that is an infrastructure gap, not a code regression.
  if (!std::ifstream(GoldPath()).good()) {
    WARN("CalcIndices gold file '"
         << GoldPath()
         << "' not present; skipping. It is downloaded from a release asset when unit "
            "tests are configured (CALC_INDICES_GOLD_SYNC), or regenerate locally with "
            "PARTHENON_REGEN_GOLD=1.");
    return;
  }

  H5F file = H5F::FromHIDCheck(H5Fopen(GoldPath().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  for (const auto &g : golds)
    CompareDim(file, g, g.concept_names);
}

#endif // ENABLE_HDF5
