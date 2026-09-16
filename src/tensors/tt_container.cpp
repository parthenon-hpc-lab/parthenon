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

#include "tensors/tt_container.hpp"

#include <memory>
#include <string>
#include <vector>

#include "interface/data_collection.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "tensors/tt_field_metadata.hpp"

namespace parthenon {

void MeshBlockTTData::InitializeFromBlock_(const std::shared_ptr<MeshBlock> &pmb,
                                           const std::vector<std::string> &fields) {
  pmy_block_ = pmb;
  fields_in_ = fields;
  map_.clear();

  const auto &resolved_packages = pmb->resolved_packages;
  PARTHENON_REQUIRE(resolved_packages != nullptr,
                    "MeshBlockTTData::Initialize requires resolved packages on the "
                    "MeshBlock.");

  auto add_field = [&](const std::string &name, const TTFieldMetadata &d) {
    // The variable builds its own train from the field's registered structure (spatial
    // core sized from the block geometry via the metadata). New fields start at all-ones
    // bond rank; physics fills in structure.
    std::vector<int> ranks(d.NCores() - 1, 1);
    map_[name] = std::make_shared<var_t>(name, d, pmb, ranks);
  };

  if (fields.empty()) {
    // Copy everything registered.
    for (const auto &q : resolved_packages->AllTTFields())
      add_field(q.first, q.second);
  } else {
    for (const auto &name : fields)
      add_field(name, resolved_packages->GetTTFieldMetadata(name));
  }
}

void MeshBlockTTData::InitializeFromContainer_(
    const std::shared_ptr<MeshBlockTTData> &src, const std::vector<std::string> &fields,
    const bool shallow) {
  pmy_block_ = src->pmy_block_;
  map_.clear();

  // If no field subset is requested, take the whole source field set.
  std::vector<std::string> names = fields;
  if (names.empty()) {
    names.reserve(src->map_.size());
    for (const auto &pair : src->map_)
      names.push_back(pair.first);
  }
  fields_in_ = names;

  for (const auto &name : names) {
    auto svar = src->Get(name);
    if (shallow) {
      // Alias the source variable (same underlying object).
      map_[name] = svar;
    } else {
      map_[name] = std::make_shared<var_t>(svar->label(), svar->field_metadata(),
                                           svar->train().DeepCopy());
    }
  }
}

IndexRange MeshBlockTTData::GetBoundsI(const IndexDomain &domain,
                                       TopologicalElement el) const {
  return GetBlockPointer()->cellbounds.GetBoundsI(domain, el);
}
IndexRange MeshBlockTTData::GetBoundsJ(const IndexDomain &domain,
                                       TopologicalElement el) const {
  return GetBlockPointer()->cellbounds.GetBoundsJ(domain, el);
}
IndexRange MeshBlockTTData::GetBoundsK(const IndexDomain &domain,
                                       TopologicalElement el) const {
  return GetBlockPointer()->cellbounds.GetBoundsK(domain, el);
}
IndexRange MeshBlockTTData::GetBoundsI(CellLevel cl, const IndexDomain &domain,
                                       TopologicalElement el) const {
  return GetBlockPointer()->GetCellBounds(cl).GetBoundsI(domain, el);
}
IndexRange MeshBlockTTData::GetBoundsJ(CellLevel cl, const IndexDomain &domain,
                                       TopologicalElement el) const {
  return GetBlockPointer()->GetCellBounds(cl).GetBoundsJ(domain, el);
}
IndexRange MeshBlockTTData::GetBoundsK(CellLevel cl, const IndexDomain &domain,
                                       TopologicalElement el) const {
  return GetBlockPointer()->GetCellBounds(cl).GetBoundsK(domain, el);
}

void MeshTTData::Initialize(const std::shared_ptr<BlockListPartition> &part,
                            const std::vector<std::string> &fields, const bool shallow) {
  PARTHENON_REQUIRE(shallow == false,
                    "Can't shallow copy when the source is not another MeshTTData.");
  SetMeshPointer(part->pmesh);
  fields_in_ = fields;
  const auto &bl = part->block_list;
  block_data_.resize(bl.size());
  for (std::size_t i = 0; i < bl.size(); ++i) {
    // The per-block TT stage lives on the owning MeshBlock's collection, exactly
    // as MeshData stores MeshBlockData on the block.
    block_data_[i] = bl[i]->tt_block_data.Add(stage_name_, bl[i], fields);
  }
  grid = part->grid;
  partition = part->partition;
}

void MeshTTData::Initialize(const std::shared_ptr<MeshTTData> &src,
                            const std::vector<std::string> &fields, const bool shallow) {
  PARTHENON_REQUIRE(src != nullptr, "MeshTTData source must be non-null.");
  SetMeshPointer(src);
  fields_in_ = fields;
  const int nblocks = src->NumBlocks();
  block_data_.resize(nblocks);
  for (int i = 0; i < nblocks; ++i) {
    auto &src_block = src->GetBlockData(i);
    auto pmb = src_block->GetBlockSharedPointer();
    block_data_[i] = pmb->tt_block_data.Add(stage_name_, src_block, fields, shallow);
  }
  grid = src->grid;
  partition = src->partition;
}

bool MeshTTData::ContainsGid(int gid) const {
  for (const auto &bd : block_data_) {
    auto pmb = bd->GetBlockSharedPointer();
    if (pmb && pmb->gid == gid) return true;
  }
  return false;
}

} // namespace parthenon
