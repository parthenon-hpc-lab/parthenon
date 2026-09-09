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

#ifndef TENSORS_TT_CONTAINER_HPP
#define TENSORS_TT_CONTAINER_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "mesh/domain.hpp"
#include "tensors/tt_types.hpp"

namespace parthenon {

// Forward declarations to avoid an interface <-> tensors include cycle. The
// heavy headers are included only in tt_container.cpp.
class MeshBlock;
class Mesh;
class StateDescriptor;
struct BlockListPartition;

// Per-block owning container for tensor-train fields, analogous to
// MeshBlockData for regular fields but holding TensorTrain objects (which have
// dynamic ranks and cannot live in fixed-size Variable<T> storage). Shaped to
// satisfy the DataCollection<T> contract: T(std::string), Initialize(src, vars,
// shallow), Contains(vars), CreatedFrom(vars). This lets MeshBlock hold a
// DataCollection<MeshBlockTTData> with an independent set of stages.
class MeshBlockTTData {
 public:
  using train_t = tensor2::TensorTrain;
  using train_ptr = std::shared_ptr<train_t>;

  MeshBlockTTData() = default;
  explicit MeshBlockTTData(std::string name) : stage_name_(std::move(name)) {}

  // DataCollection contract -------------------------------------------------

  // Initialize the "base" stage from a MeshBlock: build one all-ones-rank train
  // per registered TT field, deriving the spatial core dimension from the
  // block geometry, or initialize a named stage from another MeshBlockTTData
  // (deep copy of the trains unless shallow, which aliases the source pointers).
  template <class SRC_t>
  void Initialize(const std::shared_ptr<SRC_t> &src,
                  const std::vector<std::string> &fields = {},
                  const bool shallow = false) {
    if constexpr (std::is_same_v<SRC_t, MeshBlock>) {
      InitializeFromBlock_(src, fields);
    } else if constexpr (std::is_same_v<SRC_t, MeshBlockTTData>) {
      InitializeFromContainer_(src, fields, shallow);
    } else {
      static_assert(sizeof(SRC_t) == 0,
                    "Bad source type for MeshBlockTTData::Initialize.");
    }
  }

  // Do all of the requested field names already exist in this container?
  bool Contains(const std::vector<std::string> &names) const {
    for (const auto &n : names)
      if (map_.count(n) == 0) return false;
    return true;
  }

  // Was this container created from exactly the requested field set? Used by
  // DataCollection to detect a stage-key clash with a mismatched field list.
  bool CreatedFrom(const std::vector<std::string> &fields) const {
    return fields.size() == fields_in_.size() && Contains(fields);
  }

  // TT access ---------------------------------------------------------------

  bool Contains(const std::string &name) const { return map_.count(name) > 0; }

  train_ptr Get(const std::string &name) const {
    auto it = map_.find(name);
    PARTHENON_REQUIRE(it != map_.end(),
                      "Tensor-train field \"" + name + "\" not found in container.");
    return it->second;
  }

  // Rebind a field to a freshly-produced train (e.g. after rounding, when ranks
  // have changed and the train is a new object).
  void Set(const std::string &name, train_ptr t) {
    PARTHENON_REQUIRE(map_.count(name) > 0,
                      "Tensor-train field \"" + name + "\" not found in container.");
    map_[name] = std::move(t);
  }

  // Controlled iteration over (name, train_ptr) without exposing the raw map.
  template <class F>
  void ForEachField(F f) const {
    for (const auto &pair : map_)
      f(pair.first, pair.second);
  }

  // The trains as "variables", mirroring MeshBlockData::GetVariableVector.
  std::vector<train_ptr> GetVariableVector() const {
    std::vector<train_ptr> vars;
    vars.reserve(map_.size());
    for (const auto &pair : map_)
      vars.push_back(pair.second);
    return vars;
  }

  // Ordered list of field names held by this container (sorted, since the
  // backing map is ordered by name).
  std::vector<std::string> FieldNames() const {
    std::vector<std::string> names;
    names.reserve(map_.size());
    for (const auto &pair : map_)
      names.push_back(pair.first);
    return names;
  }

  int Size() const { return static_cast<int>(map_.size()); }
  const std::string &StageName() const { return stage_name_; }

  // Block/mesh accessors, mirroring MeshBlockData for symmetry. GetBlockPointer
  // + NumBlocks() are what the ForEachBlock SFINAE dispatch keys on, so keeping
  // them here means MeshBlockTTData can be packed like a single-block MeshData.
  std::shared_ptr<MeshBlock> GetBlockSharedPointer() const { return pmy_block_.lock(); }
  MeshBlock *GetBlockPointer() const { return GetBlockSharedPointer().get(); }
  MeshBlock *GetParentPointer() const { return GetBlockPointer(); }
  int NumBlocks() const { return 1; }
  void SetBlockPointer(std::weak_ptr<MeshBlock> pmb) { pmy_block_ = pmb; }

  // Spatial loop bounds, mirroring MeshBlockData. This is the first-class path
  // for getting index ranges for kernels over the (flattened) spatial core.
  // Defined out-of-line in tt_container.cpp because MeshBlockTTData is not a
  // class template, so these bodies would be checked eagerly against the
  // (here incomplete) MeshBlock type if defined inline.
  IndexRange GetBoundsI(const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const;
  IndexRange GetBoundsJ(const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const;
  IndexRange GetBoundsK(const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const;
  IndexRange GetBoundsI(CellLevel cl, const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const;
  IndexRange GetBoundsJ(CellLevel cl, const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const;
  IndexRange GetBoundsK(CellLevel cl, const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const;

 private:
  void InitializeFromBlock_(const std::shared_ptr<MeshBlock> &pmb,
                            const std::vector<std::string> &fields);
  void InitializeFromContainer_(const std::shared_ptr<MeshBlockTTData> &src,
                                const std::vector<std::string> &fields,
                                const bool shallow);

  std::string stage_name_{"base"};
  std::weak_ptr<MeshBlock> pmy_block_;
  std::vector<std::string> fields_in_;    // field set used to create this container
  std::map<std::string, train_ptr> map_;  // owning trains, one per field
};

// Mesh-partition container for tensor-train fields, analogous to MeshData:
// holds one MeshBlockTTData per block in the partition. Satisfies the same
// DataCollection contract so Mesh can hold a DataCollection<MeshTTData>.
class MeshTTData {
 public:
  MeshTTData() = default;
  explicit MeshTTData(std::string name) : stage_name_(std::move(name)) {}

  // Grid/partition identity, mirroring MeshData. Tensor-train physics usually
  // does not care about the grid type, but carrying it keeps MeshTTData
  // symmetric with MeshData and available if it becomes necessary (e.g. GMG).
  GridIdentifier grid;
  int partition = -1;

  // Initialize the TT stage over a partition of blocks. Mirrors
  // MeshData::Initialize: build this stage's MeshBlockTTData on each block's own
  // tt_block_data collection (from a fresh BlockListPartition) or from the
  // corresponding stage of another MeshTTData (stage-to-stage copy).
  void Initialize(const std::shared_ptr<BlockListPartition> &part,
                  const std::vector<std::string> &fields = {},
                  const bool shallow = false);
  void Initialize(const std::shared_ptr<MeshTTData> &src,
                  const std::vector<std::string> &fields = {},
                  const bool shallow = false);

  bool Contains(const std::vector<std::string> &names) const {
    for (const auto &bd : block_data_)
      if (!bd->Contains(names)) return false;
    return true;
  }
  bool CreatedFrom(const std::vector<std::string> &fields) const {
    return fields.size() == fields_in_.size();
  }

  // Spatial loop bounds, mirroring MeshData: delegate to the first block (all
  // blocks in a partition share the same logical cell shape).
  IndexRange GetBoundsI(const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const {
    if (!block_data_.empty()) return block_data_[0]->GetBoundsI(domain, el);
    return IndexRange{-1, -2};
  }
  IndexRange GetBoundsJ(const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const {
    if (!block_data_.empty()) return block_data_[0]->GetBoundsJ(domain, el);
    return IndexRange{-1, -2};
  }
  IndexRange GetBoundsK(const IndexDomain &domain,
                        TopologicalElement el = TopologicalElement::CC) const {
    if (!block_data_.empty()) return block_data_[0]->GetBoundsK(domain, el);
    return IndexRange{-1, -2};
  }

  int NumBlocks() const { return static_cast<int>(block_data_.size()); }
  // Ordered field names held by this partition (all blocks share the same set).
  std::vector<std::string> FieldNames() const {
    return block_data_.empty() ? std::vector<std::string>{}
                               : block_data_[0]->FieldNames();
  }
  const std::shared_ptr<MeshBlockTTData> &GetBlockData(int n) const {
    return block_data_[n];
  }
  std::shared_ptr<MeshBlockTTData> &GetBlockData(int n) { return block_data_[n]; }
  MeshBlockTTData *GetBlockDataRawPointer(int n) { return block_data_[n].get(); }
  const auto &GetAllBlockData() const { return block_data_; }
  const std::string &StageName() const { return stage_name_; }

  Mesh *GetMeshPointer() const { return pmy_mesh_; }
  Mesh *GetParentPointer() const { return pmy_mesh_; }
  void SetMeshPointer(Mesh *pmesh) { pmy_mesh_ = pmesh; }
  void SetMeshPointer(const std::shared_ptr<MeshTTData> &other) {
    pmy_mesh_ = other->GetMeshPointer();
  }

  bool ContainsGid(int gid) const;

 private:
  std::string stage_name_{"base"};
  Mesh *pmy_mesh_ = nullptr;
  std::vector<std::string> fields_in_;
  std::vector<std::shared_ptr<MeshBlockTTData>> block_data_;
};

} // namespace parthenon

#endif // TENSORS_TT_CONTAINER_HPP
