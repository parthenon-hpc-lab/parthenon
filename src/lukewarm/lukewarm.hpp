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
#ifndef LUKEWARM_LUKEWARM_HPP_
#define LUKEWARM_LUKEWARM_HPP_

// This file was made in part with generative AI.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm_variable.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// How many ids this rank has handed out. Every MeshData's copy of a lukewarm on a rank
// shares one, and the k-th id handed out on rank r is k * nranks + r, so ids are unique
// across ranks without any communication.
using LukewarmIdCounter = std::shared_ptr<std::uint64_t>;

// A lukewarm is a collection of particles owned by a MeshData rather than by any one
// MeshBlock. Particles are always stored densely: the live particles are exactly
// indices [0, NumParticles()), so there is no mask and no free list. Fields of the same
// data type share one (total components, Capacity()) array.
class Lukewarm {
 public:
  // Row of the int storage holding each particle's block: the local index, in the owning
  // MeshData, of the block the particle was last located in, or -1 before it is located,
  // and the gid of the block it is headed to while in transit. It is not a field, so no
  // pack or field name reaches it, but it moves with the particle like every other row.
  static constexpr int block_row = 0;

  Lukewarm(const std::string &label, const Metadata &m,
           LukewarmIdCounter next_id = std::make_shared<std::uint64_t>(0));

  const std::string &label() const { return label_; }
  const Metadata &metadata() const { return m_; }

  // Add a field. Its data type (Real, Integer, or UInt64) and shape come from the
  // metadata. Existing particles keep their values for existing fields.
  void Add(const std::string &label, const Metadata &m);

  bool Contains(const std::string &label) const { return types_.count(label) > 0; }
  std::vector<std::string> GetFieldNames() const;

  template <typename T>
  const std::shared_ptr<LukewarmVariable<T>> &GetVarPtr(const std::string &label) const {
    const auto &storage = GetStorage_<T>();
    const auto itr = storage.index.find(label);
    PARTHENON_REQUIRE_THROWS(itr != storage.index.end(), "Lukewarm " + label_ +
                                                             " has no field " + label +
                                                             " of the requested type");
    return storage.vars[itr->second];
  }
  template <typename T>
  LukewarmVariable<T> &Get(const std::string &label) const {
    return *GetVarPtr<T>(label);
  }

  // Every field of type T, in storage order
  template <typename T>
  const std::vector<std::shared_ptr<LukewarmVariable<T>>> &GetVariables() const {
    return GetStorage_<T>().vars;
  }
  // All components of every field of type T, with extents (components, Capacity()). For
  // int, that includes block_row.
  template <typename T>
  const ParArray2DRaw<T> &GetData() const {
    return GetStorage_<T>().data;
  }
  template <typename T>
  int NumComponents() const {
    return GetStorage_<T>().ncomp;
  }

  int NumParticles() const { return num_particles_; }
  int Capacity() const { return capacity_; }
  // Make room for at least n particles without moving existing ones
  void Reserve(const int n);

  // Append n new particles and give each a unique id. They start unlocated, and the
  // caller sets every other field.
  IndexRange Create(const int n);
  // Append n particles without giving them ids, e.g., to receive particles that already
  // have them. Every field is left for the caller to set.
  IndexRange Append(const int n);
  // Append counts(i) particles on behalf of each item i (a cell, say). Returns
  // counts.size() + 1 offsets, with item i's new particles at [offsets(i), offsets(i+1)).
  ParArray1DRaw<int> Create(const ParArray1DRaw<int> &counts);

  // Remove every particle, keeping the storage
  void Clear();

  // Set marks(n) = 1 in a kernel to remove particle n at the next RemoveMarked()
  const ParArray1DRaw<int> &GetRemovalMarks() const { return marks_; }
  // Remove every marked particle, moving particles from the end into the holes so
  // storage stays dense. Survivors may change index. Clears all marks.
  void RemoveMarked();

 private:
  template <typename T>
  struct Storage {
    ParArray2DRaw<T> data;
    int ncomp = 0;
    std::vector<std::shared_ptr<LukewarmVariable<T>>> vars;
    std::map<std::string, std::size_t> index;
  };
  using Storages_t = std::tuple<Storage<Real>, Storage<int>, Storage<std::uint64_t>>;

  template <typename T>
  Storage<T> &GetStorage_() {
    return std::get<LukewarmTypes::GetIdx<T>()>(storages_);
  }
  template <typename T>
  const Storage<T> &GetStorage_() const {
    return std::get<LukewarmTypes::GetIdx<T>()>(storages_);
  }
  template <typename T>
  void Add_(const std::string &label, const Metadata &m);
  // Resize every storage array to the current component counts and capacity, keeping
  // contents, and point each field's data at its rows
  void Reallocate_();

  std::string label_;
  Metadata m_;
  int num_particles_ = 0;
  int capacity_ = 0;
  LukewarmIdCounter next_id_;
  Storages_t storages_;
  ParArray1DRaw<int> marks_;
  std::map<std::string, MetadataFlag> types_;
};

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_HPP_
