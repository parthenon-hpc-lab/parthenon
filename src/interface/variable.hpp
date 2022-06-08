//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_VARIABLE_HPP_
#define INTERFACE_VARIABLE_HPP_

///
/// A Variable type for Placebo-K.
/// Builds on ParArrayNDs
/// Date: August 21, 2019
///
///
/// The variable class typically contains state data for the
/// simulation but can also hold non-mesh-based data such as physics
/// parameters, etc.  It inherits the ParArrayND class, which is used
/// for actural data storage and generation

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/cc/bvals_cc.hpp"
#include "defs.hpp"
#include "interface/metadata.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class MeshBlock;
template <typename T>
class MeshBlockData;

static constexpr int InvalidSparseID = std::numeric_limits<int>::min();

inline std::string MakeVarLabel(const std::string &base_name, int sparse_id) {
  return base_name +
         (sparse_id == InvalidSparseID ? "" : "_" + std::to_string(sparse_id));
}

struct VariableState : public empty_state_t {
  KOKKOS_INLINE_FUNCTION
  VariableState(double alloc, double dealloc)
      : sparse_allocation_threshold(alloc), sparse_deallocation_threshold(dealloc) {}

  KOKKOS_DEFAULTED_FUNCTION
  VariableState() = default;
  KOKKOS_INLINE_FUNCTION
  explicit VariableState(const empty_state_t &) {}

  double sparse_allocation_threshold;
  double sparse_deallocation_threshold;
};

template <typename T>
class CellVariable {
  // so that MeshBlock and MeshBlockData can call Allocate* and Deallocate
  friend class MeshBlock;
  friend class MeshBlockData<T>;

 public:
  CellVariable<T>(const std::string &base_name, const Metadata &metadata, int sparse_id,
                  std::weak_ptr<MeshBlock> wpmb);

  // copy fluxes and boundary variable from src CellVariable (shallow copy)
  void CopyFluxesAndBdryVar(const CellVariable<T> *src);

  // make a new CellVariable based on an existing one
  std::shared_ptr<CellVariable<T>> AllocateCopy(std::weak_ptr<MeshBlock> wpmb);

  // accessors
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION auto &operator()(Args... args) {
    assert(IsAllocated());
    return data(std::forward<Args>(args)...);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetDim(const int i) const {
    // we can't query data.GetDim() here because data may be unallocated
    assert(0 < i && i <= 6 && "ParArrayNDs are max 6D");
    return dims_[i - 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetCoarseDim(const int i) const {
    // we can't query coarse_s.GetDim() here because it may be unallocated
    assert(0 < i && i <= 6 && "ParArrayNDs are max 6D");
    return coarse_dims_[i - 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto NumComponents() const { return dims_[5] * dims_[4] * dims_[3]; }

  ///< retrieve label for variable
  inline const auto label() const { return MakeVarLabel(base_name_, sparse_id_); }
  inline const auto base_name() const { return base_name_; }

  ///< retrieve metadata for variable
  inline Metadata metadata() const { return m_; }

  /// Get Sparse ID (InvalidSparseID if not sparse)
  inline int GetSparseID() const { return IsSparse() ? sparse_id_ : InvalidSparseID; }

  inline bool IsSparse() const { return m_.IsSet(Metadata::Sparse); }

  inline std::string getAssociated() { return m_.getAssociated(); }

  /// return information string
  std::string info();

#ifdef ENABLE_SPARSE
  inline bool IsAllocated() const { return is_allocated_; }
#else
  inline constexpr bool IsAllocated() const { return true; }
#endif

  /// Repoint vbvar's var_cc array at the current variable
  inline void resetBoundary() { vbvar->var_cc = data; }

  inline bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  ParArrayND<T> data;
  ParArrayND<T> flux[4];  // used for boundary calculation
  ParArrayND<T> coarse_s; // used for sending coarse boundary calculation
  // used in case of cell boundary communication
  std::shared_ptr<CellCenteredBoundaryVariable> vbvar;
  bool mpiStatus = false;

  int dealloc_count = 0;

 private:
  // allocate data, fluxes, and boundary variable
  void Allocate(std::weak_ptr<MeshBlock> wpmb);

  // allocate data only
  void AllocateData();

  // deallocate data, fluxes, and boundary variable
  void Deallocate();

  /// allocate fluxes (if Metadata::WithFluxes is set) and coarse data if
  /// (Metadata::FillGhost is set)
  void AllocateFluxesAndCoarse(std::weak_ptr<MeshBlock> wpmb);

  Metadata m_;
  const std::string base_name_;
  const int sparse_id_;
  const std::array<int, 6> dims_, coarse_dims_;

  bool is_allocated_ = false;
  ParArray7D<T> flux_data_; // unified par array for the fluxes
};

///
/// FaceVariable extends the FaceField struct to include the metadata
/// and label so that we can refer to variables by name.  Since Athena
/// currently only has scalar Face fields, we also only allow scalar
/// face fields
template <typename T>
class FaceVariable {
 public:
  /// Initialize a face variable
  FaceVariable(const std::string &label, const std::array<int, 6> ncells,
               const Metadata &metadata)
      : data(label, ncells[5], ncells[4], ncells[3], ncells[2], ncells[1], ncells[0]),
        dims_(ncells), m_(metadata), label_(label) {
    PARTHENON_REQUIRE_THROWS(!metadata.IsSet(Metadata::Sparse),
                             "Sparse not implemented yet for FaceVariable");
  }

  /// Create an alias for the variable by making a shallow slice with max dim
  FaceVariable(const std::string &label, FaceVariable<T> &src)
      : data(src.data), dims_(src.dims_), m_(src.m_), label_(label) {}

  std::shared_ptr<FaceVariable<T>> AllocateCopy(std::weak_ptr<MeshBlock> wpmb) {
    PARTHENON_THROW("FaceVariable::AllocateCopy is not implemented yet");
  }

  // KOKKOS_FUNCTION FaceVariable() = default;
  // KOKKOS_FUNCTION FaceVariable(const FaceVariable<T>& v) = default;
  // KOKKOS_FUNCTION ~FaceVariable() = default;

  ///< retrieve label for variable
  inline const std::string &label() const { return label_; }

  ///< retrieve metadata for variable
  inline const Metadata metadata() const { return m_; }

  /// return information string
  std::string info();

  // TODO(JMM): should this be 0,1,2?
  // Should we return the reference? Or something else?
  KOKKOS_FORCEINLINE_FUNCTION
  ParArrayND<T> &Get(int i) {
    assert(1 <= i && i <= 3);
    if (i == 1) return (data.x1f);
    if (i == 2)
      return (data.x2f);
    else // i == 3
      return (data.x3f);
  }
  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(int dir, Args... args) const {
    assert(1 <= dir && dir <= 3);
    if (dir == 1) return data.x1f(std::forward<Args>(args)...);
    if (dir == 2)
      return data.x2f(std::forward<Args>(args)...);
    else // dir == 3
      return data.x3f(std::forward<Args>(args)...);
  }

  inline bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  // to mimick interface of CellVariable, currently sparse FaceVariables are not
  // implemented
  inline bool IsSparse() const { return false; }
  inline int GetSparseID() const { return InvalidSparseID; }

  FaceArray<T> data;

 private:
  std::array<int, 6> dims_;
  Metadata m_;
  std::string label_;
};

///
/// EdgeVariable extends the EdgeField struct to include the metadata
/// and label so that we can refer to variables by name.  Since Athena
/// currently only has scalar Edge fields, we also only allow scalar
/// edge fields
template <typename T>
class EdgeVariable {
 public:
  /// Initialize an edge variable
  EdgeVariable(const std::string &label, const std::array<int, 6> ncells,
               const Metadata &metadata)
      : data(label, ncells[5], ncells[4], ncells[3], ncells[2], ncells[1], ncells[0]),
        dims_(ncells), m_(metadata), label_(label) {
    assert(!metadata.IsSet(Metadata::Sparse) &&
           "Sparse not implemented yet for FaceVariable");
  }

  /// Create an alias for the variable by making a shallow slice with max dim
  EdgeVariable(const std::string &label, EdgeVariable<T> &src)
      : data(src.data), dims_(src.dims_), m_(src.m_), label_(label) {}
  ///< retrieve metadata for variable
  inline const Metadata metadata() const { return m_; }

  inline bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }
  ///< retrieve label for variable
  inline std::string label() { return label_; }

  /// return information string
  std::string info();

  EdgeArray<Real> data;

 private:
  std::array<int, 6> dims_;
  Metadata m_;
  std::string label_;
};

template <typename T>
class ParticleVariable {
 public:
  /// Initialize a particle variable
  ParticleVariable(const std::string &label, const int npool, const Metadata &metadata);

  // accessors
  KOKKOS_FORCEINLINE_FUNCTION
  ParArrayND<T> &Get() { return data; }
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION auto &operator()(Args... args) {
    return data(std::forward<Args>(args)...);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetDim(const int i) const {
    PARTHENON_DEBUG_REQUIRE(0 < i && i <= 6, "ParArrayNDGenerics are max 6D");
    return dims_[i - 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto NumComponents() const {
    return dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1];
  }

  ///< retrieve metadata for variable
  inline const Metadata metadata() const { return m_; }

  inline bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  ///< retrieve label for variable
  inline const std::string label() const { return label_; }

  /// return information string
  std::string info() const;

 private:
  Metadata m_;
  std::string label_;
  std::array<int, 6> dims_;

 public:
  ParArrayND<T> data;
};

template <typename T>
using CellVariableVector = std::vector<std::shared_ptr<CellVariable<T>>>;
template <typename T>
using FaceVector = std::vector<std::shared_ptr<FaceVariable<T>>>;

template <typename T>
using MapToCellVars = std::map<std::string, std::shared_ptr<CellVariable<T>>>;
template <typename T>
using MapToFace = std::map<std::string, std::shared_ptr<FaceVariable<T>>>;

template <typename T>
using ParticleVariableVector = std::vector<std::shared_ptr<ParticleVariable<T>>>;
template <typename T>
using MapToParticle = std::map<std::string, std::shared_ptr<ParticleVariable<T>>>;

} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
