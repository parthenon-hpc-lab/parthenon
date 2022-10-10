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

template <typename T>
class Variable {
  // so that MeshBlock and MeshBlockData can call Allocate* and Deallocate
  friend class MeshBlock;
  friend class MeshBlockData<T>;

 public:
  Variable<T>(const std::string &base_name, const Metadata &metadata, int sparse_id,
              std::weak_ptr<MeshBlock> wpmb);

  // copy fluxes and boundary variable from src Variable (shallow copy)
  void CopyBdryVar(const Variable<T> *src);

  // make a new Variable based on an existing one
  std::shared_ptr<Variable<T>> AllocateCopy(std::weak_ptr<MeshBlock> wpmb);

  // accessors
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION auto &operator()(Args... args) {
    assert(IsAllocated());
    return data(std::forward<Args>(args)...);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetDim(const int i) const {
    // we can't query data.GetDim() here because data may be unallocated
    assert(0 < i && i <= MAX_VARIABLE_DIMENSION && "ParArrayNDs are max 6D");
    return dims_[i - 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetCoarseDim(const int i) const {
    // we can't query coarse_s.GetDim() here because it may be unallocated
    assert(0 < i && i <= MAX_VARIABLE_DIMENSION && "ParArrayNDs are max 6D");
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
  inline std::string GetFluxName() { return m_.GetFluxName(); }

  /// return information string
  std::string info();

#ifdef ENABLE_SPARSE
  inline bool IsAllocated() const { return is_allocated_; }
#else
  inline constexpr bool IsAllocated() const { return true; }
#endif

  inline bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  ParArrayND<T> data;
  ParArrayND<T> coarse_s; // used for sending coarse boundary calculation

  int dealloc_count = 0;

 private:
  // allocate data, fluxes, and boundary variable
  void Allocate(std::weak_ptr<MeshBlock> wpmb);

  // allocate data only
  void AllocateData();

  // deallocate data, fluxes, and boundary variable
  void Deallocate();

  /// Allocate coarse data if
  /// (Metadata::FillGhost is set)
  void AllocateCoarse(std::weak_ptr<MeshBlock> wpmb);

  Metadata m_;
  const std::string base_name_;
  const int sparse_id_;
  const std::array<int, MAX_VARIABLE_DIMENSION> dims_, coarse_dims_;

  bool is_allocated_ = false;
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
    PARTHENON_DEBUG_REQUIRE(0 < i && i <= MAX_VARIABLE_DIMENSION,
                            "ParArrayNDGenerics are max 6D");
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
  std::array<int, MAX_VARIABLE_DIMENSION> dims_;

 public:
  ParArrayND<T> data;
};

template <typename T>
using VariableVector = std::vector<std::shared_ptr<Variable<T>>>;
template <typename T>
using MapToVars = std::map<std::string, std::shared_ptr<Variable<T>>>;

template <typename T>
using ParticleVariableVector = std::vector<std::shared_ptr<ParticleVariable<T>>>;
template <typename T>
using MapToParticle = std::map<std::string, std::shared_ptr<ParticleVariable<T>>>;

} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
