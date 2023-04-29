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
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "interface/metadata.hpp"
#include "parthenon_arrays.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"
#include "utils/unique_id.hpp"

namespace parthenon { 

class MeshBlock;
template <typename T>
class MeshBlockData;

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
  void CopyFluxesAndBdryVar(const Variable<T> *src);

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

  /// Refinement functions owned in metadata
  inline bool IsRefined() const { return m_.IsRefined(); }
  inline const refinement::RefinementFunctions_t &GetRefinementFunctions() const {
    return m_.GetRefinementFunctions();
  }

  /// Get Sparse ID (InvalidSparseID if not sparse)
  inline int GetSparseID() const { return IsSparse() ? sparse_id_ : InvalidSparseID; }

  inline bool IsSparse() const { return m_.IsSet(Metadata::Sparse); }

  inline std::string getAssociated() { return m_.getAssociated(); }

  KOKKOS_FORCEINLINE_FUNCTION
  Uid_t GetUniqueID() const { return uid_; }

  static Uid_t GetUniqueID(const std::string &var_label) { return get_uid_(var_label); }

  /// return information string
  std::string info();

#ifdef ENABLE_SPARSE
  inline bool IsAllocated() const { return is_allocated_; }
#else
  inline constexpr bool IsAllocated() const { return true; }
#endif

  inline bool IsSet(const MetadataFlag bit) const { return m_.IsSet(bit); }

  ParArrayND<T, VariableState> data;
  ParArrayND<T, VariableState> flux[4];  // used for boundary calculation
  ParArrayND<T, VariableState> coarse_s; // used for sending coarse boundary calculation
  
  template<std::size_t DIR, std::size_t... I, class... Args>
  auto GetTensorComponentImpl(ParArrayND<T, VariableState>& d, std::index_sequence<I...>, Args&&... args) { 
    return d.Get(((void) I, DIR)..., std::forward<Args>(args)...);
  }

  template<std::size_t DIR, class... Args>
  auto GetFluxTensorComponent(Args&&... args) { 
    return GetTensorComponentImpl<0>(flux[DIR], std::make_index_sequence<MAX_VARIABLE_DIMENSION - sizeof...(Args) - 3>(), std::forward<Args>(args)...);
  }
  
  template<std::size_t DIR = 0, class... Args>
  auto GetCoarseTensorComponent(Args&&... args) { 
    return GetTensorComponentImpl<DIR>(coarse_s, std::make_index_sequence<MAX_VARIABLE_DIMENSION - sizeof...(Args) - 3>(), std::forward<Args>(args)...);
  }

  template<std::size_t DIR = 0, class... Args>
  auto GetTensorComponent(Args&&... args) { 
    return GetTensorComponentImpl<DIR>(data, std::make_index_sequence<MAX_VARIABLE_DIMENSION - sizeof...(Args) - 3>(), std::forward<Args>(args)...);
  }

  int dealloc_count = 0;

  int GetAllocationStatus() {
    if (!is_allocated_) return 0;
    return num_alloc_;
  }

 private:
  // allocate data, fluxes, and boundary variable
  void Allocate(std::weak_ptr<MeshBlock> wpmb, bool flag_uninitialized = false);
  int num_alloc_ = 0;

  // allocate data only
  void AllocateData(bool flag_uninitialized = false);

  // deallocate data, fluxes, and boundary variable
  void Deallocate();

  /// allocate fluxes (if Metadata::WithFluxes is set) and coarse data if
  /// (Metadata::FillGhost is set)
  void AllocateFluxesAndCoarse(std::weak_ptr<MeshBlock> wpmb);

  VariableState MakeVariableState() const { return VariableState(m_, sparse_id_); }

  Metadata m_;
  const std::string base_name_;
  const int sparse_id_;
  const std::array<int, MAX_VARIABLE_DIMENSION> dims_, coarse_dims_;

  // Machinery for giving each variable a unique ID that is faster to
  // evaluate than a string. Safe so long as the number of MPI ranks
  // does not change while the code is running (restarts are fine).
  Uid_t uid_;
  // This generator needs to be global so that different instances of
  // variable have the same unique ID.
  inline static UniqueIDGenerator<std::string> get_uid_;

  bool is_allocated_ = false;
  ParArrayNDFlux<T> flux_data_; // unified par array for the fluxes
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

  auto GetHostMirrorAndCopy() { return data.GetHostMirrorAndCopy(); }

  auto GetHostMirrorAndCopy(int n6, int n5, int n4, int n3, int n2) {
    auto data_slice = Kokkos::subview(data, n6, n5, n4, n3, n2, Kokkos::ALL());
    return data_slice.GetHostMirrorAndCopy();
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
  
  template<std::size_t... I, class... Args>
  auto GetTensorComponentImpl(ParArrayND<T>& d, std::index_sequence<I...>, Args&&... args) { 
    return d.Get(((void) I, 0)..., std::forward<Args>(args)...);  
  }
  
  template<class... Args>
  auto GetTensorComponent(Args&&... args) { 
    return GetTensorComponentImpl(data, std::make_index_sequence<MAX_VARIABLE_DIMENSION - sizeof...(Args) - 1>(), std::forward<Args>(args)...);
  }
  
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
inline VariableVector<T> GetAnyVariables(const VariableVector<T> &cv_in,
                                         std::vector<MetadataFlag> mflags) {
  VariableVector<T> out;
  for (auto &pvar : cv_in) {
    if (std::any_of(mflags.begin(), mflags.end(),
                    [&](const auto &in) { return pvar->IsSet(in); })) {
      out.push_back(pvar);
    }
  }
  return out;
}

template <typename T>
inline VariableVector<T> GetAnyVariables(const VariableVector<T> &cv_in,
                                         std::vector<std::string> base_names) {
  VariableVector<T> out;

  std::vector<std::regex> base_regexs;
  for (auto &base_name : base_names)
    base_regexs.push_back(std::regex(base_name + ".*"));

  for (auto &pvar : cv_in) {
    if (std::any_of(base_regexs.begin(), base_regexs.end(), [&](const auto &in) {
          return std::regex_match(pvar->label(), in);
        })) {
      out.push_back(pvar);
    }
  }
  return out;
}

// Enforces uid ordering of sets/maps of variables
template <typename T>
struct VarComp {
  bool operator()(const std::shared_ptr<T> &a, const std::shared_ptr<T> &b) const {
    return ((a->GetUniqueID()) < (b->GetUniqueID()));
  }
};

template <typename T>
using VarPtr = std::shared_ptr<Variable<T>>;

template <typename T>
using MapToVars = std::map<std::string, std::shared_ptr<Variable<T>>>;

template <typename T>
using ParticleVarPtr = std::shared_ptr<ParticleVariable<T>>;
template <typename T>
using ParticleVariableVector = std::vector<ParticleVarPtr<T>>;
template <typename T>
using MapToParticle = std::map<std::string, ParticleVarPtr<T>>;
template <typename T>
using VariableSet = std::set<VarPtr<T>, VarComp<Variable<T>>>;
template <typename T>
using MetadataFlagToVariableMap = std::map<MetadataFlag, VariableSet<T>>;

} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
