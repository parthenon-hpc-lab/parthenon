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
#ifndef INTERFACE_METADATA_HPP_
#define INTERFACE_METADATA_HPP_

#include <algorithm>
#include <bitset>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "utils/error_checking.hpp"

/// The point of this macro is to generate code for each built-in flag using the
/// `PARTHENON_INTERNAL_FOR_FLAG` macro. This is to accomplish the following goals:
/// - Generate a unique value for each flag using an enum
/// - Wrap each value in a `MetadataFlag` type that can only be instantiated within
///   parthenon
/// - Make it possible to return a user-friendly string for each flag.
///
/// Having this macro means you only need to add a new flag in one place and have each of
/// these properties automatically be updated.
#define PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG                                          \
  /**  bit 0 is ignored */                                                               \
  PARTHENON_INTERNAL_FOR_FLAG(Ignore)                                                    \
  /**  no topology specified */                                                          \
  PARTHENON_INTERNAL_FOR_FLAG(None)                                                      \
  /**  Private to a package */                                                           \
  PARTHENON_INTERNAL_FOR_FLAG(Private)                                                   \
  /**  Provided by a package */                                                          \
  PARTHENON_INTERNAL_FOR_FLAG(Provides)                                                  \
  /**  Not created by a package, assumes available from another package */               \
  PARTHENON_INTERNAL_FOR_FLAG(Requires)                                                  \
  /**  does nothing if another package provides the variable */                          \
  PARTHENON_INTERNAL_FOR_FLAG(Overridable)                                               \
  /**  cell variable */                                                                  \
  PARTHENON_INTERNAL_FOR_FLAG(Cell)                                                      \
  /**  face variable */                                                                  \
  PARTHENON_INTERNAL_FOR_FLAG(Face)                                                      \
  /**  edge variable */                                                                  \
  PARTHENON_INTERNAL_FOR_FLAG(Edge)                                                      \
  /**  node variable */                                                                  \
  PARTHENON_INTERNAL_FOR_FLAG(Node)                                                      \
  /**  a vector quantity, i.e. a rank 1 contravariant tenso */                           \
  PARTHENON_INTERNAL_FOR_FLAG(Vector)                                                    \
  /**  a rank-2 tensor */                                                                \
  PARTHENON_INTERNAL_FOR_FLAG(Tensor)                                                    \
  /**  advected variable */                                                              \
  PARTHENON_INTERNAL_FOR_FLAG(Advected)                                                  \
  /**  conserved variable */                                                             \
  PARTHENON_INTERNAL_FOR_FLAG(Conserved)                                                 \
  /** intensive variable */                                                              \
  PARTHENON_INTERNAL_FOR_FLAG(Intensive)                                                 \
  /** added to restart dump */                                                           \
  PARTHENON_INTERNAL_FOR_FLAG(Restart)                                                   \
  /** is specified per-sparse index */                                                   \
  PARTHENON_INTERNAL_FOR_FLAG(Sparse)                                                    \
  /** is an independent, evolved variable */                                             \
  PARTHENON_INTERNAL_FOR_FLAG(Independent)                                               \
  /** is a derived quantity (ignored) */                                                 \
  PARTHENON_INTERNAL_FOR_FLAG(Derived)                                                   \
  /** only one copy even if multiple stages */                                           \
  PARTHENON_INTERNAL_FOR_FLAG(OneCopy)                                                   \
  /** Do boundary communication */                                                       \
  PARTHENON_INTERNAL_FOR_FLAG(FillGhost)                                                 \
  /** Communication arrays are a copy: hint to destructor */                             \
  PARTHENON_INTERNAL_FOR_FLAG(SharedComms)                                               \
  /** Boolean-valued quantity */                                                         \
  PARTHENON_INTERNAL_FOR_FLAG(Boolean)                                                   \
  /** Integer-valued quantity */                                                         \
  PARTHENON_INTERNAL_FOR_FLAG(Integer)                                                   \
  /** Real-valued quantity */                                                            \
  PARTHENON_INTERNAL_FOR_FLAG(Real)

namespace parthenon {

namespace internal {
enum class MetadataInternal {
// declare all the internal flags in an enum so that their values are unique and kept
// up to date
#define PARTHENON_INTERNAL_FOR_FLAG(name) name,
  PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG Max
#undef PARTHENON_INTERNAL_FOR_FLAG
};

class UserMetadataState;

} // namespace internal

class Metadata;

class TensorShape {
 public:
  TensorShape() {}
  explicit TensorShape(int scalar) {}
  TensorShape(int rank, int *size) { shape_.insert(shape_.end(), size, size + rank); }
  std::vector<int> shape_;
};

class MetadataFlag {
  // This allows `Metadata` and `UserMetadataState` to instantiate `MetadataFlag`.
  friend class Metadata;
  friend class internal::UserMetadataState;

 public:
  constexpr bool operator==(MetadataFlag const &other) const {
    return flag_ == other.flag_;
  }

  constexpr bool operator!=(MetadataFlag const &other) const {
    return flag_ != other.flag_;
  }

  std::string const &Name() const;

#ifdef CATCH_VERSION_MAJOR
  // Should never be used for application code - only exposed for testing.
  constexpr int InternalFlagValue() const { return flag_; }
#endif

  friend std::ostream &operator<<(std::ostream &os, const Metadata &m);
  friend std::ostream &operator<<(std::ostream &os, const MetadataFlag &flag) {
    os << flag.Name();
    return os;
  }

 private:
  // MetadataFlag can only be instantiated by Metadata
  constexpr explicit MetadataFlag(int flag) : flag_(flag) {}

  int flag_;
};

/// @brief
///
/// The metadata class is a descriptor for variables in the
/// simulation.
///
///  Can set or query attributes specifed in flags.
///
class Metadata {
 public:
  /// The flags refer to the different attributes that a variable can
  /// have.  These include the topology, IO, advection, conservation,
  /// whether it sparse, etc.  This is designed to be easily extensible.

  // this wraps all the built-in flags in the `MetadataFlag` type so that using these
  // flags is type-safe.
#define PARTHENON_INTERNAL_FOR_FLAG(name)                                                \
  constexpr static MetadataFlag name =                                                   \
      MetadataFlag(static_cast<int>(::parthenon::internal::MetadataInternal::name));

  PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

  /// Default constructor override
  Metadata() : shape_({1}), sparse_id_(-1) {}

  /// returns a new Metadata instance with set bits,
  /// set sparse_id, and fourth dimension
  explicit Metadata(const std::vector<MetadataFlag> &bits) : shape_({1}), sparse_id_(-1) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits and shape set
  explicit Metadata(const std::vector<MetadataFlag> &bits, const std::vector<int> &shape)
      : shape_(shape), sparse_id_(-1) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits and shape set
  explicit Metadata(const std::vector<MetadataFlag> &bits, const std::vector<int> &shape,
                    const std::vector<std::string> component_labels)
      : shape_(shape), sparse_id_(-1), component_labels_(component_labels) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits and sparse id set
  explicit Metadata(const std::vector<MetadataFlag> &bits, const int sparse_id)
      : shape_({1}), sparse_id_(sparse_id) {
    SetMultiple(bits);
    PARTHENON_REQUIRE_THROWS(IsSet(Sparse), "Sparse ID requires sparse metadata");
  }

  explicit Metadata(const std::vector<MetadataFlag> &bits, const std::string &associated)
      : associated_(associated) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits, shape, and sparse ID set
  explicit Metadata(const std::vector<MetadataFlag> &bits, int sparse_id,
                    const std::vector<int> &shape)
      : shape_(shape), sparse_id_(sparse_id) {
    SetMultiple(bits);
    PARTHENON_REQUIRE_THROWS(IsSet(Sparse), "Sparse ID requires sparse metadata");
  }

  explicit Metadata(const std::vector<MetadataFlag> &bits, const int sparse_id,
                    const std::string &associated)
      : sparse_id_(sparse_id), associated_(associated) {
    SetMultiple(bits);
    PARTHENON_REQUIRE_THROWS(IsSet(Sparse), "Sparse ID requires sparse metadata");
  }

  explicit Metadata(const std::vector<MetadataFlag> &bits, const std::string &associated,
                    const std::vector<int> &shape)
      : associated_(associated), shape_(shape) {
    SetMultiple(bits);
  }

  explicit Metadata(const std::vector<MetadataFlag> &bits, const int sparse_id,
                    const std::string &associated, const std::vector<int> &shape)
      : sparse_id_(sparse_id), associated_(associated), shape_(shape) {
    SetMultiple(bits);
    PARTHENON_REQUIRE_THROWS(IsSet(Sparse), "Sparse ID requires sparse metadata");
  }

  // Static routines
  static MetadataFlag AllocateNewFlag(std::string &&name);
  // Individual flag setters
  void Set(MetadataFlag f) { DoBit(f, true); }    ///< Set specific bit
  void Unset(MetadataFlag f) { DoBit(f, false); } ///< Unset specific bit

  /*--------------------------------------------------------*/
  // Getters for attributes
  /*--------------------------------------------------------*/
  /// returns the topological location of variable
  MetadataFlag Where() const {
    if (IsSet(Cell)) {
      return Cell;
    } else if (IsSet(Face)) {
      return Face;
    } else if (IsSet(Edge)) {
      return Edge;
    } else if (IsSet(Node)) {
      return Node;
    }
    /// by default return Metadata::None
    return None;
  }

  /// returns the type of the variable
  MetadataFlag Type() const {
    if (IsSet(Integer)) {
      return Integer;
    } else if (IsSet(Real)) {
      return Real;
    }
    /// by default return Metadata::None
    return None;
  }

  MetadataFlag Role() const {
    if (IsSet(Private)) {
      return Private;
    } else if (IsSet(Provides)) {
      return Provides;
    } else if (IsSet(Requires)) {
      return Requires;
    } else if (IsSet(Overridable)) {
      return Overridable;
    } else {
      return None;
    }
  }

  void SetSparseId(int id) { sparse_id_ = id; }
  int GetSparseId() const { return sparse_id_; }

  const std::vector<int> &Shape() const { return shape_; }

  /*--------------------------------------------------------*/
  // Utility functions
  /*--------------------------------------------------------*/
  /// Returns the attribute flags as a string of 1/0
  std::string MaskAsString() const {
    std::string str;
    for (auto const bit : bits_) {
      str += bit ? '1' : '0';
    }
    return str;
  }
  friend std::ostream &operator<<(std::ostream &os, const parthenon::Metadata &m);

  /**
   * @brief Returns true if any flag is set
   */
  bool AnyFlagsSet(std::vector<MetadataFlag> const &flags) const {
    return std::any_of(flags.begin(), flags.end(),
                       [this](MetadataFlag const &f) { return IsSet(f); });
  }

  bool AllFlagsSet(std::vector<MetadataFlag> const &flags) const {
    return std::all_of(flags.begin(), flags.end(),
                       [this](MetadataFlag const &f) { return IsSet(f); });
  }

  bool FlagsSet(std::vector<MetadataFlag> const &flags, bool matchAny = false) {
    return ((matchAny && AnyFlagsSet(flags)) || ((!matchAny) && AllFlagsSet(flags)));
  }

  /// returns true if bit is set, false otherwise
  bool IsSet(MetadataFlag bit) const {
    return bit.flag_ < bits_.size() && bits_[bit.flag_];
  }

  // Operators
  bool HasSameFlags(const Metadata &b) const {
    auto const &a = *this;

    // Check extra bits are unset
    auto const min_bits = std::min(a.bits_.size(), b.bits_.size());
    auto const &longer = a.bits_.size() > b.bits_.size() ? a.bits_ : b.bits_;
    for (auto i = min_bits; i < longer.size(); i++) {
      if (longer[i]) {
        // Bits are default false, so if any bit in the extraneous portion of the longer
        // bit list is set, then it cannot be equal to a.
        return false;
      }
    }

    for (size_t i = 0; i < min_bits; i++) {
      if (a.bits_[i] != b.bits_[i]) {
        return false;
      }
    }
    return true;
  }

  bool SparseEqual(const Metadata &b) const {
    return HasSameFlags(b) && (shape_ == b.shape_);
  }

  bool operator==(const Metadata &b) const {
    return (SparseEqual(b) && (sparse_id_ == b.sparse_id_));
  }

  bool operator!=(const Metadata &b) const { return !(*this == b); }

  void Associate(const std::string &name) { associated_ = name; }
  const std::string &getAssociated() const { return associated_; }

  const std::vector<std::string> getComponentLabels() const noexcept {
    return component_labels_;
  }

 private:
  /// the attribute flags that are set for the class
  std::vector<bool> bits_;
  std::vector<int> shape_;
  std::string associated_;
  int sparse_id_;

  std::vector<std::string> component_labels_;

  /*--------------------------------------------------------*/
  // Setters for the different attributes of metadata
  /*--------------------------------------------------------*/
  void SetWhere(MetadataFlag x) {
    UnsetMultiple({Cell, Face, Edge, Node, None});
    if (x == Cell) {
      DoBit(Cell, true);
    } else if (x == Face) {
      DoBit(Face, true);
    } else if (x == Edge) {
      DoBit(Edge, true);
    } else if (x == Node) {
      DoBit(Node, true);
    } else if (x == None) {
      DoBit(None, true);
    } else {
      PARTHENON_FAIL("received invalid topology flag");
    }
  } ///< Set topological element where variable is defined (None/Cell/Face/Edge/Node)

  /// Set multiple flags at the same time.
  /// Takes a comma separated set of flags from the enum above
  ///
  /// e.g. set({Face, Advected, Conserved, Sparse})
  void SetMultiple(const std::vector<MetadataFlag> &theAttributes) {
    int numTopo = 0;
    for (auto &a : theAttributes) {
      if (IsTopology(a)) { // topology flags are special
        SetWhere(a);
        numTopo++;
      } else {
        DoBit(a, true);
      }
    }
    if (numTopo > 1) {
      throw std::invalid_argument("Multiple topologies sent to SetMultiple()");
    }
  }

  /// Unset multiple flags at the same time.
  /// Takes a comma separated set of flags from the enum above
  ///
  /// e.g. unset({Face, Advected, Conserved, Sparse})
  void UnsetMultiple(const std::vector<MetadataFlag> &theAttributes) {
    for (auto &a : theAttributes) {
      DoBit(a, false);
    }
  }

  /// if flag is true set bit, clears otherwise
  void DoBit(MetadataFlag bit, bool flag) {
    if (bit.flag_ >= bits_.size()) {
      bits_.resize(bit.flag_ + 1);
    }
    bits_[bit.flag_] = flag;
  }

  /// Checks if the bit is a topology bit
  bool IsTopology(MetadataFlag bit) const {
    return ((bit == Cell) || (bit == Face) || (bit == Edge) || (bit == Node) ||
            (bit == None));
  }
};

} // namespace parthenon

#endif // INTERFACE_METADATA_HPP_
