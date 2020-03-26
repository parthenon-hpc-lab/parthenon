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
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

/// The point of this macro is to generate code for each built-in flag using the 
/// `PARTHENON_INTERNAL_FOR_FLAG` macro. This is to accomplish the following goals:
/// - Generate a unique value for each flag using an enum
/// - Wrap each value in a `MetadataFlag` type that can only be instantiated within parthenon
/// - Make it possible to return a user-friendly string for each flag.
///
/// Having this macro means you only need to add a new flag in one place and have each of these
/// properties automatically be updated.
#define PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG \
  /**  bit 0 is ignored */ \
  PARTHENON_INTERNAL_FOR_FLAG(Ignore) \
  /**  no topology specified */ \
  PARTHENON_INTERNAL_FOR_FLAG(None) \
  /**  cell variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Cell) \
  /**  face variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Face) \
  /**  edge variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Edge) \
  /**  node variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Node) \
  /**  a vector quantity, i.e. a rank 1 contravariant tenso */ \
  PARTHENON_INTERNAL_FOR_FLAG(Vector) \
  /**  a rank-2 tensor */ \
  PARTHENON_INTERNAL_FOR_FLAG(Tensor) \
  /**  advected variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Advected) \
  /**  conserved variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Conserved) \
  /** intensive variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Intensive) \
  /** added to restart dump */ \
  PARTHENON_INTERNAL_FOR_FLAG(Restart) \
  /** added to graphics dumps */ \
  PARTHENON_INTERNAL_FOR_FLAG(Graphics) \
  /** is specified per-sparse index */ \
  PARTHENON_INTERNAL_FOR_FLAG(Sparse) \
  /** is an independent, evolved variable */ \
  PARTHENON_INTERNAL_FOR_FLAG(Independent) \
  /** is a derived quantity (ignored) */ \
  PARTHENON_INTERNAL_FOR_FLAG(Derived) \
  /** only one copy even if multiple stages */ \
  PARTHENON_INTERNAL_FOR_FLAG(OneCopy) \
  /** Do boundary communication */ \
  PARTHENON_INTERNAL_FOR_FLAG(FillGhost) \
  /** Communication arrays are a copy: hint to destructor */ \
  PARTHENON_INTERNAL_FOR_FLAG(SharedComms)

namespace parthenon {

namespace internal {
  enum class MetadataInternal {
    // declare all the internal flags in an enum so that their values are unique and kept up to date
#define PARTHENON_INTERNAL_FOR_FLAG(name) name,
    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
    Max
#undef PARTHENON_INTERNAL_FOR_FLAG
  };

  class UserMetadataState;
}

class Metadata;

class TensorShape {
 public:
  TensorShape() {}
  explicit TensorShape(int scalar) {}
  TensorShape(int rank, int *size) {
    shape_.insert(shape_.end(), size, size+rank);
  }
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

  std::string const &Name() const;

#ifdef CATCH_VERSION_MAJOR
  // Should never be used for application code - only exposed for testing.
  constexpr int InternalFlagValue() const {
    return flag_;
  }
#endif
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
 private:
  // must be forward declared
  auto PackedTuple() const {
    return std::tie(bits_, shape_, sparse_id_);
  }
 public:
  /// The flags refer to the different attributes that a variable can
  /// have.  These include the topology, IO, advection, conservation,
  /// whether it sparse, etc.  This is designed to be easily extensible.


  // this wraps all the built-in flags in the `MetadataFlag` type so that using these flags is
  // type-safe.
#define PARTHENON_INTERNAL_FOR_FLAG(name) \
    constexpr static MetadataFlag name = \
      MetadataFlag(static_cast<int>(::parthenon::internal::MetadataInternal::name));

    PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

  /// Default constructor override
  Metadata() : sparse_id_(-1),
               shape_({1}) { }


  /// returns a new Metadata instance with set bits,
  /// set sparse_id, and fourth dimension
  explicit Metadata(const std::vector<MetadataFlag>& bits) :
    sparse_id_(-1),
    shape_({1}) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits and shape set
  explicit Metadata(const std::vector<MetadataFlag>& bits,
                    std::vector<int> shape) :
    sparse_id_(-1),
    shape_(shape) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits and sparse id set
  explicit Metadata(const std::vector<MetadataFlag>& bits,
                    const int sparse_id) :
    sparse_id_(sparse_id),
    shape_({1}) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits, shape, and sparse ID set
  explicit Metadata(const std::vector<MetadataFlag>& bits,
                    const int sparse_id,
                    std::vector<int> shape) :
    sparse_id_(sparse_id),
    shape_(shape) {
    SetMultiple(bits);
  }

  // Static routines
  static MetadataFlag AllocateNewFlag(std::string &&name);

  // Individual flag setters
  void Set(const MetadataFlag f) { DoBit(f, true); }             ///< Set specific bit
  void Unset(const MetadataFlag f) { DoBit(f, false); }          ///< Unset specific bit

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

  int GetSparseId() const { return sparse_id_; }

  const std::vector<int>& Shape() const { return shape_; }

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

  /**
   * @brief Returns true if any flag is set
   */
  bool AnyFlagsSet(std::vector<MetadataFlag> const &flags) const {
    return std::any_of(flags.begin(), flags.end(), [this](MetadataFlag const &f) {
      return IsSet(f);
    });
  }

  /// returns true if bit is set, false otherwise
  bool IsSet(const MetadataFlag bit) const { return bits_[bit.flag_]; }

  // Operators
  bool operator==(const Metadata &b) const {
    return PackedTuple() == b.PackedTuple();
  }

  bool operator!=(const Metadata &b) const {
    return PackedTuple() != b.PackedTuple();
  }

  void Associate(const std::string &name) { associated_ = name; }
  const std::string& getAssociated() const { return associated_; }

private:
  /// the attribute flags that are set for the class
  std::vector<bool> bits_;
  std::vector<int> shape_;
  std::string associated_;
  int sparse_id_;

  /*--------------------------------------------------------*/
  // Setters for the different attributes of metadata
  /*--------------------------------------------------------*/
  void SetWhere(MetadataFlag x) {
    UnsetMultiple({Cell, Face, Edge, Node, None});
    if ( x == Cell )      {
      DoBit(Cell, true);
    } else if ( x == Face ) {
      DoBit(Face, true);
    } else if ( x == Edge ) {
      DoBit(Edge, true);
    } else if ( x == Node ) {
      DoBit(Node, true);
    } else if ( x == None ) {
      DoBit(None, true);
    } else {
      throw std::invalid_argument ("received invalid topology flag in SetWhere()");
    }
  } ///< Set topological element where variable is defined (None/Cell/Face/Edge/Node)

    /// Set multiple flags at the same time.
    /// Takes a comma separated set of flags from the enum above
    ///
    /// e.g. set({Face, Advected, Conserved, Sparse})
  void SetMultiple(const std::vector<MetadataFlag> &theAttributes) {
    int numTopo = 0;
    for (auto &a : theAttributes) {
      if ( IsTopology(a) ) { // topology flags are special
        SetWhere(a);
        numTopo++;
      } else {
        DoBit(a, true);
      }
    }
    if (numTopo > 1) {
      throw std::invalid_argument ("Multiple topologies sent to SetMultiple()");
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
  void DoBit(MetadataFlag bit, const bool flag) {
    if (bit.flag_ >= bits_.size()) {
      bits_.resize(bit.flag_ + 1);
    }
    bits_[bit.flag_] = flag;
  }


  /// Checks if the bit is a topology bit
  bool IsTopology(MetadataFlag bit) const {
    return (
            (bit == Cell) ||
            (bit == Face) ||
            (bit == Edge) ||
            (bit == Node) ||
            (bit == None)
            );
  }
};
}
#endif // INTERFACE_METADATA_HPP_
