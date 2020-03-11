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

namespace parthenon {

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

class Flag {
  friend class Metadata;

public:
  constexpr bool operator==(Flag const &other) const {
    return flag_ == other.flag_;
  }
private:
  // Flag can only be instantiated by Metadata
  constexpr explicit Flag(int flag) : flag_(flag) {}

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
  // Apparently defining the class in-line results in the constructor being undefined at the time
  // time when the compiler is evaluating the constexpr definitions for the static flags.
  // Therefore, Flag is defined outside Metadata and then re-exported.
  using Flag = Flag;

  /// The flags refer to the different attributes that a variable can
  /// have.  These include the topology, IO, advection, conservation,
  /// whether it is sparse, etc.  This is designed to be easily extensible.
  
  constexpr static Flag Ignore      = Flag( 0); ///<  0: bit 0 is ignored
  constexpr static Flag None        = Flag( 1); ///<  1: no topology specified
  constexpr static Flag Cell        = Flag( 2); ///<  2: cell variable
  constexpr static Flag Face        = Flag( 3); ///<  3: face variable
  constexpr static Flag Edge        = Flag( 4); ///<  4: edge variable
  constexpr static Flag Node        = Flag( 5); ///<  5: node variable
  constexpr static Flag Vector      = Flag( 6); ///<  6: a vector quantity, i.e. a rank 1 contravariant tensor
  constexpr static Flag Tensor      = Flag( 7); ///<  7: a rank-2 tensor
  constexpr static Flag Advected    = Flag( 8); ///<  8: advected variable
  constexpr static Flag Conserved   = Flag( 9); ///<  9: conserved variable
  constexpr static Flag Intensive   = Flag(10); ///< 10: intensive variable
  constexpr static Flag Restart     = Flag(11); ///< 11: added to restart dump
  constexpr static Flag Graphics    = Flag(12); ///< 12: added to graphics dumps
  constexpr static Flag Sparse      = Flag(13); ///< 13: is specified per-sparse index
  constexpr static Flag Independent = Flag(14); ///< 14: is an independent, evolved variable
  constexpr static Flag Derived     = Flag(15); ///< 15: is a derived quantity (ignored)
  constexpr static Flag OneCopy     = Flag(16); ///< 16: only one copy even if multiple stages
  constexpr static Flag FillGhost   = Flag(17); ///< 17: Do boundary communication
  constexpr static Flag SharedComms = Flag(18); ///< 18: Communication arrays are a copy: hint to destructor

  // `max` must always be the final flag - place all new flags prior to this flag
  static constexpr Flag Max = Flag(19);

  /// Default constructor override
  Metadata() : sparse_id_(-1),
               shape_({1}) { }


  /// returns a new Metadata instance with set bits,
  /// set sparse_id, and fourth dimension
  explicit Metadata(const std::vector<Flag>& bits) :
    sparse_id_(-1),
    shape_({1}) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits and shape set
  explicit Metadata(const std::vector<Flag>& bits,
                    std::vector<int> shape) :
    sparse_id_(-1),
    shape_(shape) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits and sparse id set
  explicit Metadata(const std::vector<Flag>& bits,
                    const int sparse_id) :
    sparse_id_(sparse_id),
    shape_({1}) {
    SetMultiple(bits);
  }

  /// returns a metadata with bits, shape, and sparse ID set
  explicit Metadata(const std::vector<Flag>& bits,
                    const int sparse_id,
                    std::vector<int> shape) :
    sparse_id_(sparse_id),
    shape_(shape) {
    SetMultiple(bits);
  }

  // Static routines
  static Flag AllocateNewFlag();

  // Individual flag setters
  void Set(const Flag f) { DoBit(f, true); }             ///< Set specific bit
  void Unset(const Flag f) { DoBit(f, false); }          ///< Unset specific bit

  /*--------------------------------------------------------*/
  // Getters for attributes
  /*--------------------------------------------------------*/
  /// returns the topological location of variable
  Flag Where() const {
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
  bool AnyFlagsSet(std::vector<Flag> const &flags) const {
    return std::any_of(flags.begin(), flags.end(), [this](Flag const &f) {
      return IsSet(f);
    });
  }

  /// returns true if bit is set, false otherwise
  bool IsSet(const Flag bit) const { return bits_[bit.flag_]; }

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
  static int next_app_flag_;

  /// the attribute flags that are set for the class
  std::vector<bool> bits_;
  std::vector<int> shape_;
  std::string associated_;
  int sparse_id_;

  /*--------------------------------------------------------*/
  // Setters for the different attributes of metadata
  /*--------------------------------------------------------*/
  void SetWhere(Flag x) {
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
    /// e.g. set({face, advected, conserved, sparse})
  void SetMultiple(const std::vector<Flag> &theAttributes) {
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
  /// e.g. unset({face, advected, conserved, sparse})
  void UnsetMultiple(const std::vector<Flag> &theAttributes) {
    for (auto &a : theAttributes) {
      DoBit(a, false);
    }
  }

  /// if flag is true set bit, clears otherwise
  void DoBit(Flag bit, const bool flag) {
    if (bit.flag_ >= bits_.size()) {
      bits_.resize(bit.flag_ + 1);
    }
    bits_[bit.flag_] = flag;
  }


  /// Checks if the bit is a topology bit
  bool IsTopology(Flag bit) const {
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
