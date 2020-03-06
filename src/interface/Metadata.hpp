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

#include <bitset>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#define _MAXBITS_ 32

namespace parthenon {
/// @brief
///
/// The metadata class is a descriptor for variables in the
/// simulation.
///
///  Can set or query attributes specifed in flags.
///

class TensorShape {
 public:
  TensorShape() {}
  explicit TensorShape(int scalar) {}
  TensorShape(int rank, int *size) {
    shape_.insert(shape_.end(), size, size+rank);
  }
  std::vector<int> shape_;
};

class Metadata {
 public:
  /// The flags refer to the different attributes that a variable can
  /// have.  These include the topology, IO, advection, conservation,
  /// whether it is a sparse variable, etc.  This is designed to be easily extensible.
  enum flags { // *** if you modify flags, be sure to modify flag_labels as well ***
              ignore,       ///<  0: bit 0 is ignored
              none,         ///<  1: no topology specified
              cell,         ///<  2: cell variable
              face,         ///<  3: face variable
              edge,         ///<  4: edge variable
              node,         ///<  5: node variable
              vector,       ///<  6: a vector quantity, i.e. a rank 1 contravariant tensor
              tensor,       ///<  7: a rank-2 tensor
              advected,     ///<  8: advected variable
              conserved,    ///<  9: conserved variable
              intensive,    ///< 10: intensive variable
              restart,      ///< 11: added to restart dump
              graphics,     ///< 12: added to graphics dumps
              sparse,       ///< 13: is specified per sparse index
              independent,  ///< 14: is an independent, evolved variable
              derived,      ///< 15: is a derived quantity (ignored)
              oneCopy,      ///< 16: only one copy even if multiple stages
              fillGhost,    ///< 17: Do boundary communication
              sharedComms   ///< 18: Communication arrays are a copy: hint to destructor
  };

  /// Default constructor override
  Metadata() : sparse_id_(-1),
               shape_({1}) { }


  /// returns a new Metadata instance with set bits,
  /// set sparse_id, and fourth dimension
  explicit Metadata(const std::vector<flags>& bits) :
    sparse_id_(-1),
    shape_({1}) {
    setMultiple(bits);
  }

  /// returns a metadata with bits and shape set
  explicit Metadata(const std::vector<flags>& bits,
                    std::vector<int> shape) :
    sparse_id_(-1),
    shape_(shape) {
    setMultiple(bits);
  }

  /// returns a metadata with bits and sparse id set
  explicit Metadata(const std::vector<flags>& bits,
                    const int sparse_id) :
    sparse_id_(sparse_id),
    shape_({1}) {
    setMultiple(bits);
  }

  /// returns a metadata with bits, shape, and sparse ID set
  explicit Metadata(const std::vector<flags>& bits,
                    const int sparse_id,
                    std::vector<int> shape) :
    sparse_id_(sparse_id),
    shape_(shape) {
    setMultiple(bits);
  }

  /// copy constructor
  Metadata(const Metadata&m) : sparse_id_(m.sparse_id_),
                               shape_(m.shape_),
                               theBits_(m.theBits_),
                               associated_(m.associated_) { }

  void setFlags(std::bitset<_MAXBITS_> bitflags) { theBits_ = bitflags; }

  // Individual flag setters
  void set(const flags f) { doBit(f, true); }             ///< Set specific bit
  void unset(const flags f) { doBit(f, false); }          ///< Unset specific bit


  /*
  // if flag is true set bit, clears otherwise
  void setAdvected(const bool a)  {doBit(advected, a);}   ///< Set advected or not
  void setConserved(const bool a) {doBit(conserved, a);}  ///< Set conserved or not
  void setIntensive(const bool a) {doBit(intensive, a);}  ///< Set intensive or extensive
  void setRestart(const bool a)   {doBit(restart, a);}    ///< Set IO / restart
  void setGraphics(const bool a)  {doBit(graphics, a);}   ///< Set IO / graphics
  void setSparse(const bool a)    {doBit(sparse, a);}     ///< Set includes sparse variable
  void setDerived(const bool a)   {doBit(derived, a);}    ///< Set derived variable?
  void setOneCopy(const bool a)   {doBit(oneCopy, a);}    ///< Single copy across stages
  */
  void setIndexHints(int);  ///< Set indexing hints

  /*--------------------------------------------------------*/
  // Getters for attributes
  /*--------------------------------------------------------*/
  /// returns the topological location of variable
  flags where() const {
    if (isSet(cell)) {
      return cell;
    } else if (isSet(face)) {
      return face;
    } else if (isSet(edge)) {
      return edge;
    } else if (isSet(node)) {
      return node;
    }
    /// by default return topology::none
    return none;
  }

  bool isAdvected()    const { return (isSet(advected)); }   ///< true if advected
  bool isConserved()   const { return (isSet(conserved)); }  ///< true if conserved
  bool isIntensive()   const { return (isSet(intensive)); }  ///< true if intensive
  bool isRestart()     const { return (isSet(restart)); }    ///< true if restart
  bool isGraphics()    const { return (isSet(graphics)); }   ///< true if graphics
  bool isOneCopy()     const { return (isSet(oneCopy)); }    ///< true if one copy
  bool hasSparse()     const { return (isSet(sparse)); }     ///< true if it is a sparse variable
  bool fillsGhost()    const { return (isSet(fillGhost)); }
  bool isVector()      const { return (isSet(vector)); }
  bool isIndependent() const { return (isSet(independent)); }

  int  getSparseID() const { return sparse_id_; }

  const std::vector<int>& Shape() const { return shape_; }

  /*--------------------------------------------------------*/
  // Utility functions
  /*--------------------------------------------------------*/
  /// Returns the attribute flags as an unsigned long integer
  uint64_t mask() const { return theBits_.to_ulong();}

  /// Returns the attribute flags as a string of 1/0
  std::string maskAsString() const { return theBits_.to_string();}

  /// Gets a mask from a std::vector of flags
  /// This may disappear
  /// @param theFlags a vector of type flags
  /// @return unsigned long int mask of said vector
  static uint64_t getMaskForVector(const std::vector<flags> theFlags) {
    std::bitset<_MAXBITS_> myBits;
    myBits.reset();
    for (auto &bit : theFlags) {
      myBits.set(bit);
    }
    return myBits.to_ulong();
  }

  std::bitset<_MAXBITS_> getFlags() { return theBits_; }

  void upgradeFlags(Metadata& m) { setFlags(theBits_ | m.getFlags()); }

  /// returns true if bit is set, false otherwise
  bool isSet(const flags bit) const { return theBits_.test(bit); }

  // Operators
  bool operator==(const Metadata &b) const {
    return ((sparse_id_ == b.sparse_id_) && (shape_ == b.shape_) &&
            (theBits_ == b.theBits_));
  }

  bool operator!=(const Metadata &b) const {
    return ((sparse_id_ != b.sparse_id_) || (shape_ != b.shape_) ||
            (theBits_ != b.theBits_)

    );
  }

  void Associate(const std::string &name) { associated_ = name; }
  const std::string& getAssociated() const { return associated_; }

private:
  /// the attribute flags that are set for the class
  std::bitset<_MAXBITS_> theBits_;
  std::vector<int> shape_;
  std::string associated_;
  int sparse_id_;

  /*--------------------------------------------------------*/
  // Setters for the different attributes of metadata
  /*--------------------------------------------------------*/
  ///< zero out the metadata
  void reset() {theBits_.reset(); shape_.clear();}

  void setWhere(flags x) {
    unsetMultiple({cell, face, edge, node, none});
    if ( x == cell )      {
      doBit(cell, true);
    } else if ( x == face ) {
      doBit(face, true);
    } else if ( x == edge ) {
      doBit(edge, true);
    } else if ( x == node ) {
      doBit(node, true);
    } else if ( x == none ) {
      doBit(none, true);
    } else {
      throw std::invalid_argument ("received invalid topology flag in setWhere()");
    }
  } ///< Set topological element where variable is defined (none/cell/face/edge/node)

    /// Set multiple flags at the same time.
    /// Takes a comma separated set of flags from the enum above
    ///
    /// e.g. set({face, advected, conserved, sparse})
  void setMultiple(const std::vector<flags> &theAttributes) {
    int numTopo = 0;
    for (auto &a : theAttributes) {
      if ( _isTopology(a) ) { // topology flags are special
        setWhere(a);
        numTopo++;
      } else {
        doBit(a, true);
      }
    }
    if (numTopo > 1) {
      throw std::invalid_argument ("Multiple topologies sent to setMultiple()");
    }
  }

  /// Unset multiple flags at the same time.
  /// Takes a comma separated set of flags from the enum above
  ///
  /// e.g. unset({face, advected, conserved, sparse})
  void unsetMultiple(const std::vector<flags> &theAttributes) {
    for (auto &a : theAttributes) {
      doBit(a, false);
    }
  }

  /// if flag is true set bit, clears otherwise
  void doBit( Metadata::flags bit, const bool flag) {
    if( bit >0) {
      (flag? theBits_.set(bit):theBits_.reset(bit));
    }
  }


  /// Checks if the bit is a topology bit
  bool _isTopology(flags bit) const {
    return (
            (bit == cell) ||
            (bit == face) ||
            (bit == edge) ||
            (bit == node) ||
            (bit == none)
            );
  }
};
}
#endif // INTERFACE_METADATA_HPP_
