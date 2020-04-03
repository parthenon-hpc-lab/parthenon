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
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "athena.hpp"
#include "parthenon_arrays.hpp"
#include "bvals/cc/bvals_cc.hpp"
#include "Metadata.hpp"

namespace parthenon {
class MeshBlock;

template <typename T>
class CellVariable {
 public:
  /// Initialize a blank slate that will be set later
  CellVariable<T>(const std::string label, Metadata &metadata) :
    data(),
    mpiStatus(true),
    _dims({0}),
    _m(metadata),
    _label(label) {
  }

  /// Initialize with a slice from another CellVariable
  /*CellVariable<T>(const std::string label,
              CellVariable<T> &src,
              const int dim,
              const int index,
              const int nvar) :
    data(),
    mpiStatus(true),
    _m(src.metadata()),
    _label(label) {
    data->InitWithShallowSlice(src, dim, index, nvar);
    if ( _m.IsSet(Metadata::FillGhost) ) {
      _m.Set(Metadata::SharedComms);
    }
  }*/

  /// Create an alias for the variable by making a shallow slice with max dim
  /*CellVariable<T>(const std::string label, CellVariable<T> &src) :
    data(),
    mpiStatus(true),
    _m(src.metadata()),
    _label(label) {
    int dim = 6;
    int start = 0;
    int nvar = src.GetDim6();
    data->InitWithShallowSlice(src, dim, start, nvar);
    _m.Set(Metadata::SharedComms);
  }*/

  /// Initialize a 6D variable
  CellVariable<T>(const std::string label,
              const std::array<int,6> dims,
              const Metadata &metadata) :
    data(label, dims[5], dims[4], dims[3], dims[2], dims[1], dims[0]),
    mpiStatus(true),
    _dims(dims),
    _m(metadata),
    _label(label) { }

  /// copy constructor
  CellVariable<T>(const CellVariable<T>& src,
              const bool allocComms=false,
              MeshBlock *pmb=nullptr);

  // accessors

  template <class...Args>
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (Args... args) { return data(std::forward<Args>(args)...); }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetDim(const int i) const { return data.GetDim(i); }


  /// Return a new array with dimension 4 as dimension 1
  /*CellVariable<T>* createShuffle4() {
    // shuffles dim 4 to dim1

    // first construct new athena array
    const std::array<int,6> dims = {this->GetDim4(), this->GetDim1(), this->GetDim2(),
                                    this->GetDim3(), this->GetDim5(), this->GetDim6()};
    size_t stride = this->GetDim1()*this->GetDim2()*this->GetDim3();
    CellVariable<T> *vNew = new CellVariable<T>(_label, dims, _m);

    auto oldData = this->data();
    auto newData = vNew->data();
    // now fill in the data
    size_t index = 0;
    const size_t n4 = this->GetDim4();
    for (size_t i=0; i<stride; i++) {
      for (size_t j=0; j<n4; j++, newData++) {
        *newData = oldData[i+j*stride];
      }
    }
    return vNew;
  }*/



  ///< Assign label for variable
  void setLabel(const std::string label) { _label = label; }

  ///< retrieve label for variable
  const std::string label() const { return _label; }

  ///< retrieve metadata for variable
  const Metadata metadata() const { return _m; }

  std::string getAssociated() { return _m.getAssociated(); }

  /// return information string
  std::string info();

  /// allocate communication space based on info in MeshBlock
  void allocateComms(MeshBlock *pmb);

  /// Repoint vbvar's var_cc array at the current variable
  void resetBoundary() { vbvar->var_cc = data; }

  bool isSet(const MetadataFlag bit) const { return _m.IsSet(bit); }

  ParArrayND<T> data;
  ParArrayND<T> flux[3];    // used for boundary calculation
  ParArrayND<T> coarse_s;   // used for sending coarse boundary calculation
  std::shared_ptr<CellCenteredBoundaryVariable> vbvar; // used in case of cell boundary communication
  bool mpiStatus;

 private:
  std::array<int,6> _dims;
  Metadata _m;
  std::string _label;
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
  FaceVariable(const std::string label, const std::array<int,6> ncells,
    const Metadata& metadata)
    : x1f(label+".x1f",ncells[5], ncells[4], ncells[3], ncells[2], ncells[1], ncells[0]+1),
      x2f(label+".x2f",ncells[5], ncells[4], ncells[3], ncells[2], ncells[1]+1, ncells[0]),
      x3f(label+".x3f",ncells[5], ncells[4], ncells[3], ncells[2]+1, ncells[1], ncells[0]),
      _dims(ncells),
      _m(metadata),
      _label(label) {
    assert ( !metadata.IsSet(Metadata::Sparse)
             && "Sparse not implemented yet for FaceVariable" );
  }

  /// Create an alias for the variable by making a shallow slice with max dim
  FaceVariable(std::string label, FaceVariable &src)
    : x1f(src.x1f),
      x2f(src.x2f),
      x3f(src.x3f),
      _dims(src._dims),
      _m(src._m),
      _label(label) { }

  // KOKKOS_FUNCTION FaceVariable() = default;
  // KOKKOS_FUNCTION FaceVariable(const FaceVariable<T>& v) = default;
  // KOKKOS_FUNCTION ~FaceVariable() = default;

  ///< retrieve label for variable
  const std::string& label() const { return _label; }

  ///< retrieve metadata for variable
  const Metadata metadata() const { return _m; }

  /// return information string
  std::string info();

  // TODO(JMM): should this be 0,1,2?
  // Should we return the reference? Or something else?
  KOKKOS_FORCEINLINE_FUNCTION
  ParArrayND<T>& Get(int i) {
    assert( 1 <= i && i <= 3 );
    if (i == 1) return (x1f);
    if (i == 2) return (x2f);
    else return (x3f); // i == 3
    //throw std::invalid_argument("Face must be x1f, x2f, or x3f");
  }
  template<typename...Args>
  KOKKOS_FORCEINLINE_FUNCTION
  T& operator()(int dir, Args... args) const {
    assert( 1 <= dir && dir <= 3 );
    if (dir == 1) return x1f(std::forward<Args>(args)...);
    if (dir == 2) return x2f(std::forward<Args>(args)...);
    else return x3f(std::forward<Args>(args)...); // i == 3
    // throw std::invalid_argument("Face must be x1f, x2f, or x3f");
  }

  bool isSet(const MetadataFlag bit) const { return _m.IsSet(bit); }

  ParArrayND<T> x1f;
  ParArrayND<T> x2f;
  ParArrayND<T> x3f;

 private:
  std::array<int,6> _dims;
  Metadata _m;
  std::string _label;
};

///
/// EdgeVariable extends the EdgeField struct to include the metadata
/// and label so that we can refer to variables by name.  Since Athena
/// currently only has scalar Edge fields, we also only allow scalar
/// edge fields
template <typename T>
class EdgeVariable : EdgeField {
 public:
  ///< retrieve metadata for variable
  const Metadata metadata() const { return _m; }

  /// Initialize a edge variable
  EdgeVariable(const std::string label, const Metadata &metadata,
               const int ncells3, const int ncells2, const int ncells1)
    : EdgeField(ncells3, ncells2, ncells1),
      _m(metadata),
      _label(label) {
      if ( metadata.IsSet(Metadata::Sparse) ) {
        throw std::invalid_argument ("Sparse not yet implemented for EdgeVaraible");
      }
  }

  /// Create an alias for the variable by making a shallow slice with max dim
  EdgeVariable(const std::string label, const EdgeVariable &src) :
    EdgeField(src),
    _m(src.metadata()),
    _label(label) {}

  bool isSet(const MetadataFlag bit) const { return _m.IsSet(bit); }
  ///< retrieve label for variable
  std::string label() { return _label; }

  /// return information string
  std::string info();

 private:
  Metadata _m;
  std::string _label;
};


template <typename T>
using CellVariableVector = std::vector<std::shared_ptr<CellVariable<T>>>;
template <typename T>
using FaceVector = std::vector<std::shared_ptr<FaceVariable<T>>>;

template <typename T>
using MapToCellVars = std::map<std::string, std::shared_ptr<CellVariable<T>>>;
template <typename T>
using MapToFace = std::map<std::string, std::shared_ptr<FaceVariable<T>>>;

/*
template<typename T>
class VariableVector : public std::vector<std::shared_ptr<Variable<T>>> {
 public:
  Variable<T>& operator()(int m) {
    return *(*this)[m];
  }
  T& operator()(int m, int i) {
    return (*(*this)[m])(i);
  }
  T& operator()(int m, int j, int i) {
    return (*(*this)[m])(j,i);
  }
  T& operator()(int m, int k, int j, int i) {
    return (*(*this)[m])(k,j,i);
  }
  T& operator()(int m, int l, int k, int j, int i) {
    return (*(*this)[m])(l,k,j,i);
  }
  T& operator()(int m, int n, int l, int k, int j, int i) {
    return (*(*this)[m])(n,l,k,j,i);
  }
  T& operator()(int m, int g, int n, int l, int k, int j, int i) {
    return (*(*this)[m])(g,n,l,k,j,i);
  }
};*/
} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
