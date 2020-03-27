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
/// Builds on AthenaArrays
/// Date: August 21, 2019
///
///
/// The variable class typically contains state data for the
/// simulation but can also hold non-mesh-based data such as physics
/// parameters, etc.  It inherits the AthenaArray class, which is used
/// for actural data storage and generation

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "bvals/cc/bvals_cc.hpp"
#include "Metadata.hpp"
#define DATASTATUS AthenaArray<Real>::DataStatus

namespace parthenon {
class MeshBlock;

template <typename T>
class Variable {
 public:
  /// Initialize a blank slate that will be set later
  Variable<T>(const std::string label, Metadata &metadata) :
    mpiStatus(true),
    _m(metadata),
    _label(label) {
    //    std::cout << "_____CREATED VAR: " << _label << ":" << this << std::endl;
  }

  /// Initialize with a slice from another Variable
  Variable<T>(const std::string label,
              Variable<T> &src,
              const int dim,
              const int index,
              const int nvar) :
    mpiStatus(true),
    _m(src.metadata()),
    _label(label) {
    data->InitWithShallowSlice(src, dim, index, nvar);
    if ( _m.isSet(_m.fillGhost) ) {
      _m.set(_m.sharedComms);
    }
    //    std::cout << "_____CREATED VAR SLICE: " << _label << ":" << this << std::endl;
  }

  /// Create an alias for the variable by making a shallow slice with max dim
  Variable<T>(const std::string label, Variable<T> &src) :
    AthenaArray<T>(),
    mpiStatus(true),
    _m(src.metadata()),
    _label(label) {
    int dim = 6;
    int start = 0;
    int nvar = src.GetDim6();
    data->InitWithShallowSlice(src, dim, start, nvar);
    _m.set(_m.sharedComms);
    //    std::cout << "_____CREATED VAR SLICE: " << _label << ":" << this << std::endl;
  }


  /// Initialize a 6D variable
  Variable<T>(const std::string label,
              const std::array<int,6> dims,
              const Metadata &metadata) :
    data(dims[5], dims[4], dims[3], dims[2], dims[1], dims[0]),
    mpiStatus(true),
    _m(metadata),
    _label(label) {
    //    std::cout << "_____CREATED 6D VAR: " << _label << ":" << this << std::endl;
  }

  /// copy constructor
  Variable<T>(const Variable<T>& src,
              const bool allocComms=false,
              MeshBlock *pmb=nullptr);

  /// Return a new array with dimension 4 as dimension 1
  /*Variable<T>* createShuffle4() {
    // shuffles dim 4 to dim1

    // first construct new athena array
    const std::array<int,6> dims = {GetDim4(), GetDim1(), GetDim2(),
                                    GetDim3(), GetDim5(), GetDim6()};
    size_t stride = GetDim1()*GetDim2()*GetDim3();
    Variable<T> *vNew = new Variable<T>(_label, dims, _m);

    auto oldData = this->data();
    auto newData = vNew->data();
    // now fill in the data
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
  std::string label() const { return _label; }

  ///< retrieve metadata for variable
  const Metadata metadata() const { return _m; }

  std::string getAssociated() { return _m.getAssociated(); }

  bool isSet(const Metadata::flags flag) { return _m.isSet(flag); }

  ///  T *Raw(); ///< returns raw data for variable

  /// return information string
  std::string info();

  /// allocate communication space based on info in MeshBlock
  void allocateComms(MeshBlock *pmb);

  /// Repoint vbvar's var_cc array at the current variable
  inline void ResetBoundary() {
    this->vbvar->var_cc = this;
  };

  AthenaArray<T> data;
  AthenaArray<Real> flux[3];    // used for boundary calculation
  AthenaArray<Real> *coarse_s;  // used for sending coarse boundary calculation
  AthenaArray<Real> *coarse_r;  // used for sending coarse boundary calculation
  CellCenteredBoundaryVariable *vbvar; // used in case of cell boundary communication
  bool mpiStatus;

 private:
  Metadata _m;
  std::string _label;
};



///
/// FaceVariable extends the FaceField struct to include the metadata
/// and label so that we can refer to variables by name.  Since Athena
/// currently only has scalar Face fields, we also only allow scalar
/// face fields
struct FaceVariable : FaceField {
 public:
  /// Initialize a face variable
  FaceVariable(const std::string label, const Metadata &metadata,
               const std::array<int,6> ncells,
               const DATASTATUS init=DATASTATUS::allocated) :
    FaceField(ncells[5], ncells[4], ncells[3], ncells[2], ncells[1], ncells[0], init),
    _m(metadata),
    _label(label) {
    if ( metadata.hasSparse() ) {
      throw std::invalid_argument ("Sparse not yet implemented for FaceVariable");
    }
  }

  /// Create an alias for the variable by making a shallow slice with max dim
  FaceVariable(std::string label, FaceVariable &src) :
    FaceField(0,0,0,DATASTATUS::allocated),
    _m(src.metadata()),
    _label(label) {
    int dim = 6;
    int start = 0;
    this->x1f.InitWithShallowSlice(src.x1f, dim, start, src.x1f.GetDim6());
    this->x2f.InitWithShallowSlice(src.x2f, dim, start, src.x2f.GetDim6());
    this->x3f.InitWithShallowSlice(src.x3f, dim, start, src.x3f.GetDim6());
  }

  ///< retrieve label for variable
  std::string label() { return _label; }

  ///< retrieve metadata for variable
  Metadata metadata() { return _m; }

  bool isSet(const Metadata::flags flag) { return _m.isSet(flag); }

  /// return information string
  std::string info();

  // TODO(JMM): should this be 0,1,2?
  // Should we return the reference? Or something else?
  AthenaArray<Real>& Get(int i) {
    if (i == 1) return (this->x1f);
    if (i == 2) return (this->x2f);
    if (i == 3) return (this->x3f);
    throw std::invalid_argument("Face must be x1f, x2f, or x3f");
  }
  template<typename...Args>
  Real& operator()(int dir, Args... args) {
    if (dir == 1) return x1f(std::forward<Args>(args)...);
    if (dir == 2) return x2f(std::forward<Args>(args)...);
    if (dir == 3) return x3f(std::forward<Args>(args)...);
    throw std::invalid_argument("Face must be x1f, x2f, or x3f");
  }

 private:
  Metadata _m;
  std::string _label;
};

///
/// EdgeVariable extends the EdgeField struct to include the metadata
/// and label so that we can refer to variables by name.  Since Athena
/// currently only has scalar Edge fields, we also only allow scalar
/// edge fields
struct EdgeVariable : EdgeField {
 public:
  ///< retrieve metadata for variable
  Metadata metadata() const { return _m; }

  /// Initialize a edge variable
  EdgeVariable(const std::string label, const Metadata &metadata,
               const int ncells3, const int ncells2, const int ncells1,
               const DATASTATUS init=DATASTATUS::allocated) :
    EdgeField(ncells3, ncells2, ncells1, init),
    _m(metadata),
    _label(label) {
    if ( metadata.hasSparse() ) {
      throw std::invalid_argument ("Sparse not yet implemented for FaceVariable");
    }
  }

  /// Create an alias for the variable by making a shallow slice with max dim
  EdgeVariable(const std::string label, const EdgeVariable &src) :
    EdgeField(0,0,0,DATASTATUS::allocated),
    _m(src.metadata()),
    _label(label) {
    this->x1e.InitWithShallowSlice(src.x1e, 3, 0, src.x1e.GetDim3());
    this->x2e.InitWithShallowSlice(src.x2e, 3, 0, src.x2e.GetDim3());
    this->x3e.InitWithShallowSlice(src.x3e, 3, 0, src.x3e.GetDim3());
  }

  ///< retrieve label for variable
  std::string label() { return _label; }

  /// return information string
  std::string info();

 private:
  Metadata _m;
  std::string _label;
};

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
  Metadata& metadata() const { return this->begin().second->metadata();}
};
} // namespace parthenon

#endif // INTERFACE_VARIABLE_HPP_
