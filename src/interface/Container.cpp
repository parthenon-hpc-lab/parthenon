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
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>
#include "bvals/cc/bvals_cc.hpp"
#include "Container.hpp"
#include "globals.hpp" // my_rank
#include "SparseVariable.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

///
/// The new version of Add that takes the fourth dimension from
/// the metadata structure
template <typename T>
void Container<T>::Add(const std::string label, const Metadata &metadata) {
  // generate the vector and call Add
  const std::vector<int>& dims = metadata.Shape();
  Add(label, metadata, dims);
}

template <typename T>
void Container<T>::Add(const std::vector<std::string> labelArray,
                       const Metadata &metadata) {
  // generate the vector and call Add
  for (auto label : labelArray) {
    Add(label, metadata);
  }
}

template <typename T>
void Container<T>::Add(const std::vector<std::string> labelArray,
                       const Metadata &metadata,
                       const std::vector<int> dims) {
  for (auto label : labelArray) {
    Add(label, metadata, dims);
  }
}

///
/// The internal routine for allocating an array.  This subroutine
/// is topology aware and will allocate accordingly.
///
/// @param label the name of the variable
/// @param dims the size of each element
/// @param metadata the metadata associated with the variable
template <typename T>
void Container<T>::Add(const std::string label,
                       const Metadata &metadata,
                       const std::vector<int> dims) {
  std::array<int, 6> arrDims;
  calcArrDims_(arrDims, dims);

  if ( metadata.Where() == Metadata::Node ) {
    arrDims[0]++; arrDims[1]++; arrDims[2]++;
  }

  // branch on kind of variable
  if (metadata.IsSet(Metadata::Sparse)) {
    // add a sparse variable
    if (_sparseMap.find(label) == _sparseMap.end()) {
      auto sv = std::make_shared<SparseVariable<T>>(label, metadata, arrDims);
      _sparseMap[label] = sv;
      _sparseVector.push_back(sv);
    }
    int varIndex = metadata.GetSparseId();
    _sparseMap[label]->Add(varIndex);
    if (metadata.IsSet(Metadata::FillGhost)) {
      Variable<T>& v = _sparseMap[label]->Get(varIndex);
      v.allocateComms(pmy_block);
    }
  } else if ( metadata.Where() == Metadata::Edge ) {
    // add an edge variable
    std::cerr << "Accessing unliving edge array in stage" << std::endl;
    std::exit(1);
    // s->_edgeVector.push_back(
    //     new EdgeVariable(label, metadata,
    //                      pmy_block->ncells3, pmy_block->ncells2, pmy_block->ncells1));
    return;
  } else if ( metadata.Where() == (Metadata::Face) ) {
    if ( !(metadata.IsSet(Metadata::OneCopy)) ) {
      std::cerr << "Currently one one-copy face fields are supported"
                << std::endl;
      std::exit(1);
    }
    if (metadata.IsSet(Metadata::FillGhost)) {
      std::cerr << "Ghost zones not yet supported for face fields" << std::endl;
      std::exit(1);
    }
    // add a face variable
    auto pfv = std::make_shared<FaceVariable<T>>(label, metadata, arrDims);
    _faceVector.push_back(pfv);
    return;
  } else {
    // plain old variable
    if ( dims.size() > 6 || dims.size() < 1 ) {
      throw std::invalid_argument ("_addArray() must have dims between [1,5]");
    }
    for (int i=0; i<dims.size(); i++) {arrDims[5-i] = dims[i];}
    auto sv = std::make_shared<Variable<T>>(label, arrDims, metadata);
    _varVector.push_back(sv);
    _varMap[label] = sv;
    if ( metadata.IsSet(Metadata::FillGhost) ) {
      _varVector.back()->allocateComms(pmy_block);
    }
  }
}

// provides a container that has a single sparse slice
template <typename T>
Container<T> Container<T>::sparseSlice(int id) {
  Container<T> c;

  // copy in private data
  c.pmy_block = pmy_block;

  // Note that all standard arrays get added
  // add standard arrays
  for (auto v : _varVector) {
    c._varVector.push_back(v);
    c._varMap[v->label()] = v;
  }
  // for (auto v : s->_edgeVector) {
  //   EdgeVariable *vNew = new EdgeVariable(v->label(), *v);
  //   c.s->_edgeVector.push_back(vNew);
  // }
  for (auto v : _faceVector) {
    c._faceVector.push_back(v);
    c._faceMap[v->label()] = v;
  }

  // Now copy in the specific arrays
  for (auto v : _sparseVector) {
    int index = v->GetIndex(id);
    if (index >= 0) {
      Variable<T>& vmat = v->Get(id);
      auto sv = std::make_shared<Variable<T>>(vmat);
      c._varVector.push_back(sv);
      c._varMap[v->label()] = sv;
    }
  }

  return c;
}

// TODO(JMM): this could be cleaned up, I think.
// Maybe do only one loop, or do the cleanup at the end.
template <typename T>
void Container<T>::Remove(const std::string label) {
  throw std::runtime_error("Container<T>::Remove not yet implemented");
}

template <typename T>
void Container<T>::SendFluxCorrection() {
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::Independent) ) {
      v->vbvar->SendFluxCorrection();
    }
  }
  for (auto &sv : _sparseVector) {
    if ( (sv->isSet(Metadata::Independent)) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        v->vbvar->SendFluxCorrection();
      }
    }
  }
}

template <typename T>
bool Container<T>::ReceiveFluxCorrection() {
  int success=0, total=0;
  for (auto &v : _varVector) {
    if (v->isSet(Metadata::Independent)) {
      if(v->vbvar->ReceiveFluxCorrection()) success++;
      total++;
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::Independent) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        if (v->vbvar->ReceiveFluxCorrection()) success++;
        total++;
      }
    }
  }
  return (success==total);
}

template <typename T>
void Container<T>::SendBoundaryBuffers() {
  // sends the boundary
  debug=0;
  //  std::cout << "_________SEND from stage:"<<s->name()<<std::endl;
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::FillGhost) ) {
      v->resetBoundary();
      v->vbvar->SendBoundaryBuffers();
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::FillGhost) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        v->resetBoundary();
        v->vbvar->SendBoundaryBuffers();
      }
    }
  }

  return;
}

template <typename T>
void Container<T>::SetupPersistentMPI() {
  // setup persistent MPI
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::FillGhost) ) {
        v->resetBoundary();
        v->vbvar->SetupPersistentMPI();
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::FillGhost) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        v->resetBoundary();
        v->vbvar->SetupPersistentMPI();
      }
    }
  }
  return;
}

template <typename T>
bool Container<T>::ReceiveBoundaryBuffers() {
  bool ret;
  //  std::cout << "_________RECV from stage:"<<s->name()<<std::endl;
  ret = true;
  // receives the boundary
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::FillGhost) ) {
      //ret = ret & v->vbvar->ReceiveBoundaryBuffers();
      // In case we have trouble with multiple arrays causing
      // problems with task status, we should comment one line
      // above and uncomment the if block below
      if (! v->mpiStatus) {
        v->resetBoundary();
        v->mpiStatus = v->vbvar->ReceiveBoundaryBuffers();
        ret = (ret & v->mpiStatus);
      }
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::FillGhost) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        if (! v->mpiStatus) {
          v->resetBoundary();
          v->mpiStatus = v->vbvar->ReceiveBoundaryBuffers();
          ret = (ret & v->mpiStatus);
        }
      }
    }
  }

  return ret;
}

template <typename T>
void Container<T>::ReceiveAndSetBoundariesWithWait() {
  //  std::cout << "_________RSET from stage:"<<s->name()<<std::endl;
  for (auto &v : _varVector) {
    if ( (!v->mpiStatus) && v->isSet(Metadata::FillGhost) ) {
      v->resetBoundary();
      v->vbvar->ReceiveAndSetBoundariesWithWait();
      v->mpiStatus = true;
    }
  }
  for (auto &sv : _sparseVector) {
    if ( (sv->isSet(Metadata::FillGhost)) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        if (! v->mpiStatus) {
          v->resetBoundary();
          v->vbvar->ReceiveAndSetBoundariesWithWait();
          v->mpiStatus = true;
        }
      }
    }
  }
}
// This really belongs in Container.cpp. However if I put it in there,
// the meshblock file refuses to compile.  Don't know what's going on
// there, but for now this is the workaround at the expense of code
// bloat.
template <typename T>
void Container<T>::SetBoundaries() {
  //    std::cout << "in set" << std::endl;
  // sets the boundary
  //  std::cout << "_________BSET from stage:"<<s->name()<<std::endl;
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::FillGhost) ) {
      v->resetBoundary();
      v->vbvar->SetBoundaries();
      //v->mpiStatus=true;
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::FillGhost) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        v->resetBoundary();
        v->vbvar->SetBoundaries();
      }
    }
  }
}

template <typename T>
void Container<T>::ResetBoundaryVariables() {
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::FillGhost) ) {
      v->vbvar->var_cc = &(v->data);
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::FillGhost) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        v->vbvar->var_cc = &(v->data);
      }
    }
  }
}

template <typename T>
void Container<T>::StartReceiving(BoundaryCommSubset phase) {
  //    std::cout << "in set" << std::endl;
  // sets the boundary
  //  std::cout << "________CLEAR from stage:"<<s->name()<<std::endl;
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::FillGhost) ) {
      v->resetBoundary();
      v->vbvar->StartReceiving(phase);
      v->mpiStatus=false;
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::FillGhost) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        v->resetBoundary();
        v->vbvar->StartReceiving(phase);
        v->mpiStatus = false;
      }
    }
  }
}

template <typename T>
void Container<T>::ClearBoundary(BoundaryCommSubset phase) {
  //    std::cout << "in set" << std::endl;
  // sets the boundary
  //  std::cout << "________CLEAR from stage:"<<s->name()<<std::endl;
  for (auto &v : _varVector) {
    if ( v->isSet(Metadata::FillGhost) ) {
      v->vbvar->ClearBoundary(phase);
    }
  }
  for (auto &sv : _sparseVector) {
    if ( sv->isSet(Metadata::FillGhost) ) {
      VariableVector<T> vvec = sv->GetVector();
      for (auto & v : vvec) {
        v->vbvar->ClearBoundary(phase);
      }
    }
  }
}

template<typename T>
void Container<T>::print() {
  std::cout << "Variables are:\n";
  for (auto v : _varVector)  {   std::cout << " cell: " <<v->info() << std::endl; }
  for (auto v : _faceVector) {   std::cout << " face: " <<v->info() << std::endl; }
  for (auto v : _sparseVector) { std::cout << " sparse:"<<v->info() << std::endl; }
  //  for (auto v : s->_edgeVector) { std::cout << " edge: "<<v->info() << std::endl; }
}

template <typename T>
static int AddVar(Variable<T>&V, std::vector<Variable<T>>& vRet) {
  // adds aliases to vRet
  const int d6 = V.GetDim(6);
  const int d5 = V.GetDim(5);
  const int d4 = V.GetDim(5);
  const std::string label = V.label();

  for (int i6=0; i6<d6; i6++) {
    Variable<T> V6(label,V,6,0,d6);
    for (int i5=0; i5<d5; i5++) {
      Variable<T> V5(label,V6,5,0,d5);
      for (int i4=0; i4<d4; i4++) {
        vRet.push_back(Variable<T>(label,V5,4,0,d5));
      }
    }
  }
  return d6*d5*d4;
}

template<typename T>
void Container<T>::calcArrDims_(std::array<int, 6>& arrDims,
                                const std::vector<int>& dims) {
  const int N = dims.size();
  if ( N > 3 || N < 0 ) {
    // too many dimensions
    throw std::invalid_argument(std::string("Variable must be scalar or")
                                +std::string(" rank-N tensor-field, for N < 4"));
  }
  for (int i = 0; i < 6; i++) arrDims[i] = 1;
  arrDims[0] = pmy_block->ncells1;
  arrDims[1] = pmy_block->ncells2;
  arrDims[2] = pmy_block->ncells3;
  for (int i=0; i<N; i++) {arrDims[i+3] = dims[i]; }
}

template class Container<double>;

} // namespace parthenon
