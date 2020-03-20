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
  // branch on kind of variable
  if (metadata.hasSparse()) {
    // add a sparse variable
    s->_sparseVars.Add(*pmy_block, label, metadata, dims);
  } else if ( metadata.where() == (Metadata::edge) ) {
    // add an edge variable
    std::cerr << "Accessing unliving edge array in stage" << std::endl;
    std::exit(1);
    // s->_edgeArray.push_back(
    //     new EdgeVariable(label, metadata,
    //                      pmy_block->ncells3, pmy_block->ncells2, pmy_block->ncells1));
    return;
  } else if ( metadata.where() == (Metadata::face) ) {
    if ( !(metadata.isOneCopy()) ) {
      std::cerr << "Currently one one-copy face fields are supported"
                << std::endl;
      std::exit(1);
    }
    if (metadata.fillsGhost()) {
      std::cerr << "Ghost zones not yet supported for face fields" << std::endl;
      std::exit(1);
    }
    // add a face variable
    auto pfv = std::make_shared<FaceVariable>(label, metadata, arrDims);
    s->_faceArray.push_back(pfv);
    return;
  } else if ( (metadata.where() == (Metadata::cell) ) ||
            (metadata.where() == (Metadata::node) )) {
    if ( metadata.where() == (Metadata::node) ) {
      arrDims[0]++; arrDims[1]++; arrDims[2]++;
    }

    s->_varArray.push_back(std::make_shared<Variable<T>>(label, arrDims, metadata));
    if ( metadata.fillsGhost()) {
      s->_varArray.back()->allocateComms(pmy_block);
    }
  } else {
    // plain old variable
    if ( dims.size() > 6 || dims.size() < 1 ) {
      throw std::invalid_argument ("_addArray() must have dims between [1,5]");
    }
    for (int i=0; i<dims.size(); i++) {arrDims[5-i] = dims[i];}
    s->_varArray.push_back(std::make_shared<Variable<T>>(label, arrDims, metadata));
  }
}

/*
template <typename T>
void Container<T>::StageSet(std::string name) {
  s = stages[name];

  for (auto &v : s->_varArray) {
    if ( (v->metadata()).fillsGhost() ) {
      v->resetBoundary();
      //v->vbvar->var_cc = v.get();
      //      v->mpiStatus=true;
    }
  }

  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( (v.second->metadata()).fillsGhost()) {
        v.second->resetBoundary();
        //v.second->vbvar->var_cc = v.second.get();
        //v.second->mpiStatus=true;
      }
    }
  }
}*/

// provides a container that has a single sparse slice
template <typename T>
Container<T> Container<T>::sparseSlice(int id) {
  Container<T> c;

  // copy in private data
  c.pmy_block = pmy_block;

  // Note that all standard arrays get added
  // add standard arrays
  for (auto v : _varArray) {
    c._varArray.push_back(v);
  }
  // for (auto v : s->_edgeArray) {
  //   EdgeVariable *vNew = new EdgeVariable(v->label(), *v);
  //   c.s->_edgeArray.push_back(vNew);
  // }
  for (auto v : _faceArray) {
    c._faceArray.push_back(v);
  }

  // Now copy in the specific arrays
  auto null_shared_ptr = std::shared_ptr<Variable<T>>(nullptr);
  for (auto v : _sparseArray) {
    Variable<T>& vmat = v->Get(id);
    if (vmat != null_shared_ptr) {
      c._varArray.push_back(std::shared_ptr<Variable<T>>(&vmat));
    }
  }

  return std::move(c);
}

// TODO(JMM): this could be cleaned up, I think.
// Maybe do only one loop, or do the cleanup at the end.
template <typename T>
void Container<T>::Remove(const std::string label) {
  // first find the index of our
  int idx, isize;

  // Check face variables
  idx = 0;
  isize = s->_faceArray.size();
  for (auto v : s->_faceArray) {
    if ( ! label.compare(v->label()) ) break;
    idx++;
  }
  if (idx < isize) {
    s->_faceArray[idx].reset();
    isize--;
    if (isize >= 0) s->_faceArray[idx] = std::move(s->_faceArray.back());
    s->_faceArray.pop_back();
    return;
  }

  // No face match so check edge variables
  // TODO(JMM): fixme
  // idx = 0;
  // for (auto v : s->_edgeArray) {
  //   if ( ! label.compare(v->label())) {
  //     // found a match, remove it
  //     s->_edgeArray.erase(s->_edgeArray.begin() + idx);
  //     return;
  //   }
  // }

  // no face or edge, so check sized variables
  isize = s->_varArray.size();
  idx = 0;
  for (auto v : s->_varArray) {
    if ( ! label.compare(v->label())) {
      break;
    }
    idx++;
  }
  if ( idx >= isize) {
    throw std::invalid_argument ("array not found in Remove()");
  }

  // first delete the variable
  s->_varArray[idx].reset();

  // Next move the last element into idx and pop last entry
  isize--;
  if ( isize >= 0) s->_varArray[idx] = std::move(s->_varArray.back());
  s->_varArray.pop_back();
  return;
}

template <typename T>
void Container<T>::SendFluxCorrection() {
  for (auto &v : s->_varArray) {
    if ( (v->metadata()).isIndependent() ) {
      v->vbvar->SendFluxCorrection();
    }
  }
  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    for (auto &mv : myMap.second) {
      auto &v = mv.second;
      if ( (v->metadata()).isIndependent() ) {
        v->vbvar->SendFluxCorrection();
      }
    }
  }
}

template <typename T>
bool Container<T>::ReceiveFluxCorrection() {
  int success=0, total=0;
  for (auto &v : _varArray) {
    if ( (v->metadata()).isIndependent() ) {
      if(v->vbvar->ReceiveFluxCorrection()) success++;
      total++;
    }
  }
  for (auto &v : _sparseArray) {
    
  }
  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    for (auto &mv : myMap.second) {
      auto &v = mv.second;
      if ( (v->metadata()).isIndependent() ) {
        if(v->vbvar->ReceiveFluxCorrection()) success++;
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
  for (auto &v : s->_varArray) {
    if ( (v->metadata()).fillsGhost() ) {
      if ( ! v->mpiStatus ) {
        std::cout << "        sending without the receive, something's up:"
                  << v->label()
                  << std::endl;
      }
      v->vbvar->SendBoundaryBuffers();
      v->mpiStatus = false;
    }
  }

  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( ! (v.second->metadata()).fillsGhost() ) continue; // doesn't fill ghost so skip
      if ( ! v.second->mpiStatus ) {
        std::cout << "        _________Err:"
                  << v.first
                  << ":sending without the receive, something's up:"
                  << v.second->label()
                  << std::endl;
      }
      v.second->vbvar->SendBoundaryBuffers();
      v.second->mpiStatus = false;
    }
  }

  return;
}

template <typename T>
void Container<T>::SetupPersistentMPI() {
  // setup persistent MPI
  for (auto &v : s->_varArray) {
    if ( (v->metadata()).fillsGhost() ) {
        v->vbvar->SetupPersistentMPI();
      }
    }


  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( ! (v.second->metadata()).fillsGhost() ) continue; // doesn't fill ghost so skip
      if ( ! v.second->mpiStatus ) {
        v.second->vbvar->SetupPersistentMPI();
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
  for (auto &v : s->_varArray) {
    if ( (v->metadata()).fillsGhost() ) {
      //ret = ret & v->vbvar->ReceiveBoundaryBuffers();
      // In case we have trouble with multiple arrays causing
      // problems with task status, we should comment one line
      // above and uncomment the if block below
      if (! v->mpiStatus) {
        v->mpiStatus = v->vbvar->ReceiveBoundaryBuffers();
        ret = (ret & v->mpiStatus);
      }
    }
  }

  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( ! (v.second->metadata()).fillsGhost() ) continue; // doesn't fill ghost so skip
      if ( ! v.second->mpiStatus ) {
        v.second->mpiStatus = v.second->vbvar->ReceiveBoundaryBuffers();
        ret = (ret & v.second->mpiStatus);
      }
    }
  }


  return ret;
}

template <typename T>
void Container<T>::ReceiveAndSetBoundariesWithWait() {
  //  std::cout << "_________RSET from stage:"<<s->name()<<std::endl;
  for (auto &v : s->_varArray) {
    if ( (!v->mpiStatus) && ( (v->metadata()).fillsGhost()) ) {
      v->vbvar->ReceiveAndSetBoundariesWithWait();
      v->mpiStatus = true;
    }
  }
  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( ! (v.second->metadata()).fillsGhost() ) continue; // doesn't fill ghost so skip
      if ( ! v.second->mpiStatus ) {
        v.second->vbvar->ReceiveAndSetBoundariesWithWait();
        v.second->mpiStatus = true;
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
  for (auto &v : s->_varArray) {
    if ( (v->metadata()).fillsGhost() ) {
      v->vbvar->SetBoundaries();
      //v->mpiStatus=true;
    }
  }

  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( (v.second->metadata()).fillsGhost() ) {
        v.second->vbvar->SetBoundaries();
        //v.second->mpiStatus=true;
      }
    }
  }
}

template <typename T>
void Container<T>::StartReceiving(BoundaryCommSubset phase) {
  //    std::cout << "in set" << std::endl;
  // sets the boundary
  //  std::cout << "________CLEAR from stage:"<<s->name()<<std::endl;
  for (auto &v : s->_varArray) {
    if ( (v->metadata()).fillsGhost() ) {
      v->vbvar->StartReceiving(phase);
      //      v->mpiStatus=true;
    }
  }

  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( (v.second->metadata()).fillsGhost()) {
        v.second->vbvar->StartReceiving(phase);
        //v.second->mpiStatus=true;
      }
    }
  }
}

template <typename T>
void Container<T>::ClearBoundary(BoundaryCommSubset phase) {
  //    std::cout << "in set" << std::endl;
  // sets the boundary
  //  std::cout << "________CLEAR from stage:"<<s->name()<<std::endl;
  for (auto &v : s->_varArray) {
    if ( (v->metadata()).fillsGhost() ) {
      v->vbvar->ClearBoundary(phase);
      //      v->mpiStatus=true;
    }
  }

  for (auto &myMap : s->_sparseVars.getAllCellVars()) {
    // for every variable Map in the sparse variables array
    for (auto &v : myMap.second) {
      if ( (v.second->metadata()).fillsGhost()) {
        v.second->vbvar->ClearBoundary(phase);
        //v.second->mpiStatus=true;
      }
    }
  }
}

template<typename T>
void Container<T>::print() {
  std::cout << "Variables are:\n";
  for (auto v : s->_varArray)  { std::cout << " cell: "<<v->info() << std::endl; }
  for (auto v : s->_faceArray) { std::cout << " face: "<<v->info() << std::endl; }
  //  for (auto v : s->_edgeArray) { std::cout << " edge: "<<v->info() << std::endl; }
  s->_sparseVars.print();
}

template <typename T>
static int AddVar(Variable<T>&V, std::vector<Variable<T>>& vRet) {
  // adds aliases to vRet
  const int d6 = V.GetDim6();
  const int d5 = V.GetDim5();
  const int d4 = V.GetDim4();
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


/// Gets an array of real variables from container.
/// @param index_ret is returned with starting index for each name
/// @param count_ret is returned with number of arrays for each name
/// @param sparse_ids if specified, only those sparse IDs are returned
template<typename T>
int Container<T>::GetVariables(const std::vector<std::string>& names,
                               std::vector<Variable<T>>& vRet,
                               std::map<std::string,std::pair<int,int>>& indexCount,
                               const std::vector<int>& sparse_ids) {
  // First count how many entries we need and fill in index and count
  indexCount.clear();

  int index = 0;
  for (auto label : names) {
    int count = 0;
    try { // normal variable
      Variable<T>& V = Get(label);
      count += AddVar(V, vRet);
    }
    catch (const std::invalid_argument& x) {
      // Not a regular variable, so try a sparse variable
      try { // sparse variable
        SparseMap<T>& M = GetSparse(label);
        if ( M.size() > 0) {
          if ( sparse_ids.size() > 0) {
            for (auto& id : sparse_ids) {
              // Want a specific index
              auto exists = M.find(id);
              if ( exists != M.end() ) {
                auto&V = *(exists->second);
                count += AddVar(V, vRet);
              }
            }  // (auto& id : sparse_ids)
          } else { // if (sparse_ids.size() > 0)
            auto&V = *(M.begin()->second);
            count = count*V.GetDim6()*V.GetDim5()*V.GetDim4();
            for (auto& x : M) {
              auto&V = *x.second;
              count += AddVar(V, vRet);
            }
          } // else (sparse_ids.size() > 0)
        } // if (M.size() > 0)
      } catch (const std::invalid_argument& x) {
        // rethrow exception because we want to die here
        throw std::invalid_argument (" Unable to find variable " +
                                     label + " in container");
      } // sparse variable
    } // normal variable
    indexCount[label] = std::make_pair(index,count);
    index += count;
  } // (auto label : names)

  return index;
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
