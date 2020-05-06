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

#include "interface/container.hpp"

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "bvals/cc/bvals_cc.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

/// The new version of Add that takes the fourth dimension from
/// the metadata structure
template <typename T>
void Container<T>::Add(const std::string label, const Metadata &metadata) {
  // generate the vector and call Add
  const std::vector<int> &dims = metadata.Shape();
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
                       const Metadata &metadata, const std::vector<int> dims) {
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
void Container<T>::Add(const std::string label, const Metadata &metadata,
                       const std::vector<int> dims) {
  std::array<int, 6> arrDims;
  calcArrDims_(arrDims, dims, metadata);

  // branch on kind of variable
  if (metadata.IsSet(Metadata::Sparse)) {
    // add a sparse variable
    if (sparseMap_.find(label) == sparseMap_.end()) {
      auto sv = std::make_shared<SparseVariable<T>>(label, metadata, arrDims);
      sparseMap_[label] = sv;
      sparseVector_.push_back(sv);
    }
    int varIndex = metadata.GetSparseId();
    sparseMap_[label]->Add(varIndex);
    if (metadata.IsSet(Metadata::FillGhost)) {
      CellVariable<T> &v = sparseMap_[label]->Get(varIndex);
      v.allocateComms(pmy_block);
    }
  } else if (metadata.Where() == Metadata::Edge) {
    // add an edge variable
    std::cerr << "Accessing unliving edge array in stage" << std::endl;
    std::exit(1);
    // s->_edgeVector.push_back(
    //     new EdgeVariable(label, metadata,
    //                      pmy_block->ncells3, pmy_block->ncells2, pmy_block->ncells1));
  } else if (metadata.Where() == Metadata::Face) {
    if (!(metadata.IsSet(Metadata::OneCopy))) {
      std::cerr << "Currently one one-copy face fields are supported" << std::endl;
      std::exit(1);
    }
    if (metadata.IsSet(Metadata::FillGhost)) {
      std::cerr << "Ghost zones not yet supported for face fields" << std::endl;
      std::exit(1);
    }
    // add a face variable
    auto pfv = std::make_shared<FaceVariable<T>>(label, arrDims, metadata);
    faceVector_.push_back(pfv);
    faceMap_[label] = pfv;
  } else {
    auto sv = std::make_shared<CellVariable<T>>(label, arrDims, metadata);
    varVector_.push_back(sv);
    varMap_[label] = sv;
    if (metadata.IsSet(Metadata::FillGhost)) {
      sv->allocateComms(pmy_block);
    }
  }
}

// provides a container that has a single sparse slice
template <typename T>
Container<T> Container<T>::SparseSlice(int id) {
  Container<T> c;

  // copy in private data
  c.pmy_block = pmy_block;

  // Note that all standard arrays get added
  // add standard arrays
  for (auto v : varVector_) {
    c.varVector_.push_back(v);
    c.varMap_[v->label()] = v;
  }
  // for (auto v : s->_edgeVector) {
  //   EdgeVariable *vNew = new EdgeVariable(v->label(), *v);
  //   c.s->_edgeVector.push_back(vNew);
  // }
  for (auto v : faceVector_) {
    c.faceVector_.push_back(v);
    c.faceMap_[v->label()] = v;
  }

  // Now copy in the specific arrays
  for (auto v : sparseVector_) {
    int index = v->GetIndex(id);
    if (index >= 0) {
      CellVariable<T> &vmat = v->Get(id);
      auto sv = std::make_shared<CellVariable<T>>(vmat);
      c.varVector_.push_back(sv);
      c.varMap_[v->label()] = sv;
    }
  }

  return c;
}

/// Queries related to variable packs
/// TODO(JMM): Make sure this is thread-safe
/// TODO(JMM): Should the vector of names be sorted to enforce uniqueness?
template<typename T>
VariableFluxPack<T> Container<T>::PackVariablesAndFluxes_(
  const std::vector<std::string> &var_names,
  const std::vector<std::string> &flx_names,
  const vpack_types::VarList<T> &vars,
  const vpack_types::VarList<T> &fvars,
  PackIndexMap &vmap) {
  auto key = std::make_pair(var_names,flx_names);
  auto kvpair = varFluxPackMap_.find(key);
  if (kvpair == varFluxPackMap_.end()) {
    auto pack = MakeFluxPack<T>(vars, fvars, &vmap)
    varFluxPackMap_[key] = std::make_pair(pack,vmap);
    return pack;
  }
  vmap = std::get<2>(kvpair->second);
  return std::get<1>(kvpair->second);  
}

template<typename T>
VariableFluxPack<T> Container<T>::PackVariablesAndFluxes(
  const std::vector<std::string> &var_names,
  const std::vector<std::string> &flx_names,
  PackIndexMap &vmap) {
  static std::vector<std::string> all_names;
  static std::vector<std::string> all_flux_names;
  vpack_types::VarList<T> vars = MakeList(*this, var_names, {}, &all_names);
  vpack_types::VarList<T> fvars = MakeList(*this, flx_names, {} &all_flux_names);
  return this->PackVariablesAndFluxes_(all_names,all_flux_names,vars,fvars,vmap);
}
template<typename T>
VariableFluxPack<T> Container<T>::PackVariablesAndFluxes(
  const std::vector<std::string> &var_names,
  const std::vector<std::string> &flx_names) {
  PackIndexMap vmap;
  return this->PackVariablesAndFluxes(var_names, flx_names, vmap);
}
template<typename T>
VariableFluxPack<T> Container<T>::PackVariablesAndFluxes(
  const std::vector<MetadataFlag> &flags,
  PackIndexMap &vmap) {
  // TODO(JMM): making this static saves allocations per call
  // but will not be thread safe.
  static std::vector<std::string> vnams;
  vpack_types::VarList<T> vars = MakeList(*this,flags,vnams);
  return this->PackVariablesAndFluxes_(vnams,vnams,vars,vars,vmap);
}
template<typename T>
VariableFluxPack<T> Container<T>::PackVariablesAndFluxes(
  const std::vector<MetadataFlag> &flags) {
  PackIndexMap vmap;
  return this->PackVariablesAndFluxes(flags, vmap);
}

template<typename T>
VariablePack<T> Container<T>::PackVariables_(const std::vector<std::string> &names,
                                             const vpack_types::VarList<T> &vars,
                                             PackIndexMap &vmap) {
  auto kvpair = varPackMap_.find(names);
  if (kvpair == varPackMap_.end()) {
    auto pack = MakeFluxPack<T>(vars, &vmap);
    varPackMap_[key] = std::make_pair(pack,vmap);
    return pack;
  }
  vmap = std::get<2>(kvpair->second);
  return std::get<1>(kvpair->second); 
}
template<typename T>
VariablePack<T> Container<T>::PackVariables(const std::vector<std::string> &names,
                                            const std::vector<int> &sparse_ids,
                                            PackIndexMap &vmap) {
  static std::vector<std::string> all_names;
  vpack_types::VarList<T> vars = MakeList(c, names, sparse_ids, &all_names);
  return this->PackVariables(all_names, vars, vmap);
}
template <typename T>
VariablePack<T> Container<T>::PackVariables(const std::vector<std::string> &names,
                                            const std::vector<int> &sparse_ids) {
  PackIndexMap vmap;
  return this->PackVariables(names, sparse_ids, vmap);
}
template<typename T>
VariablePack<T> Container<T>::PackVariables(const std::vector<std::string> &names,
                                            PackIndexMap &vmap) {
  return this->PackVariables(names, {}, vmap);
}
template<typename T>
VariablePack<T> Container<T>::PackVariables(const std::vector<std::string> &names) {
  PackIndexMap vmap;
  return this->PackVariables(names, {}, vmap);
}
template<typename T>
VariablePack<T> Container<T>::PackVariables(const std::vector<MetadataFlag> &flags,
                                            PackIndexMap &vmap) {
  static std::vector<std::string> vnams;
  vpack_types::VarList<T> vars = MakeList(*this,flags,vnams);
  return this->PackVariables_(vnams,vars,vmap);
}
template<typename T>
VariablePack<T> Container<T>::PackVariables(const std::vector<MetadataFlag> &flags) {
  PackIndexMap &vmap;
  return this->PackVariables(flags,vmap);
}
template<typename T>
VariablePack<T> Container<T>::PackVariables(PackIndexMap &vmap) {
  static std::vector<std::string> vnams;
  vpack_types::VarList<T> vars = MakeList(*this, &vnams);
  return this->PackVariables_(vnams,vars,vmap);
}
template<typename T>
VariablePack<T> Container<T>::PackVariables() {
  PackIndexMap &vmap;
  return this->PackVariables(vmap);
}


// TODO(JMM): this could be cleaned up, I think.
// Maybe do only one loop, or do the cleanup at the end.
template <typename T>
void Container<T>::Remove(const std::string label) {
  throw std::runtime_error("Container<T>::Remove not yet implemented");
}

template <typename T>
void Container<T>::SendFluxCorrection() {
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::Independent)) {
      v->vbvar->SendFluxCorrection();
    }
  }
  for (auto &sv : sparseVector_) {
    if ((sv->IsSet(Metadata::Independent))) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
        v->vbvar->SendFluxCorrection();
      }
    }
  }
}

template <typename T>
bool Container<T>::ReceiveFluxCorrection() {
  int success = 0, total = 0;
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::Independent)) {
      if (v->vbvar->ReceiveFluxCorrection()) success++;
      total++;
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::Independent)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
        if (v->vbvar->ReceiveFluxCorrection()) success++;
        total++;
      }
    }
  }
  return (success == total);
}

template <typename T>
void Container<T>::SendBoundaryBuffers() {
  // sends the boundary
  debug = 0;
  //  std::cout << "_________SEND from stage:"<<s->name()<<std::endl;
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::FillGhost)) {
      v->resetBoundary();
      v->vbvar->SendBoundaryBuffers();
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::FillGhost)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
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
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::FillGhost)) {
      v->resetBoundary();
      v->vbvar->SetupPersistentMPI();
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::FillGhost)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
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
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::FillGhost)) {
      // ret = ret & v->vbvar->ReceiveBoundaryBuffers();
      // In case we have trouble with multiple arrays causing
      // problems with task status, we should comment one line
      // above and uncomment the if block below
      if (!v->mpiStatus) {
        v->resetBoundary();
        v->mpiStatus = v->vbvar->ReceiveBoundaryBuffers();
        ret = (ret & v->mpiStatus);
      }
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::FillGhost)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
        if (!v->mpiStatus) {
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
  for (auto &v : varVector_) {
    if ((!v->mpiStatus) && v->IsSet(Metadata::FillGhost)) {
      v->resetBoundary();
      v->vbvar->ReceiveAndSetBoundariesWithWait();
      v->mpiStatus = true;
    }
  }
  for (auto &sv : sparseVector_) {
    if ((sv->IsSet(Metadata::FillGhost))) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
        if (!v->mpiStatus) {
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
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::FillGhost)) {
      v->resetBoundary();
      v->vbvar->SetBoundaries();
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::FillGhost)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
        v->resetBoundary();
        v->vbvar->SetBoundaries();
      }
    }
  }
}

template <typename T>
void Container<T>::ResetBoundaryCellVariables() {
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::FillGhost)) {
      v->vbvar->var_cc = v->data;
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::FillGhost)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
        v->vbvar->var_cc = v->data;
      }
    }
  }
}

template <typename T>
void Container<T>::StartReceiving(BoundaryCommSubset phase) {
  //    std::cout << "in set" << std::endl;
  // sets the boundary
  //  std::cout << "________CLEAR from stage:"<<s->name()<<std::endl;
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::FillGhost)) {
      v->resetBoundary();
      v->vbvar->StartReceiving(phase);
      v->mpiStatus = false;
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::FillGhost)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
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
  for (auto &v : varVector_) {
    if (v->IsSet(Metadata::FillGhost)) {
      v->vbvar->ClearBoundary(phase);
    }
  }
  for (auto &sv : sparseVector_) {
    if (sv->IsSet(Metadata::FillGhost)) {
      CellVariableVector<T> vvec = sv->GetVector();
      for (auto &v : vvec) {
        v->vbvar->ClearBoundary(phase);
      }
    }
  }
}

template <typename T>
void Container<T>::Print() {
  std::cout << "Variables are:\n";
  for (auto v : varVector_) {
    std::cout << " cell: " << v->info() << std::endl;
  }
  for (auto v : faceVector_) {
    std::cout << " face: " << v->info() << std::endl;
  }
  for (auto v : sparseVector_) {
    std::cout << " sparse:" << v->info() << std::endl;
  }
}

template <typename T>
void Container<T>::calcArrDims_(std::array<int, 6> &arrDims, const std::vector<int> &dims,
                                const Metadata &metadata) {
  const int N = dims.size();

  if (metadata.Where() == Metadata::Cell || metadata.Where() == Metadata::Face ||
      metadata.Where() == Metadata::Edge || metadata.Where() == Metadata::Node) {
    // Let the FaceVariable, EdgeVariable, and NodeVariable
    // classes add the +1's where needed.  They all expect
    // these dimensions to be the number of cells in each
    // direction, NOT the size of the arrays
    assert(N >= 0 && N <= 3);
    arrDims[0] = pmy_block->ncells1;
    arrDims[1] = pmy_block->ncells2;
    arrDims[2] = pmy_block->ncells3;
    for (int i = 0; i < N; i++)
      arrDims[i + 3] = dims[i];
    for (int i = N; i < 3; i++)
      arrDims[i + 3] = 1;
  } else {
    // This variable is not necessarily tied to any specific
    // mesh element, so dims will be used as the actual array
    // size in each dimension
    assert(N >= 1 && N <= 6);
    for (int i = 0; i < N; i++)
      arrDims[i] = dims[i];
    for (int i = N; i < 6; i++)
      arrDims[i] = 1;
  }
}

template class Container<double>;

} // namespace parthenon
