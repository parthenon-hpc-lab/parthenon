//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
#ifndef MESH_MESHBLOCK_TREE_HPP_
#define MESH_MESHBLOCK_TREE_HPP_
//! \file meshblock_tree.hpp
//  \brief defines the LogicalLocation structure and MeshBlockTree class
//======================================================================================

#include "bvals/bvals.hpp"
#include "defs.hpp"

namespace parthenon {

class Mesh;

//--------------------------------------------------------------------------------------
//! \class MeshBlockTree
//  \brief Objects are nodes in an AMR MeshBlock tree structure

class MeshBlockTree {
  friend class Mesh;
  friend class MeshBlock;
  friend class BoundaryBase;

 public:
  explicit MeshBlockTree(Mesh *pmesh);
  MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3);
  ~MeshBlockTree();

  // accessor
  MeshBlockTree *GetLeaf(int ox1, int ox2, int ox3) {
    return pleaf_[(ox1 + (ox2 << 1) + (ox3 << 2))];
  }

  // functions
  void CreateRootGrid();
  void AddMeshBlock(LogicalLocation rloc, int &nnew);
  void AddMeshBlockWithoutRefine(LogicalLocation rloc);
  void Refine(int &nnew);
  void Derefine(int &ndel);
  MeshBlockTree *FindMeshBlock(LogicalLocation tloc);
  void CountMeshBlock(int &count);
  void GetMeshBlockList(LogicalLocation *list, int *pglist, int &count);
  MeshBlockTree *FindNeighbor(LogicalLocation myloc, int ox1, int ox2, int ox3,
                              bool amrflag = false);
  
  LogicalLocation GetLocation() const { return loc_; }
  bool IsLeaf() const { return pleaf_ == nullptr; }
  int GetGid() const { return gid_; }
 private:
  // data
  MeshBlockTree **pleaf_;
  int gid_;
  LogicalLocation loc_;

  static MeshBlockTree *proot_;
  static int nleaf_;
};

} // namespace parthenon

#endif // MESH_MESHBLOCK_TREE_HPP_
