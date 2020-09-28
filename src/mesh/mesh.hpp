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
#ifndef MESH_MESH_HPP_
#define MESH_MESH_HPP_
//! \file mesh.hpp
//  \brief defines Mesh and MeshBlock classes, and various structs used in them
//  The Mesh is the overall grid structure, and MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "application_input.hpp"
#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "domain.hpp"
#include "interface/container.hpp"
#include "interface/container_collection.hpp"
#include "interface/properties_interface.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/meshblock_tree.hpp"
#include "outputs/io_wrapper.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

// Forward declarations
class BoundaryValues;
class ParameterInput;
class RestartReader;

//----------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
  friend class RestartOutput;
  friend class HistoryOutput;
  friend class MeshBlock;
  friend class MeshBlockTree;
  friend class BoundaryBase;
  friend class BoundaryValues;
  friend class Coordinates;
  friend class MeshRefinement;

 public:
  // 2x function overloads of ctor: normal and restarted simulation
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Properties_t &properties,
       Packages_t &packages, int test_flag = 0);
  Mesh(ParameterInput *pin, ApplicationInput *app_in, RestartReader &resfile,
       Properties_t &properties, Packages_t &packages, int test_flag = 0);
  ~Mesh();

  // accessors
  int GetNumMeshBlocksThisRank(int my_rank = Globals::my_rank) const {
    return nblist[my_rank];
  }
  int GetNumMeshThreads() const { return num_mesh_threads_; }
  std::int64_t GetTotalCells() {
    auto &pmb = block_list.front();
    return static_cast<std::int64_t>(nbtotal) * pmb->block_size.nx1 *
           pmb->block_size.nx2 * pmb->block_size.nx3;
  }
  // TODO(JMM): Move block_size into mesh.
  int GetNumberOfMeshBlockCells() const {
    return block_list.front()->GetNumberOfMeshBlockCells();
  }
  const RegionSize &GetBlockSize() const { return block_list.front()->block_size; }

  // data
  bool modified;
  RegionSize mesh_size;
  BoundaryFlag mesh_bcs[6];
  const int ndim; // number of dimensions
  const bool adaptive, multilevel;
  int nbtotal, nbnew, nbdel;
  std::uint64_t mbcnt;

  int step_since_lb;
  int gflag;

  BlockList_t block_list;
  Properties_t properties;
  Packages_t packages;

  // MeshBlockPacks
  // TODO(JMM): Should these be private with a getter function?
  std::map<std::string, std::map<std::string, std::vector<MeshBlockVarPack<Real>>>>
      real_varpacks;
  std::map<std::string, std::map<std::string, std::vector<MeshBlockVarFluxPack<Real>>>>
      real_fluxpacks;

  // functions
  void Initialize(int res_flag, ParameterInput *pin, ApplicationInput *app_in);
  void SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                 BoundaryFlag *block_bcs);
  void NewTimeStep();
  void OutputCycleDiagnostics();
  void RegisterMeshBlockPack(const std::string &package, const std::string &name,
                             const VarPackingFunc<Real> &func);
  void RegisterMeshBlockPack(const std::string &package, const std::string &name,
                             const FluxPackingFunc<Real> &func);
  void BuildMeshBlockPacks();
  void LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin,
                                              ApplicationInput *app_in);
  int DefaultPackSize() {
    return default_pack_size_ < 1 ? block_list.size() : default_pack_size_;
  }
  // step 7: create new MeshBlock list (same MPI rank but diff level: create new block)
  // Moved here given Cuda/nvcc restriction:
  // "error: The enclosing parent function ("...")
  // for an extended __host__ __device__ lambda cannot have private or
  // protected access within its class"
  void FillSameRankCoarseToFineAMR(MeshBlock *pob, MeshBlock *pmb,
                                   LogicalLocation &newloc);
  void FillSameRankFineToCoarseAMR(MeshBlock *pob, MeshBlock *pmb, LogicalLocation &loc);
  int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3);

  std::shared_ptr<MeshBlock> FindMeshBlock(int tgid);

  void ApplyUserWorkBeforeOutput(ParameterInput *pin);

  // function for distributing unique "phys" bitfield IDs to BoundaryVariable objects and
  // other categories of MPI communication for generating unique MPI_TAGs
  int ReserveTagPhysIDs(int num_phys);

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  static void UserWorkAfterLoopDefault(Mesh *mesh, ParameterInput *pin,
                                       SimTime &tm); // called in main loop
  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkAfterLoop =
      &UserWorkAfterLoopDefault;
  static void UserWorkInLoopDefault(); // called in main after each cycle
  std::function<void()> UserWorkInLoop = &UserWorkInLoopDefault;
  int GetRootLevel() const noexcept { return root_level; }
  int GetMaxLevel() const noexcept { return max_level; }
  int GetCurrentLevel() const noexcept { return current_level; }
  std::vector<int> GetNbList() const noexcept { return nblist; }

 private:
  // data
  int next_phys_id_; // next unused value for encoding final component of MPI tag bitfield
  int root_level, max_level, current_level;
  int num_mesh_threads_;
  /// Maps Global Block IDs to which rank the block is mapped to.
  std::vector<int> ranklist;
  /// Maps rank to start of local block IDs.
  std::vector<int> nslist;
  /// Maps rank to count of local blocks.
  std::vector<int> nblist;
  /// Maps global block ID to its cost
  std::vector<double> costlist;
  // 8x arrays used exclusively for AMR (not SMR):
  /// Count of blocks to refine on each rank
  std::vector<int> nref;
  /// Count of blocks to de-refine on each rank
  std::vector<int> nderef;
  std::vector<int> rdisp, ddisp;
  std::vector<int> bnref, bnderef;
  std::vector<int> brdisp, bddisp;
  // the last 4x should be std::size_t, but are limited to int by MPI

  std::vector<LogicalLocation> loclist;
  MeshBlockTree tree;
  // number of MeshBlocks in the x1, x2, x3 directions of the root grid:
  // (unlike LogicalLocation.lxi, nrbxi don't grow w/ AMR # of levels, so keep 32-bit int)
  int nrbx1, nrbx2, nrbx3;
  // TODO(felker) find unnecessary static_cast<> ops. from old std::int64_t type in 2018:
  // std::int64_t nrbx1, nrbx2, nrbx3;

  // flags are false if using non-uniform or user meshgen function
  bool use_uniform_meshgen_fn_[4];

  // variables for load balancing control
  bool lb_flag_, lb_automatic_, lb_manual_;
  double lb_tolerance_;
  int lb_interval_;

  // size of default MeshBlockPacks
  int default_pack_size_;

  // functions
  MeshGenFunc MeshGenerator_[4];
  BValFunc BoundaryFunction_[6];
  AMRFlagFunc AMRFlag_;
  SrcTermFunc UserSourceTerm_;
  TimeStepFunc UserTimeStep_;
  MetricFunc UserMetric_;

  std::map<std::string, std::map<std::string, VarPackingFunc<Real>>> real_varpackers_;
  std::map<std::string, std::map<std::string, FluxPackingFunc<Real>>> real_fluxpackers_;
  void RegisterAllMeshBlockPackers(Packages_t &packages);

  void OutputMeshStructure(int dim);
  void CalculateLoadBalance(std::vector<double> const &costlist,
                            std::vector<int> &ranklist, std::vector<int> &nslist,
                            std::vector<int> &nblist);
  void ResetLoadBalanceVariables();

  void ReserveMeshBlockPhysIDs();

  // Mesh::LoadBalancingAndAdaptiveMeshRefinement() helper functions:
  void UpdateCostList();
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  bool GatherCostListAndCheckBalance();
  void RedistributeAndRefineMeshBlocks(ParameterInput *pin, ApplicationInput *app_in,
                                       int ntot);

  // Mesh::RedistributeAndRefineMeshBlocks() helper functions:
  // step 6: send
  void PrepareSendSameLevel(MeshBlock *pb, ParArray1D<Real> &sendbuf);
  void PrepareSendCoarseToFineAMR(MeshBlock *pb, ParArray1D<Real> &sendbuf,
                                  LogicalLocation &lloc);
  void PrepareSendFineToCoarseAMR(MeshBlock *pb, ParArray1D<Real> &sendbuf);
  // step 7: create new MeshBlock list (same MPI rank but diff level: create new block)
  // moved public to be called from device
  // step 8: receive
  void FinishRecvSameLevel(MeshBlock *pb, ParArray1D<Real> &recvbuf);
  void FinishRecvFineToCoarseAMR(MeshBlock *pb, ParArray1D<Real> &recvbuf,
                                 LogicalLocation &lloc);
  void FinishRecvCoarseToFineAMR(MeshBlock *pb, ParArray1D<Real> &recvbuf);

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  static void InitUserMeshDataDefault(ParameterInput *pin);
  std::function<void(ParameterInput *)> InitUserMeshData = InitUserMeshDataDefault;

  // often used (not defined) in prob file in ../pgen/
  void EnrollUserBoundaryFunction(BoundaryFace face, BValFunc my_func);
  // DEPRECATED(felker): provide trivial overload for old-style BoundaryFace enum argument
  void EnrollUserBoundaryFunction(int face, BValFunc my_func);

  void EnrollUserRefinementCondition(AMRFlagFunc amrflag);
  void EnrollUserMeshGenerator(CoordinateDirection dir, MeshGenFunc my_mg);
  void EnrollUserExplicitSourceFunction(SrcTermFunc my_func);
  void EnrollUserTimeStepFunction(TimeStepFunc my_func);
  void EnrollUserMetric(MetricFunc my_func);
};

//----------------------------------------------------------------------------------------
// \!fn Real ComputeMeshGeneratorX(std::int64_t index, std::int64_t nrange,
//                                 bool sym_interval)
// \brief wrapper fn to compute Real x logical location for either [0., 1.] or [-0.5, 0.5]
//        real cell ranges for MeshGenerator_[] functions (default/user vs. uniform)

inline Real ComputeMeshGeneratorX(std::int64_t index, std::int64_t nrange,
                                  bool sym_interval) {
  // index is typically 0, ... nrange for non-ghost boundaries
  if (!sym_interval) {
    // to map to fractional logical position [0.0, 1.0], simply divide by # of faces
    return static_cast<Real>(index) / static_cast<Real>(nrange);
  } else {
    // to map to a [-0.5, 0.5] range, rescale int indices around 0 before FP conversion
    // if nrange is even, there is an index at center x=0.0; map it to (int) 0
    // if nrange is odd, the center x=0.0 is between two indices; map them to -1, 1
    std::int64_t noffset = index - (nrange) / 2;
    std::int64_t noffset_ceil = index - (nrange + 1) / 2; // = noffset if nrange is even
    // std::cout << "noffset, noffset_ceil = " << noffset << ", " << noffset_ceil << "\n";
    // average the (possibly) biased integer indexing
    return static_cast<Real>(noffset + noffset_ceil) / (2.0 * nrange);
  }
}

//----------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX1(Real x, RegionSize rs)
// \brief x1 mesh generator function, x is the logical location; x=i/nx1, real in [0., 1.]

inline Real DefaultMeshGeneratorX1(Real x, RegionSize rs) {
  Real lw, rw;
  if (rs.x1rat == 1.0) {
    rw = x, lw = 1.0 - x;
  } else {
    Real ratn = std::pow(rs.x1rat, rs.nx1);
    Real rnx = std::pow(rs.x1rat, x * rs.nx1);
    lw = (rnx - ratn) / (1.0 - ratn);
    rw = 1.0 - lw;
  }
  // linear interp, equally weighted from left (x(xmin)=0.0) and right (x(xmax)=1.0)
  return rs.x1min * lw + rs.x1max * rw;
}

//----------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX2(Real x, RegionSize rs)
// \brief x2 mesh generator function, x is the logical location; x=j/nx2, real in [0., 1.]

inline Real DefaultMeshGeneratorX2(Real x, RegionSize rs) {
  Real lw, rw;
  if (rs.x2rat == 1.0) {
    rw = x, lw = 1.0 - x;
  } else {
    Real ratn = std::pow(rs.x2rat, rs.nx2);
    Real rnx = std::pow(rs.x2rat, x * rs.nx2);
    lw = (rnx - ratn) / (1.0 - ratn);
    rw = 1.0 - lw;
  }
  return rs.x2min * lw + rs.x2max * rw;
}

//----------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX3(Real x, RegionSize rs)
// \brief x3 mesh generator function, x is the logical location; x=k/nx3, real in [0., 1.]

inline Real DefaultMeshGeneratorX3(Real x, RegionSize rs) {
  Real lw, rw;
  if (rs.x3rat == 1.0) {
    rw = x, lw = 1.0 - x;
  } else {
    Real ratn = std::pow(rs.x3rat, rs.nx3);
    Real rnx = std::pow(rs.x3rat, x * rs.nx3);
    lw = (rnx - ratn) / (1.0 - ratn);
    rw = 1.0 - lw;
  }
  return rs.x3min * lw + rs.x3max * rw;
}

//----------------------------------------------------------------------------------------
// \!fn Real UniformMeshGeneratorX1(Real x, RegionSize rs)
// \brief x1 mesh generator function, x is the logical location; real cells in [-0.5, 0.5]

inline Real UniformMeshGeneratorX1(Real x, RegionSize rs) {
  // linear interp, equally weighted from left (x(xmin)=-0.5) and right (x(xmax)=0.5)
  return static_cast<Real>(0.5) * (rs.x1min + rs.x1max) + (x * rs.x1max - x * rs.x1min);
}

//----------------------------------------------------------------------------------------
// \!fn Real UniformMeshGeneratorX2(Real x, RegionSize rs)
// \brief x2 mesh generator function, x is the logical location; real cells in [-0.5, 0.5]

inline Real UniformMeshGeneratorX2(Real x, RegionSize rs) {
  return static_cast<Real>(0.5) * (rs.x2min + rs.x2max) + (x * rs.x2max - x * rs.x2min);
}

//----------------------------------------------------------------------------------------
// \!fn Real UniformMeshGeneratorX3(Real x, RegionSize rs)
// \brief x3 mesh generator function, x is the logical location; real cells in [-0.5, 0.5]

inline Real UniformMeshGeneratorX3(Real x, RegionSize rs) {
  return static_cast<Real>(0.5) * (rs.x3min + rs.x3max) + (x * rs.x3max - x * rs.x3min);
}

} // namespace parthenon

#endif // MESH_MESH_HPP_
