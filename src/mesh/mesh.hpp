//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "application_input.hpp"
#include "bvals/boundary_conditions.hpp"
#include "bvals/comms/tag_map.hpp"
#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "domain.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/meshblock_pack.hpp"
#include "outputs/io_wrapper.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/hash.hpp"
#include "utils/object_pool.hpp"
#include "utils/partition_stl_containers.hpp"

namespace parthenon {

// Forward declarations
class MeshBlock;
class MeshRefinement;
class ParameterInput;
class RestartReader;

// Map from LogicalLocation to (gid, rank) pair of location
using LogicalLocMap_t = std::map<LogicalLocation, std::pair<int, int>>;

//----------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
  friend class RestartOutput;
  friend class HistoryOutput;
  friend class MeshBlock;
  friend class MeshBlockTree;
  friend class MeshRefinement;

 public:
  // 2x function overloads of ctor: normal and restarted simulation
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       int test_flag = 0);
  Mesh(ParameterInput *pin, ApplicationInput *app_in, RestartReader &resfile,
       Packages_t &packages, int test_flag = 0);
  ~Mesh();

  // accessors
  int GetNumMeshBlocksThisRank(int my_rank = Globals::my_rank) const {
    return nblist[my_rank];
  }
  int GetNumMeshThreads() const { return num_mesh_threads_; }
  std::int64_t GetTotalCells();
  // TODO(JMM): Move block_size into mesh.
  int GetNumberOfMeshBlockCells() const;
  const RegionSize &GetDefaultBlockSize() const { return base_block_size; }
  RegionSize GetBlockSize(const LogicalLocation &loc) const {
    return forest.GetBlockDomain(loc);
  }
  const IndexShape &GetLeafBlockCellBounds(CellLevel level = CellLevel::same) const;

  const forest::Forest &Forest() const { return forest; }

  // data
  bool modified;
  const bool is_restart;
  RegionSize mesh_size;
  RegionSize base_block_size;
  std::array<BoundaryFlag, BOUNDARY_NFACES> mesh_bcs;
  const int ndim; // number of dimensions
  const bool adaptive, multilevel, multigrid;
  int nbtotal, nbnew, nbdel;
  std::uint64_t mbcnt;

  int step_since_lb;
  int gflag;

  BlockList_t block_list;
  Packages_t packages;
  std::shared_ptr<StateDescriptor> resolved_packages;

  DataCollection<MeshData<Real>> mesh_data;

  std::map<int, BlockList_t> gmg_block_lists;
  std::map<int, DataCollection<MeshData<Real>>> gmg_mesh_data;
  int GetGMGMaxLevel() const { return current_level; }
  int GetGMGMinLevel() const { return gmg_min_logical_level_; }

  // functions
  void Initialize(bool init_problem, ParameterInput *pin, ApplicationInput *app_in);

  bool SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                 BoundaryFlag *block_bcs);
  void OutputCycleDiagnostics();
  void LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin,
                                              ApplicationInput *app_in);
  int DefaultPackSize() {
    return default_pack_size_ < 1 ? block_list.size() : default_pack_size_;
  }
  int DefaultNumPartitions() {
    return partition::partition_impl::IntCeil(block_list.size(), DefaultPackSize());
  }
  // step 7: create new MeshBlock list (same MPI rank but diff level: create new block)
  // Moved here given Cuda/nvcc restriction:
  // "error: The enclosing parent function ("...")
  // for an extended __host__ __device__ lambda cannot have private or
  // protected access within its class"
  void FillSameRankCoarseToFineAMR(MeshBlock *pob, MeshBlock *pmb,
                                   LogicalLocation &newloc);
  void FillSameRankFineToCoarseAMR(MeshBlock *pob, MeshBlock *pmb, LogicalLocation &loc);

  std::shared_ptr<MeshBlock> FindMeshBlock(int tgid) const;

  void ApplyUserWorkBeforeOutput(Mesh *mesh, ParameterInput *pin, SimTime const &time);

  void ApplyUserWorkBeforeRestartOutput(Mesh *mesh, ParameterInput *pin,
                                        SimTime const &time, OutputParameters *pparams);

  // Boundary Functions
  BValFunc MeshBndryFnctn[BOUNDARY_NFACES] = {nullptr};
  SBValFunc SwarmBndryFnctn[BOUNDARY_NFACES] = {nullptr};
  std::array<std::vector<BValFunc>, BOUNDARY_NFACES> UserBoundaryFunctions;

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> ProblemGenerator =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> PostInitialization =
      nullptr;
  static void UserWorkAfterLoopDefault(Mesh *mesh, ParameterInput *pin,
                                       SimTime &tm); // called in main loop
  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkAfterLoop = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime &)> PreStepUserWorkInLoop =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &)> PostStepUserWorkInLoop =
      nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      UserMeshWorkBeforeOutput = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &,
                     OutputParameters *pparams)>
      UserWorkBeforeRestartOutput = nullptr;

  static void PreStepUserDiagnosticsInLoopDefault(Mesh *, ParameterInput *,
                                                  SimTime const &);
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PreStepUserDiagnosticsInLoop = PreStepUserDiagnosticsInLoopDefault;
  static void PostStepUserDiagnosticsInLoopDefault(Mesh *, ParameterInput *,
                                                   SimTime const &);
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PostStepUserDiagnosticsInLoop = PostStepUserDiagnosticsInLoopDefault;

  int GetRootLevel() const noexcept { return root_level; }
  int GetLegacyTreeRootLevel() const noexcept {
    return forest.root_level + forest.forest_level;
  }

  int GetMaxLevel() const noexcept { return max_level; }
  int GetCurrentLevel() const noexcept { return current_level; }
  std::vector<int> GetNbList() const noexcept { return nblist; }
  std::vector<LogicalLocation> GetLocList() const noexcept { return loclist; }

  std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
  GetLevelsAndLogicalLocationsFlat() const noexcept;

  void OutputMeshStructure(const int dim, const bool dump_mesh_structure = true);

  // Ordering here is important to prevent deallocation of pools before boundary
  // communication buffers
  using channel_key_t = std::tuple<int, int, std::string, int>;
  using comm_buf_t = CommBuffer<buf_pool_t<Real>::owner_t>;
  std::unordered_map<int, buf_pool_t<Real>> pool_map;
  using comm_buf_map_t =
      std::unordered_map<channel_key_t, comm_buf_t, tuple_hash<channel_key_t>>;
  comm_buf_map_t boundary_comm_map;
  TagMap tag_map;

#ifdef MPI_PARALLEL
  MPI_Comm GetMPIComm(const std::string &label) const { return mpi_comm_map_.at(label); }
#endif

  void SetAllVariablesToInitialized() {
    for (auto &sp_mb : block_list) {
      for (auto &pair : sp_mb->meshblock_data.Stages()) {
        auto &sp_mbd = pair.second;
        sp_mbd->SetAllVariablesToInitialized();
      }
    }
  }

  uint64_t GetBufferPoolSizeInBytes() const {
    std::uint64_t buffer_memory = 0;
    for (auto &p : pool_map) {
      buffer_memory += p.second.SizeInBytes();
    }
    return buffer_memory;
  }

  // expose a mesh-level call to get lists of variables from resolved_packages
  template <typename... Args>
  std::vector<std::string> GetVariableNames(Args &&...args) {
    return resolved_packages->GetVariableNames(std::forward<Args>(args)...);
  }

 private:
  // data
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
  forest::Forest forest;
  // number of MeshBlocks in the x1, x2, x3 directions of the root grid:
  // (unlike LogicalLocation.lxi, nrbxi don't grow w/ AMR # of levels, so keep 32-bit int)
  std::array<int, 3> nrbx;

  // flags are false if using non-uniform or user meshgen function
  bool use_uniform_meshgen_fn_[4];

  // variables for load balancing control
  bool lb_flag_, lb_automatic_, lb_manual_;
  double lb_tolerance_;
  int lb_interval_;

  // size of default MeshBlockPacks
  int default_pack_size_;

  int gmg_min_logical_level_ = 0;

#ifdef MPI_PARALLEL
  // Global map of MPI comms for separate variables
  std::unordered_map<std::string, MPI_Comm> mpi_comm_map_;
#endif

  // functions
  void CalculateLoadBalance(std::vector<double> const &costlist,
                            std::vector<int> &ranklist, std::vector<int> &nslist,
                            std::vector<int> &nblist);
  void ResetLoadBalanceVariables();

  // Mesh::LoadBalancingAndAdaptiveMeshRefinement() helper functions:
  void UpdateCostList();
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  bool GatherCostListAndCheckBalance();
  void RedistributeAndRefineMeshBlocks(ParameterInput *pin, ApplicationInput *app_in,
                                       int ntot);
  void BuildGMGBlockLists(ParameterInput *pin, ApplicationInput *app_in);
  void SetGMGNeighbors();
  void
  SetMeshBlockNeighbors(GridIdentifier grid_id, BlockList_t &block_list,
                        const std::vector<int> &ranklist,
                        const std::unordered_set<LogicalLocation> &newly_refined = {});

  // Optionall defined in either the problem file
  std::function<void(Mesh *, ParameterInput *)> InitUserMeshData;

  void EnrollBndryFncts_(ApplicationInput *app_in);

  // Re-used functionality in constructor
  void RegisterLoadBalancing_(ParameterInput *pin);

  void SetupMPIComms();
  void BuildTagMapAndBoundaryBuffers();
  void CommunicateBoundaries(std::string md_name = "base");
  void PreCommFillDerived();
  void FillDerived();

  // Transform from logical location coordinates to uniform mesh coordinates accounting
  // for root grid
  Real GetMeshCoordinate(CoordinateDirection dir, BlockLocation bloc,
                         const LogicalLocation &loc) const {
    auto xll = loc.LLCoord(dir, bloc);
    auto root_fac = static_cast<Real>(1 << root_level) / static_cast<Real>(nrbx[dir - 1]);
    xll *= root_fac;
    return mesh_size.xmin(dir) * (1.0 - xll) + mesh_size.xmax(dir) * xll;
  }

  std::int64_t GetLLFromMeshCoordinate(CoordinateDirection dir, int level,
                                       Real xmesh) const {
    auto root_fac = static_cast<Real>(1 << root_level) / static_cast<Real>(nrbx[dir - 1]);
    auto xLL = (xmesh - mesh_size.xmin(dir)) /
               (mesh_size.xmax(dir) - mesh_size.xmin(dir)) / root_fac;
    return static_cast<std::int64_t>((1 << std::max(level, 0)) * xLL);
  }
};

} // namespace parthenon

#endif // MESH_MESH_HPP_
