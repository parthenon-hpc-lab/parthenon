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
#include "mesh/forest/forest_topology.hpp"
#include "mesh/meshblock_pack.hpp"
#include "outputs/io_wrapper.hpp"
#include "pack/pack_descriptor.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/hash.hpp"
#include "utils/object_pool.hpp"
#include "utils/partition_stl_containers.hpp"

namespace parthenon {

// Forward declarations
class ApplicationInput;
class MeshBlock;
class MeshRefinement;
class Packages_t;
class ParameterInput;
class RestartReader;

// Map from LogicalLocation to (gid, rank) pair of location
using LogicalLocMap_t = std::map<LogicalLocation, std::pair<int, int>>;

// Base class to allow cacheing of different types of PackDescriptors

//----------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
  friend class RestartOutput;
  friend class HistoryOutput;
  friend class MeshBlock;
  friend class MeshRefinement;

  struct base_constructor_selector_t {};
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       base_constructor_selector_t);
  struct hyper_rectangular_constructor_selector_t {};
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       hyper_rectangular_constructor_selector_t);

 public:
  // 2x function overloads of ctor: normal and restarted simulation
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       int test_flag = 0);
  Mesh(ParameterInput *pin, ApplicationInput *app_in, RestartReader &resfile,
       Packages_t &packages, int test_flag = 0);
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       forest::ForestDefinition &forest_def);
  static RegionSize GetBaseMeshBlockSize(ParameterInput *pin,
                                         const RegionSize &mesh_size);
  static std::pair<RegionSize, RegionSize> GetRegionSizes(ParameterInput *pin);
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
  const IndexShape GetLeafBlockCellBounds(CellLevel level = CellLevel::same) const;

  ParArray1D<AmrTag> &GetAmrTags();

  const forest::Forest &Forest() const { return forest; }

  // data
  bool modified;
  RegionSize mesh_size;
  RegionSize base_block_size;

  BValNames_t mesh_bc_names;
  BValNames_t mesh_swarm_bc_names;

  // these are flags not boundary functions
  std::array<BoundaryFlag, BOUNDARY_NFACES> mesh_bcs;
  int ndim; // number of dimensions
  const bool adaptive, multilevel, multigrid;
  int nbtotal, nbnew, nbdel;
  std::uint64_t mbcnt;

  int step_since_lb;
  int gflag;
  int task_collection_timeout_in_seconds;

  BlockList_t block_list;
  Packages_t packages;
  std::shared_ptr<StateDescriptor> resolved_packages;

  DataCollection<MeshData<Real>> mesh_data;

  const BlockList_t &GetGMGBlockList(int level) const {
    PARTHENON_REQUIRE(multigrid, "Asking for multigrid blocks on a Mesh that was created "
                                 "without parthenon/mesh/multigrid = true set.");
    PARTHENON_REQUIRE(gmg_block_lists_.count(level),
                      "Asking for a multigrid level that doesn't exist.");
    return gmg_block_lists_.at(level);
  }
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
    if (use_pack_size_) {
      return default_pack_size_ < 1 ? std::max(static_cast<int>(block_list.size()), 1)
                                    : default_pack_size_;
    } else {
      return std::max(
          1, partition::partition_impl::IntCeil(block_list.size(), default_num_packs_));
    }
  }
  int DefaultNumPartitions() {
    if (use_pack_size_) {
      return partition::partition_impl::IntCeil(block_list.size(), DefaultPackSize());
    } else {
      return std::max(1,
                      static_cast<int>(std::min(default_num_packs_, block_list.size())));
    }
  }

  const std::vector<std::shared_ptr<BlockListPartition>> &
  GetDefaultBlockPartitions(GridIdentifier grid = GridIdentifier::leaf()) const {
    if (grid.type == GridType::two_level_composite)
      PARTHENON_REQUIRE(multigrid, "Asking for a partition of a multigrid grid when "
                                   "parthenon/mesh/multigrid = false.")
    PARTHENON_REQUIRE(
        block_partitions_.count(grid),
        "There isn't a block partition available for this grid for some reason.");
    return block_partitions_.at(grid);
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
  int GetLegacyTreeRootLevel() const {
    return forest.root_level + forest.forest_level.value();
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
  using channel_key_t = std::tuple<int, int, std::string, int, int>;
  using comm_buf_t = CommBuffer<buf_pool_t<Real>::owner_t>;
  std::unordered_map<int, buf_pool_t<Real>> pool_map;
  using comm_buf_map_t = std::unordered_map<channel_key_t, comm_buf_t>;
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

  forest::Forest forest;

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
  // Refinement tags used by MeshData checks
  ParArray1D<AmrTag> amr_tags;

  std::vector<LogicalLocation> loclist;

  // Block lists for internal nodes in the tree corresponding to multigrid levels
  std::map<int, BlockList_t> gmg_block_lists_;

  // flags are false if using non-uniform or user meshgen function
  bool use_uniform_meshgen_fn_[4];

  // variables for load balancing control
  bool lb_flag_, lb_automatic_, lb_manual_;
  double lb_tolerance_;
  int lb_interval_;

  // size of default MeshBlockPacks
  bool use_pack_size_;
  int default_pack_size_;
  std::size_t default_num_packs_;

  int gmg_min_logical_level_ = 0;

#ifdef MPI_PARALLEL
  // Global map of MPI comms for separate variables
  std::unordered_map<std::string, MPI_Comm> mpi_comm_map_;
#endif

  void SetBCNames_(ParameterInput *pin);
  std::array<BoundaryFlag, BOUNDARY_NFACES>
  GetBCsFromNames_(const BValNames_t &names) const;

  // functions
  void CheckMeshValidity() const;
  void BuildBlockList(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
                      int mesh_test,
                      const std::unordered_map<LogicalLocation, int> &dealloc_count = {});
  void DoStaticRefinement(ParameterInput *pin);
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

  // Optionally defined in the problem file
  std::function<void(Mesh *, ParameterInput *)> InitUserMeshData = nullptr;

  // Re-used functionality in constructor
  void RegisterLoadBalancing_(ParameterInput *pin);

  void SetupMPIComms();
  void BuildTagMapAndBoundaryBuffers();
  void CommunicateBoundaries(std::string md_name = "base",
                             const std::vector<std::string> &fields = {});
  void PreCommFillDerived();
  void FillDerived();

  void BuildBlockPartitions(GridIdentifier grid);
  std::map<GridIdentifier, std::vector<std::shared_ptr<BlockListPartition>>>
      block_partitions_;
};

} // namespace parthenon

#endif // MESH_MESH_HPP_
