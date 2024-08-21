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
#ifndef MESH_MESHBLOCK_HPP_
#define MESH_MESHBLOCK_HPP_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "application_input.hpp"
#include "bvals/bvals.hpp"
#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "domain.hpp"
#include "globals.hpp"
#include "interface/data_collection.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/packages.hpp"
#include "interface/swarm_container.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/forest/forest.hpp"
#include "outputs/io_wrapper.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

// Forward declarations
class ApplicationInput;
class Mesh;
class MeshBlockTree;
class MeshRefinement;
class ParameterInput;
class StateDescriptor;

// Inner loop default pattern
// - Defined outside of the MeshBlock class because it does not require an exec space
// - Not defined in kokkos_abstraction.hpp because it requires the compile time option
//   DEFAULT_INNER_LOOP_PATTERN to be set.
template <typename Function>
KOKKOS_FORCEINLINE_FUNCTION void par_for_inner(const team_mbr_t &team_member,
                                               const int &il, const int &iu,
                                               const Function &function) {
  parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, team_member, il, iu, function);
}

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//! \brief data/functions associated with a single block
class MeshBlock : public std::enable_shared_from_this<MeshBlock> {
  friend class RestartOutput;
  friend class Mesh;

 public:
  MeshBlock() = default;
  MeshBlock(const int n_side, const int ndim, bool init_coarse = true,
            bool multilevel = true); // for Kokkos testing with ghost
  ~MeshBlock();

  // Factory method deals with initialization for you
  static std::shared_ptr<MeshBlock>
  Make(int igid, int ilid, LogicalLocation iloc, RegionSize input_block,
       BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin, ApplicationInput *app_in,
       Packages_t &packages, std::shared_ptr<StateDescriptor> resolved_packages,
       int igflag, double icost = 1.0);

  // Kokkos execution space for this MeshBlock
  DevExecSpace exec_space;

  // data
  Mesh *pmy_mesh = nullptr; // ptr to Mesh containing this MeshBlock
  LogicalLocation loc;
  RegionSize block_size;
  // for convenience: "max" # of real+ghost cells along each dir for allocating "standard"
  // sized MeshBlock arrays, depending on ndim i.e.
  //
  // cellbounds.nx(X2DIR) =    nx2      + 2*Globals::nghost if   nx2 > 1
  // (entire)         (interior)               (interior)
  //
  // Assuming we have a block cells, and nx2 = 6, and Globals::nghost = 1
  //
  // <----- nx1 = 8 ---->
  //       (entire)
  //
  //     <- nx1 = 6 ->
  //       (interior)
  //
  //  - - - - - - - - - -   ^
  //  |  |  ghost    |  |   |
  //  - - - - - - - - - -   |         ^
  //  |  |     ^     |  |   |         |
  //  |  |     |     |  |  nx2 = 8    nx2 = 6
  //  |  | interior  |  | (entire)   (interior)
  //  |  |     |     |  |             |
  //  |  |     v     |  |   |         v
  //  - - - - - - - - - -   |
  //  |  |           |  |   |
  //  - - - - - - - - - -   v
  //
  IndexShape cellbounds;
  // on 1x coarser level MeshBlock i.e.
  //
  // c_cellbounds.nx(X2DIR) = cellbounds.nx(X2DIR) * 1/2 + 2*Globals::nghost, if
  // cellbounds.nx(X2DIR) >1
  //   (entire)             (interior)                          (interior)
  //
  // Assuming we have a block cells, and nx2 = 6, and Globals::nghost = 1
  //
  //          cells                              c_cells
  //
  //  - - - - - - - - - -   ^              - - - - - - - - - -     ^
  //  |  |           |  |   |              |  |           |  |     |
  //  - - - - - - - - - -   |              - - - - - - - - - -     |
  //  |  |     ^     |  |   |              |  |      ^    |  |     |
  //  |  |     |     |  |   |              |  |      |    |  |     |
  //  |  |  nx2 = 6  |  |  nx2 = 8  ====>  |  |   nx2 = 3 |  |   nx2 = 5
  //  |  |(interior) |  |  (entire)        |  | (interior)|  |  (entire)
  //  |  |     v     |  |   |              |  |      v    |  |     |
  //  - - - - - - - - - -   |              - - - - - - - - - -     |
  //  |  |           |  |   |              |  |           |  |     |
  //  - - - - - - - - - -   v              - - - - - - - - - -     v
  //
  IndexShape c_cellbounds;
  IndexShape f_cellbounds;
  int gid, lid;
  int cnghost;
  int gflag;

  const IndexShape &GetCellBounds(CellLevel cl) const {
    if (cl == CellLevel::same) {
      return cellbounds;
    } else if (cl == CellLevel::fine) {
      return f_cellbounds;
    } else if (cl == CellLevel::coarse) {
      return c_cellbounds;
    } else {
      PARTHENON_FAIL("This should not be accessible.");
      return cellbounds;
    }
  }
  // The User defined containers
  DataCollection<MeshBlockData<Real>> meshblock_data;

  Packages_t packages;
  std::shared_ptr<StateDescriptor> resolved_packages;

  std::unique_ptr<MeshBlockApplicationData> app;

  Coordinates_t coords;
  ParArray0D<Coordinates_t> coords_device;

  // mesh-related objects
  // TODO(jcd): remove all these?
  std::unique_ptr<BoundarySwarms> pbswarm;
  std::unique_ptr<MeshRefinement> pmr;

  // Block connectivity information
  std::vector<NeighborBlock> neighbors;
  std::vector<NeighborBlock> gmg_coarser_neighbors;
  std::vector<NeighborBlock> gmg_composite_finer_neighbors;
  std::vector<NeighborBlock> gmg_same_neighbors;
  std::vector<NeighborBlock> gmg_finer_neighbors;
  std::vector<NeighborBlock> gmg_leaf_neighbors;

  BoundaryFlag boundary_flag[6];

  // functions
  // Load balancing
  void SetCostForLoadBalancing(double cost);

  // Memory usage
  // TODO(JMM): Currently swarm send/receive boundaries are not counted.
  void LogMemUsage(std::int64_t delta) { mem_usage_ += delta; }

  std::uint64_t ReportMemUsage() { return mem_usage_; }

  //----------------------------------------------------------------------------------------
  //! \fn void MeshBlock::DeepCopy(const DstType& dst, const SrcType& src)
  //  \brief Deep copy between views using the exec space of the MeshBlock
  template <class DstType, class SrcType>
  void deep_copy(const DstType &dst, const SrcType &src) {
    Kokkos::deep_copy(exec_space, dst, src);
  }

  void AllocateSparse(std::string const &label, bool only_control = false,
                      bool flag_uninitialized = false);

  void AllocSparseID(std::string const &base_name, const int sparse_id) {
    AllocateSparse(MakeVarLabel(base_name, sparse_id));
  }

  void DeallocateSparse(std::string const &label);

#ifdef ENABLE_SPARSE
  inline bool IsAllocated(std::string const &label) const noexcept {
    return meshblock_data.Get()->IsAllocated(label);
  }

  inline bool IsAllocated(std::string const &base_name, int sparse_id) const noexcept {
    return IsAllocated(MakeVarLabel(base_name, sparse_id));
  }
#else
  inline constexpr bool IsAllocated(std::string const & /*label*/) const noexcept {
    return true;
  }

  inline constexpr bool IsAllocated(std::string const & /*base_name*/,
                                    int /*sparse_id*/) const noexcept {
    return true;
  }
#endif

  void SetAllVariablesToInitialized() {
    auto &stages = meshblock_data.Stages();
    std::for_each(stages.begin(), stages.end(),
                  [](auto &pair) { pair.second->SetAllVariablesToInitialized(); });
  }

  template <class... Args>
  inline void par_for(Args &&...args) {
    parthenon::par_for(std::forward<Args>(args)...);
  }

  template <class... Args>
  inline void par_reduce(Args &&...args) {
    parthenon::par_reduce(std::forward<Args>(args)...);
  }

  template <class... Args>
  inline void par_scan(Args &&...args) {
    parthenon::par_scan(std::forward<Args>(args)...);
  }

  template <typename... Args>
  inline void par_for_outer(Args &&...args) {
    parthenon::par_for_outer(std::forward<Args>(args)...);
  }

  template <typename Function>
  inline void par_for_bndry(const std::string &name, const IndexRange &nb,
                            const IndexDomain &domain, TopologicalElement el,
                            const bool coarse, const bool fine, Function &&function) {
    auto &bounds = fine ? (coarse ? cellbounds : f_cellbounds)
                        : (coarse ? c_cellbounds : cellbounds);
    auto ib = bounds.GetBoundsI(domain, el);
    auto jb = bounds.GetBoundsJ(domain, el);
    auto kb = bounds.GetBoundsK(domain, el);
    parthenon::par_for(name, nb, kb, jb, ib, std::forward<Function>(function));
  }

  int GetNumberOfMeshBlockCells() const {
    return block_size.nx(X1DIR) * block_size.nx(X2DIR) * block_size.nx(X3DIR);
  }

  void SetBlockTimestep(const Real dt) { new_block_dt_ = dt; }
  void SetAllowedDt(const Real dt) { new_block_dt_ = dt; }
  Real NewDt() const { return new_block_dt_; }

 private:
  // data
  Real new_block_dt_, new_block_dt_hyperbolic_, new_block_dt_parabolic_,
      new_block_dt_user_;
  std::vector<std::shared_ptr<Variable<Real>>> vars_cc_;

  // Initializer to set up a meshblock called with the default constructor
  // This is necessary because the back pointers can't be set up until
  // the block is allocated.
  void Initialize(int igid, int ilid, LogicalLocation iloc, RegionSize input_block,
                  BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin,
                  ApplicationInput *app_in, Packages_t &packages,
                  std::shared_ptr<StateDescriptor> resolved_packages, int igflag,
                  double icost = 1.0);

  void InitializeIndexShapesImpl(const int nx1, const int nx2, const int nx3,
                                 bool init_coarse, bool multilevel);
  void InitializeIndexShapes(const int nx1, const int nx2, const int nx3);

  // Optionally defined in the prob file or provided by ApplicationInput
  std::function<void(MeshBlock *, ParameterInput *)> ProblemGenerator = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> PostInitialization = nullptr;
  std::function<pMeshBlockApplicationData_t(MeshBlock *, ParameterInput *)>
      InitApplicationMeshBlockData = nullptr;
  std::function<void(MeshBlock *, ParameterInput *)> InitMeshBlockUserData = nullptr;
  std::function<void(MeshBlock *, ParameterInput *, const SimTime &)>
      UserWorkBeforeOutput = nullptr;

  // functions and variables for automatic load balancing based on timing
  Kokkos::Timer lb_timer;
  double cost_;
  // JMM: these are private since the timing machinery only works
  // per-meshblock nopt per-meshdata.
  void ResetTimeMeasurement();
  void StartTimeMeasurement();
  void StopTimeMeasurement();

  // memory usage on a block
  std::uint64_t mem_usage_;
};

using BlockList_t = std::vector<std::shared_ptr<MeshBlock>>;

struct BlockListPartition {
  BlockListPartition(int p, GridIdentifier g, const BlockList_t &bl, Mesh *pm)
      : partition{p}, grid{g}, block_list{bl}, pmesh{pm} {}
  const int partition;
  const GridIdentifier grid;
  const BlockList_t block_list;
  Mesh *pmesh;
};

} // namespace parthenon

#endif // MESH_MESHBLOCK_HPP_
