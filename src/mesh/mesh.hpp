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

// C headers

// C++ headers
#include <cstdint>     // int64_t
#include <functional>  // reference_wrapper
#include <map>
#include <memory>      // std::shared_ptr
#include <string>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "bvals/bvals.hpp"
#include "domain.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "interface/Container.hpp"
#include "interface/PropertiesInterface.hpp"
#include "interface/StateDescriptor.hpp"
#include "interface/Update.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh_refinement.hpp"
#include "meshblock_tree.hpp"
#include "outputs/io_wrapper.hpp"
#include "parameter_input.hpp"
#include "reconstruct/reconstruction.hpp"
#include "utils/interp_table.hpp"

namespace parthenon {
// Forward declarations
class ParameterInput;
class Mesh;
class MeshRefinement;
class MeshBlockTree;
class BoundaryValues;
class Coordinates;
class Reconstruction;

//template class Container<Real>;

// Opaque pointer to application data
class MeshBlockApplicationData {
 public:
  // make this pure virtual so that this class cannot be instantiated
  // (only derived classes can be instantiated)
  virtual ~MeshBlockApplicationData() = 0;
};
using pMeshBlockApplicationData_t = std::unique_ptr<MeshBlockApplicationData>;

// we still need to define this somewhere, though
inline MeshBlockApplicationData::~MeshBlockApplicationData() {}

//----------------------------------------------------------------------------------------
//! \class MeshBlock
//  \brief data/functions associated with a single block
class MeshBlock {
  friend class RestartOutput;
  friend class Mesh;
#ifdef HDF5OUTPUT
  friend class ATHDF5Output;
#endif

 public:
  MeshBlock(int igid, int ilid, LogicalLocation iloc, RegionSize input_size,
            BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin,
            Properties_t& properties,
            int igflag,
            bool ref_flag = false);
  MeshBlock(int igid, int ilid, Mesh *pm, ParameterInput *pin,
            Properties_t & properties,
            Packages_t& packages,
            LogicalLocation iloc,
            RegionSize input_block, BoundaryFlag *input_bcs, double icost,
            char *mbdata, int igflag);

  MeshBlock(int igid, int ilid,
            LogicalLocation iloc,
            RegionSize input_block,
            BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin,
            Properties_t& properties,
            Packages_t& packages,
            int igflag, bool ref_flag=false);
  ~MeshBlock();

  // Kokkos execution space for this MeshBlock
  DevSpace exec_space;

  // data
  Mesh *pmy_mesh;  // ptr to Mesh containing this MeshBlock
  LogicalLocation loc;
  RegionSize block_size;
  // for convenience: "max" # of real+ghost cells along each dir for allocating "standard"
  // sized MeshBlock arrays, depending on ndim (i.e. ncells2=nx2+2*NGHOST if nx2>1)
  IndexShape all_cells;
  IndexShape active_cells; 
  IndexShape all_coarse_cells;
  IndexShape active_coarse_cells;
  // on 1x coarser level MeshBlock (i.e. ncc2=nx2/2 + 2*NGHOST, if nx2>1)
  //int ncc1, ncc2, ncc3;
//  int is, ie, js, je, ks, ke;
  int gid, lid;
  //int cis, cie, cjs, cje, cks, cke, 
  int cnghost;
  int gflag;
  // At every cycle n, and field registers (u, b) are advanced from t^n -> t^{n+1},
  // the time-integration scheme may partially substep several storage register pairs
  // (u,b), (u1,b1), (u2, b2), ..., (um, bm) through the dt interval. Track their time
  // abscissae at the end of each stage (1<=l<=nstage) as (dt_m^l) relative to t^n
  Real stage_abscissae[MAX_NSTAGE+1][MAX_NREGISTER];

  // user output variables for analysis
  int nuser_out_var;
  AthenaArray<Real> user_out_var;
  std::string *user_out_var_names_;

  // user MeshBlock data that can be stored in restart files
  AthenaArray<Real> *ruser_meshblock_data;
  AthenaArray<int> *iuser_meshblock_data;

  // The User defined containers
  Container<Real> real_container;

  Properties_t properties;
  Packages_t packages;

  std::unique_ptr<MeshBlockApplicationData> app;

  // mesh-related objects
  std::unique_ptr<Coordinates> pcoord;
  std::unique_ptr<BoundaryValues> pbval;
  std::unique_ptr<MeshRefinement> pmr;
  std::unique_ptr<Reconstruction> precon;

  BoundaryFlag boundary_flag[6];

  MeshBlock *prev, *next;

  // functions

  //----------------------------------------------------------------------------------------
  //! \fn void MeshBlock::DeepCopy(const DstType& dst, const SrcType& src)
  //  \brief Deep copy between views using the exec space of the MeshBlock
  template <class DstType, class SrcType>
  void deep_copy(const DstType &dst, const SrcType &src) {
    Kokkos::deep_copy(exec_space, dst, src);
  }

  // 1D default loop pattern
  template <typename Function>
  inline void par_for(const std::string &name, const int &il, const int &iu,
                      const Function &function) {
    parthenon::par_for(name, exec_space, il, iu, function);
  }

  // 2D default loop pattern
  template <typename Function>
  inline void par_for(const std::string &name, const int &jl, const int &ju,
                      const int &il, const int &iu, const Function &function) {
    parthenon::par_for(name, exec_space, jl, ju, il, iu, function);
  }

  // 3D default loop pattern
  template <typename Function>
  inline void par_for(const std::string &name, const int &kl, const int &ku,
                      const int &jl, const int &ju, const int &il,
                      const int &iu, const Function &function) {
    parthenon::par_for(name, exec_space, kl, ku, jl, ju, il, iu, function);
  }

  // 4D default loop pattern
  template <typename Function>
  inline void par_for(const std::string &name, const int &nl, const int &nu,
                      const int &kl, const int &ku, const int &jl,
                      const int &ju, const int &il, const int &iu,
                      const Function &function) {
    parthenon::par_for(name, exec_space, nl, nu, kl, ku, jl, ju, il, iu, function);
  }

  std::size_t GetBlockSizeInBytes();
  int GetNumberOfMeshBlockCells() {
    return block_size.nx1*block_size.nx2*block_size.nx3; }
  void SearchAndSetNeighbors(MeshBlockTree &tree, int *ranklist, int *nslist);
  void WeightedAve(AthenaArray<Real> &u_out, AthenaArray<Real> &u_in1,
                   AthenaArray<Real> &u_in2, const Real wght[3]);
  void WeightedAve(FaceField &b_out, FaceField &b_in1, FaceField &b_in2,
                   const Real wght[3]);

  void ResetToIC() { ProblemGenerator(nullptr); }

  // inform MeshBlock which arrays contained in member Field, Particles,
  // ... etc. classes are the "primary" representations of a quantity. when registered,
  // that data are used for (1) load balancing (2) (future) dumping to restart file
  void RegisterMeshBlockData(Variable<Real> &pvar_cc);
  void RegisterMeshBlockData(FaceField &pvar_fc);

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  void UserWorkBeforeOutput(ParameterInput *pin); // called in Mesh fn (friend class)
  void UserWorkInLoop();                          // called in TimeIntegratorTaskList
  void SetBlockTimestep(const Real dt) {new_block_dt_ = dt;}

 private:
  // data
  Real new_block_dt_, new_block_dt_hyperbolic_, new_block_dt_parabolic_,
    new_block_dt_user_;
  int nreal_user_meshblock_data_, nint_user_meshblock_data_;
  std::vector<std::reference_wrapper<Variable<Real>>> vars_cc_;
  std::vector<std::reference_wrapper<FaceField>> vars_fc_;

  void InitializeIndexShapes();
  // functions
  void AllocateRealUserMeshBlockDataField(int n);
  void AllocateIntUserMeshBlockDataField(int n);
  void AllocateUserOutputVariables(int n);
  void SetUserOutputVariableName(int n, const char *name);
  void SetCostForLoadBalancing(double cost);

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  void ProblemGenerator(ParameterInput *pin);
  pMeshBlockApplicationData_t InitApplicationMeshBlockData(ParameterInput *pin);
  void InitUserMeshBlockData(ParameterInput *pin);

  // functions and variables for automatic load balancing based on timing
  double cost_, lb_time_;
  void ResetTimeMeasurement();
  void StartTimeMeasurement();
  void StopTimeMeasurement();
};

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
  friend class FieldDiffusion;
#ifdef HDF5OUTPUT
  friend class ATHDF5Output;
#endif

 public:
  // 2x function overloads of ctor: normal and restarted simulation
  Mesh(ParameterInput *pin,
      Properties_t &properties,
      Packages_t& packages, int test_flag=0);
  Mesh(ParameterInput *pin, IOWrapper &resfile,
      Properties_t &properties,
      Packages_t& packages, int test_flag=0);
  ~Mesh();

  // accessors
  int GetNumMeshBlocksThisRank(int my_rank) {return nblist[my_rank];}
  int GetNumMeshThreads() const {return num_mesh_threads_;}
  std::int64_t GetTotalCells() {return static_cast<std::int64_t> (nbtotal)*
      pblock->block_size.nx1*pblock->block_size.nx2*pblock->block_size.nx3;}

  // data
  RegionSize mesh_size;
  BoundaryFlag mesh_bcs[6];
  const int ndim;     // number of dimensions
  const bool adaptive, multilevel;
  Real start_time, time, tlim, dt, dt_hyperbolic, dt_parabolic, dt_user;
  int nlim, ncycle, ncycle_out, dt_diagnostics;
  int nbtotal, nbnew, nbdel;
  std::uint64_t mbcnt;

  int step_since_lb;
  int gflag;

  // ptr to first MeshBlock (node) in linked list of blocks belonging to this MPI rank:
  MeshBlock *pblock;
  Properties_t properties;
  Packages_t packages;

  AthenaArray<Real> *ruser_mesh_data;
  AthenaArray<int> *iuser_mesh_data;

  // functions
  void Initialize(int res_flag, ParameterInput *pin);
  void SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                 BoundaryFlag *block_bcs);
  void NewTimeStep();
  void OutputCycleDiagnostics();
  void LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin);
  int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3);
  MeshBlock* FindMeshBlock(int tgid);
  void ApplyUserWorkBeforeOutput(ParameterInput *pin);

  // function for distributing unique "phys" bitfield IDs to BoundaryVariable objects and
  // other categories of MPI communication for generating unique MPI_TAGs
  int ReserveTagPhysIDs(int num_phys);

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  void UserWorkAfterLoop(ParameterInput *pin);   // called in main loop
  void UserWorkInLoop(); // called in main after each cycle
  int GetRootLevel() { return root_level; }

 private:
  // data
  int next_phys_id_; // next unused value for encoding final component of MPI tag bitfield
  int root_level, max_level, current_level;
  int num_mesh_threads_;
  int *nslist, *ranklist, *nblist;
  double *costlist;
  // 8x arrays used exclusively for AMR (not SMR):
  int *nref, *nderef;
  int *rdisp, *ddisp;
  int *bnref, *bnderef;
  int *brdisp, *bddisp;
  // the last 4x should be std::size_t, but are limited to int by MPI

  LogicalLocation *loclist;
  MeshBlockTree tree;
  // number of MeshBlocks in the x1, x2, x3 directions of the root grid:
  // (unlike LogicalLocation.lxi, nrbxi don't grow w/ AMR # of levels, so keep 32-bit int)
  int nrbx1, nrbx2, nrbx3;
  // TODO(felker) find unnecessary static_cast<> ops. from old std::int64_t type in 2018:
  //std::int64_t nrbx1, nrbx2, nrbx3;

  // flags are false if using non-uniform or user meshgen function
  bool use_uniform_meshgen_fn_[3];
  int nreal_user_mesh_data_, nint_user_mesh_data_;

  int nuser_history_output_;
  std::string *user_history_output_names_;
  UserHistoryOperation *user_history_ops_;

  // variables for load balancing control
  bool lb_flag_, lb_automatic_, lb_manual_;
  double lb_tolerance_;
  int lb_interval_;

  // functions
  MeshGenFunc MeshGenerator_[3];
  BValFunc BoundaryFunction_[6];
  AMRFlagFunc AMRFlag_;
  SrcTermFunc UserSourceTerm_;
  TimeStepFunc UserTimeStep_;
  HistoryOutputFunc *user_history_func_;
  MetricFunc UserMetric_;
  FieldDiffusionCoeffFunc FieldDiffusivity_;

  void AllocateRealUserMeshDataField(int n);
  void AllocateIntUserMeshDataField(int n);
  void OutputMeshStructure(int dim);
  void CalculateLoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb);
  void ResetLoadBalanceVariables();

  void CorrectMidpointInitialCondition(std::vector<MeshBlock*> &pmb_array, int nmb);
  void ReserveMeshBlockPhysIDs();

  // Mesh::LoadBalancingAndAdaptiveMeshRefinement() helper functions:
  void UpdateCostList();
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  bool GatherCostListAndCheckBalance();
  void RedistributeAndRefineMeshBlocks(ParameterInput *pin, int ntot);

  // Mesh::RedistributeAndRefineMeshBlocks() helper functions:
  // step 6: send
  void PrepareSendSameLevel(MeshBlock* pb, Real *sendbuf);
  void PrepareSendCoarseToFineAMR(MeshBlock* pb, Real *sendbuf, LogicalLocation &lloc);
  void PrepareSendFineToCoarseAMR(MeshBlock* pb, Real *sendbuf);
  // step 7: create new MeshBlock list (same MPI rank but diff level: create new block)
  void FillSameRankFineToCoarseAMR(MeshBlock* pob, MeshBlock* pmb,
                                   LogicalLocation &loc);
  void FillSameRankCoarseToFineAMR(MeshBlock* pob, MeshBlock* pmb,
                                   LogicalLocation &newloc);
  // step 8: receive
  void FinishRecvSameLevel(MeshBlock *pb, Real *recvbuf);
  void FinishRecvFineToCoarseAMR(MeshBlock *pb, Real *recvbuf, LogicalLocation &lloc);
  void FinishRecvCoarseToFineAMR(MeshBlock *pb, Real *recvbuf);

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  void InitUserMeshData(ParameterInput *pin);

  // often used (not defined) in prob file in ../pgen/
  void EnrollUserBoundaryFunction(BoundaryFace face, BValFunc my_func);
  // DEPRECATED(felker): provide trivial overload for old-style BoundaryFace enum argument
  void EnrollUserBoundaryFunction(int face, BValFunc my_func);

  void EnrollUserRefinementCondition(AMRFlagFunc amrflag);
  void EnrollUserMeshGenerator(CoordinateDirection dir, MeshGenFunc my_mg);
  void EnrollUserExplicitSourceFunction(SrcTermFunc my_func);
  void EnrollUserTimeStepFunction(TimeStepFunc my_func);
  void AllocateUserHistoryOutput(int n);
  void EnrollUserHistoryOutput(int i, HistoryOutputFunc my_func, const char *name,
                               UserHistoryOperation op=UserHistoryOperation::sum);
  void EnrollUserMetric(MetricFunc my_func);
  void EnrollFieldDiffusivity(FieldDiffusionCoeffFunc my_func);
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
    return static_cast<Real>(index)/static_cast<Real>(nrange);
  } else {
    // to map to a [-0.5, 0.5] range, rescale int indices around 0 before FP conversion
    // if nrange is even, there is an index at center x=0.0; map it to (int) 0
    // if nrange is odd, the center x=0.0 is between two indices; map them to -1, 1
    std::int64_t noffset = index - (nrange)/2;
    std::int64_t noffset_ceil = index - (nrange+1)/2; // = noffset if nrange is even
    //std::cout << "noffset, noffset_ceil = " << noffset << ", " << noffset_ceil << "\n";
    // average the (possibly) biased integer indexing
    return static_cast<Real>(noffset + noffset_ceil)/(2.0*nrange);
  }
}

//----------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX1(Real x, RegionSize rs)
// \brief x1 mesh generator function, x is the logical location; x=i/nx1, real in [0., 1.]

inline Real DefaultMeshGeneratorX1(Real x, RegionSize rs) {
  Real lw, rw;
  if (rs.x1rat==1.0) {
    rw=x, lw=1.0-x;
  } else {
    Real ratn=std::pow(rs.x1rat,rs.nx1);
    Real rnx=std::pow(rs.x1rat,x*rs.nx1);
    lw=(rnx-ratn)/(1.0-ratn);
    rw=1.0-lw;
  }
  // linear interp, equally weighted from left (x(xmin)=0.0) and right (x(xmax)=1.0)
  return rs.x1min*lw+rs.x1max*rw;
}

//----------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX2(Real x, RegionSize rs)
// \brief x2 mesh generator function, x is the logical location; x=j/nx2, real in [0., 1.]

inline Real DefaultMeshGeneratorX2(Real x, RegionSize rs) {
  Real lw, rw;
  if (rs.x2rat==1.0) {
    rw=x, lw=1.0-x;
  } else {
    Real ratn=std::pow(rs.x2rat,rs.nx2);
    Real rnx=std::pow(rs.x2rat,x*rs.nx2);
    lw=(rnx-ratn)/(1.0-ratn);
    rw=1.0-lw;
  }
  return rs.x2min*lw+rs.x2max*rw;
}

//----------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX3(Real x, RegionSize rs)
// \brief x3 mesh generator function, x is the logical location; x=k/nx3, real in [0., 1.]

inline Real DefaultMeshGeneratorX3(Real x, RegionSize rs) {
  Real lw, rw;
  if (rs.x3rat==1.0) {
    rw=x, lw=1.0-x;
  } else {
    Real ratn=std::pow(rs.x3rat,rs.nx3);
    Real rnx=std::pow(rs.x3rat,x*rs.nx3);
    lw=(rnx-ratn)/(1.0-ratn);
    rw=1.0-lw;
  }
  return rs.x3min*lw+rs.x3max*rw;
}

//----------------------------------------------------------------------------------------
// \!fn Real UniformMeshGeneratorX1(Real x, RegionSize rs)
// \brief x1 mesh generator function, x is the logical location; real cells in [-0.5, 0.5]

inline Real UniformMeshGeneratorX1(Real x, RegionSize rs) {
  // linear interp, equally weighted from left (x(xmin)=-0.5) and right (x(xmax)=0.5)
  return static_cast<Real>(0.5)*(rs.x1min+rs.x1max) + (x*rs.x1max - x*rs.x1min);
}

//----------------------------------------------------------------------------------------
// \!fn Real UniformMeshGeneratorX2(Real x, RegionSize rs)
// \brief x2 mesh generator function, x is the logical location; real cells in [-0.5, 0.5]

inline Real UniformMeshGeneratorX2(Real x, RegionSize rs) {
  return static_cast<Real>(0.5)*(rs.x2min+rs.x2max) + (x*rs.x2max - x*rs.x2min);
}

//----------------------------------------------------------------------------------------
// \!fn Real UniformMeshGeneratorX3(Real x, RegionSize rs)
// \brief x3 mesh generator function, x is the logical location; real cells in [-0.5, 0.5]

inline Real UniformMeshGeneratorX3(Real x, RegionSize rs) {
  return static_cast<Real>(0.5)*(rs.x3min+rs.x3max) + (x*rs.x3max - x*rs.x3min);
}
} // namespace parthenon
#endif  // MESH_MESH_HPP_
