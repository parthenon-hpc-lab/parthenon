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
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "parthenon_mpi.hpp"

#include "bvals/boundary_conditions.hpp"
#include "bvals/bvals.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/update.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_tree.hpp"
#include "outputs/restart.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"
#include "utils/partition_stl_containers.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
           int mesh_test)
    : // public members:
      modified(true), is_restart(false),
      // aggregate initialization of RegionSize struct:
      mesh_size({pin->GetReal("parthenon/mesh", "x1min"),
                 pin->GetReal("parthenon/mesh", "x2min"),
                 pin->GetReal("parthenon/mesh", "x3min")},
                {pin->GetReal("parthenon/mesh", "x1max"),
                 pin->GetReal("parthenon/mesh", "x2max"),
                 pin->GetReal("parthenon/mesh", "x3max")},
                {pin->GetOrAddReal("parthenon/mesh", "x1rat", 1.0),
                 pin->GetOrAddReal("parthenon/mesh", "x2rat", 1.0),
                 pin->GetOrAddReal("parthenon/mesh", "x3rat", 1.0)},
                {pin->GetInteger("parthenon/mesh", "nx1"),
                 pin->GetInteger("parthenon/mesh", "nx2"),
                 pin->GetInteger("parthenon/mesh", "nx3")},
                {false, pin->GetInteger("parthenon/mesh", "nx2") == 1,
                 pin->GetInteger("parthenon/mesh", "nx3") == 1}),
      mesh_bcs{
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix3_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox3_bc", "reflecting"))},
      ndim((mesh_size.nx(X3DIR) > 1) ? 3 : ((mesh_size.nx(X2DIR) > 1) ? 2 : 1)),
      adaptive(pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "adaptive"
                   ? true
                   : false),
      multilevel(
          (adaptive ||
           pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "static" ||
           pin->GetOrAddString("parthenon/mesh", "multigrid", "false") == "true")
              ? true
              : false),
      multigrid(pin->GetOrAddString("parthenon/mesh", "multigrid", "false") == "true"
                    ? true
                    : false),
      nbnew(), nbdel(), step_since_lb(), gflag(), packages(packages),
      // private members:
      num_mesh_threads_(pin->GetOrAddInteger("parthenon/mesh", "num_threads", 1)),
      tree(this), use_uniform_meshgen_fn_{true, true, true, true}, lb_flag_(true),
      lb_automatic_(),
      lb_manual_(), MeshBndryFnctn{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr} {
  std::stringstream msg;
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  std::int64_t nbmax;

  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

  // check number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads=" << num_mesh_threads_
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  for (auto &[dir, label] : std::vector<std::pair<CoordinateDirection, std::string>>{
           {X1DIR, "1"}, {X2DIR, "2"}, {X3DIR, "3"}}) {
    // check number of grid cells in root level of mesh from input file.
    if (mesh_size.nx(dir) < 1) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "In mesh block in input file nx" + label + " must be >= 1, but nx" + label +
                 "="
          << mesh_size.nx(dir) << std::endl;
      PARTHENON_FAIL(msg);
    }
    // check physical size of mesh (root level) from input file.
    if (mesh_size.xmax(dir) <= mesh_size.xmin(dir)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Input x" + label + "max must be larger than x" + label + "min: x" + label +
                 "min="
          << mesh_size.xmin(dir) << " x" + label + "max=" << mesh_size.xmax(dir)
          << std::endl;
      PARTHENON_FAIL(msg);
    }
  }

  if (mesh_size.nx(X2DIR) == 1 && mesh_size.nx(X3DIR) > 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_size.nx(X3DIR)
        << ", 2D problems in x1-x3 plane not supported" << std::endl;
    PARTHENON_FAIL(msg);
  }

  // Allow for user overrides to default Parthenon functions
  if (app_in->InitUserMeshData != nullptr) {
    InitUserMeshData = app_in->InitUserMeshData;
  }
  if (app_in->MeshProblemGenerator != nullptr) {
    ProblemGenerator = app_in->MeshProblemGenerator;
  }
  if (app_in->PreStepMeshUserWorkInLoop != nullptr) {
    PreStepUserWorkInLoop = app_in->PreStepMeshUserWorkInLoop;
  }
  if (app_in->PostStepMeshUserWorkInLoop != nullptr) {
    PostStepUserWorkInLoop = app_in->PostStepMeshUserWorkInLoop;
  }
  if (app_in->PreStepDiagnosticsInLoop != nullptr) {
    PreStepUserDiagnosticsInLoop = app_in->PreStepDiagnosticsInLoop;
  }
  if (app_in->PostStepDiagnosticsInLoop != nullptr) {
    PostStepUserDiagnosticsInLoop = app_in->PostStepDiagnosticsInLoop;
  }
  if (app_in->UserWorkAfterLoop != nullptr) {
    UserWorkAfterLoop = app_in->UserWorkAfterLoop;
  }

  // check the consistency of the periodic boundaries
  if (((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic &&
        mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic) ||
       (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic &&
        mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic)) ||
      (mesh_size.nx(X2DIR) > 1 &&
       ((mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic) ||
        (mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic))) ||
      (mesh_size.nx(X3DIR) > 1 &&
       ((mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x3] != BoundaryFlag::periodic) ||
        (mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x3] == BoundaryFlag::periodic)))) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "When periodic boundaries are in use, both sides must be periodic."
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  EnrollBndryFncts_(app_in);
  for (auto &[dir, label] : std::vector<std::tuple<CoordinateDirection, std::string>>{
           {X1DIR, "nx1"}, {X2DIR, "nx2"}, {X3DIR, "nx3"}}) {
    block_size.xrat(dir) = mesh_size.xrat(dir);
    block_size.symmetry(dir) = mesh_size.symmetry(dir);
    if (!block_size.symmetry(dir)) {
      block_size.nx(dir) =
          pin->GetOrAddInteger("parthenon/meshblock", label, mesh_size.nx(dir));
    } else {
      block_size.nx(dir) = mesh_size.nx(dir);
    }
    nrbx[dir - 1] = mesh_size.nx(dir) / block_size.nx(dir);
  }
  nbmax = *std::max_element(std::begin(nrbx), std::end(nrbx));
  base_block_size = block_size;

  // check consistency of the block and mesh
  if (mesh_size.nx(X1DIR) % block_size.nx(X1DIR) != 0 ||
      mesh_size.nx(X2DIR) % block_size.nx(X2DIR) != 0 ||
      mesh_size.nx(X3DIR) % block_size.nx(X3DIR) != 0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "the Mesh must be evenly divisible by the MeshBlock" << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (block_size.nx(X1DIR) < 4 || (block_size.nx(X2DIR) < 4 && (ndim >= 2)) ||
      (block_size.nx(X3DIR) < 4 && (ndim >= 3))) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "block_size must be larger than or equal to 4 cells." << std::endl;
    PARTHENON_FAIL(msg);
  }

  // initialize user-enrollable functions
  default_pack_size_ = pin->GetOrAddInteger("parthenon/mesh", "pack_size", -1);

  // calculate the logical root level and maximum level
  for (root_level = 0; (1 << root_level) < nbmax; root_level++) {
  }
  current_level = root_level;

  tree.CreateRootGrid();

  // Load balancing flag and parameters
  RegisterLoadBalancing_(pin);

  // SMR / AMR:
  if (adaptive) {
    max_level = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 63) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63 - root_level + 1 << "." << std::endl;
      PARTHENON_FAIL(msg);
    }
  } else {
    max_level = 63;
  }

  InitUserMeshData(this, pin);

  if (multilevel) {
    if (block_size.nx(X1DIR) % 2 == 1 || (block_size.nx(X2DIR) % 2 == 1 && (ndim >= 2)) ||
        (block_size.nx(X3DIR) % 2 == 1 && (ndim >= 3))) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The size of MeshBlock must be divisible by 2 in order to use SMR or AMR."
          << std::endl;
      PARTHENON_FAIL(msg);
    }

    InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 27, "parthenon/static_refinement") == 0) {
        RegionSize ref_size;
        ref_size.xmin(X1DIR) = pin->GetReal(pib->block_name, "x1min");
        ref_size.xmax(X1DIR) = pin->GetReal(pib->block_name, "x1max");
        if (ndim >= 2) {
          ref_size.xmin(X2DIR) = pin->GetReal(pib->block_name, "x2min");
          ref_size.xmax(X2DIR) = pin->GetReal(pib->block_name, "x2max");
        } else {
          ref_size.xmin(X2DIR) = mesh_size.xmin(X2DIR);
          ref_size.xmax(X2DIR) = mesh_size.xmax(X2DIR);
        }
        if (ndim == 3) {
          ref_size.xmin(X3DIR) = pin->GetReal(pib->block_name, "x3min");
          ref_size.xmax(X3DIR) = pin->GetReal(pib->block_name, "x3max");
        } else {
          ref_size.xmin(X3DIR) = mesh_size.xmin(X3DIR);
          ref_size.xmax(X3DIR) = mesh_size.xmax(X3DIR);
        }
        int ref_lev = pin->GetInteger(pib->block_name, "level");
        int lrlev = ref_lev + root_level;
        if (lrlev > current_level) current_level = lrlev;
        // range check
        if (ref_lev < 1) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level must be larger than 0 (root level = 0)" << std::endl;
          PARTHENON_FAIL(msg);
        }
        if (lrlev > max_level) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level exceeds the maximum level (specify "
              << "'maxlevel' parameter in <parthenon/mesh> input block if adaptive)."
              << std::endl;

          PARTHENON_FAIL(msg);
        }
        if (ref_size.xmin(X1DIR) > ref_size.xmax(X1DIR) ||
            ref_size.xmin(X2DIR) > ref_size.xmax(X2DIR) ||
            ref_size.xmin(X3DIR) > ref_size.xmax(X3DIR)) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Invalid refinement region is specified." << std::endl;
          PARTHENON_FAIL(msg);
        }
        if (ref_size.xmin(X1DIR) < mesh_size.xmin(X1DIR) ||
            ref_size.xmax(X1DIR) > mesh_size.xmax(X1DIR) ||
            ref_size.xmin(X2DIR) < mesh_size.xmin(X2DIR) ||
            ref_size.xmax(X2DIR) > mesh_size.xmax(X2DIR) ||
            ref_size.xmin(X3DIR) < mesh_size.xmin(X3DIR) ||
            ref_size.xmax(X3DIR) > mesh_size.xmax(X3DIR)) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement region must be smaller than the whole mesh." << std::endl;
          PARTHENON_FAIL(msg);
        }
        std::int64_t l_region_min[3]{0, 0, 0};
        std::int64_t l_region_max[3]{1, 1, 1};
        for (auto dir : {X1DIR, X2DIR, X3DIR}) {
          if (!mesh_size.symmetry(dir)) {
            l_region_min[dir - 1] =
                GetLLFromMeshCoordinate(dir, lrlev, ref_size.xmin(dir));
            l_region_max[dir - 1] =
                GetLLFromMeshCoordinate(dir, lrlev, ref_size.xmax(dir));
            l_region_min[dir - 1] =
                std::max(l_region_min[dir - 1], static_cast<std::int64_t>(0));
            l_region_max[dir - 1] =
                std::min(l_region_max[dir - 1],
                         static_cast<std::int64_t>(nrbx[dir - 1] * (1LL << ref_lev) - 1));
            auto current_loc =
                LogicalLocation(lrlev, l_region_max[0], l_region_max[1], l_region_max[2]);
            // Remove last block if it just it's boundary overlaps with the region
            if (GetMeshCoordinate(dir, BlockLocation::Left, current_loc) ==
                ref_size.xmax(dir))
              l_region_max[dir - 1]--;
            if (l_region_min[dir - 1] % 2 == 1) l_region_min[dir - 1]--;
            if (l_region_max[dir - 1] % 2 == 0) l_region_max[dir - 1]++;
          }
        }
        for (std::int64_t k = l_region_min[2]; k < l_region_max[2]; k += 2) {
          for (std::int64_t j = l_region_min[1]; j < l_region_max[1]; j += 2) {
            for (std::int64_t i = l_region_min[0]; i < l_region_max[0]; i += 2) {
              LogicalLocation nloc(lrlev, i, j, k);
              int nnew;
              tree.AddMeshBlock(nloc, nnew);
            }
          }
        }
      }
      pib = pib->pnext;
    }
  }

  // initial mesh hierarchy construction is completed here
  tree.CountMeshBlock(nbtotal);
  loclist.resize(nbtotal);
  tree.GetMeshBlockList(loclist.data(), nullptr, nbtotal);

#ifdef MPI_PARALLEL
  // check if there are sufficient blocks
  if (nbtotal < Globals::nranks) {
    if (mesh_test == 0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal (" << nbtotal << ") < nranks ("
          << Globals::nranks << ")" << std::endl;
      PARTHENON_FAIL(msg);
    } else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal (" << nbtotal << ") < nranks ("
                << Globals::nranks << ")" << std::endl;
    }
  }
#endif

  ranklist = std::vector<int>(nbtotal);

  nslist = std::vector<int>(Globals::nranks);
  nblist = std::vector<int>(Globals::nranks);
  if (adaptive) { // allocate arrays for AMR
    nref = std::vector<int>(Globals::nranks);
    nderef = std::vector<int>(Globals::nranks);
    rdisp = std::vector<int>(Globals::nranks);
    ddisp = std::vector<int>(Globals::nranks);
    bnref = std::vector<int>(Globals::nranks);
    bnderef = std::vector<int>(Globals::nranks);
    brdisp = std::vector<int>(Globals::nranks);
    bddisp = std::vector<int>(Globals::nranks);
  }

  // initialize cost array with the simplest estimate; all the blocks are equal
  costlist = std::vector<double>(nbtotal, 1.0);

  CalculateLoadBalance(costlist, ranklist, nslist, nblist);
  PopulateLeafLocationMap();

  // Output some diagnostic information to terminal

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0) {
    if (Globals::my_rank == 0) OutputMeshStructure(ndim);
    return;
  }

  mesh_data.SetMeshPointer(this);

  resolved_packages = ResolvePackages(packages);

  // Register user defined boundary conditions
  UserBoundaryFunctions = resolved_packages->UserBoundaryFunctions;

  // Setup unique comms for each variable and swarm
  SetupMPIComms();

  // create MeshBlock list for this process
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;
  // create MeshBlock list for this process
  block_list.clear();
  block_list.resize(nbe - nbs + 1);
  for (int i = nbs; i <= nbe; i++) {
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    block_list[i - nbs] =
        MeshBlock::Make(i, i - nbs, loclist[i], block_size, block_bcs, this, pin, app_in,
                        packages, resolved_packages, gflag);
    block_list[i - nbs]->SearchAndSetNeighbors(this, tree, ranklist.data(),
                                               nslist.data());
  }
  SetSameLevelNeighbors(block_list, leaf_grid_locs, this->GetRootGridInfo(), nbs, false);
  BuildGMGHierarchy(nbs, pin, app_in);
  ResetLoadBalanceVariables();
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file
Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, RestartReader &rr,
           Packages_t &packages, int mesh_test)
    : // public members:
      // aggregate initialization of RegionSize struct:
      // (will be overwritten by memcpy from restart file, in this case)
      modified(true), is_restart(true),
      // aggregate initialization of RegionSize struct:
      mesh_size({pin->GetReal("parthenon/mesh", "x1min"),
                 pin->GetReal("parthenon/mesh", "x2min"),
                 pin->GetReal("parthenon/mesh", "x3min")},
                {pin->GetReal("parthenon/mesh", "x1max"),
                 pin->GetReal("parthenon/mesh", "x2max"),
                 pin->GetReal("parthenon/mesh", "x3max")},
                {pin->GetOrAddReal("parthenon/mesh", "x1rat", 1.0),
                 pin->GetOrAddReal("parthenon/mesh", "x2rat", 1.0),
                 pin->GetOrAddReal("parthenon/mesh", "x3rat", 1.0)},
                {pin->GetInteger("parthenon/mesh", "nx1"),
                 pin->GetInteger("parthenon/mesh", "nx2"),
                 pin->GetInteger("parthenon/mesh", "nx3")},
                {false, pin->GetInteger("parthenon/mesh", "nx2") == 1,
                 pin->GetInteger("parthenon/mesh", "nx3") == 1}),
      mesh_bcs{
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix3_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox3_bc", "reflecting"))},
      ndim((mesh_size.nx(X3DIR) > 1) ? 3 : ((mesh_size.nx(X2DIR) > 1) ? 2 : 1)),
      adaptive(pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "adaptive"
                   ? true
                   : false),
      multilevel(
          (adaptive ||
           pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "static" ||
           pin->GetOrAddString("parthenon/mesh", "multigrid", "false") == "true")
              ? true
              : false),
      multigrid(pin->GetOrAddString("parthenon/mesh", "multigrid", "false") == "true"
                    ? true
                    : false),
      nbnew(), nbdel(), step_since_lb(), gflag(), packages(packages),
      // private members:
      num_mesh_threads_(pin->GetOrAddInteger("parthenon/mesh", "num_threads", 1)),
      tree(this), use_uniform_meshgen_fn_{true, true, true, true}, lb_flag_(true),
      lb_automatic_(),
      lb_manual_(), MeshBndryFnctn{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr} {
  std::stringstream msg;
  RegionSize block_size;
  BoundaryFlag block_bcs[6];

  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

  // check the number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads=" << num_mesh_threads_
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  // read the restart file
  // the file is already open and the pointer is set to after <par_end>

  // All ranks read HDF file
  nbnew = rr.GetAttr<int>("Info", "NBNew");
  nbdel = rr.GetAttr<int>("Info", "NBDel");
  nbtotal = rr.GetAttr<int>("Info", "NumMeshBlocks");
  root_level = rr.GetAttr<int>("Info", "RootLevel");

  const auto bc = rr.GetAttrVec<std::string>("Info", "BoundaryConditions");
  for (int i = 0; i < 6; i++) {
    block_bcs[i] = GetBoundaryFlag(bc[i]);
  }

  // Allow for user overrides to default Parthenon functions
  if (app_in->InitUserMeshData != nullptr) {
    InitUserMeshData = app_in->InitUserMeshData;
  }
  if (app_in->PreStepMeshUserWorkInLoop != nullptr) {
    PreStepUserWorkInLoop = app_in->PreStepMeshUserWorkInLoop;
  }
  if (app_in->PostStepMeshUserWorkInLoop != nullptr) {
    PostStepUserWorkInLoop = app_in->PostStepMeshUserWorkInLoop;
  }
  if (app_in->PreStepDiagnosticsInLoop != nullptr) {
    PreStepUserDiagnosticsInLoop = app_in->PreStepDiagnosticsInLoop;
  }
  if (app_in->PostStepDiagnosticsInLoop != nullptr) {
    PostStepUserDiagnosticsInLoop = app_in->PostStepDiagnosticsInLoop;
  }
  if (app_in->UserWorkAfterLoop != nullptr) {
    UserWorkAfterLoop = app_in->UserWorkAfterLoop;
  }
  EnrollBndryFncts_(app_in);

  const auto grid_dim = rr.GetAttrVec<Real>("Info", "RootGridDomain");
  mesh_size.xmin(X1DIR) = grid_dim[0];
  mesh_size.xmax(X1DIR) = grid_dim[1];
  mesh_size.xrat(X1DIR) = grid_dim[2];

  mesh_size.xmin(X2DIR) = grid_dim[3];
  mesh_size.xmax(X2DIR) = grid_dim[4];
  mesh_size.xrat(X2DIR) = grid_dim[5];

  mesh_size.xmin(X3DIR) = grid_dim[6];
  mesh_size.xmax(X3DIR) = grid_dim[7];
  mesh_size.xrat(X3DIR) = grid_dim[8];

  // initialize
  loclist = std::vector<LogicalLocation>(nbtotal);

  const auto blockSize = rr.GetAttrVec<int>("Info", "MeshBlockSize");
  const auto includesGhost = rr.GetAttr<int>("Info", "IncludesGhost");
  const auto nGhost = rr.GetAttr<int>("Info", "NGhost");

  for (auto &dir : {X1DIR, X2DIR, X3DIR}) {
    block_size.xrat(dir) = mesh_size.xrat(dir);
    block_size.nx(dir) =
        blockSize[dir - 1] - (blockSize[dir - 1] > 1) * includesGhost * 2 * nGhost;
    if (block_size.nx(dir) == 1) {
      block_size.symmetry(dir) = true;
      mesh_size.symmetry(dir) = true;
    } else {
      block_size.symmetry(dir) = false;
      mesh_size.symmetry(dir) = false;
    }
    // calculate the number of the blocks
    nrbx[dir - 1] = mesh_size.nx(dir) / block_size.nx(dir);
  }
  base_block_size = block_size;

  // Load balancing flag and parameters
  RegisterLoadBalancing_(pin);

  // SMR / AMR
  if (adaptive) {
    // read from file or from input?  input for now.
    //    max_level = rr.GetAttr<int>("Info", "MaxLevel");
    max_level = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 63) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63 - root_level + 1 << "." << std::endl;
      PARTHENON_FAIL(msg);
    }
  } else {
    max_level = 63;
  }

  InitUserMeshData(this, pin);

  // Populate logical locations
  auto lx123 = rr.ReadDataset<int64_t>("/Blocks/loc.lx123");
  auto locLevelGidLidCnghostGflag =
      rr.ReadDataset<int>("/Blocks/loc.level-gid-lid-cnghost-gflag");
  current_level = -1;
  for (int i = 0; i < nbtotal; i++) {
    loclist[i] = LogicalLocation(locLevelGidLidCnghostGflag[5 * i], lx123[3 * i],
                                 lx123[3 * i + 1], lx123[3 * i + 2]);

    if (loclist[i].level() > current_level) {
      current_level = loclist[i].level();
    }
  }
  // rebuild the Block Tree
  tree.CreateRootGrid();

  for (int i = 0; i < nbtotal; i++) {
    tree.AddMeshBlockWithoutRefine(loclist[i]);
  }

  int nnb;
  // check the tree structure, and assign GID
  tree.GetMeshBlockList(loclist.data(), nullptr, nnb);
  if (nnb != nbtotal) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Tree reconstruction failed. The total numbers of the blocks do not match. ("
        << nbtotal << " != " << nnb << ")" << std::endl;
    PARTHENON_FAIL(msg);
  }

#ifdef MPI_PARALLEL
  if (nbtotal < Globals::nranks) {
    if (mesh_test == 0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal (" << nbtotal << ") < nranks ("
          << Globals::nranks << ")" << std::endl;
      PARTHENON_FAIL(msg);
    } else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal (" << nbtotal << ") < nranks ("
                << Globals::nranks << ")" << std::endl;
      return;
    }
  }
#endif
  costlist = std::vector<double>(nbtotal, 1.0);
  ranklist = std::vector<int>(nbtotal);
  nslist = std::vector<int>(Globals::nranks);
  nblist = std::vector<int>(Globals::nranks);

  if (adaptive) { // allocate arrays for AMR
    nref = std::vector<int>(Globals::nranks);
    nderef = std::vector<int>(Globals::nranks);
    rdisp = std::vector<int>(Globals::nranks);
    ddisp = std::vector<int>(Globals::nranks);
    bnref = std::vector<int>(Globals::nranks);
    bnderef = std::vector<int>(Globals::nranks);
    brdisp = std::vector<int>(Globals::nranks);
    bddisp = std::vector<int>(Globals::nranks);
  }

  CalculateLoadBalance(costlist, ranklist, nslist, nblist);
  PopulateLeafLocationMap();

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0) {
    if (Globals::my_rank == 0) OutputMeshStructure(ndim);
    return;
  }

  // allocate data buffer
  int nb = nblist[Globals::my_rank];
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nb - 1;

  mesh_data.SetMeshPointer(this);

  resolved_packages = ResolvePackages(packages);

  // Register user defined boundary conditions
  UserBoundaryFunctions = resolved_packages->UserBoundaryFunctions;

  // Setup unique comms for each variable and swarm
  SetupMPIComms();

  // Create MeshBlocks (parallel)
  block_list.clear();
  block_list.resize(nbe - nbs + 1);
  for (int i = nbs; i <= nbe; i++) {
    for (auto &v : block_bcs) {
      v = parthenon::BoundaryFlag::undef;
    }
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);

    // create a block and add into the link list
    block_list[i - nbs] =
        MeshBlock::Make(i, i - nbs, loclist[i], block_size, block_bcs, this, pin, app_in,
                        packages, resolved_packages, gflag, costlist[i]);
    block_list[i - nbs]->SearchAndSetNeighbors(this, tree, ranklist.data(),
                                               nslist.data());
  }
  SetSameLevelNeighbors(block_list, leaf_grid_locs, this->GetRootGridInfo(), nbs, false);
  BuildGMGHierarchy(nbs, pin, app_in);
  ResetLoadBalanceVariables();
}

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() {
#ifdef MPI_PARALLEL
  // Cleanup MPI comms
  for (auto &pair : mpi_comm_map_) {
    PARTHENON_MPI_CHECK(MPI_Comm_free(&(pair.second)));
  }
  mpi_comm_map_.clear();
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::OutputMeshStructure(int ndim)
//  \brief print the mesh structure information

void Mesh::OutputMeshStructure(const int ndim,
                               const bool dump_mesh_structure /*= true*/) {
  RegionSize block_size;
  BoundaryFlag block_bcs[6];

  // Write overall Mesh structure to stdout and file
  std::cout << std::endl;
  std::cout << "Root grid = " << nrbx[0] << " x " << nrbx[1] << " x " << nrbx[2]
            << " MeshBlocks" << std::endl;
  std::cout << "Total number of MeshBlocks = " << nbtotal << std::endl;
  std::cout << "Number of physical refinement levels = " << (current_level - root_level)
            << std::endl;
  std::cout << "Number of logical  refinement levels = " << current_level << std::endl;

  // compute/output number of blocks per level, and cost per level
  std::vector<int> nb_per_plevel(max_level + 1, 0);
  std::vector<int> cost_per_plevel(max_level + 1, 0);

  for (int i = 0; i < nbtotal; i++) {
    nb_per_plevel[(loclist[i].level() - root_level)]++;
    cost_per_plevel[(loclist[i].level() - root_level)] += costlist[i];
  }
  for (int i = root_level; i <= max_level; i++) {
    if (nb_per_plevel[i - root_level] != 0) {
      std::cout << "  Physical level = " << i - root_level << " (logical level = " << i
                << "): " << nb_per_plevel[i - root_level]
                << " MeshBlocks, cost = " << cost_per_plevel[i - root_level] << std::endl;
    }
  }

  if (!dump_mesh_structure) {
    return;
  }

  // compute/output number of blocks per rank, and cost per rank
  std::cout << "Number of parallel ranks = " << Globals::nranks << std::endl;
  std::vector<int> nb_per_rank(Globals::nranks, 0);
  std::vector<int> cost_per_rank(Globals::nranks, 0);

  for (int i = 0; i < nbtotal; i++) {
    nb_per_rank[ranklist[i]]++;
    cost_per_rank[ranklist[i]] += costlist[i];
  }
  for (int i = 0; i < Globals::nranks; ++i) {
    std::cout << "  Rank = " << i << ": " << nb_per_rank[i]
              << " MeshBlocks, cost = " << cost_per_rank[i] << std::endl;
  }

  FILE *fp = nullptr;

  // open 'mesh_structure.dat' file
  if ((fp = std::fopen("mesh_structure.dat", "wb")) == nullptr) {
    std::cout << "### ERROR in function Mesh::OutputMeshStructure" << std::endl
              << "Cannot open mesh_structure.dat" << std::endl;
    return;
  }

  // output relative size/locations of meshblock to file, for plotting
  double real_max = std::numeric_limits<double>::max();
  double mincost = real_max, maxcost = 0.0, totalcost = 0.0;
  for (int i = root_level; i <= max_level; i++) {
    for (int j = 0; j < nbtotal; j++) {
      if (loclist[j].level() == i) {
        SetBlockSizeAndBoundaries(loclist[j], block_size, block_bcs);
        const std::int64_t &lx1 = loclist[j].lx1();
        const std::int64_t &lx2 = loclist[j].lx2();
        const std::int64_t &lx3 = loclist[j].lx3();
        const int &ll = loclist[j].level();
        mincost = std::min(mincost, costlist[i]);
        maxcost = std::max(maxcost, costlist[i]);
        totalcost += costlist[i];
        std::fprintf(fp, "#MeshBlock %d on rank=%d with cost=%g\n", j, ranklist[j],
                     costlist[j]);
        std::fprintf(
            fp, "#  Logical level %d, location = (%" PRId64 " %" PRId64 " %" PRId64 ")\n",
            ll, lx1, lx2, lx3);
        if (ndim == 2) {
          std::fprintf(fp, "%g %g\n", block_size.xmin(X1DIR), block_size.xmin(X2DIR));
          std::fprintf(fp, "%g %g\n", block_size.xmax(X1DIR), block_size.xmin(X2DIR));
          std::fprintf(fp, "%g %g\n", block_size.xmax(X1DIR), block_size.xmax(X2DIR));
          std::fprintf(fp, "%g %g\n", block_size.xmin(X1DIR), block_size.xmax(X2DIR));
          std::fprintf(fp, "%g %g\n", block_size.xmin(X1DIR), block_size.xmin(X2DIR));
          std::fprintf(fp, "\n\n");
        }
        if (ndim == 3) {
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmax(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmax(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmax(X3DIR));
          std::fprintf(fp, "%g %g %g\n", block_size.xmin(X1DIR), block_size.xmin(X2DIR),
                       block_size.xmin(X3DIR));
          std::fprintf(fp, "\n\n");
        }
      }
    }
  }

  // close file, final outputs
  std::fclose(fp);
  std::cout << "Load Balancing:" << std::endl;
  std::cout << "  Minimum cost = " << mincost << ", Maximum cost = " << maxcost
            << ", Average cost = " << totalcost / nbtotal << std::endl
            << std::endl;
  std::cout << "See the 'mesh_structure.dat' file for a complete list"
            << " of MeshBlocks." << std::endl;
  std::cout << "Use 'python ../vis/python/plot_mesh.py' or gnuplot"
            << " to visualize mesh structure." << std::endl
            << std::endl;
}

//----------------------------------------------------------------------------------------
//  Enroll user-defined functions for boundary conditions
void Mesh::EnrollBndryFncts_(ApplicationInput *app_in) {
  static const BValFunc outflow[6] = {
      BoundaryFunction::OutflowInnerX1, BoundaryFunction::OutflowOuterX1,
      BoundaryFunction::OutflowInnerX2, BoundaryFunction::OutflowOuterX2,
      BoundaryFunction::OutflowInnerX3, BoundaryFunction::OutflowOuterX3};
  static const BValFunc reflect[6] = {
      BoundaryFunction::ReflectInnerX1, BoundaryFunction::ReflectOuterX1,
      BoundaryFunction::ReflectInnerX2, BoundaryFunction::ReflectOuterX2,
      BoundaryFunction::ReflectInnerX3, BoundaryFunction::ReflectOuterX3};

  for (int f = 0; f < BOUNDARY_NFACES; f++) {
    switch (mesh_bcs[f]) {
    case BoundaryFlag::reflect:
      MeshBndryFnctn[f] = reflect[f];
      break;
    case BoundaryFlag::outflow:
      MeshBndryFnctn[f] = outflow[f];
      break;
    case BoundaryFlag::user:
      if (app_in->boundary_conditions[f] != nullptr) {
        MeshBndryFnctn[f] = app_in->boundary_conditions[f];
      } else {
        std::stringstream msg;
        msg << "A user boundary condition for face " << f
            << " was requested. but no condition was enrolled." << std::endl;
        PARTHENON_THROW(msg);
      }
      break;
    default: // periodic/block BCs handled elsewhere.
      break;
    }

    switch (mesh_bcs[f]) {
    case BoundaryFlag::user:
      if (app_in->swarm_boundary_conditions[f] != nullptr) {
        // This is checked to be non-null later in Swarm::AllocateBoundaries, in case user
        // boundaries are requested but no swarms are used.
        SwarmBndryFnctn[f] = app_in->swarm_boundary_conditions[f];
      }
      break;
    default: // Default BCs handled elsewhere
      break;
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkBeforeOutput

void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin) {
  for (auto &pmb : block_list) {
    pmb->UserWorkBeforeOutput(pmb.get(), pin);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::Initialize(bool init_problem, ParameterInput *pin)
// \brief  initialization before the main loop as well as during remeshing

void Mesh::Initialize(bool init_problem, ParameterInput *pin, ApplicationInput *app_in) {
  PARTHENON_INSTRUMENT
  bool init_done = true;
  const int nb_initial = nbtotal;
  do {
    int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);

    // init meshblock data
    for (int i = 0; i < nmb; ++i) {
      MeshBlock *pmb = block_list[i].get();
      pmb->InitMeshBlockUserData(pmb, pin);
    }

    const int num_partitions = DefaultNumPartitions();

    // problem generator
    if (init_problem) {
      PARTHENON_REQUIRE_THROWS(
          !(ProblemGenerator != nullptr && block_list[0]->ProblemGenerator != nullptr),
          "Mesh and MeshBlock ProblemGenerators are defined. Please use only one.");

      // Call Mesh ProblemGenerator
      if (ProblemGenerator != nullptr) {
        PARTHENON_REQUIRE(num_partitions == 1,
                          "Mesh ProblemGenerator requires parthenon/mesh/pack_size=-1 "
                          "during first initialization.");

        auto &md = mesh_data.GetOrAdd("base", 0);
        ProblemGenerator(this, pin, md.get());
        // Call individual MeshBlock ProblemGenerator
      } else {
        for (int i = 0; i < nmb; ++i) {
          auto &pmb = block_list[i];
          pmb->ProblemGenerator(pmb.get(), pin);
        }
      }
      std::for_each(block_list.begin(), block_list.end(),
                    [](auto &sp_block) { sp_block->SetAllVariablesToInitialized(); });
    }

    // Pre comm fill derived
    for (int i = 0; i < nmb; ++i) {
      auto &mbd = block_list[i]->meshblock_data.Get();
      Update::PreCommFillDerived(mbd.get());
    }
    for (int i = 0; i < num_partitions; ++i) {
      auto &md = mesh_data.GetOrAdd("base", i);
      Update::PreCommFillDerived(md.get());
    }

    // Build densely populated communication tags
    tag_map.clear();
    for (int i = 0; i < num_partitions; i++) {
      auto &md = mesh_data.GetOrAdd("base", i);
      tag_map.AddMeshDataToMap<BoundaryType::any>(md);
      for (int gmg_level = 0; gmg_level < gmg_mesh_data.size(); ++gmg_level) {
        auto &mdg = gmg_mesh_data[gmg_level].GetOrAdd(gmg_level, "base", i);
        // tag_map.AddMeshDataToMap<BoundaryType::any>(mdg);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_same>(mdg);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_prolongate_send>(mdg);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_restrict_send>(mdg);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_prolongate_recv>(mdg);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_restrict_recv>(mdg);
      }
    }
    tag_map.ResolveMap();

    // Create send/recv MPI_Requests for all BoundaryData objects
    for (int i = 0; i < nmb; ++i) {
      auto &pmb = block_list[i];
      pmb->swarm_data.Get()->SetupPersistentMPI();
    }

    // send FillGhost variables
    bool can_delete;
    std::int64_t test_iters = 0;
    constexpr std::int64_t max_it = 1e10;
    do {
      can_delete = true;
      for (auto &[k, comm] : boundary_comm_map) {
        can_delete = comm.IsSafeToDelete() && can_delete;
      }
      test_iters++;
    } while (!can_delete && test_iters < max_it);
    PARTHENON_REQUIRE(
        test_iters < max_it,
        "Too many iterations waiting to delete boundary communication buffers.");

    boundary_comm_map.clear();
    boundary_comm_flxcor_map.clear();

    for (int i = 0; i < num_partitions; i++) {
      auto &md = mesh_data.GetOrAdd("base", i);
      BuildBoundaryBuffers(md);
      for (int gmg_level = 0; gmg_level < gmg_mesh_data.size(); ++gmg_level) {
        auto &mdg = gmg_mesh_data[gmg_level].GetOrAdd(gmg_level, "base", i);
        BuildBoundaryBuffers(mdg);
        BuildGMGBoundaryBuffers(mdg);
      }
    }

    std::vector<bool> sent(num_partitions, false);
    bool all_sent;
    std::int64_t send_iters = 0;
    do {
      all_sent = true;
      for (int i = 0; i < num_partitions; i++) {
        auto &md = mesh_data.GetOrAdd("base", i);
        if (!sent[i]) {
          if (SendBoundaryBuffers(md) != TaskStatus::complete) {
            all_sent = false;
          } else {
            sent[i] = true;
          }
        }
      }
      send_iters++;
    } while (!all_sent && send_iters < max_it);
    PARTHENON_REQUIRE(
        send_iters < max_it,
        "Too many iterations waiting to send boundary communication buffers.");

    // wait to receive FillGhost variables
    // TODO(someone) evaluate if ReceiveWithWait kind of logic is better, also related to
    // https://github.com/lanl/parthenon/issues/418
    std::vector<bool> received(num_partitions, false);
    bool all_received;
    std::int64_t receive_iters = 0;
    do {
      all_received = true;
      for (int i = 0; i < num_partitions; i++) {
        auto &md = mesh_data.GetOrAdd("base", i);
        if (!received[i]) {
          if (ReceiveBoundaryBuffers(md) != TaskStatus::complete) {
            all_received = false;
          } else {
            received[i] = true;
          }
        }
      }
      receive_iters++;
    } while (!all_received && receive_iters < max_it);
    PARTHENON_REQUIRE(
        receive_iters < max_it,
        "Too many iterations waiting to receive boundary communication buffers.");

    for (int i = 0; i < num_partitions; i++) {
      auto &md = mesh_data.GetOrAdd("base", i);
      // unpack FillGhost variables
      SetBoundaries(md);
    }

    //  Now do prolongation, compute primitives, apply BCs
    for (int i = 0; i < num_partitions; i++) {
      auto &md = mesh_data.GetOrAdd("base", i);
      if (multilevel) {
        ApplyBoundaryConditionsOnCoarseOrFineMD(md, true);
        ProlongateBoundaries(md);
      }
      ApplyBoundaryConditionsOnCoarseOrFineMD(md, false);
      // Call MeshData based FillDerived functions
      Update::FillDerived(md.get());
    }

    for (int i = 0; i < nmb; ++i) {
      auto &mbd = block_list[i]->meshblock_data.Get();
      // Call MeshBlockData based FillDerived functions
      Update::FillDerived(mbd.get());
    }

    if (init_problem && adaptive) {
      for (int i = 0; i < nmb; ++i) {
        block_list[i]->pmr->CheckRefinementCondition();
      }
    }

    if (init_problem && adaptive) {
      init_done = false;
      // caching nbtotal the private variable my be updated in the following function
      const int nb_before_loadbalance = nbtotal;
      LoadBalancingAndAdaptiveMeshRefinement(pin, app_in);
      if (nbtotal == nb_before_loadbalance) {
        init_done = true;
      } else if (nbtotal < nb_before_loadbalance && Globals::my_rank == 0) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
                  << "The number of MeshBlocks decreased during AMR grid initialization."
                  << std::endl
                  << "Possibly the refinement criteria have a problem." << std::endl;
      }
      if (nbtotal > 2 * nb_initial && Globals::my_rank == 0) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
                  << "The number of MeshBlocks increased more than twice during "
                     "initialization."
                  << std::endl
                  << "More computing power than you expected may be required."
                  << std::endl;
      }
    }
  } while (!init_done);

  // Initialize the "base" MeshData object
  mesh_data.Get()->Set(block_list, this);
}

/// Finds location of a block with ID `tgid`.
std::shared_ptr<MeshBlock> Mesh::FindMeshBlock(int tgid) const {
  // Attempt to simply index into the block list.
  const int nbs = block_list[0]->gid;
  const int i = tgid - nbs;
  PARTHENON_DEBUG_REQUIRE(0 <= i && i < block_list.size(),
                          "MeshBlock local index out of bounds.");
  PARTHENON_DEBUG_REQUIRE(block_list[i]->gid == tgid, "MeshBlock not found!");
  return block_list[i];
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc,
//                 RegionSize &block_size, BundaryFlag *block_bcs)
// \brief Set the physical part of a block_size structure and block boundary conditions

bool Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                     BoundaryFlag *block_bcs) {
  bool valid_region = true;
  block_size = GetBlockSize(loc);
  for (auto &dir : {X1DIR, X2DIR, X3DIR}) {
    if (!block_size.symmetry(dir)) {
      std::int64_t nrbx_ll = nrbx[dir - 1] << (loc.level() - root_level);
      if (loc.level() < root_level) {
        std::int64_t fac = 1 << (root_level - loc.level());
        nrbx_ll = nrbx[dir - 1] / fac + (nrbx[dir - 1] % fac != 0);
      }
      block_bcs[GetInnerBoundaryFace(dir)] =
          loc.l(dir - 1) == 0 ? mesh_bcs[GetInnerBoundaryFace(dir)] : BoundaryFlag::block;
      block_bcs[GetOuterBoundaryFace(dir)] = loc.l(dir - 1) == nrbx_ll - 1
                                                 ? mesh_bcs[GetOuterBoundaryFace(dir)]
                                                 : BoundaryFlag::block;
    } else {
      block_bcs[GetInnerBoundaryFace(dir)] = mesh_bcs[GetInnerBoundaryFace(dir)];
      block_bcs[GetOuterBoundaryFace(dir)] = mesh_bcs[GetOuterBoundaryFace(dir)];
    }
  }
  return valid_region;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::GetBlockSize(const LogicalLocation &loc) const
// \brief Find the (hyper-)rectangular region of the grid covered by the block at
//        logical location loc

RegionSize Mesh::GetBlockSize(const LogicalLocation &loc) const {
  RegionSize block_size = GetBlockSize();
  for (auto &dir : {X1DIR, X2DIR, X3DIR}) {
    block_size.xrat(dir) = mesh_size.xrat(dir);
    block_size.symmetry(dir) = mesh_size.symmetry(dir);
    if (!block_size.symmetry(dir)) {
      std::int64_t nrbx_ll = nrbx[dir - 1] << (loc.level() - root_level);
      if (loc.level() < root_level) {
        std::int64_t fac = 1 << (root_level - loc.level());
        nrbx_ll = nrbx[dir - 1] / fac + (nrbx[dir - 1] % fac != 0);
      }
      block_size.xmin(dir) = GetMeshCoordinate(dir, BlockLocation::Left, loc);
      block_size.xmax(dir) = GetMeshCoordinate(dir, BlockLocation::Right, loc);
      // Correct for possible overshooting, since the root grid may not cover the
      // entire logical level zero block of the mesh
      if (block_size.xmax(dir) > mesh_size.xmax(dir) || loc.level() < 0) {
        // Need integer reduction factor, so transform location back to root level
        PARTHENON_REQUIRE(loc.level() < root_level, "Something is messed up.");
        std::int64_t loc_low = loc.l(dir - 1) << (root_level - loc.level());
        std::int64_t loc_hi = (loc.l(dir - 1) + 1) << (root_level - loc.level());
        block_size.nx(dir) =
            block_size.nx(dir) * (nrbx[dir - 1] - loc_low) / (loc_hi - loc_low);
        block_size.xmax(dir) = mesh_size.xmax(dir);
      }
    } else {
      block_size.xmin(dir) = mesh_size.xmin(dir);
      block_size.xmax(dir) = mesh_size.xmax(dir);
    }
  }
  return block_size;
}

std::int64_t Mesh::GetTotalCells() {
  auto &pmb = block_list.front();
  return static_cast<std::int64_t>(nbtotal) * pmb->block_size.nx(X1DIR) *
         pmb->block_size.nx(X2DIR) * pmb->block_size.nx(X3DIR);
}
// TODO(JMM): Move block_size into mesh.
int Mesh::GetNumberOfMeshBlockCells() const {
  return block_list.front()->GetNumberOfMeshBlockCells();
}
const RegionSize &Mesh::GetBlockSize() const { return base_block_size; }

const IndexShape &Mesh::GetLeafBlockCellBounds(CellLevel level) const {
  // TODO(JMM): Luke this is for your Metadata::fine stuff.
  PARTHENON_DEBUG_REQUIRE(level != CellLevel::fine,
                          "Currently no access to finer cellbounds");
  MeshBlock *pmb = block_list[0].get();
  if (level == CellLevel::same) {
    return pmb->cellbounds;
    // TODO(JMM):
    // } else if (level == CellLevel::fine) {
    //   return pmb->fine_cellbounds;
    // }
  } else { // if (level == CellLevel::coarse) {
    return pmb->c_cellbounds;
  }
}

// Functionality re-used in mesh constructor
void Mesh::RegisterLoadBalancing_(ParameterInput *pin) {
#ifdef MPI_PARALLEL // JMM: Not sure this ifdef is needed
  const std::string balancer =
      pin->GetOrAddString("parthenon/loadbalancing", "balancer", "default",
                          std::vector<std::string>{"default", "automatic", "manual"});
  if (balancer == "automatic") {
    // JMM: I am disabling timing based load balancing, as it's not
    // threaded through the infrastructure. I think some thought needs
    // to go into doing this right with loops over meshdata rather
    // than loops over data on a single meshblock.
    PARTHENON_FAIL("Timing based load balancing is currently unavailable.");
    lb_automatic_ = true;
  } else if (balancer == "manual") {
    lb_manual_ = true;
  }
  lb_tolerance_ = pin->GetOrAddReal("parthenon/loadbalancing", "tolerance", 0.5);
  lb_interval_ = pin->GetOrAddInteger("parthenon/loadbalancing", "interval", 10);
#endif // MPI_PARALLEL
}

// Create separate communicators for all variables. Needs to be done at the mesh
// level so that the communicators for each variable across all blocks is consistent.
// As variables are identical across all blocks, we just use the info from the first.
void Mesh::SetupMPIComms() {
#ifdef MPI_PARALLEL

  for (auto &pair : resolved_packages->AllFields()) {
    auto &metadata = pair.second;
    // Create both boundary and flux communicators for everything with either FillGhost
    // or WithFluxes just to be safe
    if (metadata.IsSet(Metadata::FillGhost) || metadata.IsSet(Metadata::WithFluxes) ||
        metadata.IsSet(Metadata::ForceRemeshComm) ||
        metadata.IsSet(Metadata::GMGProlongate) ||
        metadata.IsSet(Metadata::GMGRestrict)) {
      MPI_Comm mpi_comm;
      PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
      const auto ret = mpi_comm_map_.insert({pair.first.label(), mpi_comm});
      PARTHENON_REQUIRE_THROWS(ret.second, "Communicator with same name already in map");

      if (multilevel) {
        MPI_Comm mpi_comm_flcor;
        PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_flcor));
        const auto ret =
            mpi_comm_map_.insert({pair.first.label() + "_flcor", mpi_comm_flcor});
        PARTHENON_REQUIRE_THROWS(ret.second,
                                 "Flux corr. communicator with same name already in map");
      }
    }
  }
  for (auto &pair : resolved_packages->AllSwarms()) {
    MPI_Comm mpi_comm;
    PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
    const auto ret = mpi_comm_map_.insert({pair.first, mpi_comm});
    PARTHENON_REQUIRE_THROWS(ret.second, "Communicator with same name already in map");
  }
  // TODO(everying during a sync) we should discuss what to do with face vars as they
  // are currently not handled in pmb->meshblock_data.Get()->SetupPersistentMPI(); nor
  // inserted into pmb->pbval->bvars.
#endif
}

} // namespace parthenon
