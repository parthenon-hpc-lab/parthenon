//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include "bvals/cc/bvals_cc_in_one.hpp"
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
#include "mesh/refinement_cc_in_one.hpp"
#include "outputs/restart.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"
#include "utils/partition_stl_containers.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
           int mesh_test)
    : // public members:
      modified(true),
      // aggregate initialization of RegionSize struct:
      mesh_size{pin->GetReal("parthenon/mesh", "x1min"),
                pin->GetReal("parthenon/mesh", "x2min"),
                pin->GetReal("parthenon/mesh", "x3min"),
                pin->GetReal("parthenon/mesh", "x1max"),
                pin->GetReal("parthenon/mesh", "x2max"),
                pin->GetReal("parthenon/mesh", "x3max"),
                pin->GetOrAddReal("parthenon/mesh", "x1rat", 1.0),
                pin->GetOrAddReal("parthenon/mesh", "x2rat", 1.0),
                pin->GetOrAddReal("parthenon/mesh", "x3rat", 1.0),
                pin->GetInteger("parthenon/mesh", "nx1"),
                pin->GetInteger("parthenon/mesh", "nx2"),
                pin->GetInteger("parthenon/mesh", "nx3")},
      mesh_bcs{
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix3_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox3_bc", "reflecting"))},
      ndim((mesh_size.nx3 > 1) ? 3 : ((mesh_size.nx2 > 1) ? 2 : 1)),
      adaptive(pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "adaptive"
                   ? true
                   : false),
      multilevel((adaptive ||
                  pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "static")
                     ? true
                     : false),
      nbnew(), nbdel(), step_since_lb(), gflag(), packages(packages),
      // private members:
      num_mesh_threads_(pin->GetOrAddInteger("parthenon/mesh", "num_threads", 1)),
      tree(this), use_uniform_meshgen_fn_{true, true, true, true}, lb_flag_(true),
      lb_automatic_(),
      lb_manual_(), MeshGenerator_{nullptr, UniformMeshGeneratorX1,
                                   UniformMeshGeneratorX2, UniformMeshGeneratorX3},
      MeshBndryFnctn{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr} {
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

  // check number of grid cells in root level of mesh from input file.
  if (mesh_size.nx1 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx1 must be >= 1, but nx1=" << mesh_size.nx1
        << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (mesh_size.nx2 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2=" << mesh_size.nx2
        << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (mesh_size.nx3 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3=" << mesh_size.nx3
        << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (mesh_size.nx2 == 1 && mesh_size.nx3 > 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_size.nx3
        << ", 2D problems in x1-x3 plane not supported" << std::endl;
    PARTHENON_FAIL(msg);
  }

  // check physical size of mesh (root level) from input file.
  if (mesh_size.x1max <= mesh_size.x1min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min
        << " x1max=" << mesh_size.x1max << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (mesh_size.x2max <= mesh_size.x2min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min
        << " x2max=" << mesh_size.x2max << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (mesh_size.x3max <= mesh_size.x3min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min
        << " x3max=" << mesh_size.x3max << std::endl;
    PARTHENON_FAIL(msg);
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

  // check the consistency of the periodic boundaries
  if (((mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic &&
        mesh_bcs[BoundaryFace::outer_x1] != BoundaryFlag::periodic) ||
       (mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic &&
        mesh_bcs[BoundaryFace::outer_x1] == BoundaryFlag::periodic)) ||
      (mesh_size.nx2 > 1 &&
       ((mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x2] != BoundaryFlag::periodic) ||
        (mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic &&
         mesh_bcs[BoundaryFace::outer_x2] == BoundaryFlag::periodic))) ||
      (mesh_size.nx3 > 1 &&
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

  // read and set MeshBlock parameters
  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;
  block_size.nx1 = pin->GetOrAddInteger("parthenon/meshblock", "nx1", mesh_size.nx1);
  if (ndim >= 2)
    block_size.nx2 = pin->GetOrAddInteger("parthenon/meshblock", "nx2", mesh_size.nx2);
  else
    block_size.nx2 = mesh_size.nx2;
  if (ndim >= 3)
    block_size.nx3 = pin->GetOrAddInteger("parthenon/meshblock", "nx3", mesh_size.nx3);
  else
    block_size.nx3 = mesh_size.nx3;

  // check consistency of the block and mesh
  if (mesh_size.nx1 % block_size.nx1 != 0 || mesh_size.nx2 % block_size.nx2 != 0 ||
      mesh_size.nx3 % block_size.nx3 != 0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "the Mesh must be evenly divisible by the MeshBlock" << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (block_size.nx1 < 4 || (block_size.nx2 < 4 && (ndim >= 2)) ||
      (block_size.nx3 < 4 && (ndim >= 3))) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "block_size must be larger than or equal to 4 cells." << std::endl;
    PARTHENON_FAIL(msg);
  }

  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1 / block_size.nx1;
  nrbx2 = mesh_size.nx2 / block_size.nx2;
  nrbx3 = mesh_size.nx3 / block_size.nx3;
  nbmax = (nrbx1 > nrbx2) ? nrbx1 : nrbx2;
  nbmax = (nbmax > nrbx3) ? nbmax : nrbx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0) {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR] = DefaultMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0) {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR] = DefaultMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0) {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR] = DefaultMeshGeneratorX3;
  }
  default_pack_size_ = pin->GetOrAddInteger("parthenon/mesh", "pack_size", -1);

  // calculate the logical root level and maximum level
  for (root_level = 0; (1 << root_level) < nbmax; root_level++) {
  }
  current_level = root_level;

  tree.CreateRootGrid();

  // Load balancing flag and parameters
  if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing", "tolerance", 0.5);
  lb_interval_ = pin->GetOrAddInteger("loadbalancing", "interval", 10);

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
    if (block_size.nx1 % 2 == 1 || (block_size.nx2 % 2 == 1 && (ndim >= 2)) ||
        (block_size.nx3 % 2 == 1 && (ndim >= 3))) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The size of MeshBlock must be divisible by 2 in order to use SMR or AMR."
          << std::endl;
      PARTHENON_FAIL(msg);
    }

    InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 27, "parthenon/static_refinement") == 0) {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(pib->block_name, "x1min");
        ref_size.x1max = pin->GetReal(pib->block_name, "x1max");
        if (ndim >= 2) {
          ref_size.x2min = pin->GetReal(pib->block_name, "x2min");
          ref_size.x2max = pin->GetReal(pib->block_name, "x2max");
        } else {
          ref_size.x2min = mesh_size.x2min;
          ref_size.x2max = mesh_size.x2max;
        }
        if (ndim == 3) {
          ref_size.x3min = pin->GetReal(pib->block_name, "x3min");
          ref_size.x3max = pin->GetReal(pib->block_name, "x3max");
        } else {
          ref_size.x3min = mesh_size.x3min;
          ref_size.x3max = mesh_size.x3max;
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
        if (ref_size.x1min > ref_size.x1max || ref_size.x2min > ref_size.x2max ||
            ref_size.x3min > ref_size.x3max) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Invalid refinement region is specified." << std::endl;
          PARTHENON_FAIL(msg);
        }
        if (ref_size.x1min < mesh_size.x1min || ref_size.x1max > mesh_size.x1max ||
            ref_size.x2min < mesh_size.x2min || ref_size.x2max > mesh_size.x2max ||
            ref_size.x3min < mesh_size.x3min || ref_size.x3max > mesh_size.x3max) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement region must be smaller than the whole mesh." << std::endl;
          PARTHENON_FAIL(msg);
        }
        // find the logical range in the ref_level
        // note: if this is too slow, this should be replaced with bi-section search.
        std::int64_t lx1min = 0, lx1max = 0, lx2min = 0, lx2max = 0, lx3min = 0,
                     lx3max = 0;
        std::int64_t lxmax = nrbx1 * (1LL << ref_lev);
        for (lx1min = 0; lx1min < lxmax; lx1min++) {
          Real rx =
              ComputeMeshGeneratorX(lx1min + 1, lxmax, use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) > ref_size.x1min) break;
        }
        for (lx1max = lx1min; lx1max < lxmax; lx1max++) {
          Real rx =
              ComputeMeshGeneratorX(lx1max + 1, lxmax, use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) >= ref_size.x1max) break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;
        if (ndim >= 2) { // 2D or 3D
          lxmax = nrbx2 * (1LL << ref_lev);
          for (lx2min = 0; lx2min < lxmax; lx2min++) {
            Real rx =
                ComputeMeshGeneratorX(lx2min + 1, lxmax, use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) > ref_size.x2min) break;
          }
          for (lx2max = lx2min; lx2max < lxmax; lx2max++) {
            Real rx =
                ComputeMeshGeneratorX(lx2max + 1, lxmax, use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) >= ref_size.x2max) break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }
        if (ndim == 3) { // 3D
          lxmax = nrbx3 * (1LL << ref_lev);
          for (lx3min = 0; lx3min < lxmax; lx3min++) {
            Real rx =
                ComputeMeshGeneratorX(lx3min + 1, lxmax, use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) > ref_size.x3min) break;
          }
          for (lx3max = lx3min; lx3max < lxmax; lx3max++) {
            Real rx =
                ComputeMeshGeneratorX(lx3max + 1, lxmax, use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) >= ref_size.x3max) break;
          }
          if (lx3min % 2 == 1) lx3min--;
          if (lx3max % 2 == 0) lx3max++;
        }
        // create the finest level
        if (ndim == 1) {
          for (std::int64_t i = lx1min; i < lx1max; i += 2) {
            LogicalLocation nloc;
            nloc.level = lrlev, nloc.lx1 = i, nloc.lx2 = 0, nloc.lx3 = 0;
            int nnew;
            tree.AddMeshBlock(nloc, nnew);
          }
        }
        if (ndim == 2) {
          for (std::int64_t j = lx2min; j < lx2max; j += 2) {
            for (std::int64_t i = lx1min; i < lx1max; i += 2) {
              LogicalLocation nloc;
              nloc.level = lrlev, nloc.lx1 = i, nloc.lx2 = j, nloc.lx3 = 0;
              int nnew;
              tree.AddMeshBlock(nloc, nnew);
            }
          }
        }
        if (ndim == 3) {
          for (std::int64_t k = lx3min; k < lx3max; k += 2) {
            for (std::int64_t j = lx2min; j < lx2max; j += 2) {
              for (std::int64_t i = lx1min; i < lx1max; i += 2) {
                LogicalLocation nloc;
                nloc.level = lrlev, nloc.lx1 = i, nloc.lx2 = j, nloc.lx3 = k;
                int nnew;
                tree.AddMeshBlock(nloc, nnew);
              }
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

  // Output some diagnostic information to terminal

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0) {
    if (Globals::my_rank == 0) OutputMeshStructure(ndim);
    return;
  }

  mesh_data.SetMeshPointer(this);

  resolved_packages = ResolvePackages(packages);

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
    block_list[i - nbs]->SearchAndSetNeighbors(tree, ranklist.data(), nslist.data());
  }

  ResetLoadBalanceVariables();

  // Output variables in use in this run
  if (Globals::my_rank == 0) {
    std::cout << "#Variables in use:\n" << *(resolved_packages) << std::endl;
  }
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file
Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, RestartReader &rr,
           Packages_t &packages, int mesh_test)
    : // public members:
      // aggregate initialization of RegionSize struct:
      // (will be overwritten by memcpy from restart file, in this case)
      modified(true),
      // aggregate initialization of RegionSize struct:
      mesh_size{pin->GetReal("parthenon/mesh", "x1min"),
                pin->GetReal("parthenon/mesh", "x2min"),
                pin->GetReal("parthenon/mesh", "x3min"),
                pin->GetReal("parthenon/mesh", "x1max"),
                pin->GetReal("parthenon/mesh", "x2max"),
                pin->GetReal("parthenon/mesh", "x3max"),
                pin->GetOrAddReal("parthenon/mesh", "x1rat", 1.0),
                pin->GetOrAddReal("parthenon/mesh", "x2rat", 1.0),
                pin->GetOrAddReal("parthenon/mesh", "x3rat", 1.0),
                pin->GetInteger("parthenon/mesh", "nx1"),
                pin->GetInteger("parthenon/mesh", "nx2"),
                pin->GetInteger("parthenon/mesh", "nx3")},
      mesh_bcs{
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox1_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox2_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix3_bc", "reflecting")),
          GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox3_bc", "reflecting"))},
      ndim((mesh_size.nx3 > 1) ? 3 : ((mesh_size.nx2 > 1) ? 2 : 1)),
      adaptive(pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "adaptive"
                   ? true
                   : false),
      multilevel((adaptive ||
                  pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "static")
                     ? true
                     : false),
      nbnew(), nbdel(), step_since_lb(), gflag(), packages(packages),
      // private members:
      num_mesh_threads_(pin->GetOrAddInteger("parthenon/mesh", "num_threads", 1)),
      tree(this), use_uniform_meshgen_fn_{true, true, true, true}, lb_flag_(true),
      lb_automatic_(),
      lb_manual_(), MeshGenerator_{nullptr, UniformMeshGeneratorX1,
                                   UniformMeshGeneratorX2, UniformMeshGeneratorX3},
      MeshBndryFnctn{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr} {
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
  mesh_size.x1min = grid_dim[0];
  mesh_size.x1max = grid_dim[1];
  mesh_size.x1rat = grid_dim[2];

  mesh_size.x2min = grid_dim[3];
  mesh_size.x2max = grid_dim[4];
  mesh_size.x2rat = grid_dim[5];

  mesh_size.x3min = grid_dim[6];
  mesh_size.x3max = grid_dim[7];
  mesh_size.x3rat = grid_dim[8];

  // initialize
  loclist = std::vector<LogicalLocation>(nbtotal);

  const auto blockSize = rr.GetAttrVec<int>("Info", "MeshBlockSize");
  block_size.nx1 = blockSize[0];
  block_size.nx2 = blockSize[1];
  block_size.nx3 = blockSize[2];

  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1 / block_size.nx1;
  nrbx2 = mesh_size.nx2 / block_size.nx2;
  nrbx3 = mesh_size.nx3 / block_size.nx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0) {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR] = DefaultMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0) {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR] = DefaultMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0) {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR] = DefaultMeshGeneratorX3;
  }
  default_pack_size_ = pin->GetOrAddInteger("parthenon/mesh", "pack_size", -1);

  // Load balancing flag and parameters
#ifdef MPI_PARALLEL
  if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing", "tolerance", 0.5);
  lb_interval_ = pin->GetOrAddInteger("loadbalancing", "interval", 10);
#endif

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
    loclist[i].lx1 = lx123[3 * i];
    loclist[i].lx2 = lx123[3 * i + 1];
    loclist[i].lx3 = lx123[3 * i + 2];

    loclist[i].level = locLevelGidLidCnghostGflag[5 * i];
    if (loclist[i].level > current_level) {
      current_level = loclist[i].level;
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
    block_list[i - nbs]->SearchAndSetNeighbors(tree, ranklist.data(), nslist.data());
  }

  ResetLoadBalanceVariables();

  // Output variables in use in this run
  if (Globals::my_rank == 0) {
    std::cout << "#Variables in use:\n" << *(resolved_packages) << std::endl;
  }
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
  std::cout << "Root grid = " << nrbx1 << " x " << nrbx2 << " x " << nrbx3
            << " MeshBlocks" << std::endl;
  std::cout << "Total number of MeshBlocks = " << nbtotal << std::endl;
  std::cout << "Number of physical refinement levels = " << (current_level - root_level)
            << std::endl;
  std::cout << "Number of logical  refinement levels = " << current_level << std::endl;

  // compute/output number of blocks per level, and cost per level
  std::vector<int> nb_per_plevel(max_level + 1, 0);
  std::vector<int> cost_per_plevel(max_level + 1, 0);

  for (int i = 0; i < nbtotal; i++) {
    nb_per_plevel[(loclist[i].level - root_level)]++;
    cost_per_plevel[(loclist[i].level - root_level)] += costlist[i];
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
      if (loclist[j].level == i) {
        SetBlockSizeAndBoundaries(loclist[j], block_size, block_bcs);
        std::int64_t &lx1 = loclist[j].lx1;
        std::int64_t &lx2 = loclist[j].lx2;
        std::int64_t &lx3 = loclist[j].lx3;
        int &ll = loclist[j].level;
        mincost = std::min(mincost, costlist[i]);
        maxcost = std::max(maxcost, costlist[i]);
        totalcost += costlist[i];
        std::fprintf(fp, "#MeshBlock %d on rank=%d with cost=%g\n", j, ranklist[j],
                     costlist[j]);
        std::fprintf(
            fp, "#  Logical level %d, location = (%" PRId64 " %" PRId64 " %" PRId64 ")\n",
            ll, lx1, lx2, lx3);
        if (ndim == 2) {
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "\n\n");
        }
        if (ndim == 3) {
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
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
//! \fn void Mesh::EnrollUserMeshGenerator(CoordinateDirection,MeshGenFunc my_mg)
//  \brief Enroll a user-defined function for Mesh generation

void Mesh::EnrollUserMeshGenerator(CoordinateDirection dir, MeshGenFunc my_mg) {
  std::stringstream msg;
  if (dir < 0 || dir >= 3) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (dir == X1DIR && mesh_size.x1rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x1rat = " << mesh_size.x1rat
        << " must be negative for user-defined mesh generator in X1DIR " << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (dir == X2DIR && mesh_size.x2rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x2rat = " << mesh_size.x2rat
        << " must be negative for user-defined mesh generator in X2DIR " << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (dir == X3DIR && mesh_size.x3rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x3rat = " << mesh_size.x3rat
        << " must be negative for user-defined mesh generator in X3DIR " << std::endl;
    PARTHENON_FAIL(msg);
  }
  use_uniform_meshgen_fn_[dir] = false;
  MeshGenerator_[dir] = my_mg;
  return;
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
  Kokkos::Profiling::pushRegion("Mesh::Initialize");
  bool init_done = true;
  const int nb_initial = nbtotal;
  do {
    int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);

    // init meshblock data
    for (int i = 0; i < nmb; ++i) {
      MeshBlock *pmb = block_list[i].get();
      pmb->InitMeshBlockUserData(pmb, pin);
    }

    // problem generator
    if (init_problem) {
      for (int i = 0; i < nmb; ++i) {
        auto &pmb = block_list[i];
        pmb->ProblemGenerator(pmb.get(), pin);
      }
    }

    // Create send/recv MPI_Requests for all BoundaryData objects
    for (int i = 0; i < nmb; ++i) {
      auto &pmb = block_list[i];
      // TODO(mpi people) do we still need the pbval part? Discuss also in the context of
      // other than cellvariables, see comment above on communicators.
      // BoundaryVariable objects evolved in main TimeIntegratorTaskList:
      // pmb->pbval->SetupPersistentMPI();
      pmb->meshblock_data.Get()->SetupPersistentMPI();
      pmb->swarm_data.Get()->SetupPersistentMPI();
    }

    // prepare to receive conserved variables
    for (int i = 0; i < nmb; ++i) {
      block_list[i]->meshblock_data.Get()->StartReceiving(BoundaryCommSubset::mesh_init);
    }

    const int num_partitions = DefaultNumPartitions();

    // send FillGhost variables
    for (int i = 0; i < num_partitions; i++) {
      auto &md = mesh_data.GetOrAdd("base", i);
      cell_centered_bvars::SendBoundaryBuffers(md);
    }

    // wait to receive FillGhost variables
    // TODO(someone) evaluate if ReceiveWithWait kind of logic is better, also related to
    // https://github.com/lanl/parthenon/issues/418
    bool all_received = true;
    do {
      all_received = true;
      for (int i = 0; i < num_partitions; i++) {
        auto &md = mesh_data.GetOrAdd("base", i);
        if (cell_centered_bvars::ReceiveBoundaryBuffers(md) != TaskStatus::complete) {
          all_received = false;
        }
      }
    } while (!all_received);

    // unpack FillGhost variables
    for (int i = 0; i < num_partitions; i++) {
      auto &md = mesh_data.GetOrAdd("base", i);
      cell_centered_bvars::SetBoundaries(md);
      if (multilevel) {
        cell_centered_refinement::RestrictPhysicalBounds(md.get());
      }
    }

    for (int i = 0; i < nmb; ++i) {
      block_list[i]->meshblock_data.Get()->ClearBoundary(BoundaryCommSubset::mesh_init);
    }
    // Now do prolongation, compute primitives, apply BCs
    for (int i = 0; i < nmb; ++i) {
      auto &mbd = block_list[i]->meshblock_data.Get();
      if (multilevel) {
        ProlongateBoundaries(mbd);
      }
      ApplyBoundaryConditions(mbd);
      // Call MeshBlockData based FillDerived functions
      Update::FillDerived(mbd.get());
    }
    for (int i = 0; i < num_partitions; i++) {
      auto &md = mesh_data.GetOrAdd("base", i);
      // Call MeshData based FillDerived functions
      Update::FillDerived(md.get());
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

  Kokkos::Profiling::popRegion(); // Mesh::Initialize
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

void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                     BoundaryFlag *block_bcs) {
  std::int64_t &lx1 = loc.lx1;
  int &ll = loc.level;
  std::int64_t nrbx_ll = nrbx1 << (ll - root_level);

  // calculate physical block size, x1
  if (lx1 == 0) {
    block_size.x1min = mesh_size.x1min;
    block_bcs[BoundaryFace::inner_x1] = mesh_bcs[BoundaryFace::inner_x1];
  } else {
    Real rx = ComputeMeshGeneratorX(lx1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1min = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  }
  if (lx1 == nrbx_ll - 1) {
    block_size.x1max = mesh_size.x1max;
    block_bcs[BoundaryFace::outer_x1] = mesh_bcs[BoundaryFace::outer_x1];
  } else {
    Real rx = ComputeMeshGeneratorX(lx1 + 1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1max = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  }

  // calculate physical block size, x2
  if (mesh_size.nx2 == 1) {
    block_size.x2min = mesh_size.x2min;
    block_size.x2max = mesh_size.x2max;
    block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
  } else {
    std::int64_t &lx2 = loc.lx2;
    nrbx_ll = nrbx2 << (ll - root_level);
    if (lx2 == 0) {
      block_size.x2min = mesh_size.x2min;
      block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    } else {
      Real rx = ComputeMeshGeneratorX(lx2, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2min = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    }
    if (lx2 == (nrbx_ll)-1) {
      block_size.x2max = mesh_size.x2max;
      block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
    } else {
      Real rx = ComputeMeshGeneratorX(lx2 + 1, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2max = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    }
  }

  // calculate physical block size, x3
  if (mesh_size.nx3 == 1) {
    block_size.x3min = mesh_size.x3min;
    block_size.x3max = mesh_size.x3max;
    block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
  } else {
    std::int64_t &lx3 = loc.lx3;
    nrbx_ll = nrbx3 << (ll - root_level);
    if (lx3 == 0) {
      block_size.x3min = mesh_size.x3min;
      block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    } else {
      Real rx = ComputeMeshGeneratorX(lx3, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3min = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    }
    if (lx3 == (nrbx_ll)-1) {
      block_size.x3max = mesh_size.x3max;
      block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
    } else {
      Real rx = ComputeMeshGeneratorX(lx3 + 1, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3max = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    }
  }

  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;
}

std::int64_t Mesh::GetTotalCells() {
  auto &pmb = block_list.front();
  return static_cast<std::int64_t>(nbtotal) * pmb->block_size.nx1 * pmb->block_size.nx2 *
         pmb->block_size.nx3;
}
// TODO(JMM): Move block_size into mesh.
int Mesh::GetNumberOfMeshBlockCells() const {
  return block_list.front()->GetNumberOfMeshBlockCells();
}
const RegionSize &Mesh::GetBlockSize() const { return block_list.front()->block_size; }

// Create separate communicators for all variables. Needs to be done at the mesh
// level so that the communicators for each variable across all blocks is consistent.
// As variables are identical across all blocks, we just use the info from the first.
void Mesh::SetupMPIComms() {
#ifdef MPI_PARALLEL

  for (auto &pair : resolved_packages->AllFields()) {
    auto &metadata = pair.second;
    if (metadata.IsSet(Metadata::FillGhost)) {
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
