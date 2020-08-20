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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "parthenon_mpi.hpp"

#include "bvals/boundary_conditions.hpp"
#include "bvals/bvals.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_tree.hpp"
#include "outputs/io_wrapper.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, Properties_t &properties,
           Packages_t &packages,
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
      nbnew(), nbdel(), step_since_lb(), gflag(), properties(properties),
      packages(packages),
      // private members:
      next_phys_id_(),
      num_mesh_threads_(pin->GetOrAddInteger("parthenon/mesh", "num_threads", 1)),
      tree(this), use_uniform_meshgen_fn_{true, true, true, true}, lb_flag_(true),
      lb_automatic_(),
      lb_manual_(), MeshGenerator_{nullptr, UniformMeshGeneratorX1,
                                   UniformMeshGeneratorX2, UniformMeshGeneratorX3},
      BoundaryFunction_{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}, AMRFlag_{},
      UserSourceTerm_{}, UserTimeStep_{} {
  std::stringstream msg;
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  std::int64_t nbmax;

  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

#ifdef MPI_PARALLEL
  // reserve phys=0 for former TAG_AMR=8; now hard-coded in Mesh::CreateAMRMPITag()
  next_phys_id_ = 1;
  ReserveMeshBlockPhysIDs();
#endif

  // check number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads=" << num_mesh_threads_
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  // check number of grid cells in root level of mesh from input file.
  if (mesh_size.nx1 < 4) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1=" << mesh_size.nx1
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
  if (app_in->MeshUserWorkInLoop != nullptr) {
    UserWorkInLoop = app_in->MeshUserWorkInLoop;
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

  // calculate the logical root level and maximum level
  for (root_level = 0; (1 << root_level) < nbmax; root_level++) {
  }
  current_level = root_level;

  tree.CreateRootGrid();

  // Load balancing flag and parameters
#ifdef MPI_PARALLEL
  if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing", "tolerance", 0.5);
  lb_interval_ = pin->GetOrAddReal("loadbalancing", "interval", 10);
#endif

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

  InitUserMeshData(pin);

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

  // create MeshBlock list for this process
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;
  // create MeshBlock list for this process
  for (int i = nbs; i <= nbe; i++) {
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    block_list.emplace_back(i, i - nbs, loclist[i], block_size, block_bcs, this, pin,
                            properties, packages, gflag);
    block_list.back().pbval->SearchAndSetNeighbors(tree, ranklist.data(), nslist.data());
  }

  ResetLoadBalanceVariables();
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file
#if 0
Mesh::Mesh(ParameterInput *pin, IOWrapper &resfile, Properties_t &properties,
           Packages_t &packages, int mesh_test)
    : // public members:
      // aggregate initialization of RegionSize struct:
      // (will be overwritten by memcpy from restart file, in this case)
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
      mesh_bcs{GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix1_bc", "none")),
               GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox1_bc", "none")),
               GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix2_bc", "none")),
               GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox2_bc", "none")),
               GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ix3_bc", "none")),
               GetBoundaryFlag(pin->GetOrAddString("parthenon/mesh", "ox3_bc", "none"))},
      ndim((mesh_size.nx3 > 1) ? 3 : ((mesh_size.nx2 > 1) ? 2 : 1)),
      adaptive(pin->GetOrAddString("parthenon/mesh", "refinement", "none")
                                                        == "adaptive" ? true : false),
      multilevel(
          (adaptive ||
           pin->GetOrAddString("parthenon/mesh", "refinement", "none") == "static")
              ? true
              : false),
      start_time(pin->GetOrAddReal("parthenon/time", "start_time", 0.0)),
      time(start_time),
      tlim(pin->GetReal("parthenon/time", "tlim")), dt(std::numeric_limits<Real>::max()),
      dt_hyperbolic(dt), dt_parabolic(dt), dt_user(dt),
      nlim(pin->GetOrAddInteger("parthenon/time", "nlim", -1)), ncycle(),
      ncycle_out(pin->GetOrAddInteger("parthenon/time", "ncycle_out", 1)),
      dt_diagnostics(pin->GetOrAddInteger("parthenon/time", "dt_diagnostics", -1)),
      nbnew(),
      nbdel(), step_since_lb(), gflag(), block_list(nullptr), properties(properties),
      packages(packages),
      // private members:
      next_phys_id_(),
      num_mesh_threads_(pin->GetOrAddInteger("parthenon/mesh", "num_threads", 1)),
      tree(this), use_uniform_meshgen_fn_{true, true, true}, nreal_user_mesh_data_(),
      nint_user_mesh_data_(), lb_flag_(true), lb_automatic_(),
      lb_manual_(), MeshGenerator_{UniformMeshGeneratorX1, UniformMeshGeneratorX2,
                                   UniformMeshGeneratorX3},
      BoundaryFunction_{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}, AMRFlag_{},
      UserSourceTerm_{}, UserTimeStep_{}, FieldDiffusivity_{} {
  std::stringstream msg;
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  IOWrapperSizeT *offset{};
  IOWrapperSizeT datasize, listsize, headeroffset;

  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

#ifdef MPI_PARALLEL
  // reserve phys=0 for former TAG_AMR=8; now hard-coded in Mesh::CreateAMRMPITag()
  next_phys_id_ = 1;
  ReserveMeshBlockPhysIDs();
#endif

  // check the number of OpenMP threads for mesh
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads=" << num_mesh_threads_
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  // get the end of the header
  headeroffset = resfile.GetPosition();
  // read the restart file
  // the file is already open and the pointer is set to after <par_end>
  IOWrapperSizeT headersize =
      sizeof(int) * 3 + sizeof(Real) * 2 + sizeof(RegionSize) + sizeof(IOWrapperSizeT);
  char *headerdata = new char[headersize];
  if (Globals::my_rank == 0) { // the master process reads the header data
    if (resfile.Read(headerdata, 1, headersize) != headersize) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken." << std::endl;
      PARTHENON_FAIL(msg);
    }
  }
#ifdef MPI_PARALLEL
  // then broadcast the header data
  MPI_Bcast(headerdata, headersize, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  IOWrapperSizeT hdos = 0;
  std::memcpy(&nbtotal, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&root_level, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  current_level = root_level;
  std::memcpy(&mesh_size, &(headerdata[hdos]), sizeof(RegionSize));
  hdos += sizeof(RegionSize);
  std::memcpy(&time, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&dt, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&ncycle, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&datasize, &(headerdata[hdos]), sizeof(IOWrapperSizeT));
  hdos += sizeof(IOWrapperSizeT); // (this updated value is never used)

  delete[] headerdata;

  // initialize
  loclist = std::vector<LogicalLocation>(nbtotal);
  offset = new IOWrapperSizeT[nbtotal];
  costlist = std::vector<double>(nbtotal);
  ranklist = std::vector<int>(nbtotal);
  nslist = std::vector<int>(Globals::nranks);
  nblist = std::vector<int>(Globals::nranks);

  block_size.nx1 = pin->GetOrAddInteger("parthenon/meshblock", "nx1", mesh_size.nx1);
  block_size.nx2 = pin->GetOrAddInteger("parthenon/meshblock", "nx2", mesh_size.nx2);
  block_size.nx3 = pin->GetOrAddInteger("parthenon/meshblock", "nx3", mesh_size.nx3);

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

  // Load balancing flag and parameters
#ifdef MPI_PARALLEL
  if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "automatic")
    lb_automatic_ = true;
  else if (pin->GetOrAddString("loadbalancing", "balancer", "default") == "manual")
    lb_manual_ = true;
  lb_tolerance_ = pin->GetOrAddReal("loadbalancing", "tolerance", 0.5);
  lb_interval_ = pin->GetOrAddReal("loadbalancing", "interval", 10);
#endif

  // SMR / AMR
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

  InitUserMeshData(pin);

  // read user Mesh data
  IOWrapperSizeT udsize = 0;
  for (int n = 0; n < nint_user_mesh_data_; n++)
    udsize += iuser_mesh_data[n].GetSizeInBytes();
  for (int n = 0; n < nreal_user_mesh_data_; n++)
    udsize += ruser_mesh_data[n].GetSizeInBytes();
  if (udsize != 0) {
    char *userdata = new char[udsize];
    if (Globals::my_rank == 0) { // only the master process reads the ID list
      if (resfile.Read(userdata, 1, udsize) != udsize) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "The restart file is broken." << std::endl;
        PARTHENON_FAIL(msg);
      }
    }
#ifdef MPI_PARALLEL
    // then broadcast the ID list
    MPI_Bcast(userdata, udsize, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

    IOWrapperSizeT udoffset = 0;
    for (int n = 0; n < nint_user_mesh_data_; n++) {
      std::memcpy(iuser_mesh_data[n].data(), &(userdata[udoffset]),
                  iuser_mesh_data[n].GetSizeInBytes());
      udoffset += iuser_mesh_data[n].GetSizeInBytes();
    }
    for (int n = 0; n < nreal_user_mesh_data_; n++) {
      std::memcpy(ruser_mesh_data[n].data(), &(userdata[udoffset]),
                  ruser_mesh_data[n].GetSizeInBytes());
      udoffset += ruser_mesh_data[n].GetSizeInBytes();
    }
    delete[] userdata;
  }

  // read the ID list
  listsize = sizeof(LogicalLocation) + sizeof(Real);
  // allocate the idlist buffer
  char *idlist = new char[listsize * nbtotal];
  if (Globals::my_rank == 0) { // only the master process reads the ID list
    if (resfile.Read(idlist, listsize, nbtotal) != static_cast<unsigned int>(nbtotal)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken." << std::endl;
      PARTHENON_FAIL(msg);
    }
  }
#ifdef MPI_PARALLEL
  // then broadcast the ID list
  MPI_Bcast(idlist, listsize * nbtotal, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

  int os = 0;
  for (int i = 0; i < nbtotal; i++) {
    std::memcpy(&(loclist[i]), &(idlist[os]), sizeof(LogicalLocation));
    os += sizeof(LogicalLocation);
    std::memcpy(&(costlist[i]), &(idlist[os]), sizeof(double));
    os += sizeof(double);
    if (loclist[i].level > current_level) current_level = loclist[i].level;
  }
  delete[] idlist;

  // calculate the header offset and seek
  headeroffset += headersize + udsize + listsize * nbtotal;
  if (Globals::my_rank != 0) resfile.Seek(headeroffset);

  // rebuild the Block Tree
  tree.CreateRootGrid();
  for (int i = 0; i < nbtotal; i++)
    tree.AddMeshBlockWithoutRefine(loclist[i]);
  int nnb;
  // check the tree structure, and assign GID
  tree.GetMeshBlockList(loclist, nullptr, nnb);
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
      delete[] offset;
      return;
    }
  }
#endif

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
    delete[] offset;
    return;
  }

  // allocate data buffer
  int nb = nblist[Globals::my_rank];
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nb - 1;
  char *mbdata = new char[datasize * nb];
  // load MeshBlocks (parallel)
  if (resfile.Read_at_all(mbdata, datasize, nb, headeroffset + nbs * datasize) !=
      static_cast<unsigned int>(nb)) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    PARTHENON_FAIL(msg);
  }
  for (int i = nbs; i <= nbe; i++) {
    // Match fixed-width integer precision of IOWrapperSizeT datasize
    std::uint64_t buff_os = datasize * (i - nbs);
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    block_list.emplace_back(i, i - nbs, this, pin, properties, packages, loclist[i],
                             block_size, block_bcs, costlist[i], mbdata + buff_os, gflag);
    block_list.back().pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  delete[] mbdata;
  // check consistency
  if (datasize != block_list.front().GetBlockSizeInBytes()) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  ResetLoadBalanceVariables();

  // clean up
  delete[] offset;
}
#endif

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() = default;

//----------------------------------------------------------------------------------------
//! \fn void Mesh::OutputMeshStructure(int ndim)
//  \brief print the mesh structure information

void Mesh::OutputMeshStructure(int ndim) {
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  FILE *fp = nullptr;

  // open 'mesh_structure.dat' file
  if ((fp = std::fopen("mesh_structure.dat", "wb")) == nullptr) {
    std::cout << "### ERROR in function Mesh::OutputMeshStructure" << std::endl
              << "Cannot open mesh_structure.dat" << std::endl;
    return;
  }

  // Write overall Mesh structure to stdout and file
  std::cout << std::endl;
  std::cout << "Root grid = " << nrbx1 << " x " << nrbx2 << " x " << nrbx3
            << " MeshBlocks" << std::endl;
  std::cout << "Total number of MeshBlocks = " << nbtotal << std::endl;
  std::cout << "Number of physical refinement levels = " << (current_level - root_level)
            << std::endl;
  std::cout << "Number of logical  refinement levels = " << current_level << std::endl;

  // compute/output number of blocks per level, and cost per level
  int *nb_per_plevel = new int[max_level];
  int *cost_per_plevel = new int[max_level];
  for (int i = 0; i <= max_level; ++i) {
    nb_per_plevel[i] = 0;
    cost_per_plevel[i] = 0;
  }
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

  // compute/output number of blocks per rank, and cost per rank
  std::cout << "Number of parallel ranks = " << Globals::nranks << std::endl;
  int *nb_per_rank = new int[Globals::nranks];
  int *cost_per_rank = new int[Globals::nranks];
  for (int i = 0; i < Globals::nranks; ++i) {
    nb_per_rank[i] = 0;
    cost_per_rank[i] = 0;
  }
  for (int i = 0; i < nbtotal; i++) {
    nb_per_rank[ranklist[i]]++;
    cost_per_rank[ranklist[i]] += costlist[i];
  }
  for (int i = 0; i < Globals::nranks; ++i) {
    std::cout << "  Rank = " << i << ": " << nb_per_rank[i]
              << " MeshBlocks, cost = " << cost_per_rank[i] << std::endl;
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

  delete[] nb_per_plevel;
  delete[] cost_per_plevel;
  delete[] nb_per_rank;
  delete[] cost_per_rank;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValFunc my_bc)
//  \brief Enroll a user-defined boundary function

void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValFunc my_bc) {
  throw std::runtime_error("Mesh::EnrollUserBoundaryFunction is not implemented");
}

// DEPRECATED(felker): provide trivial overloads for old-style BoundaryFace enum argument
void Mesh::EnrollUserBoundaryFunction(int dir, BValFunc my_bc) {
  EnrollUserBoundaryFunction(static_cast<BoundaryFace>(dir), my_bc);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag)
//  \brief Enroll a user-defined function for checking refinement criteria

void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag) {
  if (adaptive) AMRFlag_ = amrflag;
  return;
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
//! \fn void Mesh::EnrollUserExplicitSourceFunction(SrcTermFunc my_func)
//  \brief Enroll a user-defined source function

void Mesh::EnrollUserExplicitSourceFunction(SrcTermFunc my_func) {
  UserSourceTerm_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func)
//  \brief Enroll a user-defined time step function

void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func) {
  UserTimeStep_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMetric(MetricFunc my_func)
//  \brief Enroll a user-defined metric for arbitrary GR coordinates

void Mesh::EnrollUserMetric(MetricFunc my_func) {
  UserMetric_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkBeforeOutput

void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin) {
  for (auto &mb : block_list) {
    mb.UserWorkBeforeOutput(pin);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::Initialize(int res_flag, ParameterInput *pin)
// \brief  initialization before the main loop

void Mesh::Initialize(int res_flag, ParameterInput *pin, ApplicationInput *app_in) {
  bool iflag = true;
  int inb = nbtotal;
#ifdef OPENMP_PARALLEL
  int nthreads = GetNumMeshThreads();
#endif
  int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<MeshBlock *> pmb_array;
  do {
    // initialize a vector of MeshBlock pointers
    nmb = GetNumMeshBlocksThisRank(Globals::my_rank);

    pmb_array.clear();
    pmb_array.reserve(nmb);

    for (auto &mb : block_list) {
      pmb_array.push_back(&mb);
    }

    if (res_flag == 0) {
#pragma omp parallel for num_threads(nthreads)
      for (int i = 0; i < nmb; ++i) {
        MeshBlock *pmb = pmb_array[i];
        pmb->ProblemGenerator(pmb, pin);
      }
    }

    int call = 0;
    // Create send/recv MPI_Requests for all BoundaryData objects
#pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nmb; ++i) {
      MeshBlock *pmb = pmb_array[i];
      // BoundaryVariable objects evolved in main TimeIntegratorTaskList:
      pmb->pbval->SetupPersistentMPI();
      pmb->real_containers.Get()->SetupPersistentMPI();
    }
    call++; // 1

#pragma omp parallel num_threads(nthreads)
    {
      // prepare to receive conserved variables
#pragma omp for
      for (int i = 0; i < nmb; ++i) {
        pmb_array[i]->real_containers.Get()->StartReceiving(
            BoundaryCommSubset::mesh_init);
      }
      call++; // 2
              // send conserved variables
#pragma omp for
      for (int i = 0; i < nmb; ++i) {
        pmb_array[i]->real_containers.Get()->SendBoundaryBuffers();
      }
      call++; // 3

      // wait to receive conserved variables
#pragma omp for
      for (int i = 0; i < nmb; ++i) {
        pmb_array[i]->real_containers.Get()->ReceiveAndSetBoundariesWithWait();
      }
      call++; // 4
#pragma omp for
      for (int i = 0; i < nmb; ++i) {
        pmb_array[i]->real_containers.Get()->ClearBoundary(BoundaryCommSubset::mesh_init);
      }
      call++;
      // Now do prolongation, compute primitives, apply BCs
#pragma omp for
      for (int i = 0; i < nmb; ++i) {
        auto &pmb = pmb_array[i];
        auto &pbval = pmb->pbval;
        if (multilevel) pbval->ProlongateBoundaries(0.0, 0.0);
        // TODO(JoshuaSBrown): Dead code left in for possible future extraction of
        // primitives
        //        int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je, kl =
        //        pmb->ks,
        //           ku = pmb->ke;
        //        if (pbval->nblevel[1][1][0] != -1) il -= NGHOST;
        //        if (pbval->nblevel[1][1][2] != -1) iu += NGHOST;
        //        if (pmb->block_size.nx2 > 1) {
        //          if (pbval->nblevel[1][0][1] != -1) jl -= NGHOST;
        //          if (pbval->nblevel[1][2][1] != -1) ju += NGHOST;
        //        }
        //        if (pmb->block_size.nx3 > 1) {
        //          if (pbval->nblevel[0][1][1] != -1) kl -= NGHOST;
        //          if (pbval->nblevel[2][1][1] != -1) ku += NGHOST;
        //        }

        ApplyBoundaryConditions(pmb->real_containers.Get());
        FillDerivedVariables::FillDerived(pmb->real_containers.Get());
      }

      if (!res_flag && adaptive) {
#pragma omp for
        for (int i = 0; i < nmb; ++i) {
          pmb_array[i]->pmr->CheckRefinementCondition();
        }
      }
    } // omp parallel

    if (!res_flag && adaptive) {
      iflag = false;
      int onb = nbtotal;
      LoadBalancingAndAdaptiveMeshRefinement(pin, app_in);
      if (nbtotal == onb) {
        iflag = true;
      } else if (nbtotal < onb && Globals::my_rank == 0) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
                  << "The number of MeshBlocks decreased during AMR grid initialization."
                  << std::endl
                  << "Possibly the refinement criteria have a problem." << std::endl;
      }
      if (nbtotal > 2 * inb && Globals::my_rank == 0) {
        std::cout
            << "### Warning in Mesh::Initialize" << std::endl
            << "The number of MeshBlocks increased more than twice during initialization."
            << std::endl
            << "More computing power than you expected may be required." << std::endl;
      }
    }
  } while (!iflag);

  return;
}

/// Finds location of a block with ID `tgid`. Can provide an optional "hint" to start
/// the search at.
std::list<MeshBlock>::iterator Mesh::FindMeshBlock(int tgid) {
  // search the rest of the list
  return std::find_if(block_list.begin(), block_list.end(),
                      [tgid](MeshBlock const &bl) { return bl.gid == tgid; });
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

  return;
}

// Public function for advancing next_phys_id_ counter

// Store signed, but positive, integer corresponding to the next unused value to be used
// as unique ID for a BoundaryVariable object's single set of MPI calls (formerly "enum
// AthenaTagMPI"). 5 bits of unsigned integer representation are currently reserved
// for this "phys" part of the bitfield tag, making 0, ..., 31 legal values

int Mesh::ReserveTagPhysIDs(int num_phys) {
  // TODO(felker): add safety checks? input, output are positive, obey <= 31= MAX_NUM_PHYS
  int start_id = next_phys_id_;
  next_phys_id_ += num_phys;
  return start_id;
}

// private member fn, called in Mesh() ctor

// depending on compile- and runtime options, reserve the maximum number of "int physid"
// that might be necessary for each MeshBlock's BoundaryValues object to perform MPI
// communication for all BoundaryVariable objects

// TODO(felker): deduplicate this logic, which combines conditionals in MeshBlock ctor

void Mesh::ReserveMeshBlockPhysIDs() { return; }

} // namespace parthenon
