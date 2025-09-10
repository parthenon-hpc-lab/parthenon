//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "parthenon_mpi.hpp"

#include "amr_criteria/refinement_package.hpp"
#include "application_input.hpp"
#include "bvals/boundary_conditions.hpp"
#include "bvals/bvals.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/update.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/restart.hpp"
#include "outputs/restart_hdf5.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"
#include "utils/partition_stl_containers.hpp"

namespace parthenon {
Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
           base_constructor_selector_t)
    : // public members:
      modified(true),
      adaptive(pin->GetOrAddString("parthenon/mesh", "refinement", "none",
                                   std::vector<std::string>{"none", "static", "adaptive"},
                                   "mesh refinement mode") == "adaptive"
                   ? true
                   : false),
      multilevel((adaptive ||
                  pin->GetString("parthenon/mesh", "refinement") == "static" ||
                  pin->GetOrAddBoolean("parthenon/mesh", "multigrid", false,
                                       "enable a multigrid mesh"))),
      multigrid(pin->GetOrAddBoolean("parthenon/mesh", "multigrid", false,
                                     "enable a multigrid mesh")),
      nbnew(), nbdel(), step_since_lb(), gflag(), packages(packages),
      resolved_packages(ResolvePackages(packages)),
      task_collection_timeout_in_seconds(pin->GetOrAddInteger(
          "parthenon/mesh", "task_collection_timeout_in_seconds", 60 * 5)),
      // private members:
      num_mesh_threads_(
          pin->GetOrAddInteger("parthenon/mesh", "num_threads", 1,
                               "number of host threads for infrastructure; unused")),
      use_uniform_meshgen_fn_{true, true, true, true}, lb_flag_(true), lb_automatic_(),
      lb_manual_(), nslist(Globals::nranks), nblist(Globals::nranks),
      nref(Globals::nranks), nderef(Globals::nranks), rdisp(Globals::nranks),
      ddisp(Globals::nranks), bnref(Globals::nranks), bnderef(Globals::nranks),
      brdisp(Globals::nranks), bddisp(Globals::nranks) {
  // pack size
  bool pack_size_exists = pin->DoesParameterExist("parthenon/mesh", "pack_size");
  bool num_partitions_exists =
      pin->DoesParameterExist("parthenon/mesh", "packs_per_rank");
  // If both exists, the assumption is that packs_per_rank was added later on purpose (as
  // pack_size existed first) so the new value should take precedent.
  if (pack_size_exists && num_partitions_exists) {
    use_pack_size_ = false;
    default_num_packs_ =
        pin->GetInteger("parthenon/mesh", "packs_per_rank",
                        "number of meshblockpacks per rank, overrides pack_size");
    auto pack_size = pin->GetInteger(
        "parthenon/mesh", "pack_size",
        "default size of meshblock packs on a given rank, overwritten by packs_per_rank");
    bool are_both_default = (default_num_packs_ == 1) && (pack_size == -1);
    if (!are_both_default && (Globals::my_rank == 0)) {
      PARTHENON_WARN("Both pack_size and packs_per_rank set to non default values! "
                     "New packs_per_rank takes precedent.");
    }
    // Only one or none is set
  } else {
    if (pack_size_exists) {
      use_pack_size_ = true;
      default_pack_size_ = pin->GetInteger("parthenon/mesh", "pack_size",
                                           "default size of meshblock packs on a given "
                                           "rank, overwritten by packs_per_rank");
      // use packs_per_rank (and set default value if not set)
    } else {
      use_pack_size_ = false;
      default_num_packs_ = std::max(
          1,
          pin->GetOrAddInteger("parthenon/mesh", "packs_per_rank", 1,
                               "number of meshblockpacks per rank, overrides pack_size"));
    }
  }

  // Allow for user overrides to default Parthenon functions
  if (app_in->InitUserMeshData != nullptr) {
    InitUserMeshData = app_in->InitUserMeshData;
  }
  if (app_in->MeshProblemGenerator != nullptr) {
    ProblemGenerator = app_in->MeshProblemGenerator;
  }
  if (app_in->MeshPostInitialization != nullptr) {
    PostInitialization = app_in->MeshPostInitialization;
  }
  if (app_in->PreStepMeshUserWorkInLoop != nullptr) {
    PreStepUserWorkInLoop = app_in->PreStepMeshUserWorkInLoop;
  }
  if (app_in->PostStepMeshUserWorkInLoop != nullptr) {
    PostStepUserWorkInLoop = app_in->PostStepMeshUserWorkInLoop;
  }
  if (app_in->UserMeshWorkBeforeOutput != nullptr) {
    UserMeshWorkBeforeOutput = app_in->UserMeshWorkBeforeOutput;
  }
  if (app_in->UserWorkBeforeRestartOutput != nullptr) {
    UserWorkBeforeRestartOutput = app_in->UserWorkBeforeRestartOutput;
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

  // Default root level, may be overwritten by another constructor
  root_level = 0;
  // SMR / AMR:
  if (adaptive) {
    max_level = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1,
                                     "maximum level of refinement globally") +
                root_level - 1;
  } else {
    max_level = 63;
  }

  SetupMPIComms();

  RegisterLoadBalancing_(pin);

  mesh_data.SetMeshPointer(this);

  if (InitUserMeshData) InitUserMeshData(this, pin);
}

Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
           hyper_rectangular_constructor_selector_t)
    : Mesh(pin, app_in, packages, base_constructor_selector_t()) {

  std::tie(mesh_size, base_block_size) = GetRegionSizes(pin);

  SetBCNames_(pin);
  mesh_bcs = GetBCsFromNames_(mesh_bc_names);
  ndim = (mesh_size.nx(X3DIR) > 1) ? 3 : ((mesh_size.nx(X2DIR) > 1) ? 2 : 1);

  // Load balancing flag and parameters
  forest = forest::Forest::HyperRectangular(mesh_size, base_block_size, mesh_bcs);
  root_level = forest.root_level;
  forest.EnrollBndryFncts(app_in, mesh_bc_names, mesh_swarm_bc_names,
                          resolved_packages->UserBoundaryFunctions,
                          resolved_packages->UserSwarmBoundaryFunctions);

  // SMR / AMR:
  if (adaptive) {
    max_level = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1,
                                     "maximum level of refinement globally") +
                root_level - 1;
  } else {
    max_level = 63;
  }

  // Register user defined boundary conditions

  CheckMeshValidity();
}

Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
           forest::ForestDefinition &forest_def)
    : Mesh(pin, app_in, packages, base_constructor_selector_t()) {
  mesh_size =
      RegionSize({0, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 1, 1}, {false, false, true});

  SetBCNames_(pin);
  mesh_bcs = GetBCsFromNames_(mesh_bc_names);
  base_block_size = GetBaseMeshBlockSize(pin, mesh_size);
  forest_def.SetBlockSize(base_block_size);

  ndim = 2;
  // Load balancing flag and parameters
  forest = forest::Forest::Make2D(forest_def);
  root_level = forest.root_level;
  forest.EnrollBndryFncts(app_in, mesh_bc_names, mesh_swarm_bc_names,
                          resolved_packages->UserBoundaryFunctions,
                          resolved_packages->UserSwarmBoundaryFunctions);
  BuildBlockList(pin, app_in, packages, 0);
}

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file
Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
           int mesh_test)
    : Mesh(pin, app_in, packages, hyper_rectangular_constructor_selector_t()) {
  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

  if (multilevel) DoStaticRefinement(pin);

  // initial mesh hierarchy construction is completed here
  BuildBlockList(pin, app_in, packages, mesh_test);
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file
Mesh::Mesh(ParameterInput *pin, ApplicationInput *app_in, RestartReader &rr,
           Packages_t &packages, int mesh_test)
    : Mesh(pin, app_in, packages, hyper_rectangular_constructor_selector_t()) {
  std::stringstream msg;

  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

  // read the restart file
  // the file is already open and the pointer is set to after <par_end>

  auto mesh_info = rr.GetMeshInfo();
  // All ranks read HDF file
  nbnew = mesh_info.nbnew;
  nbdel = mesh_info.nbdel;
  nbtotal = mesh_info.nbtotal;

  const auto grid_dim = mesh_info.grid_dim;
  PARTHENON_REQUIRE(mesh_size.xmin(X1DIR) == grid_dim[0],
                    "Mesh size shouldn't change on restart.");
  PARTHENON_REQUIRE(mesh_size.xmax(X1DIR) == grid_dim[1],
                    "Mesh size shouldn't change on restart.");
  PARTHENON_REQUIRE(mesh_size.xrat(X1DIR) == grid_dim[2],
                    "Mesh size shouldn't change on restart.");

  PARTHENON_REQUIRE(mesh_size.xmin(X2DIR) == grid_dim[3],
                    "Mesh size shouldn't change on restart.");
  PARTHENON_REQUIRE(mesh_size.xmax(X2DIR) == grid_dim[4],
                    "Mesh size shouldn't change on restart.");
  PARTHENON_REQUIRE(mesh_size.xrat(X2DIR) == grid_dim[5],
                    "Mesh size shouldn't change on restart.");

  PARTHENON_REQUIRE(mesh_size.xmin(X3DIR) == grid_dim[6],
                    "Mesh size shouldn't change on restart.");
  PARTHENON_REQUIRE(mesh_size.xmax(X3DIR) == grid_dim[7],
                    "Mesh size shouldn't change on restart.");
  PARTHENON_REQUIRE(mesh_size.xrat(X3DIR) == grid_dim[8],
                    "Mesh size shouldn't change on restart.");

  for (auto &dir : {X1DIR, X2DIR, X3DIR}) {
    PARTHENON_REQUIRE(base_block_size.nx(dir) == mesh_info.block_size[dir - 1] -
                                                     (mesh_info.block_size[dir - 1] > 1) *
                                                         mesh_info.includes_ghost * 2 *
                                                         mesh_info.n_ghost,
                      "Block size not consistent on restart.");
  }

  // Populate logical locations
  loclist = std::vector<LogicalLocation>(nbtotal);
  std::unordered_map<LogicalLocation, int> dealloc_count;
  auto lx123 = mesh_info.lx123;
  auto locLevelGidLidCnghostGflag = mesh_info.level_gid_lid_cnghost_gflag;
  for (int i = 0; i < nbtotal; i++) {
    loclist[i] = forest.GetForestLocationFromLegacyTreeLocation(
        LogicalLocation(locLevelGidLidCnghostGflag[NumIDsAndFlags * i], lx123[3 * i],
                        lx123[3 * i + 1], lx123[3 * i + 2]));
    dealloc_count[loclist[i]] = mesh_info.derefinement_count[i];
  }

  // rebuild the Block Tree
  for (int i = 0; i < nbtotal; i++)
    forest.AddMeshBlock(loclist[i], false);

  int nnb = forest.CountMeshBlock();
  if (nnb != nbtotal) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Tree reconstruction failed. The total numbers of the blocks do not match. ("
        << nbtotal << " != " << nnb << ")" << std::endl;
    PARTHENON_FAIL(msg);
  }

  BuildBlockList(pin, app_in, packages, mesh_test, dealloc_count);
}

void Mesh::BuildBlockList(ParameterInput *pin, ApplicationInput *app_in,
                          Packages_t &packages, int mesh_test,
                          const std::unordered_map<LogicalLocation, int> &dealloc_count) {
  // LFR: This routine should work for general block lists
  std::stringstream msg;

  loclist = forest.GetMeshBlockListAndResolveGids();
  nbtotal = loclist.size();
  current_level = -1;
  for (const auto &loc : loclist)
    if (loc.level() > current_level) current_level = loc.level();

#ifdef MPI_PARALLEL
  // check if there are sufficient blocks
  const bool output_params_and_exit =
      pin->GetOrAddBoolean("parthenon/job", "output_params_and_exit", false);
  if (nbtotal < Globals::nranks) {
    if (mesh_test != 0) {
      if ((Globals::my_rank == 0) && !output_params_and_exit) {
        std::cout << "### Warning in Mesh constructor" << std::endl
                  << "Too few mesh blocks: nbtotal (" << nbtotal << ") < nranks ("
                  << Globals::nranks << ")" << std::endl;
      }
    }
  }
#endif

  ranklist = std::vector<int>(nbtotal);

  // initialize cost array with the simplest estimate; all the blocks are equal
  costlist = std::vector<double>(nbtotal, 1.0);

  CalculateLoadBalance(costlist, ranklist, nslist, nblist);

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0) {
    BuildBlockPartitions(GridIdentifier::leaf());
    if (Globals::my_rank == 0) OutputMeshStructure(ndim);
    return;
  }

  // create MeshBlock list for this process
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs + nblist[Globals::my_rank] - 1;

  // create MeshBlock list for this process
  block_list.clear();
  block_list.resize(nbe - nbs + 1);
  for (int i = nbs; i <= nbe; i++) {
    RegionSize block_size;
    BoundaryFlag block_bcs[6];
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    block_list[i - nbs] =
        MeshBlock::Make(i, i - nbs, loclist[i], block_size, block_bcs, this, pin, app_in,
                        packages, resolved_packages, gflag, costlist[i]);
    if (block_list[i - nbs]->pmr)
      block_list[i - nbs]->pmr->DerefinementCount() =
          dealloc_count.count(loclist[i]) ? dealloc_count.at(loclist[i]) : 0;
  }
  BuildBlockPartitions(GridIdentifier::leaf());
  BuildGMGBlockLists(pin, app_in);
  SetMeshBlockNeighbors(GridIdentifier::leaf(), block_list, ranklist);
  SetGMGNeighbors();
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
//  \brief Partition a given block list for use by MeshData

void Mesh::BuildBlockPartitions(GridIdentifier grid) {
  auto partition_blocklists = partition::ToSizeN(
      grid.type == GridType::leaf ? block_list : gmg_block_lists_[grid.logical_level],
      DefaultPackSize());
  std::vector<std::shared_ptr<BlockListPartition>> out;
  int id = 0;
  for (auto &part_bl : partition_blocklists)
    out.emplace_back(std::make_shared<BlockListPartition>(id++, grid, part_bl, this));
  block_partitions_[grid] = out;
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
  std::cout << "Number of Trees = " << forest.CountTrees() << std::endl;
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
// \!fn void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkBeforeOutput

void Mesh::ApplyUserWorkBeforeOutput(Mesh *mesh, ParameterInput *pin,
                                     SimTime const &time) {
  // call Mesh version
  if (mesh->UserMeshWorkBeforeOutput != nullptr) {
    mesh->UserMeshWorkBeforeOutput(mesh, pin, time);
  }

  // call MeshBlock version
  for (auto &pmb : block_list) {
    if (pmb->UserWorkBeforeOutput != nullptr) {
      pmb->UserWorkBeforeOutput(pmb.get(), pin, time);
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkBeforeRestartOutput
// \brief Apply Mesh and Meshblock versions of UserWorkBeforeRestartOutput
void Mesh::ApplyUserWorkBeforeRestartOutput(Mesh *mesh, ParameterInput *pin,
                                            SimTime const &time,
                                            OutputParameters *pparams) {
  if (mesh->UserWorkBeforeRestartOutput != nullptr) {
    mesh->UserWorkBeforeRestartOutput(mesh, pin, time, pparams);
  }
}

void Mesh::BuildTagMapAndBoundaryBuffers() {
  const int num_partitions = DefaultNumPartitions();
  const int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);

  // Build densely populated communication tags
  tag_map.clear();
  for (auto &partition : GetDefaultBlockPartitions()) {
    auto &md = mesh_data.Add("base", partition);
    tag_map.AddMeshDataToMap<BoundaryType::any>(md);
  }

  if (multigrid) {
    for (int gmg_level = GetGMGMinLevel(); gmg_level <= GetGMGMaxLevel(); ++gmg_level) {
      const auto grid_id = GridIdentifier::two_level_composite(gmg_level);
      for (auto &partition : GetDefaultBlockPartitions(grid_id)) {
        auto &md = mesh_data.Add("base", partition);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_same>(md);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_prolongate_send>(md);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_restrict_send>(md);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_prolongate_recv>(md);
        tag_map.AddMeshDataToMap<BoundaryType::gmg_restrict_recv>(md);
      }
    }
  }

  tag_map.ResolveMap();

  // Create send/recv MPI_Requests for swarms
  for (int i = 0; i < nmb; ++i) {
    auto &pmb = block_list[i];
    pmb->meshblock_data.Get()->GetSwarmData()->SetupPersistentMPI();
  }

  // Wait for boundary buffers to be no longer in use
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

  // Clear boundary communication buffers
  boundary_comm_map.clear();

  // Build the boundary buffers for the current mesh
  for (auto &partition : GetDefaultBlockPartitions()) {
    auto &md = mesh_data.Add("base", partition);
    BuildBoundaryBuffers(md);
  }
  if (multigrid) {
    for (int gmg_level = GetGMGMinLevel(); gmg_level <= GetGMGMaxLevel(); ++gmg_level) {
      const auto grid_id = GridIdentifier::two_level_composite(gmg_level);
      for (auto &partition : GetDefaultBlockPartitions(grid_id)) {
        auto &mdg = mesh_data.Add("base", partition);
        BuildBoundaryBuffers(mdg);
        BuildGMGBoundaryBuffers(mdg);
      }
    }
  }
}

void Mesh::CommunicateBoundaries(std::string md_name,
                                 const std::vector<std::string> &fields) {
  const int num_partitions = DefaultNumPartitions();

  TaskCollection tc;
  TaskRegion &region = tc.AddRegion(num_partitions);
  auto partitions = GetDefaultBlockPartitions();
  for (int i = 0; i < num_partitions; i++) {
    auto &md = mesh_data.Add(md_name, partitions[i], fields);
    auto bound = AddBoundaryExchangeTasks(TaskID(0), region[i], md, multilevel);
  }
  TaskListStatus status = tc.Execute(task_collection_timeout_in_seconds);
  PARTHENON_REQUIRE(status == TaskListStatus::complete,
                    "Boundary communication called internal by mesh failed.");
}

void Mesh::PreCommFillDerived() {
  const int num_partitions = DefaultNumPartitions();
  const int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  // Pre comm fill derived
  for (int i = 0; i < nmb; ++i) {
    auto &mbd = block_list[i]->meshblock_data.Get();
    Update::PreCommFillDerived(mbd.get());
  }
  for (auto &partition : GetDefaultBlockPartitions()) {
    auto &md = mesh_data.Add("base", partition);
    PARTHENON_REQUIRE(partition->pmesh == this, "Bad partition mesh pointer");
    PARTHENON_REQUIRE(md->GetParentPointer() == this, "Bad mesh pointer");
    Update::PreCommFillDerived(md.get());
  }
}

void Mesh::FillDerived() {
  const int num_partitions = DefaultNumPartitions();
  const int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  for (auto &partition : GetDefaultBlockPartitions()) {
    auto &md = mesh_data.Add("base", partition);
    // Call MeshData based FillDerived functions
    Update::FillDerived(md.get());
  }

  for (int i = 0; i < nmb; ++i) {
    auto &mbd = block_list[i]->meshblock_data.Get();
    // Call MeshBlockData based FillDerived functions
    Update::FillDerived(mbd.get());
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::Initialize(bool init_problem, ParameterInput *pin)
// \brief  initialization before the main loop

void Mesh::Initialize(bool init_problem, ParameterInput *pin, ApplicationInput *app_in) {
  PARTHENON_INSTRUMENT
  bool init_done = true;
  const int nb_initial = nbtotal;
  const bool output_params_and_exit =
      pin->GetOrAddBoolean("parthenon/job", "output_params_and_exit", false);

  do {
    int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);

    // init meshblock data
    for (int i = 0; i < nmb; ++i) {
      MeshBlock *pmb = block_list[i].get();
      if (pmb->InitMeshBlockUserData != nullptr) {
        pmb->InitMeshBlockUserData(pmb, pin);
      }
    }

    const int num_partitions = DefaultNumPartitions();

    // problem generator
    if (init_problem) {
      PARTHENON_REQUIRE_THROWS(
          !(ProblemGenerator != nullptr &&
            (nmb != 0 && block_list[0]->ProblemGenerator != nullptr)),
          "Mesh and MeshBlock ProblemGenerators are defined. Please use only one.");
      PARTHENON_REQUIRE_THROWS(
          !(PostInitialization != nullptr &&
            (nmb != 0 && block_list[0]->PostInitialization != nullptr)),
          "Mesh and MeshBlock PostInitializations are defined. Please use only one.");

      // Call Mesh ProblemGenerator
      if (ProblemGenerator != nullptr) {
        for (auto &partition : GetDefaultBlockPartitions(GridIdentifier::leaf())) {
          auto &md = mesh_data.Add("base", partition);
          ProblemGenerator(this, pin, md.get());
        }
        // Call individual MeshBlock ProblemGenerator
      } else {
        for (int i = 0; i < nmb; ++i) {
          auto &pmb = block_list[i];
          if (pmb->ProblemGenerator != nullptr) {
            pmb->ProblemGenerator(pmb.get(), pin);
          }
        }
      }

      // Call Mesh PostInitialization
      if (PostInitialization != nullptr) {
        for (auto &partition : GetDefaultBlockPartitions(GridIdentifier::leaf())) {
          auto &md = mesh_data.Add("base", partition);
          PostInitialization(this, pin, md.get());
        }
        // Call individual MeshBlock PostInitialization
      } else {
        for (int i = 0; i < nmb; ++i) {
          auto &pmb = block_list[i];
          if (pmb->PostInitialization != nullptr) {
            pmb->PostInitialization(pmb.get(), pin);
          }
        }
      }

      // Call per-package initialization
      for (const auto &[name, pkg] : packages.AllPackages()) {
        PARTHENON_REQUIRE_THROWS(
            !(pkg->PostInitializationMesh != nullptr &&
              (pkg->PostInitializationBlock != nullptr)),
            "Mesh and MeshBlock PostInitializations are defined for package " + name +
                ". Please use only one.");

        // first on the mesh...
        if (pkg->PostInitializationMesh != nullptr) {
          for (auto &partition : GetDefaultBlockPartitions(GridIdentifier::leaf())) {
            auto &md = mesh_data.Add("base", partition);
            pkg->PostInitializationMesh(this, pin, md.get());
          }
        }
        // and then per block
        if (pkg->PostInitializationBlock != nullptr) {
          for (auto &pmb : block_list) {
            pkg->PostInitializationBlock(pmb.get(), pin);
          }
        }
      }

      std::for_each(block_list.begin(), block_list.end(),
                    [](auto &sp_block) { sp_block->SetAllVariablesToInitialized(); });
    }

    PreCommFillDerived();

    BuildTagMapAndBoundaryBuffers();

    CommunicateBoundaries();

    FillDerived();

    if (init_problem && adaptive) {
      for (auto &partition : GetDefaultBlockPartitions(GridIdentifier::leaf())) {
        auto &md = mesh_data.Add("base", partition);
        Refinement::Tag(md.get());
      }
      init_done = false;
      // caching nbtotal the private variable my be updated in the following function
      const int nb_before_loadbalance = nbtotal;
      LoadBalancingAndAdaptiveMeshRefinement(pin, app_in);
      if (nbtotal == nb_before_loadbalance) {
        init_done = true;
      } else if (nbtotal < nb_before_loadbalance && Globals::my_rank == 0 &&
                 !output_params_and_exit) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
                  << "The number of MeshBlocks decreased during AMR grid initialization."
                  << std::endl
                  << "Possibly the refinement criteria have a problem." << std::endl;
      }
      if (nbtotal > 2 * nb_initial && Globals::my_rank == 0 && !output_params_and_exit) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
                  << "The number of MeshBlocks increased more than twice during "
                     "initialization."
                  << std::endl
                  << "More computing power than you expected may be required."
                  << std::endl;
      }
    }
  } while (!init_done);

#ifdef MPI_PARALLEL
  // check if there are sufficient blocks
  if (nbtotal < Globals::nranks && Globals::my_rank == 0 && !output_params_and_exit) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh Initialize" << std::endl
        << "Too few mesh blocks after initialization: nbtotal (" << nbtotal
        << ") < nranks (" << Globals::nranks << ")" << std::endl;
    PARTHENON_FAIL(msg);
  }
#endif

  // Initialize the "base" MeshData object
  mesh_data.Get()->Initialize(block_list, this);
}

/// Finds location of a block with ID `tgid`.
std::shared_ptr<MeshBlock> Mesh::FindMeshBlock(int tgid) const {
  PARTHENON_REQUIRE(block_list.size() > 0,
                    "Trying to call FindMeshBlock with empty block list");
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
  block_size = forest.GetBlockDomain(loc);
  auto bcs = forest.GetBlockBCs(loc);
  for (int i = 0; i < BOUNDARY_NFACES; ++i)
    block_bcs[i] = bcs[i];
  return valid_region;
}

std::int64_t Mesh::GetTotalCells() {
  return static_cast<std::int64_t>(nbtotal) * GetNumberOfMeshBlockCells();
}

int Mesh::GetNumberOfMeshBlockCells() const {
  return base_block_size.nx(X1DIR) * base_block_size.nx(X2DIR) *
         base_block_size.nx(X3DIR);
}

const IndexShape Mesh::GetLeafBlockCellBounds(CellLevel level) const {
  auto shapes = GetIndexShapes(
      ndim > 0 ? base_block_size.nx(X1DIR) : 0, ndim > 1 ? base_block_size.nx(X2DIR) : 0,
      ndim > 2 ? base_block_size.nx(X3DIR) : 0, multilevel, this);
  if (level == CellLevel::same) {
    return shapes[0];
  } else if (level == CellLevel::fine) {
    return shapes[1];
  } else { // if (level == CellLevel::coarse) {
    return shapes[2];
  }
}

ParArray1D<AmrTag> &Mesh::GetAmrTags() {
  const int nblocks = GetNumMeshBlocksThisRank();
  if (!amr_tags.KokkosView().is_allocated()) {
    amr_tags.KokkosView() = Kokkos::View<AmrTag *>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "amr_tags"), nblocks);
  }
  if (amr_tags.KokkosView().size() != nblocks) {
    Kokkos::realloc(amr_tags.KokkosView(), nblocks);
  }
  return amr_tags;
}

// Functionality re-used in mesh constructor
void Mesh::RegisterLoadBalancing_(ParameterInput *pin) {
  // JMM: This machinery is only used with MPI, but I don't want these
  // options hidden if the code is built without MPI, so there's no
  // ifdef.
  const std::string balancer =
      pin->GetOrAddString("parthenon/loadbalancing", "balancer", "default",
                          std::vector<std::string>{"default", "automatic", "manual"},
                          "load balancing strategy");
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
  lb_tolerance_ = pin->GetOrAddReal("parthenon/loadbalancing", "tolerance", 0.5,
                                    "load balancer tolerance");
  lb_interval_ = pin->GetOrAddInteger(
      "parthenon/loadbalancing", "interval", 10,
      "how frequently load balancing is performed if the mesh does not change");
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
    if (metadata.IsSet(Metadata::FillGhost) || metadata.IsSet(Metadata::Independent) ||
        metadata.IsSet(Metadata::ForceRemeshComm) ||
        metadata.IsSet(Metadata::GMGProlongate) ||
        metadata.IsSet(Metadata::GMGRestrict) || metadata.IsSet(Metadata::Flux)) {
      MPI_Comm mpi_comm;
      PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));
      const auto ret = mpi_comm_map_.insert({pair.first.label(), mpi_comm});
      PARTHENON_REQUIRE_THROWS(ret.second, "Communicator with same name already in map");
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

void Mesh::CheckMeshValidity() const {
  std::stringstream msg;
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

  // check consistency of the block and mesh
  if (mesh_size.nx(X1DIR) % base_block_size.nx(X1DIR) != 0 ||
      mesh_size.nx(X2DIR) % base_block_size.nx(X2DIR) != 0 ||
      mesh_size.nx(X3DIR) % base_block_size.nx(X3DIR) != 0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "the Mesh must be evenly divisible by the MeshBlock" << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (base_block_size.nx(X1DIR) < 4 || (base_block_size.nx(X2DIR) < 4 && (ndim >= 2)) ||
      (base_block_size.nx(X3DIR) < 4 && (ndim >= 3))) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "block_size must be larger than or equal to 4 cells." << std::endl;
    PARTHENON_FAIL(msg);
  }

  if (max_level > 63) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The number of the refinement level must be smaller than "
        << 63 - root_level + 1 << "." << std::endl;
    PARTHENON_FAIL(msg);
  }

  if (multilevel) {
    if (base_block_size.nx(X1DIR) % 2 == 1 ||
        (base_block_size.nx(X2DIR) % 2 == 1 && (ndim >= 2)) ||
        (base_block_size.nx(X3DIR) % 2 == 1 && (ndim >= 3))) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The size of MeshBlock must be divisible by 2 in order to use SMR or AMR."
          << std::endl;
      PARTHENON_FAIL(msg);
    }
  }
}

void Mesh::DoStaticRefinement(ParameterInput *pin) {
  std::stringstream msg;

  // TODO(LFR): Static refinement currently only works for hyper-rectangular meshes
  std::array<int, 3> nrbx;
  for (auto dir : {X1DIR, X2DIR, X3DIR})
    nrbx[dir - 1] = mesh_size.nx(dir) / base_block_size.nx(dir);

  auto GetStaticRefLLIndexRange = [](CoordinateDirection dir, int num_root_block,
                                     int ref_level, const RegionSize &ref_size,
                                     const RegionSize &mesh_size) {
    int lxtot = num_root_block * (1 << ref_level);
    int lxmin, lxmax;
    for (lxmin = 0; lxmin < lxtot; lxmin++) {
      Real r = LogicalLocation::IndexToSymmetrizedCoordinate(lxmin + 1,
                                                             BlockLocation::Left, lxtot);
      if (mesh_size.SymmetrizedLogicalToActualPosition(r, dir) > ref_size.xmin(dir))
        break;
    }
    for (lxmax = lxmin; lxmax < lxtot; lxmax++) {
      Real r = LogicalLocation::IndexToSymmetrizedCoordinate(lxmax + 1,
                                                             BlockLocation::Left, lxtot);
      if (mesh_size.SymmetrizedLogicalToActualPosition(r, dir) >= ref_size.xmax(dir))
        break;
    }
    if (lxmin % 2 == 1) lxmin--;
    if (lxmax % 2 == 0) lxmax++;
    return std::pair<int, int>{lxmin, lxmax};
  };

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
      int lrlev = ref_lev + GetLegacyTreeRootLevel();
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
          auto [lmin, lmax] =
              GetStaticRefLLIndexRange(dir, nrbx[dir - 1], ref_lev, ref_size, mesh_size);
          l_region_min[dir - 1] = lmin;
          l_region_max[dir - 1] = lmax;
        }
      }
      for (std::int64_t k = l_region_min[2]; k < l_region_max[2]; k += 2) {
        for (std::int64_t j = l_region_min[1]; j < l_region_max[1]; j += 2) {
          for (std::int64_t i = l_region_min[0]; i < l_region_max[0]; i += 2) {
            LogicalLocation nloc(lrlev, i, j, k);
            forest.AddMeshBlock(forest.GetForestLocationFromLegacyTreeLocation(nloc));
          }
        }
      }
    }
    pib = pib->pnext;
  }
}
// Return list of locations and levels for the legacy tree
// TODO(LFR): It doesn't make sense to offset the level by the
//   legacy tree root level since the location indices are defined
//   for loc.level(). It seems this level offset is required for
//   the output to agree with the legacy output though.
std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
Mesh::GetLevelsAndLogicalLocationsFlat() const noexcept {
  std::vector<std::int64_t> levels, logicalLocations;
  levels.reserve(nbtotal);
  logicalLocations.reserve(nbtotal * 3);
  for (auto loc : loclist) {
    loc = forest.GetLegacyTreeLocation(loc);
    levels.push_back(loc.level() - GetLegacyTreeRootLevel());
    logicalLocations.push_back(loc.lx1());
    logicalLocations.push_back(loc.lx2());
    logicalLocations.push_back(loc.lx3());
  }
  return std::make_pair(levels, logicalLocations);
}

void Mesh::SetBCNames_(ParameterInput *pin) {
  mesh_bc_names = {
      pin->GetOrAddString("parthenon/mesh", "ix1_bc", "outflow",
                          "global mesh boundary condition on inner X1 face"),
      pin->GetOrAddString("parthenon/mesh", "ox1_bc", "outflow",
                          "global mesh boundary condition on outer X1 face"),
      pin->GetOrAddString("parthenon/mesh", "ix2_bc", "outflow",
                          "global mesh boundary condition on inner X2 face"),
      pin->GetOrAddString("parthenon/mesh", "ox2_bc", "outflow",
                          "global mesh boundary condition on outer X2 face"),
      pin->GetOrAddString("parthenon/mesh", "ix3_bc", "outflow",
                          "global mesh boundary condition on inner X3 face"),
      pin->GetOrAddString("parthenon/mesh", "ox3_bc", "outflow",
                          "global mesh boundary condition on outer X3 face")};
  // JMM: This is needed because not all BCs are necessarily
  // implemented for swarms
  auto maybe = [](const std::string &s) {
    return ((s == "outflow") || (s == "periodic")) ? s : "outflow";
  };
  mesh_swarm_bc_names = {
      pin->GetOrAddString("parthenon/swarm", "ix1_bc", maybe(mesh_bc_names[0]),
                          "global particle boundary condition on inner X1 face"),
      pin->GetOrAddString("parthenon/swarm", "ox1_bc", maybe(mesh_bc_names[1]),
                          "global particle boundary condition on outer X1 face"),
      pin->GetOrAddString("parthenon/swarm", "ix2_bc", maybe(mesh_bc_names[2]),
                          "global particle boundary condition on inner X2 face"),
      pin->GetOrAddString("parthenon/swarm", "ox2_bc", maybe(mesh_bc_names[3]),
                          "global particle boundary condition on outer X2 face"),
      pin->GetOrAddString("parthenon/swarm", "ix3_bc", maybe(mesh_bc_names[4]),
                          "global particle boundary condition on inner X3 face"),
      pin->GetOrAddString("parthenon/swarm", "ox3_bc", maybe(mesh_bc_names[5]),
                          "global particle boundary condition on outer X3 face")};
  // JMM: A consequence of having only one boundary flag array but
  // multiple boundary function arrays is that swarms *must* be
  // periodic if the mesh is periodic but otherwise mesh and swarm
  // boundaries are decoupled.
  for (int i = 0; i < BOUNDARY_NFACES; ++i) {
    if (mesh_bc_names[i] == "periodic") {
      PARTHENON_REQUIRE(mesh_swarm_bc_names[i] == "periodic",
                        "If the mesh is periodic, swarms must be also.");
    }
  }
}

std::array<BoundaryFlag, BOUNDARY_NFACES>
Mesh::GetBCsFromNames_(const std::array<std::string, BOUNDARY_NFACES> &names) const {
  std::array<BoundaryFlag, BOUNDARY_NFACES> out;
  for (int f = 0; f < BOUNDARY_NFACES; ++f) {
    out[f] = GetBoundaryFlag(names[f]);
  }
  return out;
}

RegionSize Mesh::GetBaseMeshBlockSize(ParameterInput *pin, const RegionSize &mesh_size) {
  RegionSize base_block_size;
  for (auto &[dir, label] : std::vector<std::tuple<CoordinateDirection, std::string>>{
           {X1DIR, "nx1"}, {X2DIR, "nx2"}, {X3DIR, "nx3"}}) {
    base_block_size.xrat(dir) = mesh_size.xrat(dir);
    base_block_size.symmetry(dir) = mesh_size.symmetry(dir);
    // JMM: Just to ensure it gets written to params
    ParameterRef meshref("parthenon/mesh", label);
    base_block_size.nx(dir) = pin->GetOrAddInteger(
        "parthenon/meshblock", label, meshref,
        "logical size of a meshblock; defaults to the base size of the mesh");
    if (base_block_size.symmetry(dir)) {
      base_block_size.nx(dir) = mesh_size.nx(dir);
      pin->SetInteger("parthenon/meshblock", label, mesh_size.nx(dir));
    }
  }
  return base_block_size;
}

std::pair<RegionSize, RegionSize> Mesh::GetRegionSizes(ParameterInput *pin) {
  RegionSize mesh_size(
      {pin->GetReal("parthenon/mesh", "x1min", "minimum x1 value of domain"),
       pin->GetReal("parthenon/mesh", "x2min", "minimum x2 value of domain"),
       pin->GetReal("parthenon/mesh", "x3min", "minimum x3 value of domain")},
      {pin->GetReal("parthenon/mesh", "x1max", "maximum x1 value of domain"),
       pin->GetReal("parthenon/mesh", "x2max", "maximum x2 value of domain"),
       pin->GetReal("parthenon/mesh", "x3max", "maximum x3 value of domain")},
      {pin->GetOrAddReal("parthenon/mesh", "x1rat", 1.0, "unused"),
       pin->GetOrAddReal("parthenon/mesh", "x2rat", 1.0, "unused"),
       pin->GetOrAddReal("parthenon/mesh", "x3rat", 1.0, "unused")},
      {pin->GetInteger("parthenon/mesh", "nx1",
                       "number of cells on base mesh in x1 direction"),
       pin->GetInteger("parthenon/mesh", "nx2",
                       "number of cells on base mesh in x2 direction"),
       pin->GetInteger("parthenon/mesh", "nx3",
                       "number of cells on base mesh in x3 direction")},
      {false, pin->GetInteger("parthenon/mesh", "nx2") == 1,
       pin->GetInteger("parthenon/mesh", "nx3") == 1});

  RegionSize base_block_size = GetBaseMeshBlockSize(pin, mesh_size);

  return std::make_pair(mesh_size, base_block_size);
}

} // namespace parthenon
