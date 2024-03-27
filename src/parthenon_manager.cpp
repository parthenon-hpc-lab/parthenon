//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
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

#include "parthenon_manager.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "amr_criteria/refinement_package.hpp"
#include "config.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "utils/error_checking.hpp"
#include "utils/utils.hpp"

namespace parthenon {

ParthenonStatus ParthenonManager::ParthenonInitEnv(int argc, char *argv[]) {
  if (called_init_env_) {
    PARTHENON_THROW("ParthenonInitEnv called twice!");
  }
  called_init_env_ = true;

  // initialize MPI
#ifdef MPI_PARALLEL
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI Initialization failed." << std::endl;
    return ParthenonStatus::error;
  }
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank))) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks)) {
    std::cout << "### FATAL ERROR in main" << std::endl
              << "MPI_Comm_size failed." << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }
#else  // no MPI
  Globals::my_rank = 0;
  Globals::nranks = 1;
#endif // MPI_PARALLEL

  Kokkos::initialize(argc, argv);

  // pgrete: This is a hack to disable allocation tracking until the Kokkos
  // tools provide a more fine grained control out of the box.
  bool unused;
  if (Env::get<bool>("KOKKOS_TRACK_ALLOC_OFF", false, unused)) {
    Kokkos::Profiling::Experimental::set_allocate_data_callback(nullptr);
    Kokkos::Profiling::Experimental::set_deallocate_data_callback(nullptr);
  }

  // parse the input arguments
  ArgStatus arg_status = arg.parse(argc, argv);
  if (arg_status == ArgStatus::error) {
    return ParthenonStatus::error;
  } else if (arg_status == ArgStatus::complete) {
    return ParthenonStatus::complete;
  }

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && arg.wtlim > 0) SignalHandler::SetWallTimeAlarm(arg.wtlim);

  // Populate the ParameterInput object
  if (arg.input_filename != nullptr) {
    pinput = std::make_unique<ParameterInput>(arg.input_filename);
  } else if (arg.res_flag != 0) {
    // Read input from restart file
    restartReader = std::make_unique<RestartReader>(arg.restart_filename);

    // Load input stream
    pinput = std::make_unique<ParameterInput>();
    auto inputString = restartReader->GetAttr<std::string>("Input", "File");
    std::istringstream is(inputString);
    pinput->LoadFromStream(is);
  }

  // Modify based on command line inputs
  pinput->ModifyFromCmdline(argc, argv);
  // Set the global number of ghost zones
  Globals::nghost = pinput->GetOrAddInteger("parthenon/mesh", "nghost", 2);

  // set sparse config
  Globals::sparse_config.enabled = pinput->GetOrAddBoolean(
      "parthenon/sparse", "enable_sparse", Globals::sparse_config.enabled);
#ifndef ENABLE_SPARSE
  PARTHENON_REQUIRE_THROWS(
      !Globals::sparse_config.enabled,
      "Sparse is compile-time disabled but was requested to be enabled in input file");
#endif
  Globals::sparse_config.allocation_threshold = pinput->GetOrAddReal(
      "parthenon/sparse", "alloc_threshold", Globals::sparse_config.allocation_threshold);
  Globals::sparse_config.deallocation_threshold =
      pinput->GetOrAddReal("parthenon/sparse", "dealloc_threshold",
                           Globals::sparse_config.deallocation_threshold);
  Globals::sparse_config.deallocation_count = pinput->GetOrAddInteger(
      "parthenon/sparse", "dealloc_count", Globals::sparse_config.deallocation_count);

  // set timeout config
  Globals::receive_boundary_buffer_timeout =
      pinput->GetOrAddReal("parthenon/time", "recv_bdry_buf_timeout_sec", -1.0);

  // set boundary comms buffer switch trigger
  Globals::refinement::min_num_bufs =
      pinput->GetOrAddReal("parthenon/mesh", "refinement_in_one_min_nbufs", 64);

  return ParthenonStatus::ok;
}

void ParthenonManager::ParthenonInitPackagesAndMesh() {
  if (called_init_packages_and_mesh_) {
    PARTHENON_THROW("Called ParthenonInitPackagesAndMesh twice!");
  }
  called_init_packages_and_mesh_ = true;

  // Allow for user overrides to default Parthenon functions
  if (app_input->ProcessPackages != nullptr) {
    ProcessPackages = app_input->ProcessPackages;
  }

  // set up all the packages in the application
  auto packages = ProcessPackages(pinput);
  // always add the Refinement package
  packages.Add(Refinement::Initialize(pinput.get()));

  if (arg.res_flag == 0) {
    pmesh =
        std::make_unique<Mesh>(pinput.get(), app_input.get(), packages, arg.mesh_flag);
  } else {
    // Open restart file
    // Read Mesh from restart file and create meshblocks
    pmesh =
        std::make_unique<Mesh>(pinput.get(), app_input.get(), *restartReader, packages);

    // Read simulation time and cycle from restart file and set in input
    Real tNow = restartReader->GetAttr<Real>("Info", "Time");
    pinput->SetReal("parthenon/time", "start_time", tNow);

    Real dt = restartReader->GetAttr<Real>("Info", "dt");
    pinput->SetReal("parthenon/time", "dt", dt);

    int ncycle = restartReader->GetAttr<int>("Info", "NCycle");
    pinput->SetInteger("parthenon/time", "ncycle", ncycle);

    // Read package data from restart file
    RestartPackages(*pmesh, *restartReader);

    // close hdf5 file to prevent HDF5 hangs and corrupted files
    // if code dies after restart
    restartReader = nullptr;
  }

  // add root_level to all max_level
  for (auto const &ph : packages.AllPackages()) {
    for (auto &amr : ph.second->amr_criteria) {
      amr->max_level += pmesh->GetRootLevel();
    }
  }

  pmesh->Initialize(!IsRestart(), pinput.get(), app_input.get());

  ChangeRunDir(arg.prundir);
}

ParthenonStatus ParthenonManager::ParthenonFinalize() {
  pmesh.reset();
  Kokkos::finalize();
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
  return ParthenonStatus::complete;
}

Packages_t
ParthenonManager::ProcessPackagesDefault(std::unique_ptr<ParameterInput> &pin) {
  // In practice, this function should almost always be replaced by a version
  // that sets relevant things for the application.
  Packages_t packages;
  return packages;
}

void ParthenonManager::RestartPackages(Mesh &rm, RestartReader &resfile) {
#ifndef ENABLE_HDF5
  PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#else  // HDF5 enabled
  // Restart packages with information for blocks in ids from the restart file
  // Assumption: blocks are contiguous in restart file, may have to revisit this.
  const IndexDomain theDomain =
      (resfile.hasGhost ? IndexDomain::entire : IndexDomain::interior);
  // Get block list and temp array size
  auto &mb = *(rm.block_list.front());
  int nb = rm.GetNumMeshBlocksThisRank(Globals::my_rank);
  int nbs = mb.gid;
  int nbe = nbs + nb - 1;
  IndexRange myBlocks{nbs, nbe};

  // TODO(cleanup) why is this code here and not contained in the restart reader?
  std::cout << "Blocks assigned to rank " << Globals::my_rank << ": " << nbs << ":" << nbe
            << std::endl;

  const auto file_output_format_ver = resfile.GetOutputFormatVersion();
  if (file_output_format_ver == -1) {
    // Being extra stringent here so that we don't forget to update the machinery when
    // another change happens.
    PARTHENON_REQUIRE_THROWS(
        HDF5::OUTPUT_VERSION_FORMAT == 2 || HDF5::OUTPUT_VERSION_FORMAT == 3,
        "Auto conversion from original to format 2 or 3 not implemented yet.")

    if (Globals::my_rank == 0) {
      PARTHENON_WARN("Restarting from a old output file format. New outputs written with "
                     "this binary will use new format.")
    }
  }

  // Get an iterator on block 0 for variable listing
  IndexRange out_ib = mb.cellbounds.GetBoundsI(theDomain);
  IndexRange out_jb = mb.cellbounds.GetBoundsJ(theDomain);
  IndexRange out_kb = mb.cellbounds.GetBoundsK(theDomain);

  std::vector<size_t> bsize;
  bsize.push_back(out_ib.e - out_ib.s + 1);
  bsize.push_back(out_jb.e - out_jb.s + 1);
  bsize.push_back(out_kb.e - out_kb.s + 1);

  size_t nCells = bsize[0] * bsize[1] * bsize[2];

  // Get list of variables, they are the same for all blocks (since all blocks have the
  // same variable metadata)
  const auto indep_restart_vars =
      GetAnyVariables(mb.meshblock_data.Get()->GetVariableVector(),
                      {parthenon::Metadata::Independent, parthenon::Metadata::Restart});

  const auto sparse_info = resfile.GetSparseInfo();
  // create map of sparse field labels to index in the SparseInfo table
  std::unordered_map<std::string, int> sparse_idxs;
  for (int i = 0; i < sparse_info.num_sparse; ++i) {
    sparse_idxs.insert({sparse_info.labels[i], i});
  }

  // Allocate space based on largest vector
  int max_vlen = 1;
  int num_sparse = 0;
  for (auto &v_info : indep_restart_vars) {
    const auto &label = v_info->label();

    // check that variable is in the list of sparse fields if and only if it is sparse
    if (v_info->IsSparse()) {
      ++num_sparse;
      PARTHENON_REQUIRE_THROWS(sparse_idxs.count(label) == 1,
                               "Sparse field " + label +
                                   " is not marked as sparse in restart file");
    } else {
      PARTHENON_REQUIRE_THROWS(sparse_idxs.count(label) == 0,
                               "Dense field " + label +
                                   " is marked as sparse in restart file");
    }
    max_vlen = std::max(max_vlen, v_info->NumComponents());
  }

  // make sure we have all sparse variables that are in the restart file
  PARTHENON_REQUIRE_THROWS(
      num_sparse == sparse_info.num_sparse,
      "Mismatch between sparse fields in simulation and restart file");

  std::vector<Real> tmp(static_cast<size_t>(nb) * nCells * max_vlen);
  for (auto &v_info : indep_restart_vars) {
    const auto vlen = v_info->NumComponents();
    const auto &label = v_info->label();
    const auto &Nv = v_info->GetDim(4);
    const auto &Nu = v_info->GetDim(5);
    const auto &Nt = v_info->GetDim(6);

    if (Globals::my_rank == 0) {
      std::cout << "Var: " << label << ":" << vlen << std::endl;
    }
    // Read relevant data from the hdf file, this works for dense and sparse variables
    try {
      resfile.ReadBlocks(label, myBlocks, tmp, bsize, file_output_format_ver,
                         v_info->metadata().Where(), v_info->metadata().Shape());
    } catch (std::exception &ex) {
      std::cout << "[" << Globals::my_rank << "] WARNING: Failed to read variable "
                << label << " from restart file:" << std::endl
                << ex.what() << std::endl;
      continue;
    }

    size_t index = 0;
    for (auto &pmb : rm.block_list) {
      if (v_info->IsSparse()) {
        // check if the sparse variable is allocated on this block
        if (sparse_info.IsAllocated(pmb->gid, sparse_idxs.at(label))) {
          pmb->AllocateSparse(label);
        } else {
          // nothing to read for this block, advance reading index
          index += nCells * vlen;
          continue;
        }
      }

      auto v = pmb->meshblock_data.Get()->GetVarPtr(label);
      auto v_h = v->data.GetHostMirror();

      // Double note that this also needs to be update in case
      // we update the HDF5 infrastructure!
      if (file_output_format_ver == -1) {
        PARTHENON_WARN("This file output format version is deprecrated and will be "
                       "removed in a future release.");
        for (int k = out_kb.s; k <= out_kb.e; ++k) {
          for (int j = out_jb.s; j <= out_jb.e; ++j) {
            for (int i = out_ib.s; i <= out_ib.e; ++i) {
              for (int l = 0; l < vlen; ++l) {
                v_h(l, k, j, i) = tmp[index++];
              }
            }
          }
        }
      } else if (file_output_format_ver == 2 ||
                 file_output_format_ver == HDF5::OUTPUT_VERSION_FORMAT) {
        OutputUtils::PackOrUnpackVar(pmb.get(), v.get(), resfile.hasGhost, index, tmp,
                                     [&](auto index, int t, int u, int v, int k, int j,
                                         int i) { v_h(t, u, v, k, j, i) = tmp[index]; });
      } else {
        PARTHENON_THROW("Unknown output format version in restart file.")
      }

      v->data.DeepCopy(v_h);
    }
  }

  // Swarm data
  using FC = parthenon::Metadata::FlagCollection;
  auto flags = FC({parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
  auto swarms = (mb.meshblock_data.Get()->swarm_data.Get())->GetSwarmsByFlag(flags);
  for (auto &swarm : swarms) {
    auto swarmname = swarm->label();
    if (Globals::my_rank == 0) {
      std::cout << "Swarm: " << swarmname << std::endl;
    }
    std::vector<std::size_t> counts, offsets;
    std::size_t count_on_rank =
        resfile.GetSwarmCounts(swarmname, myBlocks, counts, offsets);
    // Compute total count and skip this swarm if total count is zero.
    std::size_t total_count = OutputUtils::MPISum(count_on_rank);
    if (total_count == 0) {
      continue;
    }
    std::size_t block_index = 0;
    // only want to do this once per block
    for (auto &pmb : rm.block_list) {
      auto pswarm_blk = (pmb->meshblock_data.Get()->swarm_data.Get())->Get(swarmname);
      pswarm_blk->AddEmptyParticles(counts[block_index]);
      block_index++;
    }
    ReadSwarmVars_<int>(swarm, rm.block_list, count_on_rank, offsets[0]);
    ReadSwarmVars_<Real>(swarm, rm.block_list, count_on_rank, offsets[0]);
  }

  // Params
  // ============================================================
  // packages and params are owned by shared pointer, so reading from
  // the mesh updates on all meshblocks.
  for (auto &[name, pkg] : rm.packages.AllPackages()) {
    auto &params = pkg->AllParams();
    resfile.ReadParams(name, params);
  }
#endif // ifdef ENABLE_HDF5
}

} // namespace parthenon
