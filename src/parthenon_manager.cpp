//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2025 The Parthenon collaboration
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

#include "parthenon_manager.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "amr_criteria/amr_criteria.hpp"
#include "amr_criteria/refinement_package.hpp"
#include "config.hpp"
#include FS_HEADER
#include "globals.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs_package.hpp"
#include "outputs/restart.hpp"
#include "outputs/restart_hdf5.hpp"
#include "utils/error_checking.hpp"
#include "utils/utils.hpp"

namespace fs = FS_NAMESPACE;

namespace parthenon {

ParthenonStatus ParthenonManager::ParthenonInitEnv(int argc, char *argv[]) {
  if (called_init_env_) {
    PARTHENON_THROW("ParthenonInitEnv called twice!");
  }
  called_init_env_ = true;

  // initialize MPI
#ifdef MPI_PARALLEL
  int mpi_initialized;
  PARTHENON_MPI_CHECK(MPI_Initialized(&mpi_initialized));
  if (!mpi_initialized && (MPI_SUCCESS != MPI_Init(&argc, &argv))) {
    std::cerr << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI Initialization failed." << std::endl;
    return ParthenonStatus::error;
  }
  // Get process id (rank) in MPI_COMM_WORLD
  PARTHENON_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &(Globals::my_rank)));

  // Get total number of MPI processes (ranks)
  PARTHENON_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &Globals::nranks));
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
  // Now that the input is parsed we can pass the info to globals
  Globals::is_restart = arg.is_restart;

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && arg.wtlim > 0) SignalHandler::SetWallTimeAlarm(arg.wtlim);

  // Populate the ParameterInput object.
  // If restart, then ParameterInput in the restart file takes precedence.
  if (arg.is_restart) {
    // Read input from restart file
    if (fs::path(arg.restart_filename).extension() == ".rhdf") {
      restartReader = std::make_unique<RestartReaderHDF5>(arg.restart_filename);
    } else {
      PARTHENON_FAIL("Unsupported restart file format.");
    }

    // Load input stream
    pinput = std::make_unique<ParameterInput>();
    auto inputString = restartReader->GetInputString();
    std::istringstream is(inputString);
    pinput->LoadFromStream(is);
  }
  // If an input file was provided
  if (arg.input_filename != nullptr) {
    // Modify info read from restart file
    if (arg.is_restart) {
      IOWrapper infile;
      infile.Open(arg.input_filename, IOWrapper::FileMode::read);
      pinput->LoadFromFile(infile);
      infile.Close();

      // Populate new object for fresh simulation
    } else {
      pinput = std::make_unique<ParameterInput>(arg.input_filename);
    }
  }

  // Modify based on command line inputs
  pinput->ModifyFromCmdline(argc, argv);

  PARTHENON_REQUIRE_THROWS(
      !pinput->DoesParameterExist("parthenon/job", "run_only_analysis") ||
          pinput->GetBoolean("parthenon/job", "run_only_analysis") == false,
      "'parthenon/job/run_only_analysis=true' input parameter was found indicating "
      "manual modification or restarting from an output written during analysis, which "
      "is undefined behavior. If you don't know how this was triggered, please contact "
      "the Parthenon developers.");
  pinput->SetBoolean("parthenon/job", "run_only_analysis", arg.analysis_flag);

  // Set the global number of ghost zones
  Globals::nghost = pinput->GetOrAddInteger("parthenon/mesh", "nghost", 2,
                                            "number of ghost zones on a block");

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
      pinput->GetOrAddInteger("parthenon/mesh", "refinement_in_one_min_nbufs", 64);

  return ParthenonStatus::ok;
}

void ParthenonManager::ParthenonInitPackagesAndMesh(
    std::optional<forest::ForestDefinition> forest_def) {
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
  packages.Add(OutputsPackage::Initialize(pinput.get()));
  if (forest_def) {
    pmesh = std::make_unique<Mesh>(pinput.get(), app_input.get(), packages, *forest_def);
  } else if (!arg.is_restart) {
    pmesh =
        std::make_unique<Mesh>(pinput.get(), app_input.get(), packages, arg.mesh_flag);
  } else {
    // Open restart file
    // Read Mesh from restart file and create meshblocks
    pmesh =
        std::make_unique<Mesh>(pinput.get(), app_input.get(), *restartReader, packages);

    // Read simulation time and cycle from restart file and set in input
    const auto time_info = restartReader->GetTimeInfo();
    Real tNow = time_info.time;
    pinput->SetReal("parthenon/time", "start_time", tNow);

    Real dt = time_info.dt;
    pinput->SetReal("parthenon/time", "dt", dt);

    int ncycle = time_info.ncycle;
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

  if (arg.mesh_flag) {
    ParthenonFinalize();
    exit(0);
  }

  if (arg.param_flag) {
    pinput->SetBoolean("parthenon/job", "output_params_and_exit", true);
    pinput->SetString("parthenon/job", "output_params_block_regex", arg.params_regex);
  }

  pmesh->Initialize(!arg.is_restart, pinput.get(), app_input.get());

  ChangeRunDir(arg.prundir);
}

ParthenonStatus ParthenonManager::ParthenonFinalize() {
  pmesh.reset();
  Kokkos::finalize();
#ifdef MPI_PARALLEL
  int mpi_finalized;
  PARTHENON_MPI_CHECK(MPI_Finalized(&mpi_finalized));
  if (!mpi_finalized) PARTHENON_MPI_CHECK(MPI_Finalize());
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
      (resfile.HasGhost() != 0 ? IndexDomain::entire : IndexDomain::interior);
  // Get block list and temp array size
  auto &mb = *(rm.block_list.front());
  int nb = rm.GetNumMeshBlocksThisRank(Globals::my_rank);
  int nbs = mb.gid;
  int nbe = nbs + nb - 1;
  IndexRange myBlocks{nbs, nbe};

  // TODO(cleanup) why is this code here and not contained in the restart reader?
  std::cout << "Blocks assigned to rank " << Globals::my_rank << ": " << nbs << ":" << nbe
            << std::endl;

  // Currently supports versions 3 and 4.
  const auto file_output_format_ver = resfile.GetOutputFormatVersion();
  if (file_output_format_ver < HDF5::OUTPUT_VERSION_FORMAT - 1) {
    std::stringstream msg;
    msg << "File format version " << file_output_format_ver << " not supported. "
        << "Current format is " << HDF5::OUTPUT_VERSION_FORMAT << std::endl;
    PARTHENON_THROW(msg)
  }

  // Get list of variables, they are the same for all blocks (since all blocks have the
  // same variable metadata)
  const auto indep_restart_vars =
      GetAnyVariables(mb.meshblock_data.Get()->GetVariableVector(),
                      {parthenon::Metadata::Independent, parthenon::Metadata::Restart});
  const auto all_vars_info =
      OutputUtils::VarInfo::GetAll(indep_restart_vars, mb.cellbounds, mb.f_cellbounds);

  const auto sparse_info = resfile.GetSparseInfo();
  // create map of sparse field labels to index in the SparseInfo table
  std::unordered_map<std::string, int> sparse_idxs;
  for (int i = 0; i < sparse_info.num_sparse; ++i) {
    sparse_idxs.insert({sparse_info.labels[i], i});
  }

  // Allocate space based on largest vector
  int num_sparse = 0;
  size_t max_fillsize = 1;
  for (const auto &v_info : all_vars_info) {
    const auto &label = v_info.label;

    // check that variable is in the list of sparse fields if and only if it is sparse
    if (v_info.is_sparse) {
      ++num_sparse;
      PARTHENON_REQUIRE_THROWS(sparse_idxs.count(label) == 1,
                               "Sparse field " + label +
                                   " is not marked as sparse in restart file");
    } else {
      PARTHENON_REQUIRE_THROWS(sparse_idxs.count(label) == 0,
                               "Dense field " + label +
                                   " is marked as sparse in restart file");
    }

    max_fillsize =
        std::max(max_fillsize, static_cast<size_t>(v_info.FillSize(theDomain)));
  }

  // make sure we have all sparse variables that are in the restart file

  // JMM: It is possible to output with more sparse variables than you
  // need, for example if you're outputting a core dump, so we
  // complain only if the number of sparse varaibles required is
  // greater than the number in the file, not if it is less.
  PARTHENON_REQUIRE_THROWS(
      num_sparse <= sparse_info.num_sparse,
      "Mismatch between sparse fields in simulation and restart file");
  std::vector<Real> tmp(static_cast<size_t>(nb) * max_fillsize);
  for (const auto &v_info : all_vars_info) {
    const auto vlen = v_info.num_components * v_info.ntop_elems;
    const auto fill_size = v_info.FillSize(theDomain);
    const auto &label = v_info.label;

    auto var_missing_on_disk = !resfile.VariableExists(label);
    if (Globals::my_rank == 0) {
      std::cout << "Var: " << label << ":" << vlen
                << (var_missing_on_disk ? " missing on disk\n" : "\n");
    }
    if (var_missing_on_disk) {
      // TODO(JMM/PG) Add failed load list of "fail/needs fix" list
      continue;
    }
    // Read relevant data from the hdf file, this works for dense and sparse variables
    // because sparse variables are currently densely written for HDF5.
    try {
      resfile.ReadBlocks(label, myBlocks, v_info, tmp, file_output_format_ver);
      // Variable does exist but could not be read. So we definitely want to fail here.
    } catch (std::exception &ex) {
      std::stringstream msg;
      msg << "[" << Globals::my_rank << "] WARNING: Failed to read variable " << label
          << " from restart file:" << std::endl
          << ex.what() << std::endl;
      PARTHENON_THROW(msg);
    }

    size_t index = 0;
    for (auto &pmb : rm.block_list) {
      if (v_info.is_sparse) {
        // check if the sparse variable is allocated on this block
        if (sparse_info.IsAllocated(pmb->gid, sparse_idxs.at(label))) {
          pmb->AllocateSparse(label);
          auto dealloc_count = sparse_info.DeallocCount(pmb->gid, sparse_idxs.at(label));
          // Warning: For this to work, it is required that the controlling variable is
          // stored in the restart files.
          pmb->meshblock_data.Get()->GetVarPtr(label)->dealloc_count = dealloc_count;
        } else {
          // nothing to read for this block, advance reading index
          index += fill_size;
          continue;
        }
      }

      auto v = pmb->meshblock_data.Get()->GetVarPtr(label);
      auto v_h = v->data.GetHostMirror();

      // Double note that this also needs to be update in case
      // we update the HDF5 infrastructure!
      if (file_output_format_ver >= HDF5::OUTPUT_VERSION_FORMAT - 1) {
        OutputUtils::PackOrUnpackVar(
            v_info, resfile.HasGhost() != 0, index,
            [&](auto index, int topo, int t, int u, int v, int k, int j, int i) {
              v_h(topo, t, u, v, k, j, i) = tmp[index];
            });
      } else {
        std::stringstream msg;
        msg << "File format version " << file_output_format_ver << " not supported. "
            << "Current format is " << HDF5::OUTPUT_VERSION_FORMAT << std::endl;
        PARTHENON_THROW(msg)
      }

      v->data.DeepCopy(v_h);
    }
  }

  // Swarm data
  using FC = parthenon::Metadata::FlagCollection;
  auto flags = FC({parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
  auto swarms = (mb.meshblock_data.Get()->GetSwarmData())->GetSwarmsByFlag(flags);
  for (auto &swarm : swarms) {
    auto swarmname = swarm->label();
    auto var_missing_on_disk = !resfile.VariableExists(swarmname);
    if (Globals::my_rank == 0) {
      std::cout << "Swarm: " << swarmname
                << (var_missing_on_disk ? " missing on disk\n" : "\n");
    }
    if (var_missing_on_disk) {
      // TODO(JMM/PG) Add failed load list of "fail/needs fix" list
      continue;
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
      auto pswarm_blk = (pmb->meshblock_data.Get()->GetSwarmData())->Get(swarmname);
      pswarm_blk->AddEmptyParticles(counts[block_index]);
      block_index++;
    }
    ReadSwarmVars_<int>(swarm, rm.block_list, count_on_rank, offsets[0]);
    ReadSwarmVars_<std::uint64_t>(swarm, rm.block_list, count_on_rank, offsets[0]);
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
