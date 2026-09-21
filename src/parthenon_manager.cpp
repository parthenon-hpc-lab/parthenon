//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

#include "parthenon_manager.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "amr_criteria/amr_criteria.hpp"
#include "amr_criteria/refinement_package.hpp"
#include "coordinates/coordinates.hpp"
#include "config.hpp"
#include FS_HEADER
#include "globals.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs_package.hpp"
#include "outputs/restart.hpp"
#include "outputs/restart_hdf5.hpp"
#ifdef PARTHENON_ENABLE_OPENPMD
#include "outputs/restart_opmd.hpp"
#endif
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

  Globals::watchdog_enabled = arg.watchdog_enabled;
  if (Globals::watchdog_enabled) {
    parthenon::WatchDog::WatchDog(arg.watchdog_timeout);
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
    const auto extension = fs::path(arg.restart_filename).extension();
    if (extension == ".rhdf" || extension == ".phdf" || extension == ".hdf5" ||
        extension == ".h5") {
#ifdef ENABLE_HDF5
      restartReader = std::make_unique<RestartReaderHDF5>(arg.restart_filename);
#else // HDF5 disabled
      PARTHENON_FAIL("Restart functionality is not available because HDF5 is disabled");
#endif
    } else if (fs::path(arg.restart_filename).extension() == ".bp" ||
               // allow user input for /path/to/restart.bp/ (with trailing `/`), e.g.,
               // from autocomplete
               (fs::is_directory(arg.restart_filename) &&
                fs::path(arg.restart_filename).parent_path().extension() == ".bp")) {
#ifdef PARTHENON_ENABLE_OPENPMD
      restartReader = std::make_unique<RestartReaderOPMD>(arg.restart_filename);
#else
      PARTHENON_FAIL("Trying to restart from OpenPMD file but OpenPMD support was not "
                     "compiled into Parthenon.");
#endif // ifdef PARTHENON_ENABLE_OPENPMD
    } else {
      PARTHENON_FAIL("Unsupported restart file format.");
    }

    if (arg.analysis_flag) {
      const auto output_mode = restartReader->GetOutputMode();
      if (output_mode == RestartReader::OutputMode::data ||
          output_mode == RestartReader::OutputMode::core) {
        analysis_data_ = true;
      } else if (output_mode == RestartReader::OutputMode::slice) {
        PARTHENON_FAIL("Analysis loading from sliced HDF5 outputs is not supported.");
      } else if (output_mode == RestartReader::OutputMode::unknown &&
                 extension != ".rhdf") {
        analysis_data_ = true;
        if (Globals::my_rank == 0) {
          PARTHENON_WARN("HDF5 file does not contain OutputMode metadata; treating the "
                         "non-.rhdf analysis input as a data dump.");
        }
      }
      PARTHENON_REQUIRE_THROWS(
          !analysis_data_ || arg.input_filename != nullptr,
          "Analysis loading from an HDF5 data dump requires '-i <analysis input file>'.");
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

  // Finalize parsing phase - parsers can no longer add parameters
  pinput->FinalizeParsing();

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
    if (analysis_data_) {
      const auto mesh_info = restartReader->GetMeshInfo();
      const auto input_nx2 = pinput->DoesParameterExist("parthenon/mesh", "nx2")
                                 ? pinput->GetInteger("parthenon/mesh", "nx2")
                                 : 1;
      const auto input_nx3 = pinput->DoesParameterExist("parthenon/mesh", "nx3")
                                 ? pinput->GetInteger("parthenon/mesh", "nx3")
                                 : 1;
      const int input_ndim = 1 + (input_nx2 > 1) + (input_nx3 > 1);
      PARTHENON_REQUIRE_THROWS(
          mesh_info.ndim == input_ndim,
          "Analysis mesh dimensionality mismatch: file has " +
              std::to_string(mesh_info.ndim) + " dimensions, input requests " +
              std::to_string(input_ndim) + ".");
      PARTHENON_REQUIRE_THROWS(
          mesh_info.coordinates == Coordinates_t::name_,
          "Analysis coordinate-system mismatch: file uses '" + mesh_info.coordinates +
              "', executable/input uses '" + Coordinates_t::name_ + "'.");
      PARTHENON_REQUIRE_THROWS(
          mesh_info.n_ghost == Globals::nghost,
          "Analysis ghost-width mismatch: file was written with nghost=" +
              std::to_string(mesh_info.n_ghost) + ", input requests nghost=" +
              std::to_string(Globals::nghost) + ".");
    }
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

    // Restart data must be present before the normal restart initialization path.
    if (!analysis_data_) RestartPackages(*pmesh, *restartReader);

    // close hdf5 file to prevent HDF5 hangs and corrupted files
    // if code dies after restart
    if (!analysis_data_) restartReader = nullptr;
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

  if (analysis_data_) {
    RestartPackages(*pmesh, *restartReader, true);
    pmesh->CommunicateBoundariesForFields(loaded_analysis_fields_);
    if (Globals::my_rank == 0) {
      std::cout << "Analysis load ghost-filled (" << loaded_analysis_fields_.size()
                << "):";
      for (const auto &name : loaded_analysis_fields_) std::cout << " " << name;
      std::cout << std::endl;
    }
    restartReader = nullptr;
  }

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

void ParthenonManager::RestartPackages(Mesh &rm, RestartReader &resfile,
                                       bool analysis_data) {
  // Restart packages with information for blocks in ids from the restart file
  // Assumption: blocks are contiguous in restart file, may have to revisit this.
  const IndexDomain theDomain = analysis_data
                                    ? IndexDomain::interior
                                    : (resfile.HasGhost() != 0 ? IndexDomain::entire
                                                               : IndexDomain::interior);
  // Get block list and temp array size
  auto &mb = *(rm.block_list.front());
  int nb = rm.GetNumMeshBlocksThisRank(Globals::my_rank);
  int nbs = mb.gid;
  int nbe = nbs + nb - 1;
  IndexRange myBlocks{nbs, nbe};

  std::cout << "Blocks assigned to rank " << Globals::my_rank << ": " << nbs << ":" << nbe
            << std::endl;

  // Get list of variables, they are the same for all blocks (since all blocks have the
  // same variable metadata)
  auto selected_vars = analysis_data
                           ? mb.meshblock_data.Get()->GetVariableVector()
                           : GetAnyVariables(
                                 mb.meshblock_data.Get()->GetVariableVector(),
                                 {parthenon::Metadata::Independent,
                                  parthenon::Metadata::Restart});
  std::set<std::string> file_fields;
  std::set<std::string> registered_fields;
  if (analysis_data) {
    const auto names = resfile.GetFieldNames();
    file_fields.insert(names.begin(), names.end());
    VariableVector<Real> intersection;
    for (const auto &var : selected_vars) {
      registered_fields.insert(var->label());
      if (file_fields.count(var->label()) != 0) intersection.push_back(var);
    }
    selected_vars = std::move(intersection);
  }
  const auto all_vars_info =
      OutputUtils::VarInfo::GetAll(selected_vars, mb.cellbounds, mb.f_cellbounds);

  const auto sparse_info = resfile.GetSparseInfo();
  // create map of sparse field labels to index in the SparseInfo table
  std::unordered_map<std::string, int> sparse_idxs;
  for (int i = 0; i < sparse_info.num_sparse; ++i) {
    sparse_idxs.insert({sparse_info.labels[i], i});
  }

  // Allocate space based on largest vector
  int num_sparse = 0;
  std::size_t max_fillsize = 1;
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
        std::max(max_fillsize, v_info.FillSize(theDomain, resfile.BlockdataIsPadded()));
  }

  // make sure we have all sparse variables that are in the restart file

  // JMM: It is possible to output with more sparse variables than you
  // need, for example if you're outputting a core dump, so we
  // complain only if the number of sparse varaibles required is
  // greater than the number in the file, not if it is less.
  if (!analysis_data) {
    PARTHENON_REQUIRE_THROWS(
        num_sparse <= sparse_info.num_sparse,
        "Mismatch between sparse fields in simulation and restart file");
  }
  std::vector<Real> tmp(static_cast<std::size_t>(nb) * max_fillsize);
  for (const auto &v_info : all_vars_info) {
    const auto vlen = v_info.num_components * v_info.ntop_elems;
    const auto fill_size = v_info.FillSize(theDomain, resfile.BlockdataIsPadded());
    const auto &label = v_info.label;

    auto var_missing_on_disk =
        !resfile.VariableExists(label, RestartReader::DataType::Field);
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
      resfile.ReadBlocks(label, myBlocks, v_info, tmp, &rm, analysis_data);
      // Variable does exist but could not be read. So we definitely want to fail here.
    } catch (std::exception &ex) {
      std::stringstream msg;
      msg << "[" << Globals::my_rank << "] WARNING: Failed to read variable " << label
          << " from restart file:" << std::endl
          << ex.what() << std::endl;
      PARTHENON_THROW(msg);
    }

    std::size_t index = 0;
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
      // we update the OpenPMD/HDF5 infrastructure!
      OutputUtils::PackOrUnpackVar(
          v_info, !analysis_data && resfile.HasGhost() != 0,
          resfile.BlockdataIsPadded(), index,
          [&](auto index, int topo, int t, int u, int v, int k, int j, int i) {
            v_h(topo, t, u, v, k, j, i) = tmp[index];
          });

      v->data.DeepCopy(v_h);
    }
    if (analysis_data) loaded_analysis_fields_.push_back(label);
  }

  if (analysis_data) {
    std::vector<std::string> missing, ignored;
    for (const auto &name : registered_fields) {
      if (file_fields.count(name) == 0) missing.push_back(name);
    }
    for (const auto &name : file_fields) {
      if (registered_fields.count(name) == 0) ignored.push_back(name);
    }
    if (Globals::my_rank == 0) {
      auto print_names = [](const char *label, const std::vector<std::string> &names) {
        std::cout << "Analysis load " << label << " (" << names.size() << "):";
        for (const auto &name : names) std::cout << " " << name;
        std::cout << std::endl;
      };
      print_names("loaded", loaded_analysis_fields_);
      print_names("missing", missing);
      print_names("ignored", ignored);
    }
    return;
  }

  // Swarm data
  using FC = parthenon::Metadata::FlagCollection;
  auto flags = FC({parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
  auto swarms = (mb.meshblock_data.Get()->GetSwarmData())->GetSwarmsByFlag(flags);
  for (auto &swarm : swarms) {
    auto swarmname = swarm->label();
    auto var_missing_on_disk =
        !resfile.VariableExists(swarmname, RestartReader::DataType::Swarm);
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
}

} // namespace parthenon
