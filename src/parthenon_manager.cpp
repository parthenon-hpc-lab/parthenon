//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "refinement/refinement.hpp"

namespace parthenon {

ParthenonStatus ParthenonManager::ParthenonInit(int argc, char *argv[]) {
  auto manager_status = ParthenonInitEnv(argc, argv);
  if (manager_status != ParthenonStatus::ok) {
    return manager_status;
  }
  ParthenonInitPackagesAndMesh();
  return ParthenonStatus::ok;
}

ParthenonStatus ParthenonManager::ParthenonInitEnv(int argc, char *argv[]) {
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
  auto *env_track_alloc = std::getenv("KOKKOS_TRACK_ALLOC_OFF");
  if (env_track_alloc != nullptr) {
    std::string env_str(env_track_alloc); // deep-copies string
    for (char &c : env_str) {
      c = toupper(c);
    }
    if ((env_str == "TRUE") || (env_str == "ON") || (env_str == "1")) {
      Kokkos::Profiling::Experimental::set_allocate_data_callback(nullptr);
      Kokkos::Profiling::Experimental::set_deallocate_data_callback(nullptr);
    }
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

  return ParthenonStatus::ok;
}

void ParthenonManager::ParthenonInitPackagesAndMesh() {
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
    pinput->SetPrecise("parthenon/time", "start_time", tNow);

    Real dt = restartReader->GetAttr<Real>("Info", "dt");
    pinput->SetPrecise("parthenon/time", "dt", dt);

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
  // Restart packages with information for blocks in ids from the restart file
  // Assumption: blocks are contiguous in restart file, may have to revisit this.
  //  const IndexDomain interior = IndexDomain::interior;
  const IndexDomain theDomain =
      (resfile.hasGhost ? IndexDomain::entire : IndexDomain::interior);
  // Get block list and temp array size
  auto &mb = *(rm.block_list.front());
  int nb = rm.GetNumMeshBlocksThisRank(Globals::my_rank);
  int nbs = mb.gid;
  int nbe = nbs + nb - 1;
  IndexRange myBlocks{nbs, nbe};

  std::cout << "Blocks assigned to rank:" << Globals::my_rank << ":" << nbs << ":" << nbe
            << std::endl;
  // Get an iterator on block 0 for variable listing
  IndexRange out_ib = mb.cellbounds.GetBoundsI(theDomain);
  IndexRange out_jb = mb.cellbounds.GetBoundsJ(theDomain);
  IndexRange out_kb = mb.cellbounds.GetBoundsK(theDomain);

  std::vector<size_t> bsize;
  bsize.push_back(out_ib.e - out_ib.s + 1);
  bsize.push_back(out_jb.e - out_jb.s + 1);
  bsize.push_back(out_kb.e - out_kb.s + 1);

  size_t nCells = bsize[0] * bsize[1] * bsize[2];

  // Get list of variables, assumed same for all blocks

  // TODO(JL) this won't work when reading true sparse variables, will be updated in PR
  // #383
  const auto indep_restart_vars =
      mb.meshblock_data.Get()
          ->GetVariablesByFlag(
              {parthenon::Metadata::Independent, parthenon::Metadata::Restart}, false)
          .vars();

  // Allocate space based on largest vector
  size_t vlen = 1;
  for (auto &v : indep_restart_vars) {
    if (v->GetDim(4) > vlen) {
      vlen = v->GetDim(4);
    }
  }
  std::vector<Real> tmp(static_cast<size_t>(nb) * nCells * vlen);
  for (auto &v : indep_restart_vars) {
    const size_t v4 = v->GetDim(4);
    const std::string vName = v->label();

    if (Globals::my_rank == 0) std::cout << "Var:" << vName << ":" << v4 << std::endl;
    // Read relevant data from the hdf file
    int stat = resfile.ReadBlocks(vName.c_str(), myBlocks, tmp, bsize, v4);
    if (stat < 0) {
      std::cout << " WARNING: Variable " << v->label() << " Not found in restart file";
      continue;
    }

    size_t index = 0;
    for (auto &pmb : rm.block_list) {
      bool found = false;
      const auto this_indep_restart_vars =
          pmb->meshblock_data.Get()
              ->GetVariablesByFlag(
                  {parthenon::Metadata::Independent, parthenon::Metadata::Restart}, false)
              .vars();
      for (auto &v : this_indep_restart_vars) {
        if (vName.compare(v->label()) == 0) {
          auto v_h = v->data.GetHostMirror();

          // Note index l transposed to interior
          for (int k = out_kb.s; k <= out_kb.e; ++k) {
            for (int j = out_jb.s; j <= out_jb.e; ++j) {
              for (int i = out_ib.s; i <= out_ib.e; ++i) {
                for (int l = 0; l < v_h.GetDim(4); ++l) {
                  v_h(l, k, j, i) = tmp[index++];
                }
              }
            }
          }

          v->data.DeepCopy(v_h);
          found = true;
          break;
        }
      }
      if (!found) {
        std::stringstream msg;
        msg << "### ERROR: Unable to find variable " << vName << std::endl;
        PARTHENON_FAIL(msg);
      }
    }
  }
}

} // namespace parthenon
