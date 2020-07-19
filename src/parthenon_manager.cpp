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

#include "parthenon_manager.hpp"

#include <utility>

#include <Kokkos_Core.hpp>

#include "driver/driver.hpp"
#include "interface/update.hpp"
#include "refinement/refinement.hpp"

namespace parthenon {

ParthenonStatus ParthenonManager::ParthenonInit(int argc, char *argv[]) {
  // initialize MPI
#ifdef MPI_PARALLEL
#ifdef OPENMP_PARALLEL
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI Initialization failed." << std::endl;
    return ParthenonStatus::error;
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for the hybrid parallelzation. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv << std::endl;
    // MPI_Finalize();
    return ParthenonStatus::error;
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in ParthenonInit" << std::endl
              << "MPI Initialization failed." << std::endl;
    return ParthenonStatus::error;
  }
#endif // OPENMP_PARALLEL
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

  // parse the input arguments
  ArgStatus arg_status = arg.parse(argc, argv);
  if (arg_status == ArgStatus::error) {
    return ParthenonStatus::error;
  } else if (arg_status == ArgStatus::complete) {
    return ParthenonStatus::complete;
  }

  // Allow for user overrides to default Parthenon functions
  if (app_input->SetFillDerivedFunctions != nullptr) {
    SetFillDerivedFunctions = app_input->SetFillDerivedFunctions;
  }
  if (app_input->ProcessProperties != nullptr) {
    ProcessProperties = app_input->ProcessProperties;
  }
  if (app_input->ProcessPackages != nullptr) {
    ProcessPackages = app_input->ProcessPackages;
  }

  // Set up the signal handler
  SignalHandler::SignalHandlerInit();
  if (Globals::my_rank == 0 && arg.wtlim > 0) SignalHandler::SetWallTimeAlarm(arg.wtlim);

  // Populate the ParameterInput object
  if (arg.input_filename != nullptr) {
    pinput = std::make_unique<ParameterInput>(arg.input_filename);
  }
  pinput->ModifyFromCmdline(argc, argv);

  // read in/set up application specific properties
  auto properties = ProcessProperties(pinput);
  // set up all the packages in the application
  auto packages = ProcessPackages(pinput);
  // always add the Refinement package
  packages["ParthenonRefinement"] = Refinement::Initialize(pinput.get());

  // TODO(jdolence): Deal with restarts
  // if (arg.res_flag == 0) {
  pmesh = std::make_unique<Mesh>(pinput.get(), app_input.get(), properties, packages,
                                 arg.mesh_flag);
  //} else {
  //  pmesh = std::make_unique<Mesh>(pinput.get(), )
  //}

  // add root_level to all max_level
  for (auto const &ph : packages) {
    for (auto &amr : ph.second->amr_criteria) {
      amr->max_level += pmesh->GetRootLevel();
    }
  }

  SetFillDerivedFunctions();

  pmesh->Initialize(Restart(), pinput.get(), app_input.get());

  ChangeRunDir(arg.prundir);

  return ParthenonStatus::ok;
}

ParthenonStatus ParthenonManager::ParthenonFinalize() {
  pmesh.reset();
  Kokkos::finalize();
#ifdef MPI_PARALLEL
  MPI_Finalize();
#endif
  return ParthenonStatus::complete;
}

void ParthenonManager::SetFillDerivedFunctionsDefault() {
  FillDerivedVariables::SetFillDerivedFunctions(nullptr, nullptr);
}

Properties_t
ParthenonManager::ProcessPropertiesDefault(std::unique_ptr<ParameterInput> &pin) {
  // In practice, this function should almost always be replaced by a version
  // that sets relevant things for the application.
  Properties_t props;
  return props;
}

Packages_t
ParthenonManager::ProcessPackagesDefault(std::unique_ptr<ParameterInput> &pin) {
  // In practice, this function should almost always be replaced by a version
  // that sets relevant things for the application.
  Packages_t packages;
  return packages;
}

} // namespace parthenon
