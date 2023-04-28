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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>

#include "driver/driver.hpp"

#include "bvals/bnd_flx_communication/bvals_in_one.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "parthenon_mpi.hpp"
#include "utils/utils.hpp"

namespace parthenon {
using SignalHandler::OutputSignal;

// Declare class static variables
Kokkos::Timer Driver::timer_main;
Kokkos::Timer Driver::timer_cycle;
Kokkos::Timer Driver::timer_LBandAMR;

void Driver::PreExecute() {
  if (Globals::my_rank == 0) {
    std::cout << std::endl << "Setup complete, executing driver...\n" << std::endl;
  }

  timer_main.reset();
}

void Driver::PostExecute(DriverStatus status) {
  if (Globals::my_rank == 0) {
    SignalHandler::CancelWallTimeAlarm();
    // Calculate and print the zone-cycles/cpu-second and wall-second
    std::uint64_t zonecycles =
        pmesh->mbcnt * static_cast<std::uint64_t>(pmesh->GetNumberOfMeshBlockCells());

    auto wtime = timer_main.seconds();
    std::cout << std::endl << "walltime used = " << wtime << std::endl;
    std::cout << "zone-cycles/wallsecond = " << static_cast<double>(zonecycles) / wtime
              << std::endl;
  }
}

DriverStatus EvolutionDriver::Execute() {
  PreExecute();
  InitializeBlockTimeStepsAndBoundaries();
  SetGlobalTimeStep();
  OutputSignal signal = OutputSignal::none;
  pouts->MakeOutputs(pmesh, pinput, &tm, signal);
  pmesh->mbcnt = 0;
  int perf_cycle_offset =
      pinput->GetOrAddInteger("parthenon/time", "perf_cycle_offset", 0);

  // Output a text file of all parameters at this point
  // Defaults must be set across all ranks
  DumpInputParameters();

  Kokkos::Profiling::pushRegion("Driver_Main");
  while (tm.KeepGoing()) {
    if (Globals::my_rank == 0) OutputCycleDiagnostics();

    pmesh->PreStepUserWorkInLoop(pmesh, pinput, tm);
    pmesh->PreStepUserDiagnosticsInLoop(pmesh, pinput, tm);

    TaskListStatus status = Step();
    if (status != TaskListStatus::complete) {
      std::cerr << "Step failed to complete all tasks." << std::endl;
      return DriverStatus::failed;
    }

    pmesh->PostStepUserWorkInLoop(pmesh, pinput, tm);
    pmesh->PostStepUserDiagnosticsInLoop(pmesh, pinput, tm);

    tm.ncycle++;
    tm.time += tm.dt;
    pmesh->mbcnt += pmesh->nbtotal;
    pmesh->step_since_lb++;

    timer_LBandAMR.reset();
    pmesh->LoadBalancingAndAdaptiveMeshRefinement(pinput, app_input);
    if (pmesh->modified) InitializeBlockTimeStepsAndBoundaries();
    time_LBandAMR += timer_LBandAMR.seconds();
    SetGlobalTimeStep();

    // check for signals
    signal = SignalHandler::CheckSignalFlags();

    if (signal == OutputSignal::final) {
      break;
    }

    // skip the final (last) output at the end of the simulation time as it happens later
    if (tm.KeepGoing()) {
      pouts->MakeOutputs(pmesh, pinput, &tm, signal);
    }

    if (tm.ncycle == perf_cycle_offset) {
      pmesh->mbcnt = 0;
      timer_main.reset();
    }
  } // END OF MAIN INTEGRATION LOOP ======================================================
  Kokkos::Profiling::popRegion(); // Driver_Main

  pmesh->UserWorkAfterLoop(pmesh, pinput, tm);

  DriverStatus status = DriverStatus::complete;

  pouts->MakeOutputs(pmesh, pinput, &tm, OutputSignal::final);
  PostExecute(status);
  return status;
}

void EvolutionDriver::PostExecute(DriverStatus status) {
  // Print diagnostic messages related to the end of the simulation
  if (Globals::my_rank == 0) {
    OutputCycleDiagnostics();
    SignalHandler::Report();
    if (status == DriverStatus::complete) {
      std::cout << std::endl << "Driver completed." << std::endl;
    } else if (status == DriverStatus::timeout) {
      std::cout << std::endl << "Driver timed out.  Restart to continue." << std::endl;
    } else if (status == DriverStatus::failed) {
      std::cout << std::endl << "Driver failed." << std::endl;
    }

    std::cout << "time=" << tm.time << " cycle=" << tm.ncycle << std::endl;
    std::cout << "tlim=" << tm.tlim << " nlim=" << tm.nlim << std::endl;

    if (pmesh->adaptive) {
      std::cout << std::endl
                << "Number of MeshBlocks = " << pmesh->nbtotal << "; " << pmesh->nbnew
                << "  created, " << pmesh->nbdel << " destroyed during this simulation."
                << std::endl;
    }
  }
  Driver::PostExecute(status);
}

void EvolutionDriver::InitializeBlockTimeStepsAndBoundaries() {
  // calculate the first time step using Block function
  for (auto &pmb : pmesh->block_list) {
    Update::EstimateTimestep(pmb->meshblock_data.Get().get());
  }
  // calculate the first time step using Mesh function
  pmesh->boundary_comm_map.clear();
  pmesh->boundary_comm_flxcor_map.clear();
  const int num_partitions = pmesh->DefaultNumPartitions();
  for (int i = 0; i < num_partitions; i++) {
    auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
    Update::EstimateTimestep(mbase.get());
    var_boundary_comm::BuildBoundaryBuffers(mbase);
  }
}

//----------------------------------------------------------------------------------------
// \!fn void EvolutionDriver::SetGlobalTimeStep()
// \brief function that loops over all MeshBlocks and find new timestep

void EvolutionDriver::SetGlobalTimeStep() {
  // don't allow dt to grow by more than 2x
  // consider making this configurable in the input
  if (tm.dt < 0.1 * std::numeric_limits<Real>::max()) {
    tm.dt *= 2.0;
  }
  Real big = std::numeric_limits<Real>::max();
  for (auto const &pmb : pmesh->block_list) {
    tm.dt = std::min(tm.dt, pmb->NewDt());
    pmb->SetAllowedDt(big);
  }

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &tm.dt, 1, MPI_PARTHENON_REAL, MPI_MIN,
                                    MPI_COMM_WORLD));
#endif

  if (tm.time < tm.tlim &&
      (tm.tlim - tm.time) < tm.dt) // timestep would take us past desired endpoint
    tm.dt = tm.tlim - tm.time;
}

void EvolutionDriver::DumpInputParameters() {
  auto archive_settings =
      pinput->GetOrAddString("parthenon/job", "archive_parameters", "false",
                             std::vector<std::string>{"true", "false", "timestamp"});
  if (archive_settings != "false" && Globals::my_rank == 0) {
    std::ostringstream ss;
    if (archive_settings == "timestamp") {
      auto itt_now =
          std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      ss << "parthinput.archive." << std::put_time(std::gmtime(&itt_now), "%FT%TZ");
    } else {
      ss << "parthinput.archive";
    }
    std::fstream pars;
    pars.open(ss.str(), std::fstream::out | std::fstream::trunc);
    pinput->ParameterDump(pars);
    pars.close();
  }
}

void EvolutionDriver::OutputCycleDiagnostics() {
  const int dt_precision = std::numeric_limits<Real>::max_digits10 - 1;
  if (tm.ncycle_out != 0) {
    if (tm.ncycle % tm.ncycle_out == 0) {
      std::uint64_t zonecycles =
          (pmesh->mbcnt - mbcnt_prev) *
          static_cast<std::uint64_t>(pmesh->GetNumberOfMeshBlockCells());
      const auto time_cycle_all = timer_cycle.seconds();
      const auto time_cycle_step = time_cycle_all - time_LBandAMR;
      const auto wtime = timer_main.seconds();
      std::cout << "cycle=" << tm.ncycle << std::scientific
                << std::setprecision(dt_precision) << " time=" << tm.time
                << " dt=" << tm.dt << std::setprecision(2) << " zone-cycles/wsec_step="
                << static_cast<double>(zonecycles) / time_cycle_step
                << " wsec_total=" << wtime << " wsec_step=" << time_cycle_step;

      // In principle load balancing based on a cost list can happens for non-AMR runs.
      // TODO(future me) fix this when this becomes important.
      if (pmesh->adaptive) {
        std::cout << " zone-cycles/wsec="
                  << static_cast<double>(zonecycles) / (time_cycle_step + time_LBandAMR)
                  << " wsec_AMR=" << time_LBandAMR;
      }

      // insert more diagnostics here
      std::cout << std::endl;

      // reset cycle related counters
      timer_cycle.reset();
      time_LBandAMR = 0.0;
      // need to cache number of MeshBlocks as AMR/load balance change it
      mbcnt_prev = pmesh->mbcnt;
    }
  }
  if (tm.ncycle_out_mesh != 0) {
    // output after mesh refinement (enabled by use of negative cycle number)
    if (tm.ncycle_out_mesh < 0 && pmesh->modified) {
      std::cout << "-------------- New Mesh structure after (de)refinement -------------";
      pmesh->OutputMeshStructure(-1, false);
      std::cout << "--------------------------------------------------------------------"
                << std::endl;
      // output in fixed intervals
    } else if (tm.ncycle % tm.ncycle_out_mesh == 0) {
      std::cout << "---------------------- Current Mesh structure ----------------------";
      pmesh->OutputMeshStructure(-1, false);
      std::cout << "--------------------------------------------------------------------"
                << std::endl;
    }
  }
}

} // namespace parthenon
