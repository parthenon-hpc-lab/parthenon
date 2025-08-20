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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "driver/driver.hpp"

#include "bvals/comms/bvals_in_one.hpp"
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

void Driver::DumpInputParameters() {
  auto archive_parameters =
      pinput->GetOrAddBoolean("parthenon/job", "archive_parameters", false);
  auto archive_timestamp =
      pinput->GetOrAddBoolean("parthenon/job", "archive_timestamp", false);
  if (archive_parameters && Globals::my_rank == 0) {
    std::ostringstream ss;
    if (archive_timestamp) {
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
  auto print_parameters =
      pinput->GetOrAddBoolean("parthenon/job", "print_parameters", false);
  if (print_parameters && Globals::my_rank == 0) {
    pinput->ParameterDump(std::cout);
  }
}

void Driver::PreExecute() {
  std::string check_orphans = pinput->GetOrAddString(
      "parthenon/job", "check_orphans", "initially",
      std::vector<std::string>{"always", "initially", "never"},
      "Print a warning if any parameters are in the input deck but not used in the code. "
      "By default this check is performed for new runs, but can also be enabled for "
      "restarts or completely disabled.");
  // Output a text file of all parameters at this point
  // Optionally also dump to console
  DumpInputParameters();

  if (Globals::my_rank == 0) {
    if ((check_orphans == "always") ||
        (!Globals::is_restart && (check_orphans == "initially"))) {
      pinput->CheckOrphans();
    }
    std::cout << "# Variables in use:\n" << *(pmesh->resolved_packages) << std::endl;
    std::cout << std::endl;
    std::cout << "Setup complete, executing driver...\n" << std::endl;
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
  // JMM: I want these before output_params_and_exit so we capture as much as possible
  OutputSignal signal = pinput->GetBoolean("parthenon/job", "run_only_analysis")
                            ? OutputSignal::analysis
                            : OutputSignal::none;
  int perf_cycle_offset = pinput->GetOrAddInteger(
      "parthenon/time", "perf_cycle_offset", 0,
      "don't measure performance for some number of initial cycles");
  const bool output_params_and_exit = pinput->GetOrAddBoolean(
      "parthenon/job", "output_params_and_exit", false,
      "output a description of all input parameters accessed and quit");
  const std::string params_block_regex =
      pinput->GetOrAddString("parthenon/job", "output_params_block_regex", "(.*)",
                             "when outputting input parameters, this selects which input "
                             "blocks to output; all are output by default");
  if (output_params_and_exit && Globals::my_rank == 0) {
    pinput->OutputParameterTable(std::cout, std::regex(params_block_regex));
    return DriverStatus::complete;
  }

  PreExecute();
  InitializeBlockTimeSteps();
  SetGlobalTimeStep();

  // Before loop do work
  { // UserWorkBeforeLoop
    PARTHENON_INSTRUMENT
    // App input version
    if (app_input->UserWorkBeforeLoop != nullptr) {
      app_input->UserWorkBeforeLoop(pmesh, pinput, tm);
    }
    // packages version
    for (auto &[name, pkg] : pmesh->packages.AllPackages()) {
      pkg->UserWorkBeforeLoop(pmesh, pinput, tm);
    }
  } // UserWorkBeforeLoop

  pouts->MakeOutputs(pmesh, pinput, &tm, signal);
  pmesh->mbcnt = 0;

  { // Main t < tmax loop region
    PARTHENON_INSTRUMENT
    while (tm.KeepGoing() && signal != OutputSignal::analysis) {
      if (Globals::my_rank == 0) OutputCycleDiagnostics();

      if (pmesh->PreStepUserWorkInLoop != nullptr) {
        pmesh->PreStepUserWorkInLoop(pmesh, pinput, tm);
      }
      if (pmesh->PreStepUserDiagnosticsInLoop != nullptr) {
        pmesh->PreStepUserDiagnosticsInLoop(pmesh, pinput, tm);
      }

      TaskListStatus status = Step();
      if (status != TaskListStatus::complete) {
        std::cerr << "Step failed to complete all tasks." << std::endl;
        return DriverStatus::failed;
      }

      if (pmesh->PostStepUserWorkInLoop != nullptr) {
        pmesh->PostStepUserWorkInLoop(pmesh, pinput, tm);
      }
      if (pmesh->PostStepUserDiagnosticsInLoop != nullptr) {
        pmesh->PostStepUserDiagnosticsInLoop(pmesh, pinput, tm);
      }

      tm.ncycle++;
      tm.time += tm.dt;
      pmesh->mbcnt += pmesh->nbtotal;
      pmesh->step_since_lb++;

      // skip the final (last) output at the end of the simulation time as it happens
      // later
      if (output_before_amr && tm.KeepGoing()) {
        pouts->MakeOutputs(pmesh, pinput, &tm, signal);
      }

      timer_LBandAMR.reset();
      pmesh->LoadBalancingAndAdaptiveMeshRefinement(pinput, app_input);
      if (pmesh->modified) InitializeBlockTimeSteps();
      time_LBandAMR += timer_LBandAMR.seconds();
      SetGlobalTimeStep();

      // check for signals
      signal = SignalHandler::CheckSignalFlags();

      if (signal == OutputSignal::final) {
        break;
      }

      // skip the final (last) output at the end of the simulation time as it happens
      // later
      if (!output_before_amr && tm.KeepGoing()) {
        pouts->MakeOutputs(pmesh, pinput, &tm, signal);
      }

      if (tm.ncycle == perf_cycle_offset) {
        pmesh->mbcnt = 0;
        timer_main.reset();
      }
    } // END OF MAIN INTEGRATION LOOP
      // ======================================================
  }   // Main t < tmax loop region

  if (pmesh->UserWorkAfterLoop != nullptr) {
    pmesh->UserWorkAfterLoop(pmesh, pinput, tm);
  }

  DriverStatus status = tm.KeepGoing() ? DriverStatus::timeout : DriverStatus::complete;
  // Do *not* write the "final" output, if this is analysis run.
  // The analysis output itself has already been written above before the main loop.
  if (signal != OutputSignal::analysis) {
    pouts->MakeOutputs(pmesh, pinput, &tm, OutputSignal::final);
  }
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

void EvolutionDriver::InitializeBlockTimeSteps() {
  // calculate the first time step using Block function
  for (auto &pmb : pmesh->block_list) {
    Update::EstimateTimestep(pmb->meshblock_data.Get().get());
  }
  // calculate the first time step using Mesh function
  for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
    auto &mbase = pmesh->mesh_data.Add("base", partition);
    Update::EstimateTimestep(mbase.get());
  }
}

//----------------------------------------------------------------------------------------
// \!fn void EvolutionDriver::SetGlobalTimeStep()
// \brief function that loops over all MeshBlocks and find new timestep

void EvolutionDriver::SetGlobalTimeStep() {
  if (dt_force > 0.0) {
    // Check if user wants to force the value
    tm.dt = dt_force;
  } else if ((tm.ncycle == 0) && (dt_init_force) && (dt_init > 0.0)) {
    // Check if user wants to force the first cycle value
    tm.dt = dt_init;
  } else {
    if (tm.dt < 0.1 * std::numeric_limits<Real>::max()) {
      // don't allow dt to grow by more than 2x
      // consider making this configurable in the input
      tm.dt *= dt_factor;
    }
    // Check for special first cycle value
    if (tm.ncycle == 0) {
      tm.dt = std::min(tm.dt, dt_init);
    }
    // Allow the meshblocks to vote
    for (auto const &pmb : pmesh->block_list) {
      tm.dt = std::min(tm.dt, pmb->NewDt());
      pmb->SetAllowedDt(std::numeric_limits<Real>::max());
    }
    // Force timestep to be in the allowable range
    tm.dt = std::max(dt_floor, std::min(tm.dt, dt_ceil));
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &tm.dt, 1, MPI_PARTHENON_REAL,
                                      MPI_MIN, MPI_COMM_WORLD));
#endif
  }

  // Check that we have not gone off the rails
  if (tm.dt <= dt_min) {
    if (++dt_min_count >= dt_min_count_max) {
      std::stringstream msg;
      msg << "Timesetep has fallen bellow minimum (parthenon/time/dt_min=" << dt_min
          << ") for more than " << dt_min_count_max << " steps";
      PARTHENON_FAIL(msg);
    }
  } else {
    dt_min_count = 0;
  }
  if (tm.dt >= dt_max) {
    if (++dt_max_count >= dt_max_count_max) {
      std::stringstream msg;
      msg << "Timesetep has risen above maximum (parthenon/time/dt_max=" << dt_max
          << ") for more than " << dt_max_count_max << " steps";
      PARTHENON_FAIL(msg);
    }
  } else {
    dt_max_count = 0;
  }

  // Limit timestep if it would take us past desired endpoint
  // This comes after the bounds checking so that we don't fail when epsilon
  // away from tlim
  if (tm.time < tm.tlim && (tm.tlim - tm.time) < tm.dt) {
    tm.dt = tm.tlim - tm.time;
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
