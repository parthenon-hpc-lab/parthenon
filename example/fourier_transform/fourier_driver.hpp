//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_FOURIER_TRANSFORM_FOURIER_DRIVER_HPP_
#define EXAMPLE_FOURIER_TRANSFORM_FOURIER_DRIVER_HPP_

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>

namespace fourier_transform {
using namespace parthenon::driver::prelude;

/**
 * @brief Constructs a driver to demonstrate the use of Fourier transforms in Parthenon.
 * The driver will compute the Fourier transform of a 3D array, compute the inverse
 * Fourier transform to recover the original array, and compute the maximum error between
 * the original and recovered arrays.
 */
class FourierDriver : public Driver {
 public:
  FourierDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm) {
    InitializeOutputs();
  }

  /// MakeTaskList and MakeTasks aren't virtual routines on `Driver`,
  // but each driver is expected to implement at least one of them.
  /// TaskList MakeTaskList(MeshBlock *pmb);
  template <typename T>
  TaskCollection MakeTaskCollection(T &blocks);

  /// `Execute` cycles until simulation completion.
  DriverStatus Execute() override;
};

} // namespace fourier_transform

#endif // EXAMPLE_FOURIER_TRANSFORM_FOURIER_DRIVER_HPP_
