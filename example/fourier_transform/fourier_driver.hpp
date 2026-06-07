//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

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
  FourierDriver(ParameterInput *pin, ApplicationInput *fin, Mesh *pm)
      : Driver(pin, fin, pm) {
    InitializeOutputs();
  }

  /// MakeTaskList and MakeTasks aren't virtual routines on `Driver`,
  // but each driver is expected to implement at least one of them.
  /// TaskList MakeTaskList(MeshBlock *pmb);
  template <typename T>
  TaskCollection MakeTaskCollection(T &blocks);

  /// `Execute` cycles until simulation completion.
  DriverStatus Execute() override;

 protected:
  void FourierPostExecute(Real max_error);
};

} // namespace fourier_transform

#endif // EXAMPLE_FOURIER_TRANSFORM_FOURIER_DRIVER_HPP_
