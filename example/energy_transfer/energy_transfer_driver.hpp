#ifndef EXAMPLE_ENERGY_TRANSFER_ENERGY_TRANSFER_DRIVER_HPP_
#define EXAMPLE_ENERGY_TRANSFER_ENERGY_TRANSFER_DRIVER_HPP_

#include <memory>
#include <string>
#include <vector>

#include <parthenon/driver.hpp>

namespace energy_transfer {
using namespace parthenon::driver::prelude;

class EnergyTransferDriver : public Driver {
 public:
  EnergyTransferDriver(ParameterInput *pin, ApplicationInput *fin, Mesh *pm)
      : Driver(pin, fin, pm) {
    InitializeOutputs();
  }

  template <typename T>
  TaskCollection MakeTaskCollection(T &blocks);

  DriverStatus Execute() override;
};

} // namespace energy_transfer

#endif // EXAMPLE_ENERGY_TRANSFER_ENERGY_TRANSFER_DRIVER_HPP_
