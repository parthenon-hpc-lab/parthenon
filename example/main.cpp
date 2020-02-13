// Third Party Includes
#include <mpi.h>

// Athena++ headers
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//  \brief Parthenon Sample main program

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  ParameterInput pin;
  pin.SetInteger("mesh", "nx1", 50);
  pin.SetReal("mesh", "x1min", -0.5);
  pin.SetReal("mesh", "x1max", 0.5);

  pin.SetInteger("mesh", "nx2", 1);
  pin.SetReal("mesh", "x2min", -0.5);
  pin.SetReal("mesh", "x2max", 0.5);

  pin.SetInteger("mesh", "nx3", 1);
  pin.SetReal("mesh", "x3min", -0.5);
  pin.SetReal("mesh", "x3max", 0.5);

  pin.SetReal("time", "tlim", 1.0);

  std::vector<std::shared_ptr<MaterialPropertiesInterface>> mats;
  std::map<std::string, std::shared_ptr<StateDescriptor>> physics;
  Mesh m(&pin, mats, physics, [](Container<Real> &) {});

  MPI_Finalize();
}
