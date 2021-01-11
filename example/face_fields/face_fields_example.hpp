//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_FACE_FIELDS_FACE_FIELDS_EXAMPLE_HPP_
#define EXAMPLE_FACE_FIELDS_FACE_FIELDS_EXAMPLE_HPP_

#include <memory>

#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"

namespace FaceFields {

parthenon::TaskStatus fill_faces(parthenon::MeshBlock *pmb);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);
void ProblemGenerator(parthenon::MeshBlock *pmb, parthenon::ParameterInput *pin);

class FaceFieldExample : public parthenon::Driver {
 public:
  FaceFieldExample(parthenon::ParameterInput *pin, parthenon::ApplicationInput *app_in,
                   parthenon::Mesh *pm)
      : parthenon::Driver(pin, app_in, pm) {
    InitializeOutputs();
  }
  parthenon::TaskList MakeTaskList(parthenon::MeshBlock *pmb);
  parthenon::DriverStatus Execute();
};

} // namespace FaceFields

#endif // EXAMPLE_FACE_FIELDS_FACE_FIELDS_EXAMPLE_HPP_
