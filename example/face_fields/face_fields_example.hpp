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

#ifndef EXAMPLE_FACE_FIELDS_FACE_FIELDS_EXAMPLE_HPP_
#define EXAMPLE_FACE_FIELDS_FACE_FIELDS_EXAMPLE_HPP_

#include <memory>

#include "driver/driver.hpp"
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"

namespace parthenon {

class FaceFieldExample : public Driver {
 public:
  FaceFieldExample(ParameterInput *pin, Mesh *pm, Outputs *pout)
      : Driver(pin, pm, pout) {}
  TaskList MakeTaskList(MeshBlock *pmb);
  DriverStatus Execute();
};

} // namespace parthenon

namespace FaceFields {

parthenon::TaskStatus fill_faces(parthenon::MeshBlock *pmb);

} // namespace FaceFields

#endif // EXAMPLE_FACE_FIELDS_FACE_FIELDS_EXAMPLE_HPP_
