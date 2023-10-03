//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef SOLVERS_MG_SOLVER_HPP_
#define SOLVERS_MG_SOLVER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/solver_utils.hpp"
#include "tasks/task_id.hpp"
#include "tasks/task_list.hpp"

namespace parthenon {

namespace solvers {

struct MGParams { 
  int max_iters = 10; 
  Real residual_tolerance = 1.e-12;  
}; 

#define MGVARIABLE(base, varname)                                                        \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return base::name() + "." #varname; }                    \
  }

template <class u, class rhs> 
class MGSolver {
  MGVARIABLE(u, res_err);
  MGVARIABLE(u, temp);
 public:
  MGSolver(StateDescriptor *pkg, MGParams params_in) : params_(params_in) { 
    using namespace parthenon::refinement_ops;
    auto mres_err = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                              Metadata::GMGRestrict, Metadata::GMGProlongate, Metadata::OneCopy});
    mres_err.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(res_err::name(), mres_err);

    auto mtemp = Metadata(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes, Metadata::OneCopy});
    mtemp.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(temp::name(), mtemp);
  }

  TaskID AddTasks(TaskList &tl, int partition, Mesh *pmesh, int reg_dep_id) { 
    iter_counter = 0;
    auto itl = tl.AddIteration("MG." + u::name());
    return TaskID(0);
  }

 protected: 
  MGParams params_;
  int iter_counter;

};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_MG_SOLVER_HPP_
