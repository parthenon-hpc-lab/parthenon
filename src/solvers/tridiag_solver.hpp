//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef SOLVERS_TRIDIAG_SOLVER_HPP_
#define SOLVERS_TRIDIAG_SOLVER_HPP_

#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/mg_solver.hpp"
#include "solvers/solver_base.hpp"
#include "solvers/solver_utils.hpp"
#include "tasks/tasks.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

namespace solvers {

// The equations class must include a template method
//
//   template <class x_t, class y_t, class TL_t>
//   TaskID Ax(TL_t &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md)
//
// that takes a field associated with x_t and applies
// the matrix A to it and stores the result in y_t.
template <class equations>
class TridiagSolver : public SolverBase {
  using FieldTL = typename equations::IndependentVars;

  std::vector<std::string> sol_fields;
  // Name of user defined container that should contain information required to
  // calculate the matrix part of the matrix vector product
  std::string container_base;
  // User defined container in which the solution will reside, only needs to contain
  // sol_fields
  // TODO(LFR): Also allow for an initial guess to come in here
  std::string container_u;
  // User defined container containing the rhs vector, only needs to contain sol_fields
  std::string container_rhs;
  // Internal containers for solver which create deep copies of sol_fields
  std::string container_100, container_010, container_001, container_100_out,
      container_010_out, container_001_out, container_Aup, container_Adi, container_Alo,
      container_r;

  static inline std::size_t id{0};

 public:
  TridiagSolver(const std::string &container_base, const std::string &container_u,
                const std::string &container_rhs, ParameterInput *pin,
                const std::string &input_block, const equations &eq_in = equations())
      : container_base(container_base), container_u(container_u),
        container_rhs(container_rhs), iter_counter(0), eqs_(eq_in),
        print_solution_(pin->GetOrAddBoolean(input_block, "print_solution", false)) {
    FieldTL::IterateTypes(
        [this](auto t) { this->sol_fields.push_back(decltype(t)::name()); });
    PARTHENON_REQUIRE(sol_fields.size() == 1,
                      "Tridiagonal solver only works for a single field on a single "
                      "one-dimensional block.");
    std::string solver_id = "tridiag" + std::to_string(id++);
    container_100 = solver_id + "_100";
    container_010 = solver_id + "_010";
    container_001 = solver_id + "_001";

    container_100_out = solver_id + "_100_out";
    container_010_out = solver_id + "_010_out";
    container_001_out = solver_id + "_001_out";

    container_Alo = solver_id + "_Alo";
    container_Adi = solver_id + "_Adiag";
    container_Aup = solver_id + "_Aup";
    container_r = solver_id + "_r";
  }

  TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition, Mesh *pmesh) {
    return dependence;
  }

  static TaskStatus SetMasks(const std::shared_ptr<MeshData<Real>> &md_100,
                             const std::shared_ptr<MeshData<Real>> &md_010,
                             const std::shared_ptr<MeshData<Real>> &md_001) {
    IndexRange ib = md_100->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md_100->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md_100->GetBoundsK(IndexDomain::interior);

    PARTHENON_REQUIRE(md_100->GetMeshPointer()->nbtotal == 1,
                      "Tridiag only works for a single block.");
    PARTHENON_REQUIRE(jb.s == jb.e, "Must be one dimensional.");
    PARTHENON_REQUIRE(kb.s == kb.e, "Must be one dimensional.");

    static auto desc = parthenon::MakePackDescriptorFromTypeList<FieldTL>(md_100.get());
    auto pack_100 = desc.GetPack(md_100.get());
    auto pack_010 = desc.GetPack(md_010.get());
    auto pack_001 = desc.GetPack(md_001.get());

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetMasks", DevExecSpace(), 0, pack_100.GetNBlocks() - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack_100(b, 0, k, j, i) = ((i - ib.s + 3) % 3) == 0;
          pack_010(b, 0, k, j, i) = ((i - ib.s + 2) % 3) == 0;
          pack_001(b, 0, k, j, i) = ((i - ib.s + 1) % 3) == 0;
        });

    return TaskStatus::complete;
  }

  static TaskStatus SetDiagonals(const std::shared_ptr<MeshData<Real>> &md_100_out,
                                 const std::shared_ptr<MeshData<Real>> &md_010_out,
                                 const std::shared_ptr<MeshData<Real>> &md_001_out,
                                 const std::shared_ptr<MeshData<Real>> &md_Alo,
                                 const std::shared_ptr<MeshData<Real>> &md_Adi,
                                 const std::shared_ptr<MeshData<Real>> &md_Aup) {
    IndexRange ib = md_Alo->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md_Alo->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md_Alo->GetBoundsK(IndexDomain::interior);

    PARTHENON_REQUIRE(md_100_out->GetMeshPointer()->nbtotal == 1,
                      "Tridiag only works for a single block.");
    PARTHENON_REQUIRE(jb.s == jb.e, "Must be one dimensional.");
    PARTHENON_REQUIRE(kb.s == kb.e, "Must be one dimensional.");

    static auto desc =
        parthenon::MakePackDescriptorFromTypeList<FieldTL>(md_100_out.get());
    auto pack_100 = desc.GetPack(md_100_out.get());
    auto pack_010 = desc.GetPack(md_010_out.get());
    auto pack_001 = desc.GetPack(md_001_out.get());
    auto pack_lo = desc.GetPack(md_Alo.get());
    auto pack_di = desc.GetPack(md_Adi.get());
    auto pack_up = desc.GetPack(md_Aup.get());

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetDiagonalsBasedOnMasks", DevExecSpace(), 0,
        pack_100.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          if ((i - ib.s) % 3 == 0) {
            pack_di(b, 0, k, j, i) = pack_100(b, 0, k, j, i);
            pack_up(b, 0, k, j, i) = pack_010(b, 0, k, j, i);
            pack_lo(b, 0, k, j, i) = pack_001(b, 0, k, j, i);
          } else if ((i - ib.s) % 3 == 1) {
            pack_lo(b, 0, k, j, i) = pack_100(b, 0, k, j, i);
            pack_di(b, 0, k, j, i) = pack_010(b, 0, k, j, i);
            pack_up(b, 0, k, j, i) = pack_001(b, 0, k, j, i);
          } else {
            pack_up(b, 0, k, j, i) = pack_100(b, 0, k, j, i);
            pack_lo(b, 0, k, j, i) = pack_010(b, 0, k, j, i);
            pack_di(b, 0, k, j, i) = pack_001(b, 0, k, j, i);
          }
        });

    return TaskStatus::complete;
  }

  static TaskStatus Solve(const std::shared_ptr<MeshData<Real>> &md_Alo,
                          const std::shared_ptr<MeshData<Real>> &md_Adi,
                          const std::shared_ptr<MeshData<Real>> &md_Aup,
                          const std::shared_ptr<MeshData<Real>> &md_rhs,
                          const std::shared_ptr<MeshData<Real>> &md_out) {
    IndexRange ib = md_Alo->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md_Alo->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md_Alo->GetBoundsK(IndexDomain::interior);

    PARTHENON_REQUIRE(md_Alo->GetMeshPointer()->nbtotal == 1,
                      "Tridiag only works for a single block.");
    PARTHENON_REQUIRE(jb.s == jb.e, "Must be one dimensional.");
    PARTHENON_REQUIRE(kb.s == kb.e, "Must be one dimensional.");

    static auto desc = parthenon::MakePackDescriptorFromTypeList<FieldTL>(md_Alo.get());
    auto pack_lo = desc.GetPack(md_Alo.get());
    auto pack_di = desc.GetPack(md_Adi.get());
    auto pack_up = desc.GetPack(md_Aup.get());
    auto pack_rhs = desc.GetPack(md_rhs.get());
    auto pack_out = desc.GetPack(md_out.get());

    const int b = 0;
    const int k = kb.s;
    const int j = jb.s;

    // Since this needs to be sequential, we launch an outer loop of size one. Obviously
    // this would be really inefficient on device
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "DotProduct", DevExecSpace(), 0, 0,
        KOKKOS_LAMBDA(const int) {
          pack_di(b, 0, k, j, ib.s - 1) = 1.0;
          for (int i = ib.s; i <= ib.e; ++i) {
            pack_di(b, 0, k, j, i) -= pack_lo(b, 0, k, j, i) *
                                      pack_up(b, 0, k, j, i - 1) /
                                      pack_di(b, 0, k, j, i - 1);
            pack_rhs(b, 0, k, j, i) -= pack_lo(b, 0, k, j, i) *
                                       pack_rhs(b, 0, k, j, i - 1) /
                                       pack_di(b, 0, k, j, i - 1);
          }
          pack_out(b, 0, k, j, ib.e + 1) = 0.0;
          for (int i = ib.e; i >= ib.s; --i) {
            pack_out(b, 0, k, j, i) =
                (pack_rhs(b, 0, k, j, i) -
                 pack_up(b, 0, k, j, i) * pack_out(b, 0, k, j, i + 1)) /
                pack_di(b, 0, k, j, i);
          }
          pack_out(b, 0, k, j, ib.s - 1) = 0.0;
        });

    return TaskStatus::complete;
  }

  static TaskStatus PrintSolution(const std::shared_ptr<MeshData<Real>> &md_base,
                                  const std::shared_ptr<MeshData<Real>> &md_u,
                                  const std::shared_ptr<MeshData<Real>> &md_r,
                                  const std::shared_ptr<MeshData<Real>> &md_rhs) {
    IndexRange ib = md_r->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md_r->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md_r->GetBoundsK(IndexDomain::interior);

    PARTHENON_REQUIRE(md_r->GetMeshPointer()->nbtotal == 1,
                      "Tridiag only works for a single block.");
    PARTHENON_REQUIRE(jb.s == jb.e, "Must be one dimensional.");
    PARTHENON_REQUIRE(kb.s == kb.e, "Must be one dimensional.");

    static auto desc = parthenon::MakePackDescriptorFromTypeList<FieldTL>(md_r.get());
    auto pack_u = desc.GetPack(md_u.get());
    auto pack_base = desc.GetPack(md_base.get());
    auto pack_r = desc.GetPack(md_r.get());
    auto pack_rhs = desc.GetPack(md_rhs.get());

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "PrintSolution", DevExecSpace(), 0, 0, kb.s, kb.e, jb.s,
        jb.e, ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          printf("row %i: %e %e %e  %e \n", i, pack_u(b, 0, k, j, i),
                 pack_base(b, 0, k, j, i), pack_r(b, 0, k, j, i),
                 pack_rhs(b, 0, k, j, i));
        });
    return TaskStatus::complete;
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, const int partition, Mesh *pmesh) {
    using namespace utils;
    TaskID none;
    auto partitions = pmesh->GetDefaultBlockPartitions();
    // Should contain all fields necessary for applying the matrix to a give state vector,
    // e.g. diffusion coefficients and diagonal, these will not be modified by the solvers
    auto &md_base = pmesh->mesh_data.Add(container_base, partitions[partition]);
    // Container in which the solution is stored and with which the downstream user can
    // interact. This container only requires the fields in sol_fields
    auto &md_u = pmesh->mesh_data.Add(container_u, partitions[partition], sol_fields);
    // Container of the rhs, only requires fields in sol_fields
    auto &md_rhs = pmesh->mesh_data.Add(container_rhs, partitions[partition], sol_fields);
    // Internal solver containers
    auto &md_100 = pmesh->mesh_data.Add(container_100, partitions[partition], sol_fields);
    auto &md_010 = pmesh->mesh_data.Add(container_010, partitions[partition], sol_fields);
    auto &md_001 = pmesh->mesh_data.Add(container_001, partitions[partition], sol_fields);

    auto &md_100_out =
        pmesh->mesh_data.Add(container_100_out, partitions[partition], sol_fields);
    auto &md_010_out =
        pmesh->mesh_data.Add(container_010_out, partitions[partition], sol_fields);
    auto &md_001_out =
        pmesh->mesh_data.Add(container_001_out, partitions[partition], sol_fields);

    auto &md_Alo = pmesh->mesh_data.Add(container_Alo, partitions[partition], sol_fields);
    auto &md_Adi = pmesh->mesh_data.Add(container_Adi, partitions[partition], sol_fields);
    auto &md_Aup = pmesh->mesh_data.Add(container_Aup, partitions[partition], sol_fields);
    auto &md_r = pmesh->mesh_data.Add(container_r, partitions[partition], sol_fields);

    iter_counter = 0;
    bool multilevel = pmesh->multilevel;

    auto masks = tl.AddTask(dependence, SetMasks, md_100, md_010, md_001);
    auto Ax100 = eqs_.Ax(tl, masks, md_base, md_100, md_100_out);
    auto Ax010 = eqs_.Ax(tl, Ax100, md_base, md_010, md_010_out);
    auto Ax001 = eqs_.Ax(tl, Ax010, md_base, md_001, md_001_out);
    auto explicit_A = tl.AddTask(Ax001, SetDiagonals, md_100_out, md_010_out, md_001_out,
                                 md_Alo, md_Adi, md_Aup);

    auto copy_rhs = tl.AddTask(explicit_A, utils::CopyData<FieldTL>, md_rhs, md_r);
    auto sol = tl.AddTask(copy_rhs, Solve, md_Alo, md_Adi, md_Aup, md_r, md_u);

    sol = tl.AddTask(sol, utils::SetToZero<FieldTL>, md_r);
    auto Ax_check = eqs_.Ax(tl, sol, md_base, md_u, md_r);
    if (print_solution_)
      Ax_check = tl.AddTask(Ax_check, PrintSolution, md_base, md_u, md_r, md_rhs);

    return Ax_check;
  }

  Real GetSquaredResidualSum() const { return 0.0; }
  int GetCurrentIterations() const { return 1; }

 protected:
  int iter_counter;
  Real ru_old;
  equations eqs_;
  bool print_solution_;
};

} // namespace solvers
} // namespace parthenon

#endif // SOLVERS_TRIDIAG_SOLVER_HPP_
