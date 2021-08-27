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
#ifndef SOLVERS_CG_SOLVER_HPP_
#define SOLVERS_CG_SOLVER_HPP_

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

struct CG_Counter {
  static int global_num_cg_solvers;
};

template <typename SPType>
class CG_Solver : public CG_Counter {
 public:
  CG_Solver() {}
  CG_Solver(StateDescriptor *pkg, const Real error_tol_in, const Stencil<Real> &st)
      : error_tol(error_tol_in), stencil(st) {
    use_sparse_accessor = false;
    Init(pkg);
  }
  CG_Solver(StateDescriptor *pkg, const Real error_tol_in, const SparseMatrixAccessor &sp)
      : error_tol(error_tol_in), sp_accessor(sp) {
    use_sparse_accessor = true;
    Init(pkg);
  }

  void Init(StateDescriptor *pkg) {
    // add a couple of vectors for solver..
    auto mcdo = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    spm_name = pkg->Param<std::string>("spm_name");
    sol_name = pkg->Param<std::string>("sol_name");
    rhs_name = pkg->Param<std::string>("rhs_name");
    const std::string cg_id(std::to_string(global_num_cg_solvers));
    zk = "zk" + cg_id;
    res = "res" + cg_id;
    apk = "apk" + cg_id;
    xk = sol_name;
    pk = "pk" + cg_id;
    pkg->AddField(zk, mcdo);
    pkg->AddField(res, mcdo);
    pkg->AddField(apk, mcdo);

    // ghost exchange required...
    auto mcif = Metadata({Metadata::Cell, Metadata::FillGhost});
    pkg->AddField(pk, mcif);

    global_num_cg_solvers++;
  }

  std::vector<std::string> SolverState() const {
    if (use_sparse_accessor) {
      return std::vector<std::string>({zk, res, apk, pk, spm_name, rhs_name});
    } else {
      return std::vector<std::string>({zk, res, apk, pk, rhs_name});
    }
  }
  std::string label() const {
    std::string lab;
    for (const auto &s : SolverState())
      lab += s;
    return lab;
  }

  TaskID createCGTaskList(TaskID &begin, const int i, int &j, TaskList &tl,
                          TaskRegion &solver_region, IterativeTasks &solver,
                          std::shared_ptr<MeshData<Real>> md,
                          std::shared_ptr<MeshData<Real>> mout) {
    TaskID none(0);

    // these are values shared across lists
    auto rz0 = (i == 0 ? tl.AddTask(
                             none,
                             [](Real *val, Real *beta, int *cntr) {
                               *val = 0;
                               *beta = 0;
                               *cntr = 0;
                               return TaskStatus::complete;
                             },
                             &r_dot_z.val, &betak.val, &cg_cntr)
                       : tl.AddTask(none, &CG_Solver<SPType>::DoNothing, this));
    solver_region.AddRegionalDependencies(j, i, rz0);
    j++;

    // x=0;
    // b = dV*rho
    // r=b-Ax;
    // z = Minv*r;
    auto res0 = tl.AddTask(begin | rz0, &CG_Solver<SPType>::DiagScaling<MeshData<Real>>,
                           this, md.get(), mout.get(), &r_dot_z.val);
    solver_region.AddRegionalDependencies(j, i, res0);
    j++;

    // r.z;
    auto start_global_rz =
        (i == 0 ? tl.AddTask(res0, &AllReduce<Real>::StartReduce, &r_dot_z, MPI_SUM)
                : res0);

    auto finish_global_rz =
        tl.AddTask(start_global_rz, &AllReduce<Real>::CheckReduce, &r_dot_z);

    ////////////////////////////////////////////////////////////////////////////////
    // CG
    // this will move to somewhere..

    // initialization only happens once.

    /////////////////////////////////////////////
    // Iteration starts here.
    // p = beta*p+z;
    auto axpy1 =
        solver.AddTask(res0 | finish_global_rz, &CG_Solver<SPType>::Axpy1<MeshData<Real>>,
                       this, md.get(), &betak.val);
    // matvec Ap = J*p
    auto pAp0 = (i == 0 ? solver.AddTask(
                              none,
                              [](Real *val) {
                                *val = 0;
                                return TaskStatus::complete;
                              },
                              &p_dot_ap.val)
                        : solver.AddTask(none, &CG_Solver<SPType>::DoNothing, this));
    solver_region.AddRegionalDependencies(j, i, pAp0);
    j++;

    auto start_recv = solver.AddTask(none, &MeshData<Real>::StartReceiving, md.get(),
                                     BoundaryCommSubset::all);
    // ghost exchange.
    auto send =
        solver.AddTask(axpy1, parthenon::cell_centered_bvars::SendBoundaryBuffers, md);
    auto recv = solver.AddTask(
        start_recv, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, md);
    auto setb =
        solver.AddTask(recv | axpy1, parthenon::cell_centered_bvars::SetBoundaries, md);
    auto clear = solver.AddTask(send | setb, &MeshData<Real>::ClearBoundary, md.get(),
                                BoundaryCommSubset::all);
    auto matvec = solver.AddTask(clear | pAp0, &CG_Solver<SPType>::MatVec<MeshData<Real>>,
                                 this, md.get(), &p_dot_ap.val);
    solver_region.AddRegionalDependencies(j, i, matvec);
    j++;

    // reduce p.Ap
    auto start_global_pAp =
        (i == 0
             ? solver.AddTask(matvec, &AllReduce<Real>::StartReduce, &p_dot_ap, MPI_SUM)
             : matvec);
    auto finish_global_pAp =
        solver.AddTask(start_global_pAp, &AllReduce<Real>::CheckReduce, &p_dot_ap);

    // alpha = r.z/p.Ap
    auto alpha = (i == 0 ? solver.AddTask(
                               finish_global_pAp | finish_global_rz,
                               [](Real *val, Real *rz, Real *pAp, Real *rznew) {
                                 *val = (*rz) / (*pAp);
                                 *rznew = 0;
                                 return TaskStatus::complete;
                               },
                               &alphak.val, &r_dot_z.val, &p_dot_ap.val, &r_dot_z_new.val)
                         : solver.AddTask(none, &CG_Solver<SPType>::DoNothing, this));
    solver_region.AddRegionalDependencies(j, i, alpha);
    j++;

    // x = x+alpha*p
    // r = r-alpha*Apk
    // z = M^-1*r
    // r.z-new
    auto double_axpy =
        solver.AddTask(alpha, &CG_Solver<SPType>::DoubleAxpy<MeshData<Real>>, this,
                       md.get(), mout.get(), &alphak.val, &r_dot_z_new.val);
    solver_region.AddRegionalDependencies(j, i, double_axpy);
    j++;

    // reduce p.Ap
    auto start_global_rz_new =
        (i == 0 ? solver.AddTask(double_axpy, &AllReduce<Real>::StartReduce, &r_dot_z_new,
                                 MPI_SUM)
                : double_axpy);
    auto finish_global_rz_new =
        solver.AddTask(start_global_rz_new, &AllReduce<Real>::CheckReduce, &r_dot_z_new);

    // beta= rz_new/rz
    // and check convergence..
    auto beta = (i == 0 ? solver.AddTask(
                              finish_global_rz_new,
                              [](Real *beta, Real *rz_new, Real *rz, Real *res_global,
                                 Real *gres0, int *cntr) {
                                *beta = (*rz_new) / (*rz);
                                *res_global = sqrt(*rz_new);

                                if (*cntr == 0) *gres0 = *res_global;

                                *cntr = *cntr + 1;
                                (*rz) = (*rz_new);

                                return TaskStatus::complete;
                              },
                              &betak.val, &r_dot_z_new.val, &r_dot_z.val, &res_global,
                              &global_res0, &cg_cntr)
                        : solver.AddTask(none, &CG_Solver<SPType>::DoNothing, this));
    solver_region.AddRegionalDependencies(j, i, beta);
    j++;

    auto check =
        (i == 0 ? solver.SetCompletionTask(
                      beta,
                      [](Real *res_global, Real *gres0, Real *err_tol, int *cntr) {
                        auto status =
                            (*res_global / (*gres0) < (*err_tol) ? TaskStatus::complete
                                                                 : TaskStatus::iterate);

                        if (parthenon::Globals::my_rank == 0)
                          std::cout << parthenon::Globals::my_rank << " its= " << *cntr
                                    << " relative res: " << *res_global / (*gres0)
                                    << " absolute-res " << *res_global
                                    << " relerr-tol: " << (*err_tol) << std::endl
                                    << std::flush;

                        return status;
                      },
                      &res_global, &global_res0, &error_tol, &cg_cntr)
                : solver.SetCompletionTask(none, &CG_Solver<SPType>::DoNothing, this));
    solver_region.AddGlobalDependencies(j, i, check);
    j++;

    return check;
  }

  TaskStatus DoNothing() { return TaskStatus::complete; }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Utility tasks for solver..
  /////////////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus Axpy1(T *u, Real *beta) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({zk, pk});
    const auto &v = u->PackVariables(vars, imap);

    // this get cell variable..
    const int izk = imap[zk].first;
    const int ipk = imap[pk].first;

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "axpy1", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e,
        jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          v(b, ipk, k, j, i) = (*beta) * v(b, ipk, k, j, i) + v(b, izk, k, j, i);
        });

    return TaskStatus::complete;
  } // Axpy1
  /////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus DiagScaling(T *u, T *du, Real *reduce_sum) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({zk, pk, res, rhs_name, spm_name});
    const auto &v = u->PackVariables(vars, imap);

    // this get cell variable..
    const int izk = imap[zk].first;
    const int ipk = imap[pk].first;
    const int ires = imap[res].first;
    const int irhs = imap[rhs_name].first;
    const int isp_lo = imap[spm_name].first;
    const int isp_hi = imap[spm_name].second;
    int diag;
    if (use_sparse_accessor) {
      diag = sp_accessor.ndiag + isp_lo;
    } else {
      diag = stencil.ndiag;
    }

    // assume solution is in "dv"
    const std::vector<std::string> var2({sol_name});
    PackIndexMap imap2;
    const auto &dv = du->PackVariables(var2, imap2);
    const int ixk = imap2[sol_name].first;

    Real sum(0);
    Real gsum(0);

    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "diag_scaling", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          // x=0
          dv(b, ixk, k, j, i) = 0;

          // res = rhs
          v(b, ires, k, j, i) = v(b, irhs, k, j, i);

          // z=r/J_ii
          Real J_ii = (use_sparse_accessor ? v(b, diag, k, j, i) : stencil.w(diag));
          v(b, izk, k, j, i) = v(b, ires, k, j, i) / J_ii;
          // p=z
          v(b, ipk, k, j, i) = v(b, izk, k, j, i);
          // r.z
          lsum += v(b, ires, k, j, i) * v(b, izk, k, j, i);
        },
        Kokkos::Sum<Real>(gsum));

    *reduce_sum += gsum;
    return TaskStatus::complete;
  } // DiagScaling

  /////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus MatVec(T *u, Real *reduce_sum) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({pk, apk, spm_name});
    const auto &v = u->PackVariables(vars, imap);

    const int ipk = imap[pk].first;
    const int iapk = imap[apk].first;

    const int isp_lo = imap[spm_name].first;
    const int isp_hi = imap[spm_name].second;

    int ndim = v.GetNdim();
    Real dot(0);

    // const auto &sp_accessor =
    //  pkg->Param<parthenon::solvers::SparseMatrixAccessor>("sparse_accessor");

    if (use_sparse_accessor) {
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, "mat_vec", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
            // ap = A*p;
            v(b, iapk, k, j, i) =
                (use_sparse_accessor
                     ? sp_accessor.MatVec(v, isp_lo, isp_hi, v, ipk, b, k, j, i)
                     : stencil.MatVec(v, ipk, b, k, j, i));

            // p.Ap
            lsum += v(b, ipk, k, j, i) * v(b, iapk, k, j, i);
          },
          Kokkos::Sum<Real>(dot));
    } else {
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, "mat_vec", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
            // ap = A*p;
            v(b, iapk, k, j, i) = stencil.MatVec(v, ipk, b, k, j, i);

            // p.Ap
            lsum += v(b, ipk, k, j, i) * v(b, iapk, k, j, i);
          },
          Kokkos::Sum<Real>(dot));
    }
    *reduce_sum += dot;

    return TaskStatus::complete;
  } // MatVec

  /////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus DoubleAxpy(T *u, T *du, Real *palphak, Real *reduce_sum) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({pk, apk, res, zk, spm_name});
    const auto &v = u->PackVariables(vars, imap);

    const int ipk = imap[pk].first;
    const int iapk = imap[apk].first;

    const int ires = imap[res].first;
    const int izk = imap[zk].first;

    const int isp_lo = imap[spm_name].first;
    int diag;
    if (use_sparse_accessor) {
      diag = sp_accessor.ndiag + isp_lo;
    } else {
      diag = stencil.ndiag;
    }

    const std::vector<std::string> var2({sol_name});
    PackIndexMap imap2;
    const auto &dv = du->PackVariables(var2, imap2);
    const int ixk = imap2[sol_name].first;

    Real sum(0);
    // make a local copy so it's captured in the kernel
    const Real alphak = *palphak;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "double_axpy", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          // x = x+alpha*p
          dv(b, ixk, k, j, i) += alphak * v(b, ipk, k, j, i);
          // r = r-alpha*Ap
          v(b, ires, k, j, i) -= alphak * v(b, iapk, k, j, i);
          // z = r/J_ii;(precon..)
          Real J_ii = (use_sparse_accessor ? v(b, diag, k, j, i) : stencil.w(diag));
          v(b, izk, k, j, i) = v(b, ires, k, j, i) / J_ii;
          // r.z

          lsum += v(b, ires, k, j, i) * v(b, izk, k, j, i);
        },
        Kokkos::Sum<Real>(sum));

    *reduce_sum += sum;
    return TaskStatus::complete;
  } // DoubleAxpy

 private:
  AllReduce<Real> p_dot_ap;
  AllReduce<Real> r_dot_z;
  AllReduce<Real> r_dot_z_new;
  AllReduce<Real> alphak;
  AllReduce<Real> betak;
  Real res_global;

  int cg_cntr;
  Real global_res0;
  std::string zk, res, apk, xk, pk;
  std::string spm_name, sol_name, rhs_name;
  Real error_tol;
  Stencil<Real> stencil;
  SparseMatrixAccessor sp_accessor;
  bool use_sparse_accessor;
};

} // namespace solvers
} // namespace parthenon

#endif // SOLVERS_CG_SOLVER_HPP_
