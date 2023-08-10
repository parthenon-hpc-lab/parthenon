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
      : error_tol(error_tol_in), stencil(st),
        max_iters(pkg->Param<int>("cg_max_iterations")),
        check_interval(pkg->Param<int>("cg_check_interval")),
        fail_flag(pkg->Param<bool>("cg_abort_on_fail")),
        warn_flag(pkg->Param<bool>("cg_warn_on_fail")) {
    use_sparse_accessor = false;
    Init(pkg);
  }
  CG_Solver(StateDescriptor *pkg, const Real error_tol_in, const SparseMatrixAccessor &sp)
      : error_tol(error_tol_in), sp_accessor(sp),
        max_iters(pkg->Param<int>("cg_max_iterations")),
        check_interval(pkg->Param<int>("cg_check_interval")),
        fail_flag(pkg->Param<bool>("cg_abort_on_fail")),
        warn_flag(pkg->Param<bool>("cg_warn_on_fail")) {
    use_sparse_accessor = true;
    Init(pkg);
  }

  enum Precon_Type { NONE = 1, DIAG_SCALING = 2, ICC = 3, ERROR = 4 };

  void Init(StateDescriptor *pkg) {
    // add a couple of vectors for solver..
    auto mcdo = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    spm_name = pkg->Param<std::string>("spm_name");
    sol_name = pkg->Param<std::string>("sol_name");
    rhs_name = pkg->Param<std::string>("rhs_name");
    pcm_name = "";

    const std::string cg_id(std::to_string(global_num_cg_solvers));
    solver_name = "internal_cg_" + cg_id;
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

    // setting up preconditioner.
    precon_name = pkg->Param<std::string>("precon_name");

    if (precon_name == "none") {
      precon_type = Precon_Type::NONE;
    } else if (precon_name == "diag") {
      precon_type = Precon_Type::DIAG_SCALING;
    } else if (precon_name == "icc") {
      precon_type = Precon_Type::ICC;
      pcm_name = pkg->Param<std::string>("pcm_name");
    } else {
      precon_type = Precon_Type::ERROR;
    }

    global_num_cg_solvers++;
  }

  std::vector<std::string> SolverState() const {
    if (use_sparse_accessor) {
      if (precon_type == Precon_Type::ICC)
        return std::vector<std::string>({zk, res, apk, pk, spm_name, pcm_name, rhs_name});
      else
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
  int MaxIters() const { return max_iters; }
  int CheckInterval() const { return check_interval; }
  bool GetFail() const { return fail_flag; }
  bool GetWarn() const { return warn_flag; }

  TaskID createTaskList(const TaskID &begin, const int i, TaskRegion &tr,
                        std::shared_ptr<MeshData<Real>> md,
                        std::shared_ptr<MeshData<Real>> mout) {
    auto &solver = tr[i].AddIteration(solver_name);
    solver.SetMaxIterations(max_iters);
    solver.SetCheckInterval(check_interval);
    solver.SetFailWithMaxIterations(fail_flag);
    solver.SetWarnWithMaxIterations(warn_flag);
    return createTaskList(begin, i, tr, solver, md, mout);
  }

  TaskID createTaskList(const TaskID &begin, const int i, TaskRegion &tr,
                        IterativeTasks &solver, std::shared_ptr<MeshData<Real>> md,
                        std::shared_ptr<MeshData<Real>> mout) {
    TaskID none(0);
    TaskList &tl = tr[i];
    RegionCounter reg(solver_name);

    // these are values shared across lists
    r_dot_z.val = 0.0;
    betak = 0.0;
    cg_cntr = 0;
    p_dot_ap.val = 0.0;

    // x=0;
    // b = dV*rho
    // r=b-Ax;
    // z = Minv*r;
    auto init_cg = tl.AddTask(begin, &CG_Solver<SPType>::InitializeCG<MeshData<Real>>,
                              this, md.get(), mout.get());

    auto precon0 =
        tl.AddTask(init_cg, &CG_Solver<SPType>::Precon<MeshData<Real>>, this, md.get());

    auto rdotz0 = tl.AddTask(precon0, &CG_Solver<SPType>::RdotZ<MeshData<Real>>, this,
                             md.get(), &r_dot_z.val);

    tr.AddRegionalDependencies(reg.ID(), i, rdotz0);

    // r.z;
    auto start_global_rz =
        (i == 0 ? tl.AddTask(rdotz0, &AllReduce<Real>::StartReduce, &r_dot_z, MPI_SUM)
                : rdotz0);

    auto finish_global_rz =
        tl.AddTask(start_global_rz, &AllReduce<Real>::CheckReduce, &r_dot_z);

    ////////////////////////////////////////////////////////////////////////////////
    // CG

    // initialization only happens once.

    /////////////////////////////////////////////
    // Iteration starts here.
    // p = beta*p+z; NOTE: at the first iteration, beta=0 so p=z;
    auto axpy1 =
        solver.AddTask(init_cg | finish_global_rz,
                       &CG_Solver<SPType>::Axpy1<MeshData<Real>>, this, md.get());

    // ghost exchange.
    auto send = solver.AddTask(axpy1, parthenon::SendBoundaryBuffers, md);
    auto recv = solver.AddTask(none, parthenon::ReceiveBoundaryBuffers, md);
    auto setb = solver.AddTask(recv | axpy1, parthenon::SetBoundaries, md);

    // matvec Ap = J*p
    auto matvec =
        solver.AddTask(setb, &CG_Solver<SPType>::MatVec<MeshData<Real>>, this, md.get());
    tr.AddRegionalDependencies(reg.ID(), i, matvec);

    // reduce p.Ap
    auto start_global_pAp =
        (i == 0
             ? solver.AddTask(matvec, &AllReduce<Real>::StartReduce, &p_dot_ap, MPI_SUM)
             : matvec);
    auto finish_global_pAp =
        solver.AddTask(start_global_pAp, &AllReduce<Real>::CheckReduce, &p_dot_ap);

    // alpha = r.z/p.Ap
    auto alpha = solver.AddTask(finish_global_pAp | finish_global_rz,
                                &CG_Solver<SPType>::UpdateAlpha, this, i);
    tr.AddRegionalDependencies(reg.ID(), i, alpha);

    // x = x+alpha*p
    // r = r-alpha*Apk
    // z = M^-1*r
    // r.z-new
    auto double_axpy =
        solver.AddTask(alpha, &CG_Solver<SPType>::DoubleAxpy<MeshData<Real>>, this,
                       md.get(), mout.get());
    auto precon = solver.AddTask(double_axpy, &CG_Solver<SPType>::Precon<MeshData<Real>>,
                                 this, md.get());
    auto rdotz = solver.AddTask(precon, &CG_Solver<SPType>::RdotZ<MeshData<Real>>, this,
                                md.get(), &r_dot_z_new.val);
    tr.AddRegionalDependencies(reg.ID(), i, rdotz);

    // reduce p.Ap
    auto start_global_rz_new =
        (i == 0
             ? solver.AddTask(rdotz, &AllReduce<Real>::StartReduce, &r_dot_z_new, MPI_SUM)
             : rdotz);
    auto finish_global_rz_new =
        solver.AddTask(start_global_rz_new, &AllReduce<Real>::CheckReduce, &r_dot_z_new);

    // beta= rz_new/rz
    // and check convergence..
    auto beta = solver.AddTask(finish_global_rz_new, &CG_Solver<SPType>::UpdateInternals,
                               this, i);
    tr.AddRegionalDependencies(reg.ID(), i, beta);

    auto check = solver.SetCompletionTask(beta, &CG_Solver<SPType>::CheckConvergence,
                                          this, i, false);
    tr.AddGlobalDependencies(reg.ID(), i, check);

    return check;
  }

  TaskStatus DoNothing() { return TaskStatus::complete; }
  TaskStatus UpdateAlpha(const int &i) {
    if (i == 0) {
      alphak = r_dot_z.val / p_dot_ap.val;
      r_dot_z_new.val = 0.0;
      p_dot_ap.val = 0.0;
    }
    return TaskStatus::complete;
  }
  TaskStatus UpdateInternals(const int &i) {
    if (i == 0) {
      betak = r_dot_z_new.val / r_dot_z.val;
      res_global = std::sqrt(r_dot_z.val);
      if (cg_cntr == 0) res_global0 = res_global;
      cg_cntr++;
      r_dot_z.val = r_dot_z_new.val;
    }
    return TaskStatus::complete;
  }
  TaskStatus CheckConvergence(const int &i, bool report) {
    if (i != 0) return TaskStatus::complete;
    if (report) {
      if (parthenon::Globals::my_rank == 0) {
        std::cout << parthenon::Globals::my_rank << " its= " << cg_cntr
                  << " relative res: " << res_global / res_global0 << " absolute-res "
                  << res_global << " relerr-tol: " << error_tol << std::endl
                  << std::flush;
      }
    }
    return (res_global / res_global0 < error_tol ? TaskStatus::complete
                                                 : TaskStatus::iterate);
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Utility tasks for solver..
  /////////////////////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus Axpy1(T *u) {
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

    // make local copy.
    const Real bk = betak;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "axpy1", DevExecSpace(), 0, v.GetDim(5) - 1, kb.s, kb.e,
        jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          v(b, ipk, k, j, i) = bk * v(b, ipk, k, j, i) + v(b, izk, k, j, i);
        });

    return TaskStatus::complete;
  } // Axpy1
  /////////////////////////////////////////////////////////////////////////

  template <typename T>
  TaskStatus Precon(T *u) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    std::vector<std::string> vars({zk, res, spm_name});

    if (precon_type == Precon_Type::ICC) vars.push_back(pcm_name);

    const auto &v = u->PackVariables(vars, imap);

    // this get cell variable..
    const int izk = imap[zk].first;
    const int ires = imap[res].first;
    const int isp_lo = imap[spm_name].first;
    const int isp_hi = imap[spm_name].second;
    int diag;
    if (use_sparse_accessor) {
      diag = sp_accessor.ndiag + isp_lo;
    } else {
      diag = stencil.ndiag;
    }

    // Real sum(0);
    // Real gsum(0);

    bool &r_use_sparse_accessor = use_sparse_accessor;
    Stencil<Real> &r_stencil = stencil;

    switch (precon_type) {
    case Precon_Type::NONE:
      parthenon::par_for(
          parthenon::loop_pattern_mdrange_tag, "noprecon", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            // z=r/J_ii
            v(b, izk, k, j, i) = v(b, ires, k, j, i);
          });
      break;
    case Precon_Type::DIAG_SCALING:
      parthenon::par_for(
          parthenon::loop_pattern_mdrange_tag, "diag_scaling", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            // z=r/J_ii
            Real J_ii = (r_use_sparse_accessor ? v(b, diag, k, j, i) : r_stencil.w(diag));
            v(b, izk, k, j, i) = v(b, ires, k, j, i) / J_ii;
          });
      break;
    case Precon_Type::ICC: {
      int nx = (ib.e - ib.s) + 1;
      int ny = (jb.e - jb.s) + 1;
      // int nz = (kb.e - kb.s) + 1;
      int nxny = nx * ny;
      const auto ioff = sp_accessor.ioff;
      const auto joff = sp_accessor.joff;
      const auto koff = sp_accessor.koff;

      const int ipcm_lo = imap[pcm_name].first;
      const int ipcm_hi = imap[pcm_name].second;
      int pc_diag = sp_accessor.ndiag + ipcm_lo;

      // int nstencil = sp_accessor.nstencil;

      // first copy r into z.
      parthenon::par_for(
          parthenon::loop_pattern_mdrange_tag, "noprecon", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            // z=r/J_ii
            // int l = (i - ib.s) + nx * (j - jb.s) + nxny * (k - kb.s);
            v(b, izk, k, j, i) = v(b, ires, k, j, i);
          });

      // step1: x=(D+L)^-1 *r
      for (int b = 0; b < v.GetDim(5); ++b) {
        for (int k = kb.s; k < kb.e + 1; ++k) {
          for (int j = jb.s; j < jb.e + 1; ++j) {
            for (int i = ib.s; i < ib.e + 1; ++i) {
              Real sum(0);

              int l = (i - ib.s) + nx * (j - jb.s) + nxny * (k - kb.s);
              for (int col = ipcm_lo; col <= ipcm_hi; col++) { // m-loop
                // a(l,m) = v(col,k,j,i)
                // a(m,l) = v(off_inv+ipcm_lo,k2,j2,i2)
                const int off = col - ipcm_lo;
                // get neighbor id.
                int i2 = i + ioff(off);
                int j2 = j + joff(off);
                int k2 = k + koff(off);
                int m = (i2 - ib.s) + nx * (j2 - jb.s) + nxny * (k2 - kb.s);
                if (m < l) {
                  sum += v(b, col, k, j, i) * v(b, izk, k2, j2, i2);
                }
              } // ncol
              v(b, izk, k, j, i) = (v(b, izk, k, j, i) - sum) / v(b, pc_diag, k, j, i);
            } // i
          }   // j
        }     // k
      }       // b
#if 0
      //step2 y = Dx
      for(int b=0; b<v.GetDim(5); ++b) {
        for(int k=kb.s; k<kb.e+1; ++k) {
          for(int j=jb.s; j<jb.e+1; ++j) {
            for(int i=ib.s; i<ib.e+1; ++i) {
              v(b,izk,k,j,i) *= v(b,pc_diag,k,j,i);
            }//i
          }//j
        }//k
      }//b
#endif
      // step3: z=(D+U)^-1 *y
      for (int b = 0; b < v.GetDim(5); ++b) {
        for (int k = kb.e; k >= kb.s; --k) {
          for (int j = jb.e; j >= jb.s; --j) {
            for (int i = ib.e; i >= ib.s; --i) {
              Real sum(0);

              int l = (i - ib.s) + nx * (j - jb.s) + nxny * (k - kb.s);
              for (int col = ipcm_lo; col <= ipcm_hi; col++) { // m-loop
                // a(l,m) = v(col,k,j,i)
                // a(m,l) = v(off_inv+ipcm_lo,k2,j2,i2)
                const int off = col - ipcm_lo;
                // get neighbor id.
                int i2 = i + ioff(off);
                int j2 = j + joff(off);
                int k2 = k + koff(off);
                int m = (i2 - ib.s) + nx * (j2 - jb.s) + nxny * (k2 - kb.s);
                if (m > l) sum += v(b, col, k, j, i) * v(b, izk, k2, j2, i2);
              } // ncol
              v(b, izk, k, j, i) = (v(b, izk, k, j, i) - sum) / v(b, pc_diag, k, j, i);
              //  std::cout<< " step: " << v(b,izk,k,j,i)
              //   <<" " << v(b,ires,k,j,i)<<std::endl;
            } // i
          }   // j
        }     // k
      }       // b
    }

    break;

    default:
      std::cout << "Preconditiong invalid...." << std::endl;
      throw;
      break;
    }

    return TaskStatus::complete;
  } // Precon
  /////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus RdotZ(T *u, Real *reduce_sum) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({zk, res});
    const auto &v = u->PackVariables(vars, imap);

    // this get cell variable..
    const int izk = imap[zk].first;
    const int ires = imap[res].first;

    // Real sum(0);
    Real gsum(0);

    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "r_dot_z", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          lsum += v(b, ires, k, j, i) * v(b, izk, k, j, i);
        },
        Kokkos::Sum<Real>(gsum));

    *reduce_sum += gsum;
    return TaskStatus::complete;
  } // RdotZ

  /////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus InitializeCG(T *u, T *du) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    int nx = (ib.e - ib.s) + 1;
    int ny = (jb.e - jb.s) + 1;
    // int nz = (kb.e - kb.s) + 1;
    int nxny = nx * ny;

    PackIndexMap imap;
    std::vector<std::string> vars({zk, pk, res, rhs_name, spm_name});

    if (precon_type == Precon_Type::ICC) vars.push_back(pcm_name);

    const auto &v = u->PackVariables(vars, imap);

    // this get cell variable..
    const int izk = imap[zk].first;
    const int ipk = imap[pk].first;
    const int ires = imap[res].first;
    const int irhs = imap[rhs_name].first;
    const int isp_lo = imap[spm_name].first;
    const int isp_hi = imap[spm_name].second;
    const int ipcm_lo = imap[pcm_name].first;
    const int ipcm_hi = imap[pcm_name].second;
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

    // this runs i(inner-most),j,k,b(outer-most) in the order..
    const auto ioff = sp_accessor.ioff;
    const auto joff = sp_accessor.joff;
    const auto koff = sp_accessor.koff;
    const auto inv_entries = sp_accessor.inv_entries;
    parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "initialize_cg", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          // x=0
          dv(b, ixk, k, j, i) = 0;

          // res = rhs
          v(b, ires, k, j, i) = v(b, irhs, k, j, i);
        });

    // do incomplete cholesky factorization..
    // pcm="preconditioning matrix"
    if (precon_type == Precon_Type::ICC) {
      //  const auto ioff = sp_accessor.ioff;
      // const auto joff = sp_accessor.joff;
      // const auto koff = sp_accessor.koff;

      // const int ipcm_lo = imap[pcm_name].first;
      // const int ipcm_hi = imap[pcm_name].second;
      int pc_diag = sp_accessor.ndiag + ipcm_lo;

      // int nstencil = sp_accessor.nstencil;

      // copy matrix into precon matrix.
      parthenon::par_for(
          parthenon::loop_pattern_mdrange_tag, "icc_copy", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            for (int n = ipcm_lo, n2 = 0; n <= ipcm_hi; n++, n2++) {
              v(b, n, k, j, i) = v(b, isp_lo + n2, k, j, i);
            } // n
          });

      // This is only works (for now) for 5pt (2d) or 7pt(3d) stencil ...
      // outer loop.
      int left(0), right(2), bottom(3), top(5);

      for (int b = 0; b < v.GetDim(5); ++b) {
        for (int k = kb.s; k < kb.e + 1; ++k) {
          for (int j = jb.s; j < jb.e + 1; ++j) {
            for (int i = ib.s; i < ib.e + 1; ++i) {
              // recurrance.
              // diag_i=diag_i-left_i*right_i-1/diag_i-1-bottom_i*top_j-1/diag_j-1
              Real val = v(b, diag, k, j, i);
              Real val_i = -v(b, ipcm_lo + left, k, j, i) *
                           v(b, ipcm_lo + right, k, j, i - 1) /
                           v(b, pc_diag, k, j, i - 1);
              Real val_j = -v(b, ipcm_lo + bottom, k, j, i) *
                           v(b, ipcm_lo + top, k, j - 1, i) / v(b, pc_diag, k, j - 1, i);
              if (i == ib.s) val_i = 0;
              if (j == jb.s) val_j = 0;

              v(b, diag, k, j, i) = val + val_i + val_j;
            } // i
          }   // j
        }     // k
      }       // b

#if 0
// This is only works (for now) for 5pt (2d) or 7pt(3d) stencil ...
//outer loop.

      for(int b=0; b<v.GetDim(5); ++b) {
        for(int k=kb.s; k<kb.e+1; ++k) {
          for(int j=jb.s; j<jb.e+1; ++j) {
            for(int i=ib.s; i<ib.e+1; ++i) {
              //a(l,l) = v(pc_diag,k,j,i)
              v(b, pc_diag, k, j, i) = sqrt( v(b,pc_diag,k,j,i) );

              Real denom = v(b, pc_diag, k, j, i);
              int l = (i-ib.s)+nx*(j-jb.s)+nxny*(k-kb.s);

              //for( m=(l+1):ncells)
              //{
              //  if (a(m,l)!=0)
              //    a(m,l) = a(m,l)/a(l,l);
              //}//m
              for(int col=ipcm_lo; col<=ipcm_hi; col++) { // m-loop
                // a(l,m) = v(col,k,j,i)
                // a(m,l) = v(off_inv+ipcm_lo,k2,j2,i2)
                const int off = col - ipcm_lo;
                // get neighbor id.
                int i2 = i+ioff(off);
                int j2 = j+joff(off);
                int k2 = k+koff(off);
                int m = (i2-ib.s)+nx*(j2-jb.s)+nxny*(k2-kb.s);
                if( m > l ) {
                  int off_inv = inv_entries(off);
                  int col_inv = ipcm_lo+off_inv;
                  v(b,col_inv,k2,j2,i2) = v(b,col_inv,k2,j2,i2)/denom;
                }//m>l
              }//col

              //for( n=(l+1):ncells)
              //{
              //  for( m=n:ncells)
              //  {
              //    if (a(m,n)!=0)
              //      a(m,n) = a(m,n)-a(m,l)*a(n,l);
              //  }//m
              //}//n
              for(int col=ipcm_lo; col<=ipcm_hi; col++) { //n
                // a(l,n) = v(col,k,j,i);
                // a(n,l) = v(off_inv+ipcm_lo,k2,j2,i2);
                const int off = col - ipcm_lo;
                int off_inv = inv_entries(off);
                int col_inv = ipcm_lo+off_inv;

                // get neighbor id.
                int i2 = i+ioff(off);
                int j2 = j+joff(off);
                int k2 = k+koff(off);
                int n = (i2-ib.s)+nx*(j2-jb.s)+nxny*(k2-kb.s);
                Real a_nl = v(b,off_inv+ipcm_lo,k2,j2,i2);
                if( n> l ) {
                  // only diag is common.. a(m,n) !=0 iff m=n.
                  // so a(n,n) = a(n,n)-a(n,l)*a(n,l);
                  v(b,pc_diag,k2,j2,i2) = v(b,pc_diag,k2,j2,i2)-a_nl*a_nl;
                }//n>l
              }//n
            }//i
          }//j
        }//k
      }//b
#endif
      // because of the sparse matrix, we fill upper trianglar..
      // l<m
      for (int b = 0; b < v.GetDim(5); ++b) {
        for (int k = kb.s; k < kb.e + 1; ++k) {
          for (int j = jb.s; j < jb.e + 1; ++j) {
            for (int i = ib.s; i < ib.e + 1; ++i) {
              int l = (i - ib.s) + nx * (j - jb.s) + nxny * (k - kb.s);
              for (int col = ipcm_lo; col <= ipcm_hi; col++) { // m-loop
                // a(l,m) = v(col,k,j,i)
                // a(m,l) = v(off_inv+ipcm_lo,k2,j2,i2)
                const int off = col - ipcm_lo;
                // get neighbor id.
                int i2 = i + ioff(off);
                int j2 = j + joff(off);
                int k2 = k + koff(off);
                int m = (i2 - ib.s) + nx * (j2 - jb.s) + nxny * (k2 - kb.s);
                if (l < m) {
                  int off_inv = inv_entries(off);
                  int col_inv = ipcm_lo + off_inv;
                  v(b, col, k, j, i) = v(b, col_inv, k2, j2, i2);
                  //           std::cout <<"v-up: " << col << " k " << k << " " << j << "
                  //           " << i
                  //            << " "  << v(b,col,k,j,i)<<std::endl;
                } // m>l
              }   // col
            }     // i
          }       // j
        }         // k
      }           // b
#if 0
      parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "icc_0", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          //a(l,l) = v(pc_diag,k,j,i)
          v(b, pc_diag, k, j, i) = sqrt( v(b,pc_diag,k,j,i) );

          Real denom = v(b, pc_diag, k, j, i);
          int l = (i-ib.s)+nx*(j-jb.s)+nxny*(k-kb.s);

          //for( m=(l+1):ncells)
          //{
          //  if (a(m,l)!=0)
          //    a(m,l) = a(m,l)/a(l,l);
          //}//m
          for(int col=ipcm_lo; col<=ipcm_hi; col++) { // m-loop
            // a(l,m) = v(col,k,j,i)
            // a(m,l) = v(off_inv+ipcm_lo,k2,j2,i2)
            const int off = col - ipcm_lo;
            // get neighbor id.
            int i2 = i+ioff(off);
            int j2 = j+joff(off);
            int k2 = k+koff(off);
            int m = (i2-ib.s)+nx*(j2-jb.s)+nxny*(k2-kb.s);
            if( m > l ) {
              int off_inv = inv_entries[off];
              int col_inv = ipcm_lo+off_inv;
              v(b,col_inv,k2,j2,i2) = v(b,col_inv,k2,j2,i2)/denom;
            }//m>l
          }//col

          //for( n=(l+1):ncells)
          //{
          //  for( m=n:ncells)
          //  {
          //    if (a(m,n)!=0)
          //      a(m,n) = a(m,n)-a(m,l)*a(n,l);
          //  }//m
          //}//n
          for(int col=ipcm_lo; col<=ipcm_hi; col++) { //n
            // a(l,n) = v(col,k,j,i);
            // a(n,l) = v(off_inv+ipcm_lo,k2,j2,i2);
            const int off = col - ipcm_lo;
            int off_inv = inv_entries[off];
            int col_inv = ipcm_lo+off_inv;

            // get neighbor id.
            int i2 = i+ioff(off);
            int j2 = j+joff(off);
            int k2 = k+koff(off);
            int n = (i2-ib.s)+nx*(j2-jb.s)+nxny*(k2-kb.s);
            Real a_nl = v(b,off_inv+ipcm_lo,k2,j2,i2);
            if( n> l ) {
              // only diag is common.. a(m,n) !=0 iff m=n.
              // so a(n,n) = a(n,n)-a(n,l)*a(n,l);
              v(b,pc_diag,k2,j2,i2) = v(b,pc_diag,k2,j2,i2)-a_nl*a_nl;
            }//n>l
          }//n
        });

      // because of the sparse matrix, we fill upper trianglar..
      // l<m
      parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "icc_0", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          int l = (i-ib.s)+nx*(j-jb.s)+nxny*(k-kb.s);
          for(int col=ipcm_lo; col<=ipcm_hi; col++) { // m-loop
            // a(l,m) = v(col,k,j,i)
            // a(m,l) = v(off_inv+ipcm_lo,k2,j2,i2)
            const int off = col - ipcm_lo;
            // get neighbor id.
            int i2 = i+ioff(off);
            int j2 = j+joff(off);
            int k2 = k+koff(off);
            int m = (i2-ib.s)+nx*(j2-jb.s)+nxny*(k2-kb.s);
            if(  l < m ) {
              int off_inv = inv_entries[off];
              int col_inv = ipcm_lo+off_inv;
              v(b,col,k,j,i) = v(b,col_inv,k2,j2,i2);
            }//m>l
          }//col
        });

#endif
    }
    return TaskStatus::complete;
  } // initializeCG;

  /////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus MatVec(T *u) {
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
    if (use_sparse_accessor) {
      bool &r_use_sparse_accessor = use_sparse_accessor;
      SparseMatrixAccessor &r_sp_accessor = sp_accessor;
      Stencil<Real> &r_stencil = stencil;
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, "mat_vec", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
            // ap = A*p;
            v(b, iapk, k, j, i) =
                (r_use_sparse_accessor
                     ? r_sp_accessor.MatVec(v, isp_lo, isp_hi, v, ipk, b, k, j, i)
                     : r_stencil.MatVec(v, ipk, b, k, j, i));

            // p.Ap
            lsum += v(b, ipk, k, j, i) * v(b, iapk, k, j, i);
          },
          Kokkos::Sum<Real>(dot));
    } else {
      Stencil<Real> &r_stencil = stencil;
      parthenon::par_reduce(
          parthenon::loop_pattern_mdrange_tag, "mat_vec", DevExecSpace(), 0,
          v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
            // ap = A*p;
            v(b, iapk, k, j, i) = r_stencil.MatVec(v, ipk, b, k, j, i);

            // p.Ap
            lsum += v(b, ipk, k, j, i) * v(b, iapk, k, j, i);
          },
          Kokkos::Sum<Real>(dot));
    }
    p_dot_ap.val += dot;

    return TaskStatus::complete;
  } // MatVec

  /////////////////////////////////////////////////////////////////////////
  template <typename T>
  TaskStatus DoubleAxpy(T *u, T *du) {
    auto pm = u->GetParentPointer();
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({pk, apk, res});
    const auto &v = u->PackVariables(vars, imap);

    const int ipk = imap[pk].first;
    const int iapk = imap[apk].first;

    const int ires = imap[res].first;

    const std::vector<std::string> var2({sol_name});
    PackIndexMap imap2;
    const auto &dv = du->PackVariables(var2, imap2);
    const int ixk = imap2[sol_name].first;

    // make a local copy so it's captured in the kernel
    const Real ak = alphak;
    parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "double_axpy", DevExecSpace(), 0,
        v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          // x = x+alpha*p
          dv(b, ixk, k, j, i) += ak * v(b, ipk, k, j, i);
          // r = r-alpha*Ap
          v(b, ires, k, j, i) -= ak * v(b, iapk, k, j, i);
        });
    return TaskStatus::complete;
  } // DoubleAxpy

 private:
  AllReduce<Real> p_dot_ap;
  AllReduce<Real> r_dot_z;
  AllReduce<Real> r_dot_z_new;
  Real alphak, betak, res_global, res_global0;

  int cg_cntr;
  std::string solver_name;
  std::string zk, res, apk, xk, pk;
  std::string spm_name, sol_name, rhs_name, precon_name;
  std::string pcm_name;

  Real error_tol;
  Stencil<Real> stencil;
  SparseMatrixAccessor sp_accessor;
  int max_iters, check_interval;
  bool fail_flag, warn_flag;
  bool use_sparse_accessor;
  Precon_Type precon_type;
};

} // namespace solvers
} // namespace parthenon

#endif // SOLVERS_CG_SOLVER_HPP_
