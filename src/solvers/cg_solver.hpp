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


#include <string>
#include <vector>

#include "kokkos_abstraction.hpp"

#include "solvers/solver_utils.hpp"

namespace parthenon {

namespace solvers {


/////////////////////////////////////////////////////////////////////////////////////////  
//Utility tasks for solver..
/////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
TaskStatus Axpy1( T* u, Real *beta)
{
  if( parthenon::Globals::my_rank==0)
    std::cout <<"in axpy1"<<std::endl;
  auto pm = u->GetParentPointer();
  const auto &ib = u->GetBoundsI(IndexDomain::interior);
  const auto &jb = u->GetBoundsJ(IndexDomain::interior);
  const auto &kb = u->GetBoundsK(IndexDomain::interior);
  
  PackIndexMap imap;
  const std::vector<std::string> vars({"zk", "pk"});
  const auto &v = u->PackVariables(vars,imap);
  
  // this get cell variable..
  const int izk = imap["zk"].first;
  const int ipk = imap["pk"].first;

  parthenon::par_for(
    DEFAULT_LOOP_PATTERN, "axpy1", DevExecSpace(), 0, v.GetDim(5)-1, 
    kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
      v(b,ipk,k,j,i) = (*beta)*v(b,ipk,k,j,i) + v(b,izk,k,j,i);
    });  
  if( parthenon::Globals::my_rank==0)
    std::cout <<"out axpy1"<<std::endl;
  
  return TaskStatus::complete;
    
}//Axpy1
/////////////////////////////////////////////////////////////////////////
template<typename T> 
TaskStatus DiagScaling(T *u, T*du, Real *reduce_sum)
{
  if( parthenon::Globals::my_rank==0)
    std::cout <<"in diagscaling"<<std::endl;
  
  auto pm = u->GetParentPointer();
  const auto &ib = u->GetBoundsI(IndexDomain::interior);
  const auto &jb = u->GetBoundsJ(IndexDomain::interior);
  const auto &kb = u->GetBoundsK(IndexDomain::interior);

  PackIndexMap imap;
  const std::vector<std::string> vars({"density", "zk", "pk" ,"res","rhs","poisson_sparse_matrix"});
  const auto &v = u->PackVariables(vars,imap);

  // this get cell variable..
  const int irho = imap["density"].first;
  
  const int izk = imap["zk"].first;
  const int ipk = imap["pk"].first;
  const int ires = imap["res"].first;
  const int irhs = imap["rhs"].first;
  const int isp_lo = imap["poisson_sparse_matrix"].first;
  const int isp_hi = imap["poisson_sparse_matrix"].second;
  const int diag = isp_lo+1;
  
  const std::vector<std::string> var2({"xk"});
  PackIndexMap imap2;
  const auto &dv = du->PackVariables(var2, imap2);
  const int ixk = imap2["xk"].first;

  Real sum(0);
  Real gsum(0);

  using PackType = decltype(v);

//  auto coords = GetCoords(pm);
  const int ndim = v.GetNdim();
//  const Real dx = coords.Dx(X1DIR);
//  for (int i = X2DIR; i <= ndim; i++) {
//    const Real dy = coords.Dx(i);
//    PARTHENON_REQUIRE_THROWS(dx == dy,
//                             "DiagScaling requires that DX be equal in all directions.");
//  }
//  const Real dV = std::pow(dx, ndim);

  
  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  const auto &sp_accessor =
    pkg->Param<parthenon::solvers::SparseMatrixAccessor>("sparse_accessor");

  parthenon::par_reduce(
    parthenon::loop_pattern_mdrange_tag, "diag_scaling", DevExecSpace(), 0,
    v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum)
    {
      //r0 = b = dV*rho;(for this test.)
//      v(b,ires,k,j,i) = dV*v(b,irho,k,j,i);
      
      //
      //x=0
      dv(b,ixk,k,j,i) = 0;

      // res = rhs
      v(b,ires,k,j,i) = v(b,irhs,k,j,i);
      
      // z=r/J_ii
      v(b,izk,k,j,i) = v(b,ires,k,j,i)/v(b,diag,k,j,i);
      //p=z
      v(b,ipk,k,j,i) = v(b,izk,k,j,i);
      //r.z
      lsum += v(b,ires,k,j,i)*v(b,izk,k,j,i);
    },Kokkos::Sum<Real>(gsum));  

  *reduce_sum += gsum;
  std::cout <<parthenon::Globals::my_rank<< " diag-scale sum: " << *reduce_sum<<std::endl;
  
  return TaskStatus::complete;

}//DiagScaling

  
/////////////////////////////////////////////////////////////////////////
template<typename T>
TaskStatus MatVec(T* u, Real *reduce_sum)
{
  if( parthenon::Globals::my_rank==0)
    std::cout <<"in matvec: " <<std::endl;
  
  auto pm = u->GetParentPointer();
  const auto &ib = u->GetBoundsI(IndexDomain::interior);
  const auto &jb = u->GetBoundsJ(IndexDomain::interior);
  const auto &kb = u->GetBoundsK(IndexDomain::interior);
    
  PackIndexMap imap;
  const std::vector<std::string> vars({"pk", "apk","poisson_sparse_matrix"});
  const auto &v = u->PackVariables(vars,imap);

  const int ipk  = imap["pk"].first;
  const int iapk = imap["apk"].first;
    
  const int isp_lo = imap["poisson_sparse_matrix"].first;
  const int isp_hi = imap["poisson_sparse_matrix"].second;

  int ndim =v.GetNdim() ;
  Real dot(0);

  StateDescriptor *pkg = pm->packages.Get("poisson_package").get();
  const auto &sp_accessor =
    pkg->Param<parthenon::solvers::SparseMatrixAccessor>("sparse_accessor");

  parthenon::par_reduce(
    parthenon::loop_pattern_mdrange_tag, "mat_vec", DevExecSpace(), 0,
    v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum)
    {
      //ap = A*p;
      v(b,iapk, k,j,i) = sp_accessor.MatVec(v, isp_lo, isp_hi, v, ipk, b, k, j, i);

      //p.Ap
      lsum += v(b,ipk,k,j,i)*v(b,iapk,k,j,i);
    },Kokkos::Sum<Real>(dot));  
  *reduce_sum += dot;

  std::cout <<parthenon::Globals::my_rank<<" matvec: " << *reduce_sum<<std::endl;
  
  return TaskStatus::complete;
}//MatVec
  

/////////////////////////////////////////////////////////////////////////
template<typename T>
TaskStatus DoubleAxpy(T* u, T* du, Real *alphak, Real *reduce_sum)
{
  std::cout <<parthenon::Globals::my_rank<<" in double axpy "<<*alphak<<std::endl;
  
  auto pm = u->GetParentPointer();
  const auto &ib = u->GetBoundsI(IndexDomain::interior);
  const auto &jb = u->GetBoundsJ(IndexDomain::interior);
  const auto &kb = u->GetBoundsK(IndexDomain::interior);
    
  PackIndexMap imap;
  const std::vector<std::string> vars({"pk", "apk", "res", "zk","jac", "poisson_sparse_matrix"});
  const auto &v = u->PackVariables(vars,imap);

  const int ipk  = imap["pk"].first;
  const int iapk = imap["apk"].first;
    
  const int ires = imap["res"].first;
  const int izk  = imap["zk"].first;
    
  const int isp_lo = imap["poisson_sparse_matrix"].first;
  const int diag = isp_lo+1;

  const std::vector<std::string> var2({"xk"});
  PackIndexMap imap2;
  const auto &dv = du->PackVariables(var2, imap2);
  const int ixk = imap2["xk"].first;

  Real sum(0);

  parthenon::par_reduce(
    parthenon::loop_pattern_mdrange_tag, "double_axpy", DevExecSpace(), 0,
    v.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum)
    {
      //x = x+alpha*p
      dv(b, ixk, k, j, i) += *alphak*v(b, ipk, k, j, i);
      //r = r-alpha*Ap
      v(b, ires,k, j, i) -= *alphak*v(b, iapk, k, j, i);
      //z = r/J_ii;(precon..)
      v(b, izk, k, j, i)  = v(b, ires, k, j, i)/v(b, diag, k, j, i);
      //r.z

      lsum += v(b, ires, k, j, i)*v(b, izk, k, j, i);
    },Kokkos::Sum<Real>(sum));
    
  *reduce_sum += sum;
  std::cout<<parthenon::Globals::my_rank <<" doubleaxpy: " << *reduce_sum<<std::endl;
//  std::cin.get();
  
  return TaskStatus::complete;
    
}//DoubleAxpy

template TaskStatus Axpy1<MeshData<Real>>(MeshData<Real> *, Real *);
template TaskStatus DiagScaling<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *, Real *);
template TaskStatus MatVec<MeshData<Real>>(MeshData<Real> *, Real *);
template TaskStatus DoubleAxpy<MeshData<Real>>(MeshData<Real> *, MeshData<Real> *, Real *, Real *);
  
template TaskStatus Axpy1<MeshBlockData<Real>>(MeshBlockData<Real> *, Real *);
template TaskStatus DiagScaling<MeshBlockData<Real>>(MeshBlockData<Real> *, MeshBlockData<Real> *, Real *);
template TaskStatus MatVec<MeshBlockData<Real>>(MeshBlockData<Real> *, Real *);
template TaskStatus DoubleAxpy<MeshBlockData<Real>>(MeshBlockData<Real> *, MeshBlockData<Real> *, Real *, Real *);
  
struct CG_Solver_Helper
{
  void init(std::shared_ptr<StateDescriptor> pkg)
  {
    //add a couple of vectors for solver..
    auto mcdo = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField("zk", mcdo);
    pkg->AddField("res", mcdo);
    pkg->AddField("apk", mcdo);
    pkg->AddField("xk", mcdo);
        
    //ghost exchange required...
    auto mcif = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
    pkg->AddField("pk", mcif);
  }
};
  


  
  
struct CG_Solver
{
    
  TaskID createCGTaskList(TaskID &begin, int &i, int &j,
                          TaskCollection &tc, TaskList &tl,
                          TaskRegion &solver_region, IterativeTasks &solver,
                          std::shared_ptr<MeshData<Real>> md, std::shared_ptr<MeshData<Real>> mdelta)
  {

//    auto &solver = tl.AddIteration("poisson solver");
//    solver.SetMaxIterations(max_iters);
//    solver.SetCheckInterval(check_interval);
//    solver.SetFailWithMaxIterations(fail_flag);a
//    solver.SetWarnWithMaxIterations(warn_flag);

    auto rz0=  tl.AddTask(begin,
                          [](Real *val, Real *beta, int *cntr) {
                            *val=0;
                            *beta=0;
                            *cntr=0;
                            return TaskStatus::complete;
                          },
                          &r_dot_z.val, &betak.val, &cg_cntr);

    // x=0;
    // b = dV*rho
    // r=b-Ax;
    // z = Minv*r;
    auto res0 =
      tl.AddTask( rz0 | begin, DiagScaling<MeshData<Real>>, md.get(), mdelta.get(), &r_dot_z.val);

    // r.z;
    auto start_global_rz =
      (i == 0 ? tl.AddTask(res0,  &AllReduce<Real>::StartReduce, &r_dot_z, MPI_SUM)
       : begin);

    auto finish_global_rz =
      tl.AddTask(start_global_rz, &AllReduce<Real>::CheckReduce, &r_dot_z);

    //synch.
    solver_region.AddRegionalDependencies(j, i, finish_global_rz);
    j++;
    

    

////////////////////////////////////////////////////////////////////////////////
    //CG
    // this will move to somewhere..
    
    // initialization only happens once.

    /////////////////////////////////////////////
    // Iteration starts here.
    // p = beta*p+z;
    auto axpy1 = solver.AddTask(begin, Axpy1<MeshData<Real>>, md.get(), &betak.val);
    
    // matvec Ap = J*p
    auto pAp0=  solver.AddTask(begin,
                               [](Real *val) {
                                 if( parthenon::Globals::my_rank==0)
                                   std::cout <<" in pap0"<<std::endl;
                                 
                                 *val=0;
                                 return TaskStatus::complete;
                               },
                               &p_dot_ap.val);
    
    auto start_recv = solver.AddTask(begin, &MeshData<Real>::StartReceiving, md.get(),
                                     BoundaryCommSubset::all);
    
    //ghost exchange.
    auto send =
      solver.AddTask(axpy1, parthenon::cell_centered_bvars::SendBoundaryBuffers, md);

    auto recv = solver.AddTask(
      start_recv, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, md);

    auto setb =
      solver.AddTask(recv|axpy1, parthenon::cell_centered_bvars::SetBoundaries, md);

    auto clear = solver.AddTask(send | setb, &MeshData<Real>::ClearBoundary, md.get(),
                                BoundaryCommSubset::all);


    auto matvec = solver.AddTask(clear, MatVec<MeshData<Real>>, md.get(), &p_dot_ap.val);
    
    //solver_region.AddRegionalDependencies(j, i, matvec);
    //j++;
    
    // reduce p.Ap
//    auto start_global_pAp =
//      (i == 0 ? solver.AddTask(matvec, &AllReduce<Real>::StartReduce, &p_dot_ap, MPI_SUM)
//       : matvec);

//    auto finish_global_pAp =
//      tl.AddTask(start_global_pAp, &AllReduce<Real>::CheckReduce, &p_dot_ap);


    // reduce p.Ap
    auto start_global_pAp =
      (i == 0 ? solver.AddTask(matvec, &AllReduce<Real>::StartReduce, &p_dot_ap, MPI_SUM)
       : begin);

    auto finish_global_pAp =
      solver.AddTask(start_global_pAp, &AllReduce<Real>::CheckReduce, &p_dot_ap);



    //    solver_region.AddRegionalDependencies(3, i, finish_global_pAp);

    // alpha = r.z/p.Ap
    auto alpha=  solver.AddTask(finish_global_pAp,
                                [](Real *val,Real *rz, Real *pAp, Real *rznew) {
                                  if( parthenon::Globals::my_rank==0)
                                    std::cout <<"in aplha"<<std::endl;
                                  
                                  *val=(*rz)/(*pAp);
                                  *rznew=0;
                                  std::cout<<parthenon::Globals::my_rank <<" alpha: " << *val
                                  << " rz: " << *rz << " pap: " << *pAp<<std::endl;
                                  
                                  return TaskStatus::complete;
                                },
                                &alphak.val, &r_dot_z.val, &p_dot_ap.val, &r_dot_z_new.val );
    
    // x = x+alpha*p
    // r = r-alpha*Apk
    // z = M^-1*r
    // r.z-new

    auto double_axpy = solver.AddTask(alpha, DoubleAxpy<MeshData<Real>>, md.get(), mdelta.get(),
                                      &alphak.val, &r_dot_z_new.val);
    
    // reduce p.Ap
    auto start_global_rz_new =
      (i == 0 ? solver.AddTask(double_axpy, &AllReduce<Real>::StartReduce, &r_dot_z_new, MPI_SUM)
       : begin);

    auto finish_global_rz_new =
      solver.AddTask(start_global_rz_new, &AllReduce<Real>::CheckReduce, &r_dot_z_new);

//    solver_region.AddRegionalDependencies(4, i, finish_global_rz_new);
    
    // beta= rz_new/rz
    // and check convergence..
    auto beta = solver.SetCompletionTask(finish_global_rz_new,
                                         [](Real *beta, Real *rz_new, Real *rz, Real *res_global, Real *gres0, int* cntr)
                                         {
                                           if( parthenon::Globals::my_rank==0)
                                           std::cout <<"in beta "<<std::endl;
                                           
                                           *beta=(*rz_new)/(*rz);
                                           *res_global = sqrt(*rz_new);
                                           
                                           if( *cntr == 0 )
                                             *gres0 = *res_global;

                                           *cntr= *cntr+1;
                                           (*rz) = (*rz_new);
                                           
                                           Real err_tol = 1e-2;//pkg->Param<Real>("error_tolerance");
                                           //Real res0    = 1000;//pkg->Param<Real>("res0");
                                           auto status = (*res_global/(*gres0) < err_tol ?
                                                          TaskStatus::complete : TaskStatus::iterate);
                                           std::cout <<parthenon::Globals::my_rank<< "  " <<*cntr<<" res: "
                                           << *res_global/(*gres0)<< "  " << *res_global
                                           << " err: " << err_tol<<std::endl;
                                           
                                           return status;
                                         },
                                         &betak.val, &r_dot_z_new.val, &r_dot_z.val, &res_global.val, &global_res0, &cg_cntr);

    solver_region.AddRegionalDependencies(j, i, beta);
    j++;
    

    return beta;
    

  }


  private:
    AllReduce<Real> p_dot_ap;
    AllReduce<Real> r_dot_z;

    AllReduce<Real> r_dot_z_new;
    AllReduce<Real> alphak;
    AllReduce<Real> betak;
    AllReduce<Real> res_global;

    int cg_cntr;
    Real global_res0;
  };
  
    
  
}//solvers
}//parthenon

#endif// SOLVERS_CG_SOLVER_HPP_
