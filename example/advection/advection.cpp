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

#include <memory>
#include <string>
#include <utility>

#include "advection.hpp"
#include "better_boundaries/boundary_conditions.hpp"
#include "better_refinement/better_refinement.hpp"
#include "driver/multistage.hpp"
#include "interface/Params.hpp"
#include "interface/StateDescriptor.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_manager.hpp"
#include "reconstruct/reconstruction.hpp"
#include "bvals/bvals.hpp"

using parthenon::ParthenonManager;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//
namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput>& pin) {
  Packages_t packages;
  packages["Advection"] = Advection::Initialize(pin.get());
  return packages;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Container<Real>& rc = real_containers.Get();
  Variable<Real>& q = rc.Get("advected");

  for (int k=0; k<ncells3; k++) {
    for (int j=0; j<ncells2; j++) {
      for (int i=0; i<ncells1; i++) {
        Real rsq = std::pow(pcoord->x1v(i),2) + std::pow(pcoord->x2v(j),2);
        q(k,j,i) = (rsq < 0.15*0.15 ? 1.0 : 0.0);
      }
    }
  }
}

void ParthenonManager::SetFillDerivedFunctions() {
  parthenon::FillDerivedVariables::SetFillDerivedFunctions(Advection::PreFill, Advection::PostFill);
}

} // namespace parthenon

// *************************************************//
// define the "physics" package Advect, which      *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//
namespace Advection {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Advection");
  Params& params = pkg->AllParams();

  Real cfl = pin->GetOrAddReal("Advection", "cfl", 0.45);
  params.Add("cfl", cfl);
  Real vx  = pin->GetOrAddReal("Advection", "vx", 1.0);
  params.Add("vx", vx);
  Real vy  = pin->GetOrAddReal("Advection", "vy", 1.0);
  params.Add("vy", vy);
  Real refine_tol  = pin->GetOrAddReal("Advection", "refine_tol", 0.3);
  params.Add("refine_tol", refine_tol);
  Real derefine_tol  = pin->GetOrAddReal("Advection", "derefine_tol", 0.03);
  params.Add("derefine_tol", derefine_tol);

  std::string field_name = "advected";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::Graphics, Metadata::FillGhost});
  pkg->AddField(field_name, m);

  field_name = "one_minus_advected";
  m = Metadata({Metadata::Cell, Metadata::Graphics, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  field_name = "one_minus_advected_sq";
  pkg->AddField(field_name, m);

  field_name = "one_minus_sqrt_one_minus_advected_sq";
  pkg->AddField(field_name, m);

  pkg->FillDerived = SquareIt;
  pkg->CheckRefinement = CheckRefinement;
  pkg->EstimateTimestep = EstimateTimestep;

  return pkg;
}

int CheckRefinement(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  // refine on advected, for example.  could also be a derived quantity
  Variable<Real>& v = rc.Get("advected");
  Real vmin = 1.0;
  Real vmax = 0.0;
  for (int k=0; k<pmb->ncells3; k++) {
    for (int j=0; j<pmb->ncells2; j++) {
      for (int i=0; i<pmb->ncells1; i++) {
        vmin = (v(k,j,i) < vmin ? v(k,j,i) : vmin);
        vmax = (v(k,j,i) > vmax ? v(k,j,i) : vmax);
      }
    }
  }
  auto pkg = pmb->packages["Advection"];
  const auto& refine_tol = pkg->Param<Real>("refine_tol");
  const auto& derefine_tol = pkg->Param<Real>("derefine_tol");

  if (vmax > refine_tol && vmin < derefine_tol) return 1;
  if (vmax < derefine_tol) return -1;
  return 0;
}

// demonstrate usage of a "pre" fill derived routine
void PreFill(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  int is = 0; int js = 0; int ks = 0;
  int ie = pmb->ncells1-1; int je = pmb->ncells2-1; int ke = pmb->ncells3-1;
  Variable<Real>& qin = rc.Get("advected");
  Variable<Real>& qout = rc.Get("one_minus_advected");
  for (int i=is; i<=ie; i++) {
    for (int j=js; j<=je; j++) {
      for (int k=ks; k<=ke; k++) {
        qout(k,j,i) = 1.0 - qin(k,j,i);
      }
    }
  }
}

// this is the package registered function to fill derived
void SquareIt(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  int is = 0; int js = 0; int ks = 0;
  int ie = pmb->ncells1-1; int je = pmb->ncells2-1; int ke = pmb->ncells3-1;
  Variable<Real>& qin = rc.Get("one_minus_advected");
  Variable<Real>& qout = rc.Get("one_minus_advected_sq");
  for (int i=is; i<=ie; i++) {
    for (int j=js; j<=je; j++) {
      for (int k=ks; k<=ke; k++) {
        qout(k,j,i) = qin(k,j,i)*qin(k,j,i);
      }
    }
  }
}

// demonstrate usage of a "post" fill derived routine
void PostFill(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  int is = 0; int js = 0; int ks = 0;
  int ie = pmb->ncells1-1; int je = pmb->ncells2-1; int ke = pmb->ncells3-1;
  Variable<Real>& qin = rc.Get("one_minus_advected_sq");
  Variable<Real>& qout = rc.Get("one_minus_sqrt_one_minus_advected_sq");
  for (int i=is; i<=ie; i++) {
    for (int j=js; j<=je; j++) {
      for (int k=ks; k<=ke; k++) {
        qout(k,j,i) = 1.0 - sqrt(qin(k,j,i));
      }
    }
  }  
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestep(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  auto pkg = pmb->packages["Advection"];
  const auto& cfl = pkg->Param<Real>("cfl");
  const auto& vx  = pkg->Param<Real>("vx");
  const auto& vy  = pkg->Param<Real>("vy");

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Real min_dt = std::numeric_limits<Real>::max();
  ParArrayND<Real> dx0("dx0",pmb->ncells1);
  ParArrayND<Real> dx1("dx1",pmb->ncells1);

  // this is obviously overkill for this constant velocity problem
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CenterWidth1(k, j, is, ie, dx0);
      pmb->pcoord->CenterWidth2(k, j, is, ie, dx1);
      for (int i=is; i<=ie; i++) {
        min_dt = std::min(min_dt, dx0(i)/std::abs(vx));
        min_dt = std::min(min_dt, dx1(i)/std::abs(vy));
      }
    }
  }

  return cfl*min_dt;
}

// Compute fluxes at faces given the constant velocity field and 
// some field "advected" that we are pushing around.
// This routine implements all the "physics" in this example
TaskStatus CalculateFluxes(Container<Real>& rc) {
  MeshBlock *pmb = rc.pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  Variable<Real>& q = rc.Get("advected");
  auto pkg = pmb->packages["Advection"];
  const auto& vx  = pkg->Param<Real>("vx");
  const auto& vy  = pkg->Param<Real>("vy");

  int maxdim = std::max(std::max(pmb->ncells1, pmb->ncells2), pmb->ncells3);
  ParArrayND<Real> ql("ql",maxdim);
  ParArrayND<Real> qr("qr",maxdim);
  ParArrayND<Real> qltemp("qltemp",maxdim);

  // get x-fluxes
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      // get reconstructed state on faces
      pmb->precon->DonorCellX1(k, j, is-1, ie+1, q.data, ql, qr);
      if (vx > 0.0) {
        for (int i=is; i<=ie+1; i++) {
          q.flux[0](k,j,i) = ql(i)*vx;
        }
      } else {
        for (int i=is; i<=ie+1; i++) {
          q.flux[0](k,j,i) = qr(i)*vx;
        }
      }
    }
  }
  // get y-fluxes
  if (pmb->pmy_mesh->ndim>=2) {
    for (int k=ks; k<=ke; k++) {
      pmb->precon->DonorCellX2(k, js-1, is, ie, q.data, ql, qr);
      for (int j=js; j<=je+1; j++) {
        pmb->precon->DonorCellX2(k, j, is, ie, q.data, qltemp, qr);
        if (vy > 0.0) {
          for (int i=is; i<=ie; i++) {
            q.flux[1](k,j,i) = ql(i)*vy;
          }
        } else {
          for (int i=is; i<=ie; i++) {
            q.flux[1](k,j,i) = qr(i)*vy;
          }
        }
        auto temp = ql;
        ql = qltemp;
        qltemp = temp;
      }
    }
  }

  // TODO(jcd): implement z-fluxes

  return TaskStatus::success;

}

} // namespace Advection

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//
// first some helper tasks
TaskStatus UpdateContainer(MeshBlock *pmb, int stage, std::vector<std::string>& stage_name, Integrator* integrator) {
  //const Real beta = stage_wghts[stage-1].beta;
  const Real beta = integrator->_beta[stage-1];
  Container<Real>& base = pmb->real_containers.Get();
  Container<Real>& cin = pmb->real_containers.Get(stage_name[stage-1]);
  Container<Real>& cout = pmb->real_containers.Get(stage_name[stage]);
  Container<Real>& dudt = pmb->real_containers.Get("dUdt");
  parthenon::Update::AverageContainers(cin, base, beta);
  parthenon::Update::UpdateContainer(cin, dudt, beta*pmb->pmy_mesh->dt, cout);
  return TaskStatus::success;
}


TaskList AdvectionDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;
  // we're going to populate our last with multiple kinds of tasks
  // these lambdas just clean up the interface to adding tasks of the relevant kinds
  auto AddMyTask = [&tl, pmb, stage, this] (BlockStageNamesIntegratorTaskFunc func, TaskID dep) {
    return tl.AddTask<BlockStageNamesIntegratorTask>(func,dep,pmb,stage,stage_name,integrator);
  };
  auto AddContainerTask = [&tl] (ContainerTaskFunc func, TaskID dep, Container<Real>& rc) {
    return tl.AddTask<ContainerTask>(func,dep,rc);
  };
  auto AddTwoContainerTask = [&tl] (TwoContainerTaskFunc func, TaskID dep, Container<Real>& rc1, Container<Real>& rc2) {
    return tl.AddTask<TwoContainerTask>(func,dep,rc1,rc2);
  };

  TaskID none(0);
  // first make other useful containers
  if (stage == 1) {
    Container<Real>& base = pmb->real_containers.Get();
    pmb->real_containers.Add("dUdt", base);
    for (int i=1; i<integrator->_nstages; i++) pmb->real_containers.Add(stage_name[i], base);
  }

  Container<Real>& sc0  = pmb->real_containers.Get(stage_name[stage-1]);
  Container<Real>& sc1  = pmb->real_containers.Get(stage_name[stage]);
  Container<Real>& dudt = pmb->real_containers.Get("dUdt");


  auto start_recv = AddContainerTask([](Container<Real>& rc) {
    rc.StartReceiving(parthenon::BoundaryCommSubset::all);
    return TaskStatus::success;
  }, none, sc1);

  auto advect_flux = AddContainerTask(Advection::CalculateFluxes, none, sc0);

  auto send_flux = AddContainerTask([](Container<Real>& rc) {
    rc.SendFluxCorrection();
    return TaskStatus::success;
  }, advect_flux, sc0);

  auto recv_flux = AddContainerTask([](Container<Real>& rc) {
      if (!rc.ReceiveFluxCorrection()) return TaskStatus::fail;
      return TaskStatus::success;
    }, advect_flux, sc0);

  // compute the divergence of fluxes of conserved variables
  auto flux_div = AddTwoContainerTask([](Container<Real>& u, Container<Real>& du) {
    parthenon::Update::FluxDivergence(u, du);
    return TaskStatus::success;
  }, recv_flux, sc0, dudt);

  // apply du/dt to all independent fields in the container
  auto update_container = AddMyTask(UpdateContainer, flux_div);

  // update ghost cells
  auto send = AddContainerTask([](Container<Real>& rc) {
    rc.SendBoundaryBuffers();
    return TaskStatus::success;
  }, update_container, sc1);
  auto recv = AddContainerTask([](Container<Real>& rc) {
    if ( !rc.ReceiveBoundaryBuffers() ) return TaskStatus::fail;
    return TaskStatus::success;
  }, send, sc1);
  auto setC = AddContainerTask([](Container<Real>& rc) {
    rc.SetBoundaries();
    rc.ClearBoundary(parthenon::BoundaryCommSubset::all);
    return TaskStatus::success;
  }, recv, sc1);
  auto prolongBound = AddContainerTask([](Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    BoundaryValues *pbval = pmb->pbval.get();
    pbval->ProlongateBoundaries(0.0, 0.0);
    return TaskStatus::success;
  }, setC, sc1);

  // set physical boundaries
  auto set_bc = AddContainerTask([](Container<Real>& rc) {
    parthenon::ApplyBoundaryConditions(rc);
    return TaskStatus::success;
  }, prolongBound, sc1);

  // fill in derived fields
  auto fill_derived = AddContainerTask([](Container<Real>& rc) {
    parthenon::FillDerivedVariables::FillDerived(rc);
    return TaskStatus::success;
  }, set_bc, sc1);

  // estimate next time step
  if (stage == integrator->_nstages) {
    auto new_dt = AddContainerTask([](Container<Real>& rc) {
      MeshBlock *pmb = rc.pmy_block;
      pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
      return TaskStatus::success;
    }, fill_derived, sc1);

    // Update refinement
    if (pmesh->adaptive) {
      auto tag_refine = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
        pmb->pmr->CheckRefinementCondition();
        return TaskStatus::success;
      }, fill_derived, pmb);
    }
    // Purge stages
    auto purge_stages = tl.AddTask<BlockTask>([](MeshBlock *pmb) {
      pmb->real_containers.PurgeNonBase();
      return TaskStatus::success;
    }, fill_derived, pmb);

  }
  return tl;
}
