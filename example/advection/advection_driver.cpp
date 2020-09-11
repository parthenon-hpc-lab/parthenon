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
#include <vector>

// Local Includes
#include "advection_driver.hpp"
#include "advection_package.hpp"
#include "mesh/mesh_pack.hpp"
#include "parthenon/driver.hpp"

using namespace parthenon::driver::prelude;

namespace advection_example {

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
AdvectionDriver::AdvectionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageBlockTaskDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");
  pin->CheckDesired("Advection", "cfl");
  pin->CheckDesired("Advection", "vx");
  pin->CheckDesired("Advection", "refine_tol");
  pin->CheckDesired("Advection", "derefine_tol");
}

// first some helper tasks
TaskStatus UpdateContainer(std::vector<MeshBlock *> &blocks, const int stage,
                           std::vector<std::string> stage_name, Integrator *integrator) {
  // const Real beta = stage_wghts[stage-1].beta;
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  parthenon::Update::AverageContainers(blocks, stage_name[stage - 1], "base", beta);
  parthenon::Update::UpdateContainer(blocks, stage_name[stage - 1], "dUdt", beta * dt,
                                     stage_name[stage]);
  return TaskStatus::complete;
}

struct BndInfo {
  bool is_used = false;
  int si = 0;
  int ei = 0;
  int sj = 0;
  int ej = 0;
  int sk = 0;
  int ek = 0;
  parthenon::ParArray1D<Real> buf;
};

// send boundary buffers with MeshBlockPack support
// TODO(pgrete) should probaly be moved to the bvals or interface folders
auto SendBoundaryBuffers(std::vector<MeshBlock *> &blocks,
                         const std::string &container_name) -> TaskStatus {
  auto var_pack = parthenon::PackVariablesOnMesh(
      blocks, container_name,
      std::vector<parthenon::MetadataFlag>{parthenon::Metadata::FillGhost});

  // TODO(?) talk about whether the number of buffers should be a compile time const
  const int num_buffers = 56;
  parthenon::ParArray2D<BndInfo> boundary_info("boundary_info", blocks.size(),
                                               num_buffers);
  auto boundary_info_h = Kokkos::create_mirror_view(boundary_info);

  for (int b = 0; b < blocks.size(); b++) {
    auto *pmb = blocks[b];
    auto &rc = pmb->real_containers.Get(container_name);

    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(parthenon::Metadata::FillGhost)) {
        v->resetBoundary();
      }
    }

    int mylevel = pmb->loc.level;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      // TODO(?) currently this only works for a single "Variable" per container.
      // Need to update the buffer sizes so that it matches the packed Variables.
      auto *bd_var_ = rc->GetCellVariableVector()[0]->vbvar->GetBdVar();
      if (bd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed) continue;
      boundary_info_h(b, n).is_used = true;

      if (nb.snb.level == mylevel) {
        IndexDomain interior = IndexDomain::interior;
        const parthenon::IndexShape &cellbounds = pmb->cellbounds;
        boundary_info_h(b, n).si = (nb.ni.ox1 > 0)
                                       ? (cellbounds.ie(interior) - NGHOST + 1)
                                       : cellbounds.is(interior);
        boundary_info_h(b, n).ei = (nb.ni.ox1 < 0)
                                       ? (cellbounds.is(interior) + NGHOST - 1)
                                       : cellbounds.ie(interior);
        boundary_info_h(b, n).sj = (nb.ni.ox2 > 0)
                                       ? (cellbounds.je(interior) - NGHOST + 1)
                                       : cellbounds.js(interior);
        boundary_info_h(b, n).ej = (nb.ni.ox2 < 0)
                                       ? (cellbounds.js(interior) + NGHOST - 1)
                                       : cellbounds.je(interior);
        boundary_info_h(b, n).sk = (nb.ni.ox3 > 0)
                                       ? (cellbounds.ke(interior) - NGHOST + 1)
                                       : cellbounds.ks(interior);
        boundary_info_h(b, n).ek = (nb.ni.ox3 < 0)
                                       ? (cellbounds.ks(interior) + NGHOST - 1)
                                       : cellbounds.ke(interior);
      } else if (nb.snb.level < mylevel) {
        // ssize = LoadBoundaryBufferToCoarser(bd_var_.send[nb.bufid], nb);
      } else {
        // ssize = LoadBoundaryBufferToFiner(bd_var_.send[nb.bufid], nb);
      }
      // on the same process fill the target buffer directly
      if (nb.snb.rank == parthenon::Globals::my_rank) {
        auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
        // TODO(?) again hardcoded 0 index for single Variable
        boundary_info_h(b, n).buf =
            target_block->pbval->bvars[0]->GetBdVar()->recv[nb.targetid];
      } else {
        boundary_info_h(b, n).buf = bd_var_->send[nb.bufid];
      }
    }
  }
  // TODO(?) track which buffers are actually used, extract subview, and only
  // copy/loop over that
  Kokkos::deep_copy(boundary_info, boundary_info_h);

  const int NbNb = blocks.size() * num_buffers;
  const int Nv = var_pack.GetDim(4);

  Kokkos::parallel_for(
      "SendBoundaryBuffers",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), NbNb, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / num_buffers;
        const int n = team_member.league_rank() - b * num_buffers;
        if (boundary_info(b, n).is_used) {
          const int si = boundary_info(b, n).si;
          const int ei = boundary_info(b, n).ei;
          const int sj = boundary_info(b, n).sj;
          const int ej = boundary_info(b, n).ej;
          const int sk = boundary_info(b, n).sk;
          const int ek = boundary_info(b, n).ek;
          const int Ni = ei + 1 - si;
          const int Nj = ej + 1 - sj;
          const int Nk = ek + 1 - sk;
          const int NvNkNj = Nv * Nk * Nj;
          const int NkNj = Nk * Nj;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, NvNkNj), [&](const int idx) {
                const int v = idx / NkNj;
                int k = (idx - v * NkNj) / Nj;
                int j = idx - v * NkNj - k * Nj;
                k += sk;
                j += sj;

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, si, ei + 1), [&](const int i) {
                      boundary_info(b, n).buf(i - si +
                                              Ni * (j - sj + Nj * (k - sk + Nk * v))) =
                          var_pack(b, v, k, j, i);
                    });
              });
        }
      });

  Kokkos::fence();
  for (auto *pmb : blocks) {
    auto &rc = pmb->real_containers.Get(container_name);

    int mylevel = pmb->loc.level;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      // TODO(?) currently this only works for a single "Variable" per container.
      // Need to update the buffer sizes so that it matches the packed Variables.
      auto *bd_var_ = rc->GetCellVariableVector()[0]->vbvar->GetBdVar();
      if (bd_var_->sflag[nb.bufid] == parthenon::BoundaryStatus::completed) continue;

      // on the same rank the data has been directly copied to the target buffer
      if (nb.snb.rank == parthenon::Globals::my_rank) {
        // TODO(?) check performance of FindMeshBlock. Could be caching from call above.
        auto target_block = pmb->pmy_mesh->FindMeshBlock(nb.snb.gid);
        target_block->pbval->bvars[0]->GetBdVar()->flag[nb.targetid] =
            parthenon::BoundaryStatus::arrived;
      } else {
#ifdef MPI_PARALLEL
        MPI_Start(&(bd_var_->req_send[nb.bufid]));
#endif
      }

      bd_var_->sflag[nb.bufid] = parthenon::BoundaryStatus::completed;
    }
  }

  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

auto ReceiveBoundaryBuffers(std::vector<MeshBlock *> &blocks,
                            const std::string &container_name) -> TaskStatus {
  bool ret = true;
  for (auto *pmb : blocks) {
    auto &rc = pmb->real_containers.Get(container_name);
    // receives the boundary
    for (auto &v : rc->GetCellVariableVector()) {
      if (!v->mpiStatus) {
        if (v->IsSet(parthenon::Metadata::FillGhost)) {
          // ret = ret & v->vbvar->ReceiveBoundaryBuffers();
          // In case we have trouble with multiple arrays causing
          // problems with task status, we should comment one line
          // above and uncomment the if block below
          v->resetBoundary();
          // v->mpiStatus = v->vbvar->ReceiveBoundaryBuffers();
          v->mpiStatus = true;
          ret = (ret & v->mpiStatus);
        }
      }
    }
  }

  // TODO(?) reintroduce sparse logic (or merge with above)
  if (ret) return TaskStatus::complete;
  return TaskStatus::incomplete;
}

// set boundaries from buffers with MeshBlockPack support
// TODO(pgrete) should probaly be moved to the bvals or interface folders
auto SetBoundaries(std::vector<MeshBlock *> &blocks, const std::string &container_name)
    -> TaskStatus {
  auto var_pack = parthenon::PackVariablesOnMesh(
      blocks, container_name,
      std::vector<parthenon::MetadataFlag>{parthenon::Metadata::FillGhost});

  // TODO(?) talk about whether the number of buffers should be a compile time const
  const int num_buffers = 56;
  parthenon::ParArray2D<BndInfo> boundary_info("boundary_info", blocks.size(),
                                               num_buffers);
  auto boundary_info_h = Kokkos::create_mirror_view(boundary_info);

  auto CalcIndices = [](int ox, int &s, int &e, const IndexRange &bounds) {
    if (ox == 0) {
      s = bounds.s;
      e = bounds.e;
    } else if (ox > 0) {
      s = bounds.e + 1;
      e = bounds.e + NGHOST;
    } else {
      s = bounds.s - NGHOST;
      e = bounds.s - 1;
    }
  };

  for (int b = 0; b < blocks.size(); b++) {
    auto *pmb = blocks[b];
    auto &rc = pmb->real_containers.Get(container_name);

    int mylevel = pmb->loc.level;
    for (int n = 0; n < pmb->pbval->nneighbor; n++) {
      parthenon::NeighborBlock &nb = pmb->pbval->neighbor[n];
      // TODO(?) currently this only works for a single "Variable" per container.
      // Need to update the buffer sizes so that it matches the packed Variables.
      auto *bd_var_ = rc->GetCellVariableVector()[0]->vbvar->GetBdVar();

      if (nb.snb.level == mylevel) {
        IndexDomain interior = IndexDomain::interior;
        const parthenon::IndexShape &cellbounds = pmb->cellbounds;
        CalcIndices(nb.ni.ox1, boundary_info_h(b, n).si, boundary_info_h(b, n).ei,
                    cellbounds.GetBoundsI(interior));
        CalcIndices(nb.ni.ox2, boundary_info_h(b, n).sj, boundary_info_h(b, n).ej,
                    cellbounds.GetBoundsJ(interior));
        CalcIndices(nb.ni.ox3, boundary_info_h(b, n).sk, boundary_info_h(b, n).ek,
                    cellbounds.GetBoundsK(interior));
      } else if (nb.snb.level < mylevel) {
        // SetBoundaryFromCoarser(bd_var_.recv[nb.bufid], nb);
      } else {
        // SetBoundaryFromFiner(bd_var_.recv[nb.bufid], nb);
      }
      boundary_info_h(b, n).buf = bd_var_->recv[nb.bufid];
      boundary_info_h(b, n).is_used = true;
      // safe to set completed here as the kernel updating all buffers is
      // called immediately afterwards
      bd_var_->flag[nb.bufid] = parthenon::BoundaryStatus::completed;
    }
  }
  Kokkos::deep_copy(boundary_info, boundary_info_h);

  const int NbNb = blocks.size() * num_buffers;
  const int Nv = var_pack.GetDim(4);

  Kokkos::parallel_for(
      "SetBoundaries",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), NbNb, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / num_buffers;
        const int n = team_member.league_rank() - b * num_buffers;
        if (boundary_info(b, n).is_used) {
          const int si = boundary_info(b, n).si;
          const int ei = boundary_info(b, n).ei;
          const int sj = boundary_info(b, n).sj;
          const int ej = boundary_info(b, n).ej;
          const int sk = boundary_info(b, n).sk;
          const int ek = boundary_info(b, n).ek;
          const int Ni = ei + 1 - si;
          const int Nj = ej + 1 - sj;
          const int Nk = ek + 1 - sk;
          const int NvNkNj = Nv * Nk * Nj;
          const int NkNj = Nk * Nj;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team_member, NvNkNj), [&](const int idx) {
                const int v = idx / NkNj;
                int k = (idx - v * NkNj) / Nj;
                int j = idx - v * NkNj - k * Nj;
                k += sk;
                j += sj;

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team_member, si, ei + 1), [&](const int i) {
                      var_pack(b, v, k, j, i) = boundary_info(b, n).buf(
                          i - si + Ni * (j - sj + Nj * (k - sk + Nk * v)));
                    });
              });
        }
      });

  // TODO(?) reintroduce sparse logic (or merge with above)
  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection AdvectionDriver::MakeTaskCollection(std::vector<MeshBlock *> &blocks,
                                                   const int stage) {
  TaskCollection tc;

  TaskID none(0);

  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto *pmb = blocks[i];
    auto &tl = async_region1[i];
    // first make other useful containers
    if (stage == 1) {
      auto &base = pmb->real_containers.Get();
      pmb->real_containers.Add("dUdt", base);
      for (int i = 1; i < integrator->nstages; i++)
        pmb->real_containers.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto &sc0 = pmb->real_containers.Get(stage_name[stage - 1]);
    // pull out a container we'll use to store dU/dt.
    // This is just -flux_divergence in this example
    auto &dudt = pmb->real_containers.Get("dUdt");
    // pull out the container that will hold the updated state
    // effectively, sc1 = sc0 + dudt*dt
    auto &sc1 = pmb->real_containers.Get(stage_name[stage]);

    auto start_recv = tl.AddTask(&Container<Real>::StartReceiving, sc1.get(), none,
                                 BoundaryCommSubset::all);

    auto advect_flux = tl.AddTask(advection_package::CalculateFluxes, none, sc0);

    auto send_flux =
        tl.AddTask(&Container<Real>::SendFluxCorrection, sc0.get(), advect_flux);
    auto recv_flux =
        tl.AddTask(&Container<Real>::ReceiveFluxCorrection, sc0.get(), advect_flux);
  }

  // note that task within this region that contains only a single task list
  // could still be executed in parallel
  TaskRegion &single_tasklist_region = tc.AddRegion(1);
  {
    auto &tl = single_tasklist_region[0];
    // compute the divergence of fluxes of conserved variables
    auto flux_div = tl.AddTask(parthenon::Update::FluxDivergenceMesh, none, blocks,
                               stage_name[stage - 1], "dUdt");
    // apply du/dt to all independent fields in the container
    auto update_container =
        tl.AddTask(UpdateContainer, flux_div, blocks, stage, stage_name, integrator);

    // update ghost cells
    auto send =
        tl.AddTask(SendBoundaryBuffers, update_container, blocks, stage_name[stage]);

    auto recv = tl.AddTask(ReceiveBoundaryBuffers, send, blocks, stage_name[stage]);
    auto fill_from_bufs = tl.AddTask(SetBoundaries, recv, blocks, stage_name[stage]);
  }
  TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto *pmb = blocks[i];
    auto &tl = async_region2[i];
    auto &sc1 = pmb->real_containers.Get(stage_name[stage]);

    auto clear_comm_flags = tl.AddTask(&Container<Real>::ClearBoundary, sc1.get(), none,
                                       BoundaryCommSubset::all);

    auto prolongBound = tl.AddTask(
        [](MeshBlock *pmb) {
          pmb->pbval->ProlongateBoundaries(0.0, 0.0);
          return TaskStatus::complete;
        },
        none, pmb);

    // set physical boundaries
    auto set_bc = tl.AddTask(parthenon::ApplyBoundaryConditions, prolongBound, sc1);

    // fill in derived fields
    auto fill_derived =
        tl.AddTask(parthenon::FillDerivedVariables::FillDerived, set_bc, sc1);

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          [](std::shared_ptr<Container<Real>> &rc) {
            MeshBlock *pmb = rc->pmy_block;
            pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
            return TaskStatus::complete;
          },
          fill_derived, sc1);

      // Update refinement
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            [](MeshBlock *pmb) {
              pmb->pmr->CheckRefinementCondition();
              return TaskStatus::complete;
            },
            fill_derived, pmb);
      }
    }
  }
  return tc;
}

} // namespace advection_example
